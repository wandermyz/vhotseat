"""Fake Strata, ComfyUI, GPU and host for the tests.

The two servers speak just enough of the real APIs (the parts gpu-workload and
comfy-render use) over real HTTP on localhost, so the drivers run unchanged.
The GPU and host fakes read the servers' state: Strata holds 14000 MiB while
loaded; ComfyUI holds whatever its `vram` says.
"""

import base64
import hashlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

STRATA_VRAM = 14000
COMFY_CONTEXT_VRAM = 400   # CUDA context, no models
COMFY_MODEL_VRAM = 7000


class _Server:
    """A threaded HTTP server on a free port; subclasses route requests."""

    def __init__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                return json.loads(raw) if raw else None

            def send_json(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                outer.route(self, "GET", self._url(), None)

            def do_POST(self):
                outer.route(self, "POST", self._url(), self._body())

            def _url(self):
                u = urlsplit(self.path)
                return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.calls: list[str] = []
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeStrata(_Server):
    """/health, /unload?wait=N, /load. `on_load` is its before_load hook: return
    False to refuse (Strata answers 503 and doesn't load)."""

    def __init__(self):
        self.loaded = False
        self.loading = False
        self.model = "qwen3.8-flash-next-iq3_xxs"
        self.answering = 0          # requests being answered
        self.ignores_wait = False   # /unload answers busy at once (Strata before engine 0.1.43.1)
        self.load_s = 0.05
        self.on_load = lambda: True
        super().__init__()

    @property
    def vram(self):
        return STRATA_VRAM if self.loaded else 0

    def _load(self) -> bool:
        """Load the model through the before_load hook, as Strata does; False if refused."""
        if self.loaded:
            return True
        self.loading = True
        try:
            if not self.on_load():
                return False
            time.sleep(self.load_s)
            self.loaded = True
            return True
        finally:
            self.loading = False

    def request(self, seconds: float = 0.0) -> int:
        """A chat request: wakes the lazy model (503 if the hook refuses), then
        answers for `seconds`. Returns the HTTP status."""
        self.answering += 1
        try:
            if not self._load():
                return 503
            time.sleep(seconds)
            return 200
        finally:
            self.answering -= 1

    def request_async(self, seconds: float = 0.0):
        """request() on a thread; .result holds the status once it's joined."""
        t = threading.Thread(target=lambda: setattr(t, "result", self.request(seconds)), daemon=True)
        t.result = None
        t.start()
        return t

    def route(self, h, method, url, body):
        path, q = url
        self.calls.append(f"{method} {path}")
        if path == "/health":
            return h.send_json(200, {"status": "ok", "loaded": self.loaded, "model": self.model})
        if path == "/_vram":  # for a gpu-workload running in a subprocess (tests/run_cli.py)
            return h.send_json(200, {"vram": self.vram})
        if path == "/unload" and method == "POST":
            wait = 0 if self.ignores_wait else float(q.get("wait", 0))
            deadline = time.monotonic() + wait
            while (self.answering or self.loading) and time.monotonic() < deadline:
                time.sleep(0.01)
            if self.answering or self.loading:
                return h.send_json(409, {"status": "busy"})
            was, self.loaded = self.loaded, False
            return h.send_json(200, {"status": "unloaded" if was else "not loaded"})
        if path == "/load" and method == "POST":
            if not self._load():
                return h.send_json(503, {"error": "before_load refused"})
            return h.send_json(200, {"status": "loaded"})
        h.send_json(404, {"error": path})


class FakeComfy(_Server):
    """/queue, /free, /history, /prompt and the /ws progress websocket.

    Jobs run one at a time on a worker thread. `guard(prompt_id)`, if set, runs
    first, as the guard node does; it returns an error string to fail the job.
    What a job does comes from its workflow's class types:
      Fail     raises during execution        Reject   fails /prompt validation
      Slow     takes `slow_s` seconds          SaveImage saves one file
    """

    def __init__(self):
        self.vram = COMFY_CONTEXT_VRAM
        self.running: list[str] = []
        self.pending: list[str] = []
        self.history: dict[str, dict] = {}
        self.workflows: dict[str, dict] = {}
        self.clients: dict[str, dict] = {}       # prompt_id -> client_id
        self.down = False                       # /queue answers 500
        self.stuck_vram = None                  # /free leaves this much (an extension that doesn't unload)
        self.job_arrives_on_free = False        # a job is queued just as /free arrives
        self.job_s = 0.05
        self.slow_s = 1.0
        self.guard = None
        self.lose_history = False
        self.models_dir = "/comfy/models/checkpoints"
        self.sockets: dict[str, list] = {}      # client_id -> [socket]
        self._work = threading.Event()
        super().__init__()
        threading.Thread(target=self._worker, daemon=True).start()

    # --- state helpers used by tests ---

    def queue_job(self, workflow=None) -> str:
        pid = str(uuid.uuid4())
        self.workflows[pid] = workflow or {"1": {"class_type": "SaveImage", "inputs": {}}}
        self.pending.append(pid)
        self._work.set()
        return pid

    def finish_history(self, pid: str, *, ago: float = 0.0, ok: bool = True, outputs=None, messages=None):
        t = int((time.time() - ago) * 1000)
        msgs = messages or [["execution_start", {"prompt_id": pid, "timestamp": t - 10}],
                            ["execution_success" if ok else "execution_error", {"prompt_id": pid, "timestamp": t}]]
        self.history[pid] = {"prompt": [0, pid, self.workflows.get(pid, {})], "outputs": outputs or {},
                             "status": {"status_str": "success" if ok else "error", "completed": ok,
                                        "messages": msgs}}

    def restarted(self):
        """What a restart does: models, history and queue gone."""
        self.vram = COMFY_CONTEXT_VRAM
        self.stuck_vram = None
        self.history.clear()

    # --- worker ---

    def _send(self, pid, kind, data):
        for sock in self.sockets.get(self.clients.get(pid), []):
            payload = json.dumps({"type": kind, "data": {"prompt_id": pid, **data}}).encode()
            head = bytes([0x81]) + (bytes([len(payload)]) if len(payload) < 126
                                    else bytes([126]) + len(payload).to_bytes(2, "big"))
            try:
                sock.sendall(head + payload)
            except OSError:
                pass

    def _worker(self):
        while True:
            self._work.wait(0.05)
            self._work.clear()
            while self.pending:
                pid = self.pending[0]
                self.running.append(pid)
                self.pending.pop(0)
                self._run(pid)
                self.running.remove(pid)

    def _run(self, pid):
        wf = self.workflows[pid]
        if self.guard is not None and (err := self.guard(pid)):
            self.finish_history(pid, ok=False, messages=[
                ["execution_start", {"prompt_id": pid, "timestamp": int(time.time() * 1000)}],
                ["execution_error", {"prompt_id": pid, "node_id": "1", "node_type": "",
                                     "exception_message": f"GPU busy, job not run: {err}",
                                     "timestamp": int(time.time() * 1000)}]])
            return
        self.vram = COMFY_MODEL_VRAM
        outputs = {}
        for nid, node in wf.items():
            cls = node.get("class_type")
            self._send(pid, "executing", {"node": nid})
            if cls == "Slow":
                for step in range(1, 4):
                    time.sleep(self.slow_s / 3)
                    self._send(pid, "progress", {"node": nid, "value": step, "max": 3})
            if cls == "Fail":
                self.finish_history(pid, ok=False, messages=[
                    ["execution_error", {"prompt_id": pid, "node_id": nid, "node_type": cls,
                                         "exception_message": "boom", "traceback": ["  File x\n"],
                                         "timestamp": int(time.time() * 1000)}]])
                return
            if cls == "SaveImage":
                outputs[nid] = {"images": [{"filename": f"{pid[:8]}_{nid}.png", "subfolder": "", "type": "output"}]}
                self._send(pid, "executed", {"node": nid, "display_node": nid, "output": outputs[nid]})
        time.sleep(self.job_s)
        self._send(pid, "execution_success", {"timestamp": int(time.time() * 1000)})
        if self.lose_history:     # restarted between the job ending and anyone reading /history
            return
        self.finish_history(pid, outputs=outputs)

    # --- HTTP ---

    def route(self, h, method, url, body):
        path, q = url
        self.calls.append(f"{method} {path}")
        if path == "/ws":
            return self._websocket(h, q.get("clientId", ""))
        if path == "/_vram":
            return h.send_json(200, {"vram": self.vram})
        if path == "/internal/folder_paths":
            return h.send_json(200, {"checkpoints": [self.models_dir, "/comfy/output/checkpoints"]})
        if path == "/queue":
            if self.down:
                return h.send_json(500, {"error": "down"})
            return h.send_json(200, {"queue_running": [[0, p] for p in self.running],
                                     "queue_pending": [[1, p] for p in self.pending]})
        if path == "/free" and method == "POST":
            if self.job_arrives_on_free:
                self.queue_job({"1": {"class_type": "Slow", "inputs": {}}})
            if not (self.running or self.pending):   # /free only acts between jobs
                self.vram = self.stuck_vram if self.stuck_vram is not None else COMFY_CONTEXT_VRAM
            return h.send_json(200, {})
        if path == "/history":
            if q.get("max_items") == "1" and self.history:
                pid = list(self.history)[-1]
                return h.send_json(200, {pid: self.history[pid]})
            return h.send_json(200, {} if q.get("max_items") else self.history)
        if path.startswith("/history/"):
            pid = path.split("/", 2)[2]
            return h.send_json(200, {pid: self.history[pid]} if pid in self.history else {})
        if path == "/prompt" and method == "POST":
            wf = body["prompt"]
            bad = [nid for nid, n in wf.items() if n.get("class_type") == "Reject"]
            if bad:
                return h.send_json(400, {"error": {"message": "Prompt outputs failed validation"},
                                         "node_errors": {bad[0]: {"class_type": "Reject", "errors": [
                                             {"message": "Value not in list", "details": "ckpt_name: nope"}]}}})
            pid = self.queue_job(wf)
            self.clients[pid] = body.get("client_id")
            return h.send_json(200, {"prompt_id": pid, "number": 0, "node_errors": {}})
        h.send_json(404, {"error": path})

    def _websocket(self, h, client_id):
        key = h.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        h.send_response(101)
        h.send_header("Upgrade", "websocket")
        h.send_header("Connection", "Upgrade")
        h.send_header("Sec-WebSocket-Accept", accept.decode())
        h.end_headers()
        h.wfile.flush()
        self.sockets.setdefault(client_id, []).append(h.connection)
        h.close_connection = True
        while h.connection in self.sockets[client_id]:
            try:
                if not h.connection.recv(1024):
                    break
            except OSError:
                break


class FakeGpu:
    """A GPU backend whose processes are the fake servers' allocations."""

    name = "fake"

    def __init__(self, strata: FakeStrata, comfy: FakeComfy):
        self.strata, self.comfy = strata, comfy
        self.other = 0
        self.error = None

    def snapshot(self, selectors: dict) -> dict:
        if self.error:
            return {"total_mib": None, "free_mib": None, **{f"{n}_mib": 0 for n in selectors}, "other_mib": 0,
                    "error": self.error}
        if getattr(self, "strata_as_other", False):   # a workload the backend can't attribute
            used = {"strata_mib": 0, "comfy_mib": self.comfy.vram, "other_mib": self.other + self.strata.vram}
        else:
            used = {"strata_mib": self.strata.vram, "comfy_mib": self.comfy.vram, "other_mib": self.other}
        return {"total_mib": 16303, "free_mib": 16303 - sum(used.values()), **used}


class _FakeService:
    def __init__(self, host, name):
        self.host, self.name, self.unit = host, name, f"{name}.service"

    def is_active(self):
        return self.host.active.get(self.name, False)

    def start(self):
        self.host.starts.append(self.name)
        self.host.active[self.name] = True

    def restart(self):
        self.host.restarts.append(self.name)
        self.host.active[self.name] = True
        if cb := self.host.on_restart.get(self.name):
            cb()

    def memory(self):
        return dict(self.host.mem.get(self.name, {"ram_mib": None, "swap_mib": None, "file_mib": None}))


class FakeHost:
    """systemd --user stand-in: units start active; restarting `comfyui` calls
    `on_restart["comfyui"]`. `service(unit)` gives the object a workload holds."""

    def __init__(self):
        self.active = {"strata": True, "comfyui": True}
        self.mem = {"strata": {"ram_mib": 100, "swap_mib": 0, "file_mib": 0},
                    "comfyui": {"ram_mib": 800, "swap_mib": 0, "file_mib": 0}}
        self.available_mib = 50000
        self.on_restart: dict = {}
        self.restarts: list[str] = []
        self.starts: list[str] = []

    def service(self, unit: str):
        return _FakeService(self, unit)

    def machine_ram(self):
        return {"total_mib": 61952, "available_mib": self.available_mib, "swap_total_mib": 8192, "swap_used_mib": 0}

    def install(self, arbiter):
        """Give every workload in `arbiter` its fake service."""
        for w in arbiter.workloads.values():
            w.service = self.service(w.service.name)
        arbiter.machine_ram = self.machine_ram
        return arbiter
