"""The generic `http` driver: a workload described by HTTP requests in the config.

    strata:
      driver: http
      url: http://127.0.0.1:2335
      resident: {get: /health, path: loaded}          # loaded = this field is true
      busy:     {get: /status, path: busy}            # optional; else learned from release
      release:
        pre: {post: http://127.0.0.1:11434/_vhotseat/drain?wait={wait}}   # optional:
                                                      # first, e.g. a proxy stops forwarding
        post: /unload?wait={wait}
        always: true                                  # ask even when it looks unloaded
        busy_when: [{status: 409}, {path: status, in: [busy]}]
        ok_when:   [{path: status, in: [unloaded, not loaded]}]
      confirm: {vram_at_most: 0, timeout: 30}         # default: its VRAM ≤ idle_vram_mib
      load: {post: /load, timeout: 300}               # optional
      model: {get: /health, path: model}              # optional: the model it serves,
                                                      # for admission.by_model

Paths are dotted (`a.b.0.c`; a leading `$.` is allowed). `{wait}` in a URL is
replaced by the seconds the release may wait for running work. A request path
starting with http:// goes to that address instead of `url` (a proxy in front
of the server, say).
"""

import time
from dataclasses import dataclass, field

from .. import util
from ..errors import Busy, SwitchError
from .base import Workload

MISSING = object()


def lookup(body, path: str):
    """`a.b.0.c` (or `$.a.b[0].c`) in parsed JSON, MISSING if absent."""
    cur = body
    for part in path.removeprefix("$.").replace("[", ".").replace("]", "").split("."):
        if not part:
            continue
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return MISSING
    return cur


def matches(rule: dict, code, body) -> bool:
    """One `*_when` rule: {status: 409}, {path: x}, {path: x, in: [...]} or {path: x, equals: v}."""
    if "status" in rule and code != rule["status"]:
        return False
    if "path" in rule:
        value = lookup(body, rule["path"])
        if "in" in rule:
            return value in rule["in"]
        if "equals" in rule:
            return value == rule["equals"]
        return value is not MISSING and bool(value)
    return True


def _request(spec: dict):
    method = "POST" if "post" in spec else "GET"
    return method, spec.get("post") or spec.get("get")


@dataclass
class HttpWorkload(Workload):
    spec: dict = field(default_factory=dict)

    def __post_init__(self):
        self.health_path = _request(self.spec["resident"])[1]
        self.release_always = bool(self.spec["release"].get("always", False))

    def _call(self, spec: dict, timeout: float = 5.0, **fmt):
        method, path = _request(spec)
        body = {} if method == "POST" else None
        url = path if path.startswith(("http://", "https://")) else self.url + path
        return util.http(method, url.format(**fmt), body, timeout=timeout)

    def probe(self, gpu: dict) -> dict:
        st = {"service": self.service_active(), "reachable": False, "loaded": False}
        if st["service"]:
            code, body = self._call(self.spec["resident"])
            if code == 200:
                value = lookup(body, self.spec["resident"]["path"])
                st.update(reachable=True, loaded=value is not MISSING and bool(value))
        return st

    def active(self, st: dict, gpu: dict) -> bool:
        # The GPU is the tiebreaker if its API is unreachable but it's on the card.
        return st["loaded"] or gpu[f"{self.name}_mib"] > self.idle_vram_mib

    def describe(self, st: dict) -> str:
        return super().describe(st) or ("model loaded" if st["loaded"] else "model unloaded")

    def model(self) -> str | None:
        spec = self.spec.get("model")
        if not spec or not self.service_active():
            return None
        code, body = self._call(spec)
        value = lookup(body, spec["path"]) if code == 200 else MISSING
        return value if isinstance(value, str) and value else None

    def busy(self) -> str | None:
        spec = self.spec.get("busy")
        if not spec or not self.service_active():
            return None
        code, body = self._call(spec)
        if code != 200:
            return f"{self.label}'s API isn't answering, so whether it's busy is unknown"
        value = lookup(body, spec["path"])
        return f"{self.label} is busy" if value not in (MISSING, False, 0, None) else None

    def release(self, wait: float, timeout: float) -> list[str]:
        spec = self.spec["release"]
        if pre := spec.get("pre"):
            # e.g. a proxy: stop forwarding new requests, let running ones finish (up to wait).
            code, body = self._call(pre, timeout=wait + 30, wait=int(wait))
            if code == 409 or (code is not None and code >= 500 and lookup(body, "status") == "busy"):
                raise Busy(f"{self.label} is still answering (use --wait SECONDS to let it finish)")
            if code != 200:
                raise SwitchError(f"{self.label} {_request(pre)[1].split('?')[0]} failed: {code} {body}")
        start = time.monotonic()
        code, body = self._call(spec, timeout=wait + spec.get("timeout", 30), wait=int(wait))
        if any(matches(r, code, body) for r in spec.get("busy_when", [])):
            raise Busy(self._busy_reason(spec, int(wait), time.monotonic() - start))
        ok = spec.get("ok_when") or [{"status": 200}]
        if not any(matches(r, code, body) for r in ok):
            raise SwitchError(f"{self.label} {_request(spec)[1].split('?')[0]} failed: {code} {body}")
        confirm = self.spec.get("confirm", {})
        limit = confirm.get("vram_at_most", self.idle_vram_mib)
        util.wait_until(lambda: self.vram_mib() <= limit, confirm.get("timeout", 30),
                        f"{self.label}'s engine to leave the GPU")
        return []

    def _busy_reason(self, spec: dict, wait: int, took: float) -> str:
        """Why a release answered busy. "Use --wait" only helps when no wait was
        given; a server that answers busy well before the wait it was asked for
        doesn't wait at all (Strata before engine 0.1.43.1 ignores /unload?wait=)."""
        what = f"{self.label} is answering or loading"
        if not wait:
            return f"{what} (use --wait SECONDS to let it finish)"
        path = _request(spec)[1].split("?")[0]
        if "{wait}" not in _request(spec)[1]:
            return (f"{what}, and its release request ({path}) has no {{wait}} to wait with; "
                    "add ?wait={wait} to it if the server supports one")
        if took < wait / 2:
            when = "at once" if took < 1 else f"after {took:.0f} s"
            return (f"{what}: it answered busy {when} though asked to wait up to {wait} s, "
                    f"so its {path} seems to ignore ?wait= (Strata needs engine 0.1.43.1 or later)")
        return f"{what}, still after waiting {wait} s"

    def can_load(self) -> bool:
        return "load" in self.spec

    def load(self, wait: float) -> None:
        spec = self.spec["load"]
        util.log(f"loading {self.label}'s model")
        # Synchronous: answers once it's up (Strata's cold load reads ~33 GB of experts).
        code, body = self._call(spec, timeout=max(spec.get("timeout", 300), wait))
        if code != 200:
            raise SwitchError(f"{self.label} {_request(spec)[1]} failed: {code} {body}")
