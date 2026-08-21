#!/usr/bin/env python3
"""Session-aware proxy adding disk-backed llama.cpp slot snapshots.

Pi/Zed requests get one serialized operation, an optional slot restore, explicit
prompt caching, and a post-response snapshot. Requests without an explicit
session header use a stable-prefix-derived local affinity; media requests skip
disk snapshots because their prompt prefix is not reusable.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cache_core import build_prefix_payload, cache_filename, cache_key, with_slot_cache


LOGGER = logging.getLogger("pi-llama-cache")
DEFAULT_CACHE_DIR = str(Path.home() / ".llama-slot-cache")
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class SnapshotPlan:
    session_id: str
    slot_id: int
    prefix_key: str
    session_file: Path
    prefix_file: Path
    prefix_payload: dict[str, Any]
    prefix_was_present: bool


@dataclass(frozen=True)
class SlotState:
    slot_id: int
    prefix_key: str
    n_tokens: int


class LlamaCacheProxy:
    def __init__(
        self,
        upstream: str = "http://127.0.0.1:8080",
        cache_dir: str | None = None,
        max_cache_gib: float = 12.0,
        wait_seconds: float = 120.0,
        enable_prefix_seeding: bool = True,
        prefix_seed_delay_seconds: float = 2.0,
    ) -> None:
        parsed = urlsplit(upstream)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("upstream must be an http URL")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or 80
        self.upstream_prefix = parsed.path.rstrip("/")
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = int(max_cache_gib * 1024**3)
        self.wait_seconds = wait_seconds
        self.enable_prefix_seeding = enable_prefix_seeding
        self.prefix_seed_delay_seconds = prefix_seed_delay_seconds
        self.operation_lock = threading.Lock()
        self.session_states: dict[str, SlotState] = {}
        self.prefix_states: dict[str, SlotState] = {}
        self.prefix_seed_lock = threading.Lock()
        self.prefix_seeds_in_flight: set[Path] = set()
        self.foreground_waiters = 0
        self.foreground_waiters_lock = threading.Lock()

    def prepare(self, body: dict[str, Any], session_id: str) -> tuple[dict[str, Any], SnapshotPlan]:
        prefix_key = cache_key(body)
        session_file = self.cache_dir / cache_filename(session_id, body, "session")
        prefix_file = self.cache_dir / cache_filename("prefix", body, "prefix")
        prefix_payload = build_prefix_payload(body)
        hot_state = self._hot_state(self.session_states.get(session_id), prefix_key)
        restored_source = None
        if hot_state is not None:
            slot_id = hot_state.slot_id
            LOGGER.info("reusing hot session %s in slot %d", session_id, slot_id)
        else:
            hot_state = self._hot_state(self.prefix_states.get(prefix_key), prefix_key)
            if hot_state is not None:
                slot_id = hot_state.slot_id
                self._forget_slot(slot_id)
                LOGGER.info("reusing hot prefix %s in slot %d", prefix_key[:12], slot_id)
            else:
                slot_id = self._wait_for_idle_slot()
                self._forget_slot(slot_id)
                sources = [path for path in (session_file, prefix_file) if path.exists()]
                for source in sources:
                    try:
                        self._restore(slot_id, source)
                    except RuntimeError as error:
                        LOGGER.warning("could not restore %s: %s; continuing without disk restore", source.name, error)
                        continue
                    restored_source = source
                    LOGGER.info("restored %s into slot %d", source.name, slot_id)
                    break
        plan = SnapshotPlan(
            session_id=session_id,
            slot_id=slot_id,
            prefix_key=prefix_key,
            session_file=session_file,
            prefix_file=prefix_file,
            prefix_payload=prefix_payload,
            prefix_was_present=prefix_file.exists() and (restored_source is not None or hot_state is not None),
        )
        return with_slot_cache(body, slot_id), plan

    def finish(self, plan: SnapshotPlan, status: int) -> None:
        if status < 200 or status >= 300:
            return
        n_saved = self._save(plan.slot_id, plan.session_file)
        state = SlotState(plan.slot_id, plan.prefix_key, n_saved)
        self.session_states[plan.session_id] = state
        self.prefix_states[plan.prefix_key] = state
        if not plan.prefix_was_present:
            self._schedule_prefix_seed(plan)
        self._prune()

    def _hot_state(self, state: SlotState | None, prefix_key: str) -> SlotState | None:
        if state is None or state.prefix_key != prefix_key:
            return None
        for slot in self._slots():
            if int(slot.get("id", -1)) != state.slot_id:
                continue
            if slot.get("is_processing"):
                return None
            if int(slot.get("n_prompt_tokens") or 0) != state.n_tokens:
                return None
            return state
        return None

    def _forget_slot(self, slot_id: int) -> None:
        self.session_states = {
            key: state for key, state in self.session_states.items() if state.slot_id != slot_id
        }
        self.prefix_states = {
            key: state for key, state in self.prefix_states.items() if state.slot_id != slot_id
        }

    @staticmethod
    def _prefix_seed_payload(body: dict[str, Any]) -> dict[str, Any]:
        payload = build_prefix_payload(body)
        payload.update(
            {
                "add_generation_prompt": False,
                "cache_prompt": False,
                "n_predict": 0,
                "stream": False,
            }
        )
        return payload

    def _schedule_prefix_seed(self, plan: SnapshotPlan) -> None:
        if not self.enable_prefix_seeding:
            return
        if not plan.prefix_payload.get("messages") and not plan.prefix_payload.get("tools"):
            return
        with self.prefix_seed_lock:
            if plan.prefix_file in self.prefix_seeds_in_flight:
                return
            self.prefix_seeds_in_flight.add(plan.prefix_file)
        threading.Thread(
            target=self._seed_prefix,
            args=(plan.prefix_payload, plan.prefix_file, plan.slot_id),
            name="pi-prefix-seed",
            daemon=True,
        ).start()

    def _seed_prefix(self, prefix_payload: dict[str, Any], prefix_file: Path, excluded_slot_id: int) -> None:
        try:
            time.sleep(self.prefix_seed_delay_seconds)
            deadline = time.monotonic() + self.wait_seconds
            while time.monotonic() < deadline:
                if self._has_foreground_waiters():
                    time.sleep(0.5)
                    continue
                if not self.operation_lock.acquire(blocking=False):
                    time.sleep(0.5)
                    continue
                try:
                    reserved_slots = {excluded_slot_id}
                    reserved_slots.update(state.slot_id for state in self.session_states.values())
                    idle_slots = [
                        slot
                        for slot in self._slots()
                        if not slot.get("is_processing") and int(slot.get("id", -1)) not in reserved_slots
                    ]
                    if not idle_slots:
                        time.sleep(1.0)
                        continue
                    slot_id = min(idle_slots, key=lambda slot: int(slot.get("n_prompt_tokens") or 0)).get("id", 0)
                    request = self._prefix_seed_payload(prefix_payload)
                    request["id_slot"] = int(slot_id)
                    self._json_request("POST", "/v1/chat/completions", request)
                    self._save(int(slot_id), prefix_file)
                    self._forget_slot(int(slot_id))
                    self._prune()
                    LOGGER.info("seeded stable prefix -> %s", prefix_file.name)
                    return
                finally:
                    self.operation_lock.release()
            LOGGER.info("prefix seed timed out waiting for a safe idle slot: %s", prefix_file.name)
        except Exception:
            LOGGER.exception("prefix seed failed: %s", prefix_file.name)
        finally:
            with self.prefix_seed_lock:
                self.prefix_seeds_in_flight.discard(prefix_file)

    def _has_foreground_waiters(self) -> bool:
        with self.foreground_waiters_lock:
            return self.foreground_waiters > 0

    @contextmanager
    def foreground_operation(self):
        with self.foreground_waiters_lock:
            self.foreground_waiters += 1
        acquired = False
        try:
            self.operation_lock.acquire()
            acquired = True
            with self.foreground_waiters_lock:
                self.foreground_waiters -= 1
            yield
        finally:
            if not acquired:
                with self.foreground_waiters_lock:
                    self.foreground_waiters -= 1
            if acquired:
                self.operation_lock.release()

    def forward(self, handler: BaseHTTPRequestHandler, method: str, path: str, body: bytes) -> int:
        upstream = HTTPConnection(self.upstream_host, self.upstream_port, timeout=1200)
        headers = self._forward_headers(handler)
        upstream.request(method, f"{self.upstream_prefix}{path}", body=body, headers=headers)
        response = upstream.getresponse()
        setattr(handler, "_proxy_response_started", True)
        handler.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            handler.send_header(key, value)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        try:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                handler.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                handler.wfile.write(chunk)
                handler.wfile.write(b"\r\n")
                handler.wfile.flush()
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()
            return response.status
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.debug("client disconnected while proxying %s %s", method, path)
            return response.status
        finally:
            upstream.close()

    def _forward_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        headers = {"Content-Type": handler.headers.get("Content-Type", "application/json")}
        for name in ("Authorization", "Accept", "User-Agent"):
            value = handler.headers.get(name)
            if value:
                headers[name] = value
        return headers

    def _json_request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        connection = HTTPConnection(self.upstream_host, self.upstream_port, timeout=120)
        try:
            connection.request(
                method,
                f"{self.upstream_prefix}{path}",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"llama {method} {path} returned {response.status}: {raw[:240]!r}")
            return json.loads(raw or b"{}")
        finally:
            connection.close()

    def _slots(self) -> list[dict[str, Any]]:
        connection = HTTPConnection(self.upstream_host, self.upstream_port, timeout=10)
        try:
            connection.request("GET", f"{self.upstream_prefix}/slots")
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"llama GET /slots returned {response.status}")
            data = json.loads(response.read())
            return data if isinstance(data, list) else []
        finally:
            connection.close()

    def _wait_for_idle_slot(self) -> int:
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            slots = [slot for slot in self._slots() if not slot.get("is_processing")]
            if slots:
                return min(slots, key=lambda slot: int(slot.get("n_prompt_tokens") or 0)).get("id", 0)
            time.sleep(0.5)
        raise TimeoutError("no idle llama slot became available")

    def _restore(self, slot_id: int, filename: Path) -> None:
        self._json_request("POST", f"/slots/{slot_id}?action=restore", {"filename": filename.name})

    def _save(self, slot_id: int, target: Path) -> int:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        result = self._json_request(
            "POST",
            f"/slots/{slot_id}?action=save",
            {"filename": temporary.name},
        )
        if int(result.get("n_saved") or 0) <= 0 or not temporary.exists():
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"llama saved no tokens for slot {slot_id}")
        temporary.replace(target)
        n_saved = int(result["n_saved"])
        LOGGER.info("saved slot %d -> %s (%s tokens)", slot_id, target.name, n_saved)
        return n_saved

    def _prune(self) -> None:
        files = sorted(
            self.cache_dir.glob("pi-*.bin"),
            key=lambda path: path.stat().st_mtime,
        )
        total = sum(path.stat().st_size for path in files)
        while total > self.max_cache_bytes and files:
            victim = files.pop(0)
            size = victim.stat().st_size
            victim.unlink(missing_ok=True)
            total -= size
            LOGGER.info("pruned %s (%d bytes)", victim.name, size)


def _session_id(handler: BaseHTTPRequestHandler) -> str | None:
    for header in ("X-Session-Affinity", "X-Pi-Session-Id", "X-Client-Request-Id"):
        value = handler.headers.get(header)
        if value:
            return value.strip()
    return None


def _body_session_id(body: dict[str, Any]) -> str | None:
    for field in ("session_id", "prompt_cache_key"):
        value = body.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _without_proxy_affinity_fields(body: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(body)
    request.pop("session_id", None)
    request.pop("prompt_cache_key", None)
    return request


def _anonymous_session_id(body: dict[str, Any]) -> str:
    return f"anonymous-{cache_key(body)}"


def _has_media(body: dict[str, Any]) -> bool:
    if body.get("images"):
        return True
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(item, dict) and item.get("type") != "text" for item in content):
            return True
    return False


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    proxy: LlamaCacheProxy

    def do_GET(self) -> None:
        try:
            self.proxy.forward(self, "GET", self.path, b"")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def _send_upstream_error(self, error: Exception) -> None:
        LOGGER.exception("upstream request failed")
        if not getattr(self, "_proxy_response_started", False) and not self.wfile.closed:
            self.send_error(503, f"upstream unavailable: {error}")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path.rstrip("/") != "/v1/chat/completions":
            try:
                self.proxy.forward(self, "POST", self.path, raw)
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "request body must be JSON")
            return
        body = _without_proxy_affinity_fields(body)
        session_id = _session_id(self) or _body_session_id(body) or _anonymous_session_id(body)
        if _has_media(body):
            body["cache_prompt"] = True
            try:
                self.proxy.forward(self, "POST", self.path, json.dumps(body).encode("utf-8"))
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        try:
            with self.proxy.foreground_operation():
                request, plan = self.proxy.prepare(body, session_id)
                status = self.proxy.forward(self, "POST", self.path, json.dumps(request).encode("utf-8"))
                try:
                    self.proxy.finish(plan, status)
                except Exception:
                    LOGGER.exception("response succeeded but snapshot finalization failed")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("PI_LLAMA_CACHE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    proxy = LlamaCacheProxy(
        upstream=os.environ.get("PI_LLAMA_UPSTREAM", "http://127.0.0.1:8080"),
        cache_dir=os.environ.get("PI_LLAMA_CACHE_DIR", DEFAULT_CACHE_DIR),
        max_cache_gib=float(os.environ.get("PI_LLAMA_CACHE_MAX_GIB", "12")),
        wait_seconds=float(os.environ.get("PI_LLAMA_CACHE_WAIT_SECONDS", "120")),
        prefix_seed_delay_seconds=float(os.environ.get("PI_LLAMA_CACHE_PREFIX_SEED_DELAY", "2")),
    )
    ProxyHandler.proxy = proxy
    host = os.environ.get("PI_LLAMA_CACHE_HOST", "127.0.0.1")
    port = int(os.environ.get("PI_LLAMA_CACHE_PORT", "8081"))
    server = ThreadingHTTPServer((host, port), ProxyHandler)
    LOGGER.info("listening on http://%s:%d -> http://%s:%d", host, port, proxy.upstream_host, proxy.upstream_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
