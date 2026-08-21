import json
import os
import runpy
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import cache_proxy
from cache_core import cache_filename, cache_key
from cache_proxy import (
    LlamaCacheProxy,
    ProxyHandler,
    SnapshotPlan,
    SlotState,
    _anonymous_session_id,
    _body_session_id,
    _has_media,
    _session_id,
    _without_proxy_affinity_fields,
)


class FakeLlamaHandler(BaseHTTPRequestHandler):
    cache_dir = None
    restored = []
    saved = []
    restore_failed = False
    slot_tokens = 0

    def do_GET(self):
        if self.path == "/slots":
            self._json([{"id": 0, "is_processing": False, "n_prompt_tokens": self.slot_tokens}])
            return
        self._json({"status": "ok"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if "action=restore" in self.path:
            self.restored.append(payload["filename"])
            if self.restore_failed:
                self._json({"error": "incompatible snapshot"}, status=500)
                return
            type(self).slot_tokens = 10
            self._json({"n_restored": 10})
            return
        if "action=save" in self.path:
            filename = payload["filename"]
            self.saved.append(filename)
            Path(self.cache_dir, filename).write_bytes(b"snapshot")
            type(self).slot_tokens = 10
            self._json({"n_saved": 10})
            return
        self._json({"status": "ok"})

    def _json(self, value, status=200):
        raw = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args):
        pass


class UnavailableProxy:
    @contextmanager
    def foreground_operation(self):
        yield

    def prepare(self, *_args):
        raise ConnectionRefusedError("llama is unavailable")


class CapturingProxy:
    def __init__(self):
        self.session_ids = []
        self.forwarded = []

    @contextmanager
    def foreground_operation(self):
        yield

    def prepare(self, body, session_id):
        self.session_ids.append(session_id)
        return body, None

    def forward(self, handler, *_args):
        self.forwarded.append(_args)
        raw = b"{}"
        setattr(handler, "_proxy_response_started", True)
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
        return 200

    def finish(self, *_args):
        pass


class FailingForwardProxy(CapturingProxy):
    def forward(self, *_args):
        raise OSError("upstream unavailable")


class FinishFailingProxy(CapturingProxy):
    def finish(self, *_args):
        raise RuntimeError("snapshot finalization failed")


class FakeResponse:
    def __init__(self, status=200, raw=b"{}", headers=None, chunks=None):
        self.status = status
        self.reason = "OK"
        self._raw = raw
        self._headers = headers or []
        self._chunks = list(chunks) if chunks is not None else None

    def getheaders(self):
        return self._headers

    def read(self, _size=None):
        if self._chunks is None:
            return self._raw
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class FakeConnection:
    def __init__(self, _host, _port, timeout=None, response=None):
        self.timeout = timeout
        self.response = response or FakeResponse()
        self.requests = []
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class RecordingWriter:
    def __init__(self, broken=False):
        self.broken = broken
        self.closed = False
        self.data = []

    def write(self, data):
        if self.broken:
            raise BrokenPipeError("client disconnected")
        self.data.append(data)
        return len(data)

    def flush(self):
        pass


class RecordingHandler:
    def __init__(self, headers=None, broken=False):
        self.headers = headers or {}
        self.wfile = RecordingWriter(broken=broken)
        self.responses = []
        self.sent_headers = []
        self.errors = []
        self.ended = False

    def send_response(self, status, reason=None):
        self.responses.append((status, reason))

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        self.ended = True

    def send_error(self, status, message):
        self.errors.append((status, message))


class CacheProxyTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        FakeLlamaHandler.cache_dir = self.tempdir.name
        FakeLlamaHandler.restored = []
        FakeLlamaHandler.saved = []
        FakeLlamaHandler.restore_failed = False
        FakeLlamaHandler.slot_tokens = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=1,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        self.body = {
            "model": "qwen3.8:27b",
            "messages": [
                {"role": "system", "content": "project rules"},
                {"role": "user", "content": "request"},
            ],
            "tools": [],
        }

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tempdir.cleanup()

    def test_session_snapshot_is_saved_and_prefix_snapshot_is_reused(self):
        request, plan = self.proxy.prepare(self.body, "session-a")
        self.assertEqual(request["id_slot"], 0)
        self.proxy.finish(plan, 200)

        prefix_file = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        prefix_file.write_bytes(b"prefix snapshot")
        files = {path.name for path in Path(self.tempdir.name).glob("local-llm-*.bin")}
        self.assertEqual(len(files), 2)

        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        cold_proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=1,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        cold_proxy.prepare(other_body, "session-b")
        self.assertEqual(len(FakeLlamaHandler.restored), 1)

    def test_incompatible_snapshot_falls_back_to_current_slot(self):
        request, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        prefix_file = Path(self.tempdir.name, cache_filename("prefix", self.body, "prefix"))
        prefix_file.write_bytes(b"prefix snapshot")
        FakeLlamaHandler.restore_failed = True

        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        cold_proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=1,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        request, plan = cold_proxy.prepare(other_body, "session-b")

        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(len(FakeLlamaHandler.restored), 1)

    def test_same_session_reuses_hot_slot_without_disk_restore(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(plan, 200)
        restored_before = len(FakeLlamaHandler.restored)

        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        request, _ = self.proxy.prepare(other_body, "session-a")

        self.assertEqual(request["id_slot"], 0)
        self.assertEqual(len(FakeLlamaHandler.restored), restored_before)

    def test_invalid_upstream_url_is_rejected(self):
        with self.assertRaises(ValueError):
            LlamaCacheProxy(upstream="https://example.test")

    def test_finish_ignores_unsuccessful_response(self):
        _, plan = self.proxy.prepare(self.body, "session-a")

        self.proxy.finish(plan, 500)

        self.assertEqual(self.proxy.session_states, {})
        self.assertEqual(FakeLlamaHandler.saved, [])

    def test_finish_does_not_seed_when_prefix_snapshot_is_present(self):
        _, plan = self.proxy.prepare(self.body, "session-a")
        plan = SnapshotPlan(
            plan.session_id,
            plan.slot_id,
            plan.prefix_key,
            plan.session_file,
            plan.prefix_file,
            plan.prefix_payload,
            True,
        )
        self.proxy._schedule_prefix_seed = Mock()

        self.proxy.finish(plan, 200)

        self.proxy._schedule_prefix_seed.assert_not_called()

    def test_hot_state_rejects_stale_slot_states(self):
        state = SlotState(1, cache_key(self.body), 10)

        self.assertIsNone(self.proxy._hot_state(None, state.prefix_key))
        self.assertIsNone(self.proxy._hot_state(state, "different-prefix"))

        with patch.object(self.proxy, "_slots", return_value=[{"id": 0, "is_processing": False}]):
            self.assertIsNone(self.proxy._hot_state(state, state.prefix_key))
        with patch.object(self.proxy, "_slots", return_value=[{"id": 1, "is_processing": True}]):
            self.assertIsNone(self.proxy._hot_state(state, state.prefix_key))
        with patch.object(
            self.proxy,
            "_slots",
            return_value=[{"id": 1, "is_processing": False, "n_prompt_tokens": 9}],
        ):
            self.assertIsNone(self.proxy._hot_state(state, state.prefix_key))
        with patch.object(
            self.proxy,
            "_slots",
            return_value=[{"id": 1, "is_processing": False, "n_prompt_tokens": 10}],
        ):
            self.assertEqual(self.proxy._hot_state(state, state.prefix_key), state)

    def test_forget_slot_removes_only_that_session(self):
        self.proxy.session_states = {
            "session-a": SlotState(0, "a", 1),
            "session-b": SlotState(1, "b", 2),
        }

        self.proxy._forget_slot(0)

        self.assertEqual(list(self.proxy.session_states), ["session-b"])

    def test_schedule_prefix_seed_skips_empty_and_duplicate_work(self):
        empty_plan = SnapshotPlan(
            "session-a",
            0,
            "key",
            Path(self.tempdir.name, "session.bin"),
            Path(self.tempdir.name, "prefix.bin"),
            {"messages": [], "tools": []},
            False,
        )
        self.proxy.enable_prefix_seeding = True
        with patch("cache_proxy.threading.Thread") as thread:
            self.proxy._schedule_prefix_seed(empty_plan)
            thread.assert_not_called()

        seeded_plan = SnapshotPlan(
            "session-a",
            0,
            "key",
            empty_plan.session_file,
            empty_plan.prefix_file,
            {"messages": [{"role": "system", "content": "rules"}], "tools": []},
            False,
        )
        self.proxy.prefix_seeds_in_flight.add(seeded_plan.prefix_file)
        with patch("cache_proxy.threading.Thread") as thread:
            self.proxy._schedule_prefix_seed(seeded_plan)
            thread.assert_not_called()
        self.proxy.prefix_seeds_in_flight.clear()

        with patch("cache_proxy.threading.Thread") as thread:
            thread.return_value.start = Mock()
            self.proxy._schedule_prefix_seed(seeded_plan)
            thread.assert_called_once()
            thread.return_value.start.assert_called_once_with()

    def test_foreground_operation_releases_lock_on_success_and_error(self):
        with self.proxy.foreground_operation():
            self.assertEqual(self.proxy.foreground_waiters, 0)
        self.assertEqual(self.proxy.foreground_waiters, 0)

        class RaisingLock:
            def acquire(self):
                raise RuntimeError("lock failed")

        self.proxy.operation_lock = RaisingLock()
        with self.assertRaises(RuntimeError):
            with self.proxy.foreground_operation():
                pass
        self.assertEqual(self.proxy.foreground_waiters, 0)

        self.proxy.operation_lock = threading.Lock()
        with self.assertRaises(ValueError):
            with self.proxy.foreground_operation():
                raise ValueError("request failed")
        self.assertEqual(self.proxy.foreground_waiters, 0)

    def test_session_snapshot_wins_after_switching_same_prefix_session(self):
        _, first_plan = self.proxy.prepare(self.body, "session-a")
        self.proxy.finish(first_plan, 200)
        _, second_plan = self.proxy.prepare(self.body, "session-b")
        self.proxy.finish(second_plan, 200)
        FakeLlamaHandler.restored = []

        self.proxy.prepare(self.body, "session-a")

        self.assertEqual(
            FakeLlamaHandler.restored,
            [cache_filename("session-a", self.body, "session")],
        )

    def test_new_session_uses_unowned_idle_slot_before_evicting_session(self):
        self.proxy.session_states["session-a"] = SlotState(1, cache_key(self.body), 10)
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 0},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 10},
        ]

        with patch.object(self.proxy, "_slots", return_value=slots):
            request, _ = self.proxy.prepare(self.body, "session-b")

        self.assertEqual(request["id_slot"], 0)

    def test_prefix_seed_skips_when_no_safe_idle_slot_exists(self):
        self.proxy.session_states["session-a"] = SlotState(1, cache_key(self.body), 10)
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": True, "n_prompt_tokens": 10},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 10},
            ]
        )
        self.proxy._json_request = Mock()
        self.proxy._save = Mock()

        self.proxy._seed_prefix(
            {"messages": [{"role": "system", "content": "rules"}]},
            Path(self.tempdir.name, "prefix.bin"),
            excluded_slot_id=1,
        )

        self.proxy._json_request.assert_not_called()
        self.proxy._save.assert_not_called()

    def test_prefix_seed_uses_unowned_idle_slot(self):
        self.proxy.prefix_seed_delay_seconds = 0
        self.proxy._slots = Mock(
            return_value=[
                {"id": 0, "is_processing": False, "n_prompt_tokens": 3},
                {"id": 1, "is_processing": False, "n_prompt_tokens": 10},
            ]
        )
        self.proxy._json_request = Mock(return_value={})
        self.proxy._save = Mock(return_value=10)
        self.proxy._forget_slot = Mock()
        self.proxy._prune = Mock()
        prefix_file = Path(self.tempdir.name, "prefix.bin")

        self.proxy._seed_prefix(
            {"messages": [{"role": "system", "content": "rules"}]},
            prefix_file,
            excluded_slot_id=1,
        )

        request = self.proxy._json_request.call_args.args[2]
        self.assertEqual(request["id_slot"], 0)
        self.proxy._save.assert_called_once_with(0, prefix_file)
        self.proxy._prune.assert_called_once_with()

    def test_prefix_seed_skips_when_foreground_is_waiting_or_proxy_is_busy(self):
        self.proxy.prefix_seed_delay_seconds = 0
        prefix_file = Path(self.tempdir.name, "prefix.bin")
        self.proxy.foreground_waiters = 1
        self.proxy._json_request = Mock()
        self.proxy._seed_prefix({"messages": [{"role": "system", "content": "rules"}]}, prefix_file, 1)
        self.proxy._json_request.assert_not_called()

        self.proxy.foreground_waiters = 0
        busy_lock = Mock()
        busy_lock.acquire.return_value = False
        self.proxy.operation_lock = busy_lock
        self.proxy._seed_prefix({"messages": [{"role": "system", "content": "rules"}]}, prefix_file, 1)
        busy_lock.release.assert_not_called()

    def test_prefix_seed_contains_and_cleans_up_failures(self):
        self.proxy.prefix_seed_delay_seconds = 0
        prefix_file = Path(self.tempdir.name, "prefix.bin")
        self.proxy._slots = Mock(side_effect=RuntimeError("slots unavailable"))
        self.proxy._seed_prefix({"messages": [{"role": "system", "content": "rules"}]}, prefix_file, 1)
        self.assertNotIn(prefix_file, self.proxy.prefix_seeds_in_flight)

    def test_prefix_seed_payload_stops_before_user_message(self):
        body = {
            **self.body,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        seed = self.proxy._prefix_seed_payload(body)

        self.assertEqual(seed["messages"], [self.body["messages"][0]])
        self.assertEqual(seed["n_predict"], 0)
        self.assertFalse(seed["add_generation_prompt"])
        self.assertFalse(seed["cache_prompt"])

    def test_anonymous_affinity_follows_project_prefix_not_user_message(self):
        other_body = {
            **self.body,
            "messages": [self.body["messages"][0], {"role": "user", "content": "new"}],
        }
        changed_project = {
            **self.body,
            "messages": [{"role": "system", "content": "other project"}, self.body["messages"][1]],
        }

        self.assertEqual(_anonymous_session_id(self.body), _anonymous_session_id(other_body))
        self.assertNotEqual(_anonymous_session_id(self.body), _anonymous_session_id(changed_project))

    def test_affinity_helpers_use_priority_and_strip_proxy_fields(self):
        handler = RecordingHandler(
            headers={
                "X-Session-Affinity": "  header-session  ",
                "X-Pi-Session-Id": "fallback-session",
            }
        )
        self.assertEqual(_session_id(handler), "header-session")
        self.assertIsNone(_session_id(RecordingHandler()))
        self.assertEqual(_body_session_id({"session_id": " body-session "}), "body-session")
        self.assertEqual(_body_session_id({"prompt_cache_key": " prompt-key "}), "prompt-key")
        self.assertIsNone(_body_session_id({"session_id": 42, "prompt_cache_key": ""}))

        body = {"session_id": "session", "prompt_cache_key": "key", "messages": []}
        stripped = _without_proxy_affinity_fields(body)
        self.assertEqual(stripped, {"messages": []})
        self.assertEqual(body["session_id"], "session")

    def test_media_detection_covers_images_and_message_content(self):
        self.assertTrue(_has_media({"images": ["image-data"]}))
        self.assertTrue(
            _has_media(
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url"}]},
                    ]
                }
            )
        )
        self.assertFalse(
            _has_media(
                {
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                    ]
                }
            )
        )
        self.assertFalse(_has_media({"messages": [{"role": "user", "content": "hello"}]}))

    def test_wait_for_idle_slot_prefers_unowned_and_times_out(self):
        self.proxy.session_states["session-a"] = SlotState(1, cache_key(self.body), 10)
        slots = [
            {"id": 0, "is_processing": False, "n_prompt_tokens": 3},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 1},
        ]
        with patch.object(self.proxy, "_slots", return_value=slots):
            self.assertEqual(self.proxy._wait_for_idle_slot(), 0)

        self.proxy.wait_seconds = 0
        with patch.object(self.proxy, "_slots", return_value=[]):
            with self.assertRaises(TimeoutError):
                self.proxy._wait_for_idle_slot()

        self.proxy.wait_seconds = 1
        with patch.object(self.proxy, "_slots", return_value=[]), patch(
            "cache_proxy.time.monotonic", side_effect=[0, 0, 2]
        ), patch("cache_proxy.time.sleep") as sleep:
            with self.assertRaises(TimeoutError):
                self.proxy._wait_for_idle_slot()
        sleep.assert_called_once_with(0.5)

    def test_forward_streams_body_and_filters_hop_by_hop_headers(self):
        response = FakeResponse(
            status=201,
            headers=[("Content-Type", "text/plain"), ("Connection", "close")],
            chunks=[b"ok"],
        )
        connection = FakeConnection("host", 80, response=response)
        handler = RecordingHandler(
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer token",
                "Accept": "text/event-stream",
                "User-Agent": "test",
            }
        )

        with patch("cache_proxy.HTTPConnection", return_value=connection):
            status = self.proxy.forward(handler, "POST", "/v1/test", b"body")

        self.assertEqual(status, 201)
        self.assertEqual(connection.requests[0][0:2], ("POST", "/v1/test"))
        self.assertEqual(connection.requests[0][2], b"body")
        self.assertIn(("Content-Type", "text/plain"), handler.sent_headers)
        self.assertNotIn(("Connection", "close"), handler.sent_headers)
        self.assertIn(("Transfer-Encoding", "chunked"), handler.sent_headers)
        self.assertEqual(b"".join(handler.wfile.data), b"2\r\nok\r\n0\r\n\r\n")
        self.assertTrue(connection.closed)

    def test_forward_returns_status_when_client_disconnects(self):
        connection = FakeConnection("host", 80, response=FakeResponse(status=200, chunks=[b"ok"]))
        handler = RecordingHandler(broken=True)

        with patch("cache_proxy.HTTPConnection", return_value=connection):
            status = self.proxy.forward(handler, "GET", "/health", b"")

        self.assertEqual(status, 200)
        self.assertTrue(connection.closed)

    def test_forward_headers_preserve_supported_request_headers(self):
        handler = RecordingHandler(
            headers={
                "Content-Type": "application/custom",
                "Authorization": "token",
                "Accept": "application/json",
                "User-Agent": "agent",
            }
        )

        headers = self.proxy._forward_headers(handler)

        self.assertEqual(
            headers,
            {
                "Content-Type": "application/custom",
                "Authorization": "token",
                "Accept": "application/json",
                "User-Agent": "agent",
            },
        )

    def test_json_request_and_slots_handle_success_empty_and_error(self):
        success = FakeConnection("host", 80, response=FakeResponse(raw=b'{"ok": true}'))
        with patch("cache_proxy.HTTPConnection", return_value=success):
            self.assertEqual(self.proxy._json_request("POST", "/test", {"x": 1}), {"ok": True})
        self.assertTrue(success.closed)

        empty = FakeConnection("host", 80, response=FakeResponse(raw=b""))
        with patch("cache_proxy.HTTPConnection", return_value=empty):
            self.assertEqual(self.proxy._json_request("POST", "/test", {}), {})

        failed = FakeConnection("host", 80, response=FakeResponse(status=500, raw=b"bad"))
        with patch("cache_proxy.HTTPConnection", return_value=failed):
            with self.assertRaises(RuntimeError):
                self.proxy._json_request("POST", "/test", {})
        self.assertTrue(failed.closed)

        slots = FakeConnection("host", 80, response=FakeResponse(raw=b'[{"id": 0}]'))
        with patch("cache_proxy.HTTPConnection", return_value=slots):
            self.assertEqual(self.proxy._slots(), [{"id": 0}])

        non_list = FakeConnection("host", 80, response=FakeResponse(raw=b"{}"))
        with patch("cache_proxy.HTTPConnection", return_value=non_list):
            self.assertEqual(self.proxy._slots(), [])

        failed_slots = FakeConnection("host", 80, response=FakeResponse(status=503))
        with patch("cache_proxy.HTTPConnection", return_value=failed_slots):
            with self.assertRaises(RuntimeError):
                self.proxy._slots()

    def test_save_rejects_zero_tokens(self):
        self.proxy._json_request = Mock(return_value={"n_saved": 0})

        with self.assertRaises(RuntimeError):
            self.proxy._save(0, Path(self.tempdir.name, "session.bin"))

    def test_proxy_handler_get_and_non_chat_post_forward_successfully(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = CapturingProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()

            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("POST", "/v1/models", body=b"{}", headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_get_and_non_chat_post_return_503_on_upstream_error(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = FailingForwardProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for method, path in (("GET", "/health"), ("POST", "/v1/models")):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                body = b"{}" if method == "POST" else None
                headers = {"Content-Type": "application/json"} if body is not None else {}
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 503)
                connection.close()
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_rejects_invalid_json_and_forwards_media(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        proxy = CapturingProxy()
        ProxyHandler.proxy = proxy
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("POST", "/v1/chat/completions", body=b"not-json")
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 400)
            connection.close()

            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
            self.assertEqual(proxy.forwarded[-1][0], "POST")
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_returns_503_for_media_upstream_error(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = FailingForwardProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"images": ["image-data"]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 503)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_proxy_handler_contains_snapshot_finalization_error(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = FinishFailingProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_error_response_does_not_write_second_response_after_disconnect(self):
        handler = RecordingHandler()
        handler._proxy_response_started = False
        ProxyHandler._send_upstream_error(handler, RuntimeError("downstream failed"))
        self.assertEqual(handler.errors[0][0], 503)

        started = RecordingHandler()
        started._proxy_response_started = True
        ProxyHandler._send_upstream_error(started, RuntimeError("downstream failed"))
        self.assertEqual(started.errors, [])

    def test_proxy_handler_log_message_uses_client_address(self):
        handler = RecordingHandler()
        handler.address_string = lambda: "127.0.0.1"
        with patch.object(cache_proxy.LOGGER, "info") as info:
            ProxyHandler.log_message(handler, "status %s", 200)
        info.assert_called_once_with("%s - %s", "127.0.0.1", "status 200")

    def test_main_builds_server_from_environment_and_closes_on_interrupt(self):
        servers = []

        class MainServer:
            def __init__(self, address, handler):
                self.address = address
                self.handler = handler
                self.closed = False
                servers.append(self)

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                self.closed = True

        env = {
            "PI_LLAMA_UPSTREAM": "http://127.0.0.1:9999/api",
            "PI_LLAMA_CACHE_HOST": "127.0.0.1",
            "PI_LLAMA_CACHE_PORT": "19082",
            "PI_LLAMA_CACHE_DIR": self.tempdir.name,
            "PI_LLAMA_CACHE_MAX_GIB": "1",
            "PI_LLAMA_CACHE_WAIT_SECONDS": "1",
            "PI_LLAMA_CACHE_PREFIX_SEED_DELAY": "0",
        }
        with patch.dict(os.environ, env, clear=False), patch("http.server.ThreadingHTTPServer", MainServer):
            runpy.run_path(cache_proxy.__file__, run_name="__main__")

        self.assertEqual(servers[0].address, ("127.0.0.1", 19082))
        self.assertTrue(servers[0].closed)

    def test_upstream_unavailable_returns_503(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = UnavailableProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "rules"},
                            {"role": "user", "content": "hello"},
                        ]
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 503)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_prune_removes_local_llm_snapshots_over_limit(self):
        proxy = LlamaCacheProxy(
            upstream=f"http://127.0.0.1:{self.server.server_port}",
            cache_dir=self.tempdir.name,
            max_cache_gib=0,
            wait_seconds=1,
            enable_prefix_seeding=False,
        )
        snapshot = Path(self.tempdir.name, "local-llm-session-old.bin")
        snapshot.write_bytes(b"snapshot")

        proxy._prune()

        self.assertFalse(snapshot.exists())

    def test_body_session_id_is_used_for_affinity(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        capturing_proxy = CapturingProxy()
        ProxyHandler.proxy = capturing_proxy
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "session_id": "body-session",
                        "messages": [
                            {"role": "system", "content": "rules"},
                            {"role": "user", "content": "hello"},
                        ],
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            self.assertEqual(capturing_proxy.session_ids, ["body-session"])
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy

    def test_non_object_json_returns_400(self):
        previous_proxy = getattr(ProxyHandler, "proxy", None)
        ProxyHandler.proxy = UnavailableProxy()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body="[]",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            self.assertEqual(response.status, 400)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            if previous_proxy is None:
                delattr(ProxyHandler, "proxy")
            else:
                ProxyHandler.proxy = previous_proxy


if __name__ == "__main__":
    unittest.main()
