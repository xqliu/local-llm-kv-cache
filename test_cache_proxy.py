import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cache_core import cache_filename
from cache_proxy import LlamaCacheProxy, ProxyHandler, _anonymous_session_id


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

    @contextmanager
    def foreground_operation(self):
        yield

    def prepare(self, body, session_id):
        self.session_ids.append(session_id)
        return body, None

    def forward(self, handler, *_args):
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
