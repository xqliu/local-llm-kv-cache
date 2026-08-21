import unittest

from cache_core import (
    build_prefix_payload,
    cache_key,
    cache_filename,
    with_slot_cache,
)


class CacheCoreTests(unittest.TestCase):
    def setUp(self):
        self.body = {
            "model": "qwen3.8:27b",
            "messages": [
                {"role": "system", "content": "project rules"},
                {"role": "user", "content": "first request"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        }

    def test_user_message_does_not_change_project_prefix_key(self):
        other = {**self.body, "messages": [
            self.body["messages"][0],
            {"role": "user", "content": "different request"},
        ]}

        self.assertEqual(cache_key(self.body), cache_key(other))

    def test_project_rules_change_invalidates_prefix_key(self):
        other = {**self.body, "messages": [
            {"role": "system", "content": "changed project rules"},
            self.body["messages"][1],
        ]}

        self.assertNotEqual(cache_key(self.body), cache_key(other))

    def test_tool_schema_change_invalidates_prefix_key(self):
        other = {**self.body, "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file with a changed contract",
                    "parameters": {"type": "object"},
                },
            }
        ]}

        self.assertNotEqual(cache_key(self.body), cache_key(other))

    def test_prefix_payload_keeps_only_stable_leading_messages(self):
        prefix = build_prefix_payload(self.body)

        self.assertEqual(prefix["messages"], [self.body["messages"][0]])
        self.assertEqual(prefix["tools"], self.body["tools"])
        self.assertNotIn("first request", str(prefix))

    def test_slot_request_forces_prompt_cache_and_slot_affinity(self):
        request = with_slot_cache(self.body, 1)

        self.assertTrue(request["cache_prompt"])
        self.assertEqual(request["id_slot"], 1)
        self.assertEqual(request["messages"], self.body["messages"])

    def test_cache_filename_is_safe_and_stable(self):
        name = cache_filename("session/with spaces", self.body, "session")

        self.assertRegex(name, r"^pi-session-[0-9a-f]{64}\.bin$")
        self.assertNotIn("/", name)
        self.assertEqual(name, cache_filename("session/with spaces", self.body, "session"))


if __name__ == "__main__":
    unittest.main()
