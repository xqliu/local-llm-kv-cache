"""Pure cache-key and request helpers for the Pi llama proxy."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


_PREFIX_FIELDS = (
    "model",
    "tools",
    "tool_choice",
    "chat_template_kwargs",
    "chat_template_args",
    "enable_thinking",
    "reasoning_effort",
    "reasoning_format",
    "response_format",
    "json_schema",
    "grammar",
    "add_generation_prompt",
    "continue_final_message",
    "parallel_tool_calls",
)
_CACHE_FORMAT_VERSION = "2"
_SYSTEM_ROLES = {"system", "developer"}


def build_prefix_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Return request fields that affect the stable prompt prefix."""
    messages = []
    for message in body.get("messages") or []:
        if message.get("role") not in _SYSTEM_ROLES:
            break
        messages.append(copy.deepcopy(message))

    prefix = {"messages": messages}
    for field in _PREFIX_FIELDS:
        if field in body:
            prefix[field] = copy.deepcopy(body[field])
    return prefix


def cache_key(body: dict[str, Any]) -> str:
    """Hash a canonical stable prefix so project changes invalidate it."""
    encoded = json.dumps(
        build_prefix_payload(body),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_filename(identity: str, body: dict[str, Any], kind: str) -> str:
    """Build a filesystem-safe filename scoped to an identity and prefix."""
    material = f"{_CACHE_FORMAT_VERSION}\0{kind}\0{identity}\0{cache_key(body)}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return f"local-llm-{kind}-{digest}.bin"


def with_slot_cache(body: dict[str, Any], slot_id: int) -> dict[str, Any]:
    """Copy a request and force llama.cpp prompt cache plus slot affinity."""
    request = copy.deepcopy(body)
    request["cache_prompt"] = True
    request["id_slot"] = int(slot_id)
    return request
