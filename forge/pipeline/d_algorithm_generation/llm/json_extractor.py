"""JSON extraction for structured LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def extract_json_object(
    text: str,
    *,
    preferred_keys: tuple[str, ...] = (),
) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract a JSON object, preferring objects that match the caller's contract.

    LLM responses sometimes contain a small JSON example or diagnostic object
    before the actual payload.  Returning the first decodable object in that
    situation creates a false parse rejection.  ``preferred_keys`` lets a
    structured caller rank all decoded objects without weakening JSON parsing.
    A JSON string containing one JSON object is also accepted because some
    OpenAI-compatible gateways double-encode structured responses.
    """

    parsed, errors, _repairs = extract_json_object_with_audit(
        text,
        preferred_keys=preferred_keys,
    )
    return parsed, errors


def extract_json_object_with_audit(
    text: str,
    *,
    preferred_keys: tuple[str, ...] = (),
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, str]]]:
    """Extract an object and report non-semantic transport recovery actions."""

    errors: list[str] = []
    text = text or ""
    candidates = _candidate_texts(text)
    parsed_objects: list[tuple[dict[str, Any], int]] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"json parse failed: {exc.msg}")
            continue
        parsed, nested_depth = _decode_nested_json_object(parsed)
        if isinstance(parsed, dict):
            parsed_objects.append((parsed, nested_depth))
            continue
        errors.append("top-level JSON value must be an object")
    if parsed_objects:
        selected_index = 0
        if preferred_keys:
            key_priority = {key: index for index, key in enumerate(preferred_keys)}
            ranked = [
                (
                    min(
                        (key_priority[key] for key in item[0] if key in key_priority),
                        default=len(key_priority),
                    ),
                    index,
                    item,
                )
                for index, item in enumerate(parsed_objects)
            ]
            _rank, selected_index, selected = min(
                ranked, key=lambda item: (item[0], item[1])
            )
        else:
            selected = parsed_objects[0]
        parsed, nested_depth = selected
        repairs: list[dict[str, str]] = []
        if nested_depth:
            repairs.append(
                {
                    "path": "$",
                    "action": f"decoded a JSON object nested inside {nested_depth} JSON string layer(s)",
                }
            )
        if selected_index:
            repairs.append(
                {
                    "path": "$",
                    "action": "selected the contract-matching JSON object after unrelated response text",
                }
            )
        return parsed, [], repairs
    return None, errors or ["no JSON object found"], []


def _decode_nested_json_object(
    value: Any, *, maximum_depth: int = 2
) -> tuple[Any, int]:
    """Decode the common double-encoded JSON-object transport shape."""

    decoded = value
    depth = 0
    for _ in range(maximum_depth):
        if not isinstance(decoded, str):
            break
        candidate = decoded.strip()
        if not candidate.startswith("{"):
            break
        try:
            decoded = json.loads(candidate)
            depth += 1
        except json.JSONDecodeError:
            break
    return decoded, depth


def ensure_json_only_response(obj: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(obj, dict):
        return False, ["response is not a JSON object"]
    return True, []


def _candidate_texts(text: str) -> list[str]:
    output: list[str] = []
    stripped = text.strip()
    if stripped:
        output.append(stripped)
    for match in FENCE_RE.finditer(text):
        output.append(match.group(1).strip())
    balanced = _balanced_object_candidates(text)
    output.extend(balanced)
    unique: list[str] = []
    seen: set[str] = set()
    for item in output:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _balanced_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start_positions = [index for index, char in enumerate(text) if char == "{"]
    decoder = json.JSONDecoder()
    for start in start_positions:
        try:
            _obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        candidates.append(text[start : start + end])
    return candidates
