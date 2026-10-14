"""Tolerant JSON extraction from model output."""
from __future__ import annotations

import json
import re
from typing import Any

from llm_router.errors import InvalidJSON

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _light_repair(blob: str) -> Any:
    """Minimal fixes for the most common defects (trailing commas, missing closers)."""
    fixed = blob.rstrip()
    depth_obj = fixed.count("{") - fixed.count("}")
    depth_arr = fixed.count("[") - fixed.count("]")
    if 0 <= depth_arr <= 8 and 0 <= depth_obj <= 8:
        fixed += "]" * depth_arr + "}" * depth_obj
    return json.loads(_TRAILING_COMMA.sub(r"\1", fixed))


def parse_json(text: str) -> dict[str, Any]:
    """Return the JSON object contained in ``text``.

    Layers: (1) strip a markdown fence, (2) strict ``json.loads`` on the outermost
    ``{...}``, (3) ``json_repair`` when installed (extra ``repair``), otherwise a light
    built-in repair. Raises ``InvalidJSON`` when no object can be recovered, which the
    router treats as a retryable failure of that attempt.
    """
    text = (text or "").strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise InvalidJSON("response contains no JSON object")
    end = text.rfind("}")
    blob = text[start:end + 1] if end > start else text[start:]
    try:
        value = json.loads(blob)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json  # optional dependency

            value = repair_json(blob, return_objects=True)
        except ImportError:
            try:
                value = _light_repair(blob)
            except json.JSONDecodeError:
                raise InvalidJSON("unrepairable JSON in model response") from None
    if not isinstance(value, dict) or not value:
        raise InvalidJSON("model response is not a non-empty JSON object")
    return value
