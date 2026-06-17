"""Deterministic state hashing utilities.

Produces reproducible SHA-256 hashes of agent state content so that
identical states always map to the same digest — used for loop detection
and state deduplication.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _make_deterministic(obj: Any) -> Any:
    """Recursively sort dicts and convert sets to sorted lists for deterministic JSON."""
    if isinstance(obj, dict):
        return {k: _make_deterministic(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_make_deterministic(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_make_deterministic(item) for item in obj)
    return obj


def compute_state_hash(
    action: dict[str, Any],
    observation: str,
    memory: list[dict[str, Any]],
    scratchpad: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
) -> str:

    payload = _make_deterministic(
        {
            "action": action,
            "observation": observation,
            "scratchpad": scratchpad,
            "inputs": inputs or [],
            "outputs": outputs or [],
        }
    )
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
