"""In-memory state store.

Default persistence backend backed by a plain ``dict``. Thread-safe via a
``threading.Lock`` — suitable for single-process agent runs.
"""

from __future__ import annotations

import threading
from typing import Optional

from agents_driftguard.core.state import AgentState


class InMemoryStateStore:
    """In-memory implementation of the ``StateStore`` protocol.

    Stores agent states in a dict keyed by ``step_id``.  All operations
    are guarded by a re-entrant lock for thread safety.
    """

    def __init__(self) -> None:
        self._store: dict[int, AgentState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # StateStore protocol implementation
    # ------------------------------------------------------------------

    def save(self, state: AgentState) -> None:
        """Persist an agent state.

        Args:
            state: The state to save.
        """
        with self._lock:
            self._store[state.step_id] = state

    def get(self, step_id: int) -> Optional[AgentState]:
        """Retrieve a state by step ID.

        Args:
            step_id: The step index to look up.

        Returns:
            The matching ``AgentState``, or ``None`` if not found.
        """
        with self._lock:
            return self._store.get(step_id)

    def get_all(self) -> list[AgentState]:
        """Return all stored states, ordered by ``step_id``.

        Returns:
            Sorted list of all ``AgentState`` objects.
        """
        with self._lock:
            return [self._store[k] for k in sorted(self._store.keys())]

    def get_last_n(self, n: int) -> list[AgentState]:
        """Return the most recent *n* states.

        Args:
            n: Number of recent states to retrieve.

        Returns:
            Up to *n* most-recent states, ordered by ``step_id``.
        """
        with self._lock:
            keys = sorted(self._store.keys())
            return [self._store[k] for k in keys[-n:]]

    def clear(self) -> None:
        """Remove all stored states."""
        with self._lock:
            self._store.clear()

    def truncate_after(self, step_id: int) -> int:
        """Remove all states with ``step_id`` strictly greater than the given value.

        Args:
            step_id: Keep states up to and including this step.

        Returns:
            Number of states removed.
        """
        with self._lock:
            to_remove = [k for k in self._store if k > step_id]
            for k in to_remove:
                del self._store[k]
            return len(to_remove)

    def __len__(self) -> int:
        """Return the number of stored states."""
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"InMemoryStateStore(states={len(self)})"
