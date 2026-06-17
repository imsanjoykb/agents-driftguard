from __future__ import annotations
import threading
from typing import Optional
from agents_driftguard.core.state import AgentState

class InMemoryStateStore:
    def __init__(self) -> None:
        self._store: dict[int, AgentState] = {}
        self._lock = threading.Lock()

    def save(self, state: AgentState) -> None:
        with self._lock:
            self._store[state.step_id] = state

    def get(self, step_id: int) -> Optional[AgentState]:
        with self._lock:
            return self._store.get(step_id)

    def get_all(self) -> list[AgentState]:
        with self._lock:
            return [self._store[k] for k in sorted(self._store.keys())]

    def get_last_n(self, n: int) -> list[AgentState]:
        with self._lock:
            keys = sorted(self._store.keys())
            return [self._store[k] for k in keys[-n:]]

    def clear(self) -> None:
        """Remove all stored states."""
        with self._lock:
            self._store.clear()

    def truncate_after(self, step_id: int) -> int:
        with self._lock:
            to_remove = [k for k in self._store if k > step_id]
            for k in to_remove:
                del self._store[k]
            return len(to_remove)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"InMemoryStateStore(states={len(self)})"
