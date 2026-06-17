from __future__ import annotations
import time
from functools import wraps
from typing import Any, Callable
from pydantic import BaseModel, Field
from agents_driftguard.core.state import AgentState

class RollbackEvent(BaseModel):

    timestamp: float = Field(default_factory=time.time)
    event_type: str = ""
    details: str = ""


class RollbackResult(BaseModel):
    success: bool = False
    rolled_back_to: int = -1
    steps_undone: int = 0
    events: list[RollbackEvent] = Field(default_factory=list)

class ReversibleToolRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, tool_name: str, undo_fn: Callable[..., Any]) -> None:
        self._registry[tool_name] = undo_fn

    def get_undo(self, tool_name: str) -> Callable[..., Any] | None:
        return self._registry.get(tool_name)

    def has_undo(self, tool_name: str) -> bool:
        return tool_name in self._registry

    @property
    def registered_tools(self) -> list[str]:
        """List all tool names with registered undo hooks."""
        return list(self._registry.keys())

_global_registry = ReversibleToolRegistry()


def get_global_registry() -> ReversibleToolRegistry:
    """Return the module-level ``ReversibleToolRegistry`` singleton."""
    return _global_registry
def reversible(
    undo_fn: Callable[..., Any] | None = None,
    *,
    tool_name: str | None = None,
) -> Callable[..., Any]:

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = tool_name or fn.__name__

        if undo_fn is not None:
            _global_registry.register(name, undo_fn)

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        # Attach metadata
        wrapper._driftguard_reversible = True  # type: ignore[attr-defined]
        wrapper._driftguard_tool_name = name  # type: ignore[attr-defined]
        return wrapper

    return decorator

class RollbackController:

    def __init__(
        self,
        registry: ReversibleToolRegistry | None = None,
        verbose: bool = False,
    ) -> None:
        self.registry = registry or _global_registry
        self.verbose = verbose

    def rollback(
        self,
        to_state: AgentState,
        store: Any,  # StateStore protocol
        states_to_undo: list[AgentState] | None = None,
    ) -> RollbackResult:
        events: list[RollbackEvent] = []

        events.append(RollbackEvent(
            event_type="rollback_started",
            details=f"Rolling back to step {to_state.step_id} ({to_state.node_id})",
        ))

        # 1. Invoke reversible hooks for states being undone
        undone_states = states_to_undo or []
        for state in reversed(undone_states):  # undo in reverse order
            tool_name = state.action.get("tool_name", "")
            undo_fn = self.registry.get_undo(tool_name) if tool_name else None
            if undo_fn is not None:
                try:
                    undo_fn(state.action.get("args", {}), state.observation)
                    events.append(RollbackEvent(
                        event_type="hook_invoked",
                        details=f"Undo hook called for tool '{tool_name}' at step {state.step_id}",
                    ))
                except Exception as exc:
                    events.append(RollbackEvent(
                        event_type="hook_failed",
                        details=f"Undo hook for '{tool_name}' at step {state.step_id} failed: {exc}",
                    ))

        # 2. Truncate the store
        removed = store.truncate_after(to_state.step_id)

        events.append(RollbackEvent(
            event_type="store_truncated",
            details=f"Removed {removed} state(s) from store",
        ))

        events.append(RollbackEvent(
            event_type="rollback_completed",
            details=f"Successfully restored to step {to_state.step_id}",
        ))

        return RollbackResult(
            success=True,
            rolled_back_to=to_state.step_id,
            steps_undone=removed,
            events=events,
        )
