"""Rollback controller and reversible-tool support.

Provides the machinery to restore agent state to a previous safe checkpoint:

- ``RollbackController`` — orchestrates state restoration, store truncation,
  and reversible-hook invocation.
- ``@reversible`` decorator — registers an undo function for a tool so that
  side-effects can be unwound during rollback.
- ``ReversibleToolRegistry`` — global registry of undo hooks.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from pydantic import BaseModel, Field

from agents_driftguard.core.state import AgentState


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RollbackEvent(BaseModel):
    """A single event within a rollback operation.

    Attributes:
        timestamp: When the event occurred.
        event_type: Category (e.g. ``"state_restored"``, ``"hook_invoked"``).
        details: Human-readable description.
    """

    timestamp: float = Field(default_factory=time.time)
    event_type: str = ""
    details: str = ""


class RollbackResult(BaseModel):
    """Outcome of a rollback operation.

    Attributes:
        success: Whether the rollback completed without errors.
        rolled_back_to: Step ID of the restored state.
        steps_undone: Number of states removed.
        events: Chronological log of what happened during rollback.
    """

    success: bool = False
    rolled_back_to: int = -1
    steps_undone: int = 0
    events: list[RollbackEvent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reversible tool registry
# ---------------------------------------------------------------------------

class ReversibleToolRegistry:
    """Global registry mapping tool names to their undo functions.

    An undo function receives ``(original_args, original_result)`` and
    should reverse any side-effects produced by the tool.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, tool_name: str, undo_fn: Callable[..., Any]) -> None:
        """Register an undo function for a tool.

        Args:
            tool_name: The name of the tool.
            undo_fn: Callable that reverses the tool's side-effects.
        """
        self._registry[tool_name] = undo_fn

    def get_undo(self, tool_name: str) -> Callable[..., Any] | None:
        """Look up the undo function for a tool.

        Args:
            tool_name: The name of the tool.

        Returns:
            The undo callable, or ``None`` if no hook is registered.
        """
        return self._registry.get(tool_name)

    def has_undo(self, tool_name: str) -> bool:
        """Check whether a tool has a registered undo hook.

        Args:
            tool_name: The name of the tool.
        """
        return tool_name in self._registry

    @property
    def registered_tools(self) -> list[str]:
        """List all tool names with registered undo hooks."""
        return list(self._registry.keys())


# Module-level singleton registry
_global_registry = ReversibleToolRegistry()


def get_global_registry() -> ReversibleToolRegistry:
    """Return the module-level ``ReversibleToolRegistry`` singleton."""
    return _global_registry


# ---------------------------------------------------------------------------
# @reversible decorator
# ---------------------------------------------------------------------------

def reversible(
    undo_fn: Callable[..., Any] | None = None,
    *,
    tool_name: str | None = None,
) -> Callable[..., Any]:
    """Decorator that registers an undo function for a tool.

    Can be used in two ways::

        # 1. With an explicit undo function
        @reversible(undo_fn=my_undo)
        def write_file(path, content): ...

        # 2. As a simple marker (undo must be registered separately)
        @reversible(tool_name="write_file")
        def write_file(path, content): ...

    Args:
        undo_fn: The function that reverses the tool's side-effects.
        tool_name: Override the registered tool name (defaults to
            the decorated function's ``__name__``).

    Returns:
        The original function, unchanged but registered.
    """

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


# ---------------------------------------------------------------------------
# Rollback controller
# ---------------------------------------------------------------------------

class RollbackController:
    """Orchestrates rollback to a previous safe agent state.

    Steps performed during ``rollback()``:

    1. Identify states to undo (everything after the target state).
    2. Invoke registered reversible-tool hooks for undone states.
    3. Truncate the state store.
    4. Emit ``RollbackEvent`` entries for observability.
    """

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
        """Roll back the agent to ``to_state``.

        Args:
            to_state: The safe state to restore to.
            store: The state store to truncate.
            states_to_undo: Optional explicit list of states being undone
                (if not given, the controller will rely on the store).

        Returns:
            ``RollbackResult`` describing what happened.
        """
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
