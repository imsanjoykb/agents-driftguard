"""Tests for core/rollback.py — RollbackController and reversible tools."""

from __future__ import annotations

import pytest

from agents_driftguard.core.rollback import (
    ReversibleToolRegistry,
    RollbackController,
    get_global_registry,
    reversible,
)
from agents_driftguard.core.state import AgentState
from agents_driftguard.store.memory import InMemoryStateStore


def _make_state(step_id: int, tool_name: str = "tool") -> AgentState:
    """Helper to create a test state."""
    return AgentState(
        step_id=step_id,
        action={"tool_name": tool_name},
        observation=f"result_{step_id}",
    )


class TestReversibleToolRegistry:
    """Test the reversible tool registry."""

    def test_register_and_get(self) -> None:
        """Register an undo function and retrieve it."""
        reg = ReversibleToolRegistry()
        undo = lambda args, result: None  # noqa: E731
        reg.register("write_file", undo)
        assert reg.has_undo("write_file")
        assert reg.get_undo("write_file") is undo

    def test_missing_undo(self) -> None:
        """Missing undo returns None."""
        reg = ReversibleToolRegistry()
        assert reg.get_undo("unknown") is None
        assert not reg.has_undo("unknown")

    def test_registered_tools(self) -> None:
        """registered_tools lists all registered tool names."""
        reg = ReversibleToolRegistry()
        reg.register("tool_a", lambda a, r: None)
        reg.register("tool_b", lambda a, r: None)
        assert set(reg.registered_tools) == {"tool_a", "tool_b"}


class TestReversibleDecorator:
    """Test the @reversible decorator."""

    def test_decorator_preserves_function(self) -> None:
        """Decorated function still works normally."""
        call_log: list[str] = []

        def undo_fn(args: dict, result: str) -> None:
            call_log.append("undone")

        @reversible(undo_fn=undo_fn, tool_name="test_tool")
        def my_tool(x: int) -> int:
            return x * 2

        assert my_tool(5) == 10
        assert my_tool._driftguard_reversible is True  # type: ignore[attr-defined]

    def test_decorator_registers_undo(self) -> None:
        """Decorator registers the undo function in the global registry."""
        undo = lambda args, result: None  # noqa: E731

        @reversible(undo_fn=undo, tool_name="decorator_test_tool")
        def my_tool() -> str:
            return "done"

        reg = get_global_registry()
        assert reg.has_undo("decorator_test_tool")


class TestRollbackController:
    """Test the RollbackController."""

    def test_basic_rollback(self) -> None:
        """Rollback truncates store and reports success."""
        store = InMemoryStateStore()
        s0 = _make_state(0)
        s1 = _make_state(1)
        s2 = _make_state(2)
        s3 = _make_state(3)

        for s in [s0, s1, s2, s3]:
            store.save(s)

        controller = RollbackController()
        result = controller.rollback(
            to_state=s1,
            store=store,
            states_to_undo=[s2, s3],
        )

        assert result.success
        assert result.rolled_back_to == 1
        assert result.steps_undone == 2
        assert len(store.get_all()) == 2  # s0 and s1

    def test_rollback_invokes_undo_hooks(self) -> None:
        """Undo hooks are called during rollback."""
        undo_log: list[str] = []

        reg = ReversibleToolRegistry()
        reg.register("dangerous_tool", lambda args, result: undo_log.append(f"undone: {result}"))

        store = InMemoryStateStore()
        s0 = _make_state(0)
        s1 = _make_state(1, tool_name="dangerous_tool")
        store.save(s0)
        store.save(s1)

        controller = RollbackController(registry=reg)
        result = controller.rollback(
            to_state=s0,
            store=store,
            states_to_undo=[s1],
        )

        assert result.success
        assert len(undo_log) == 1
        assert "undone" in undo_log[0]

    def test_rollback_events_logged(self) -> None:
        """Rollback produces event log entries."""
        store = InMemoryStateStore()
        s0 = _make_state(0)
        s1 = _make_state(1)
        store.save(s0)
        store.save(s1)

        controller = RollbackController()
        result = controller.rollback(to_state=s0, store=store, states_to_undo=[s1])

        event_types = [e.event_type for e in result.events]
        assert "rollback_started" in event_types
        assert "store_truncated" in event_types
        assert "rollback_completed" in event_types

    def test_rollback_to_zero(self) -> None:
        """Can roll back all the way to the initial state."""
        store = InMemoryStateStore()
        states = [_make_state(i) for i in range(5)]
        for s in states:
            store.save(s)

        controller = RollbackController()
        result = controller.rollback(
            to_state=states[0],
            store=store,
            states_to_undo=states[1:],
        )

        assert result.success
        assert result.rolled_back_to == 0
        assert len(store.get_all()) == 1
