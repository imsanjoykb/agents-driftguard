"""Tests for core/state.py — AgentState model and hashing."""

from __future__ import annotations

import pytest

from agents_driftguard.core.state import AgentState
from agents_driftguard.utils.hashing import compute_state_hash


class TestAgentState:
    """Test AgentState creation, hashing, and serialization."""

    def test_create_minimal_state(self) -> None:
        """Create a state with minimal fields."""
        state = AgentState(step_id=0)
        assert state.step_id == 0
        assert state.diverged is False
        assert state.state_hash != ""
        assert state.node_id == "S0"

    def test_auto_hash_computed(self) -> None:
        """State hash is auto-computed on creation."""
        state = AgentState(
            step_id=1,
            action={"tool_name": "search", "args": {"q": "test"}},
            observation="Found 5 results",
            memory=[{"role": "user", "content": "search for test"}],
            scratchpad="I should search for test",
        )
        assert state.state_hash != ""
        assert len(state.state_hash) == 64  # SHA-256 hex digest

    def test_same_content_same_hash(self) -> None:
        """Identical content produces identical hashes."""
        kwargs = dict(
            step_id=1,
            action={"tool_name": "search", "args": {"q": "test"}},
            observation="result",
            memory=[],
            scratchpad="thinking",
        )
        s1 = AgentState(**kwargs)
        s2 = AgentState(**kwargs)
        assert s1.state_hash == s2.state_hash

    def test_different_content_different_hash(self) -> None:
        """Different content produces different hashes."""
        s1 = AgentState(step_id=1, observation="result A")
        s2 = AgentState(step_id=2, observation="result B")
        assert s1.state_hash != s2.state_hash

    def test_explicit_hash_not_overwritten(self) -> None:
        """If state_hash is explicitly set, it is preserved."""
        state = AgentState(step_id=0, state_hash="custom_hash")
        assert state.state_hash == "custom_hash"

    def test_node_id_format(self) -> None:
        """node_id returns the expected 'S{step_id}' format."""
        state = AgentState(step_id=42)
        assert state.node_id == "S42"

    def test_serialization_roundtrip(self) -> None:
        """State can be serialized to JSON and deserialized back."""
        state = AgentState(
            step_id=3,
            action={"tool_name": "write", "args": {"path": "/tmp/x"}},
            observation="file written",
            memory=[{"role": "assistant", "content": "writing file"}],
            scratchpad="need to write",
            goal_embedding=[0.1, 0.2, 0.3],
            state_embedding=[0.4, 0.5, 0.6],
            metadata={"attempt": 1},
        )
        json_str = state.model_dump_json()
        restored = AgentState.model_validate_json(json_str)
        assert restored.step_id == state.step_id
        assert restored.state_hash == state.state_hash
        assert restored.goal_embedding == state.goal_embedding

    def test_diverged_flag(self) -> None:
        """Can create a state with diverged=True."""
        state = AgentState(step_id=5, diverged=True)
        assert state.diverged is True

    def test_timestamp_auto_set(self) -> None:
        """Timestamp is auto-populated."""
        state = AgentState(step_id=0)
        assert state.timestamp > 0

    def test_default_empty_fields(self) -> None:
        """Default fields are empty but present."""
        state = AgentState(step_id=0)
        assert state.action == {}
        assert state.observation == ""
        assert state.memory == []
        assert state.scratchpad == ""
        assert state.goal_embedding == []
        assert state.state_embedding == []
        assert state.metadata == {}


class TestComputeStateHash:
    """Test the standalone hashing function."""

    def test_deterministic(self) -> None:
        """Same inputs always produce the same hash."""
        h1 = compute_state_hash({"a": 1}, "obs", [{"k": "v"}], "pad")
        h2 = compute_state_hash({"a": 1}, "obs", [{"k": "v"}], "pad")
        assert h1 == h2

    def test_dict_key_order_irrelevant(self) -> None:
        """Dict key order does not affect the hash."""
        h1 = compute_state_hash({"b": 2, "a": 1}, "obs", [], "")
        h2 = compute_state_hash({"a": 1, "b": 2}, "obs", [], "")
        assert h1 == h2

    def test_different_inputs_different_hash(self) -> None:
        """Different inputs produce different hashes."""
        h1 = compute_state_hash({}, "obs1", [], "")
        h2 = compute_state_hash({}, "obs2", [], "")
        assert h1 != h2
