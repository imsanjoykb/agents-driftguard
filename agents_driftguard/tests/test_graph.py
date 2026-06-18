from __future__ import annotations
import pytest
from agents_driftguard.core.graph import CausalGraph
from agents_driftguard.core.state import AgentState


def _make_state(
    step_id: int,
    tool_name: str = "tool",
    observation: str = "ok",
    diverged: bool = False,
    goal_emb: list[float] | None = None,
    state_emb: list[float] | None = None,
) -> AgentState:
    """Helper to create a test AgentState."""
    return AgentState(
        step_id=step_id,
        action={"tool_name": tool_name, "type": "step"},
        observation=observation,
        goal_embedding=goal_emb or [1.0, 0.0, 0.0],
        state_embedding=state_emb or [1.0, 0.0, 0.0],
        diverged=diverged,
    )


class TestCausalGraph:
    """Test causal graph transitions, queries, and rollback-point search."""

    def test_add_state_and_transition(self) -> None:
        """Add states and transitions, verify graph structure."""
        g = CausalGraph()
        s0 = _make_state(0, "__start__")
        s1 = _make_state(1, "search")
        s2 = _make_state(2, "compute")

        g.add_transition(s0, s1)
        g.add_transition(s1, s2)

        assert len(g) == 3
        assert 0 in g
        assert 1 in g
        assert 2 in g

    def test_get_state(self) -> None:
        """Retrieve a state by step_id."""
        g = CausalGraph()
        s0 = _make_state(0)
        g.add_state(s0)
        assert g.get_state(0) is not None
        assert g.get_state(0).step_id == 0  # type: ignore[union-attr]
        assert g.get_state(99) is None

    def test_get_ancestors(self) -> None:
        """Ancestors are returned in step_id order."""
        g = CausalGraph()
        s0 = _make_state(0)
        s1 = _make_state(1)
        s2 = _make_state(2)
        s3 = _make_state(3)

        g.add_transition(s0, s1)
        g.add_transition(s1, s2)
        g.add_transition(s2, s3)

        ancestors = g.get_ancestors(3)
        assert [a.step_id for a in ancestors] == [0, 1, 2]

    def test_get_descendants(self) -> None:
        """Descendants are returned in step_id order."""
        g = CausalGraph()
        s0 = _make_state(0)
        s1 = _make_state(1)
        s2 = _make_state(2)

        g.add_transition(s0, s1)
        g.add_transition(s1, s2)

        descendants = g.get_descendants(0)
        assert [d.step_id for d in descendants] == [1, 2]

    def test_mark_node_diverged(self) -> None:
        """Marking a node as diverged updates both the graph and state."""
        g = CausalGraph()
        s0 = _make_state(0)
        s1 = _make_state(1)
        g.add_transition(s0, s1)

        g.mark_node_diverged(1)
        state = g.get_state(1)
        assert state is not None
        assert state.diverged is True

    def test_get_safe_rollback_point_simple(self) -> None:
        """Finds the most recent non-diverged ancestor."""
        g = CausalGraph()
        s0 = _make_state(0, "__start__")
        s1 = _make_state(1, "search")
        s2 = _make_state(2, "compute")
        s3 = _make_state(3, "write", diverged=True)

        g.add_transition(s0, s1)
        g.add_transition(s1, s2)
        g.add_transition(s2, s3)
        g.mark_node_diverged(3)

        safe = g.get_safe_rollback_point(3)
        assert safe is not None
        assert safe.step_id == 2

    def test_get_safe_rollback_point_skips_diverged(self) -> None:
        """Skips ancestors that are themselves diverged."""
        g = CausalGraph()
        s0 = _make_state(0)
        s1 = _make_state(1, diverged=True)
        s2 = _make_state(2)
        s3 = _make_state(3, diverged=True)

        g.add_transition(s0, s1)
        g.add_transition(s1, s2)
        g.add_transition(s2, s3)
        g.mark_node_diverged(1)
        g.mark_node_diverged(3)

        safe = g.get_safe_rollback_point(3)
        assert safe is not None
        # S2 is safe (not diverged), and its subtree between it and S3
        # contains no diverged intermediaries
        assert safe.step_id == 2

    def test_get_safe_rollback_point_none_if_all_diverged(self) -> None:
        """Returns None if all ancestors are diverged."""
        g = CausalGraph()
        s0 = _make_state(0, diverged=True)
        s1 = _make_state(1, diverged=True)

        g.add_transition(s0, s1)
        g.mark_node_diverged(0)
        g.mark_node_diverged(1)

        safe = g.get_safe_rollback_point(1)
        assert safe is None

    def test_edge_data(self) -> None:
        """Edge data contains action_type and tool_used."""
        g = CausalGraph()
        s0 = _make_state(0)
        s1 = _make_state(1, "search")

        g.add_transition(s0, s1)

        data = g.get_edge_data("S0", "S1")
        assert data is not None
        assert data["tool_used"] == "search"
        assert data["action_type"] == "step"

    def test_visualize_returns_mermaid(self) -> None:
        """visualize() returns a non-empty Mermaid string."""
        g = CausalGraph()
        s0 = _make_state(0, "__start__")
        s1 = _make_state(1, "search")

        g.add_transition(s0, s1)
        mermaid = g.visualize()

        assert "flowchart TD" in mermaid
        assert "S0" in mermaid
        assert "S1" in mermaid

    def test_empty_graph(self) -> None:
        """Operations on an empty graph return empty results."""
        g = CausalGraph()
        assert len(g) == 0
        assert g.get_state(0) is None
        assert g.get_ancestors(0) == []
        assert g.get_descendants(0) == []
        assert g.get_safe_rollback_point(0) is None
