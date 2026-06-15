"""Causal graph over agent state transitions.

Wraps a ``networkx.DiGraph`` where each node is an ``AgentState`` and each
edge ``Si → Sj`` means *Sj causally depends on Si*. Edges carry metadata
such as the tool used and a confidence score.
"""

from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]

from agents_driftguard.core.state import AgentState
from agents_driftguard.utils.embeddings import cosine_similarity


class CausalGraph:
    """Directed acyclic graph tracking causal dependencies between agent states.

    Each node stores the full ``AgentState`` as a node attribute. Edges are
    annotated with ``{action_type, tool_used, confidence}``.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_state(self, state: AgentState) -> None:
        """Add a standalone state node (no incoming edge).

        Args:
            state: The agent state to add.
        """
        self._graph.add_node(
            state.node_id,
            state=state,
            diverged=state.diverged,
        )

    def add_transition(
        self,
        from_state: AgentState,
        to_state: AgentState,
        confidence: float = 1.0,
    ) -> None:
        """Record a causal transition ``from_state → to_state``.

        Both nodes are upserted (created if absent). The edge carries
        metadata extracted from ``to_state.action``.

        Args:
            from_state: The predecessor state.
            to_state: The successor state.
            confidence: Confidence score for this transition (0.0–1.0).
        """
        # Upsert nodes
        for s in (from_state, to_state):
            self._graph.add_node(s.node_id, state=s, diverged=s.diverged)

        tool_used = to_state.action.get("tool_name", "unknown")
        action_type = to_state.action.get("type", "step")

        self._graph.add_edge(
            from_state.node_id,
            to_state.node_id,
            action_type=action_type,
            tool_used=tool_used,
            confidence=confidence,
        )

    def mark_node_diverged(self, step_id: int) -> None:
        """Mark a node as diverged.

        Args:
            step_id: The step ID whose node should be marked.
        """
        node_id = f"S{step_id}"
        if node_id in self._graph:
            self._graph.nodes[node_id]["diverged"] = True
            state: AgentState = self._graph.nodes[node_id]["state"]
            state.diverged = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_state(self, step_id: int) -> AgentState | None:
        """Retrieve the ``AgentState`` for a given step.

        Args:
            step_id: The step ID to look up.

        Returns:
            The ``AgentState`` or ``None`` if not found.
        """
        node_id = f"S{step_id}"
        if node_id in self._graph:
            return self._graph.nodes[node_id]["state"]
        return None

    def get_ancestors(self, step_id: int) -> list[AgentState]:
        """Return all ancestor states of the given step, ordered by step_id.

        Args:
            step_id: The step whose ancestors to find.

        Returns:
            List of ancestor ``AgentState`` objects, sorted by ``step_id``.
        """
        node_id = f"S{step_id}"
        if node_id not in self._graph:
            return []
        ancestor_ids = nx.ancestors(self._graph, node_id)
        states = [self._graph.nodes[nid]["state"] for nid in ancestor_ids]
        return sorted(states, key=lambda s: s.step_id)

    def get_descendants(self, step_id: int) -> list[AgentState]:
        """Return all descendant states of the given step.

        Args:
            step_id: The step whose descendants to find.

        Returns:
            List of descendant ``AgentState`` objects, sorted by ``step_id``.
        """
        node_id = f"S{step_id}"
        if node_id not in self._graph:
            return []
        desc_ids = nx.descendants(self._graph, node_id)
        states = [self._graph.nodes[nid]["state"] for nid in desc_ids]
        return sorted(states, key=lambda s: s.step_id)

    def get_safe_rollback_point(
        self,
        from_step_id: int,
        goal_sim_threshold: float = 0.75,
    ) -> AgentState | None:
        """Find the most recent safe ancestor to roll back to.

        A state is "safe" when:
        1. It is not marked as diverged.
        2. Its ``goal_embedding`` ↔ ``state_embedding`` cosine similarity
           exceeds ``goal_sim_threshold``.
        3. No descendant between it and ``from_step_id`` used a tool that
           is also marked as failed (heuristic: any diverged node in its
           subtree has a failed tool).

        Args:
            from_step_id: The step from which to start searching backwards.
            goal_sim_threshold: Minimum cosine similarity between goal and
                state embeddings.

        Returns:
            The safest recent ``AgentState`` to roll back to, or ``None``
            if no safe point exists.
        """
        ancestors = self.get_ancestors(from_step_id)
        # Walk backwards (most recent first)
        for state in reversed(ancestors):
            if state.diverged:
                continue

            # Check goal alignment
            if state.goal_embedding and state.state_embedding:
                sim = cosine_similarity(state.goal_embedding, state.state_embedding)
                if sim < goal_sim_threshold:
                    continue

            # Check subtree for diverged nodes (between this state and from_step_id)
            subtree = self.get_descendants(state.step_id)
            subtree_diverged = any(
                s.diverged
                for s in subtree
                if s.step_id < from_step_id and s.step_id > state.step_id
                # Only check nodes that are NOT the from_step_id itself
                # (since we already know that one is diverged)
            )
            # If there are diverged nodes in the subtree between this state
            # and the divergence point, skip — the path is tainted
            if subtree_diverged:
                continue

            return state

        return None

    @property
    def nodes(self) -> list[str]:
        """Return all node IDs."""
        return list(self._graph.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Return all edges as ``(from_id, to_id)`` pairs."""
        return list(self._graph.edges)

    def get_edge_data(self, from_id: str, to_id: str) -> dict[str, Any] | None:
        """Return edge metadata between two nodes.

        Args:
            from_id: Source node ID.
            to_id: Target node ID.

        Returns:
            Edge data dict or ``None``.
        """
        return self._graph.get_edge_data(from_id, to_id)

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._graph)

    def __contains__(self, step_id: int) -> bool:
        """Check if a step ID exists in the graph."""
        return f"S{step_id}" in self._graph

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, rollback_path: list[int] | None = None) -> str:
        """Export the causal graph as a Mermaid flowchart string.

        Args:
            rollback_path: Optional list of step_ids forming the rollback path
                (edges on this path will be rendered in bold).

        Returns:
            Mermaid-formatted flowchart string.
        """
        from agents_driftguard.utils.visualize import visualize_graph

        return visualize_graph(self, rollback_path=rollback_path)
