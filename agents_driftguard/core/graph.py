from __future__ import annotations
from typing import Any
import networkx as nx  # type: ignore[import-untyped]
from agents_driftguard.core.state import AgentState
from agents_driftguard.utils.embeddings import cosine_similarity

class CausalGraph:

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_state(self, state: AgentState) -> None:
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
        node_id = f"S{step_id}"
        if node_id in self._graph:
            self._graph.nodes[node_id]["diverged"] = True
            state: AgentState = self._graph.nodes[node_id]["state"]
            state.diverged = True

    def get_state(self, step_id: int) -> AgentState | None:
        node_id = f"S{step_id}"
        if node_id in self._graph:
            return self._graph.nodes[node_id]["state"]
        return None

    def get_ancestors(self, step_id: int) -> list[AgentState]:
        node_id = f"S{step_id}"
        if node_id not in self._graph:
            return []
        ancestor_ids = nx.ancestors(self._graph, node_id)
        states = [self._graph.nodes[nid]["state"] for nid in ancestor_ids]
        return sorted(states, key=lambda s: s.step_id)

    def get_descendants(self, step_id: int) -> list[AgentState]:
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
        failed_resource: str | None = None,
    ) -> AgentState | None:
        ancestors = self.get_ancestors(from_step_id)

        # 1. Dependency-aware rollback traversal
        if failed_resource:
            for state in reversed(ancestors):
                if failed_resource in state.outputs or failed_resource in state.inputs:
                    target_step = state.step_id - 1
                    target_state = self.get_state(target_step) if target_step >= 0 else None
                    if target_state and not target_state.diverged:
                        return target_state
                    elif target_step == -1 or target_step == 0:
                        return self.get_state(0)

        # 2. Standard safe rollback point logic
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
            )

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
        return self._graph.get_edge_data(from_id, to_id)

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._graph)

    def __contains__(self, step_id: int) -> bool:
        """Check if a step ID exists in the graph."""
        return f"S{step_id}" in self._graph

    def visualize(self, rollback_path: list[int] | None = None) -> str:
        from agents_driftguard.utils.visualize import visualize_graph

        return visualize_graph(self, rollback_path=rollback_path)
