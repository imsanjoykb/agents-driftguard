"""Mermaid flowchart visualization of a ``CausalGraph``.

Produces a Mermaid-formatted string where:
- **Green** nodes (``:::safe``) = non-diverged states
- **Red** nodes (``:::diverged``) = diverged states
- **Bold** edges = rollback path (if provided)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents_driftguard.core.graph import CausalGraph


def visualize_graph(
    causal_graph: CausalGraph,
    rollback_path: list[int] | None = None,
) -> str:
    """Export a ``CausalGraph`` as a Mermaid flowchart string.

    Args:
        causal_graph: The causal graph to visualize.
        rollback_path: Optional list of step_ids whose connecting edges
            should be rendered with bold/thick lines.

    Returns:
        A Mermaid-formatted string that can be pasted into any
        Mermaid-compatible renderer.

    Example output::

        ```mermaid
        flowchart TD
            classDef safe fill:#2d6a4f,stroke:#1b4332,color:#fff
            classDef diverged fill:#d00000,stroke:#6a040f,color:#fff
            S0["S0: __start__"]:::safe
            S1["S1: search"]:::safe
            S2["S2: compute"]:::safe
            S3["S3: write_file"]:::diverged
            S0 --> S1
            S1 --> S2
            S2 --> S3
        ```
    """
    lines: list[str] = [
        "flowchart TD",
        "    classDef safe fill:#2d6a4f,stroke:#1b4332,color:#fff",
        "    classDef diverged fill:#d00000,stroke:#6a040f,color:#fff",
        "    classDef rollback fill:#e85d04,stroke:#dc2f02,color:#fff",
    ]

    rollback_edges: set[tuple[str, str]] = set()
    if rollback_path and len(rollback_path) > 1:
        for i in range(len(rollback_path) - 1):
            rollback_edges.add((f"S{rollback_path[i]}", f"S{rollback_path[i + 1]}"))

    # Nodes
    for node_id in causal_graph.nodes:
        state = causal_graph.get_state(int(node_id[1:]))  # strip "S" prefix
        if state is None:
            continue
        tool_name = state.action.get("tool_name", "?")
        label = f'{node_id}["{node_id}: {_escape_mermaid(tool_name)}"]'
        css_class = "diverged" if state.diverged else "safe"
        lines.append(f"    {label}:::{css_class}")

    # Edges
    for from_id, to_id in causal_graph.edges:
        edge_data = causal_graph.get_edge_data(from_id, to_id)
        if (from_id, to_id) in rollback_edges:
            # Bold edge for rollback path
            lines.append(f"    {from_id} ==> {to_id}")
        else:
            # Normal edge
            if edge_data and edge_data.get("tool_used"):
                tool = _escape_mermaid(edge_data["tool_used"])
                lines.append(f"    {from_id} -->|{tool}| {to_id}")
            else:
                lines.append(f"    {from_id} --> {to_id}")

    return "\n".join(lines)


def _escape_mermaid(text: str) -> str:
    """Escape special characters for Mermaid labels.

    Args:
        text: Raw text to escape.

    Returns:
        Mermaid-safe string.
    """
    # Remove quotes and brackets that break Mermaid syntax
    return (
        text.replace('"', "'")
        .replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
        .replace("<", "‹")
        .replace(">", "›")
    )
