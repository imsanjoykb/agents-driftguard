"""Agent state model and abstract state-store protocol.

Defines the ``AgentState`` Pydantic v2 model that captures a full snapshot of
the agent at one step, plus the ``StateStore`` protocol for persistence backends.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from agents_driftguard.utils.hashing import compute_state_hash


class AgentState(BaseModel):
    """Immutable snapshot of an agent's state at a single step.

    Attributes:
        step_id: Monotonically-increasing step index (0-based).
        timestamp: Unix epoch timestamp of when this state was created.
        action: Dict describing the action taken (e.g. ``{tool_name, args}``).
        observation: The tool/environment output string.
        memory: Full conversation history snapshot (list of message dicts).
        scratchpad: Agent reasoning trace / chain-of-thought.
        goal_embedding: Embedding of the original goal (fixed for a run).
        state_embedding: Embedding of the current context (changes each step).
        metadata: Arbitrary extra information.
        diverged: Whether this state has been marked as diverged.
        state_hash: Deterministic hash auto-computed from core fields.
    """

    step_id: int
    timestamp: float = Field(default_factory=time.time)
    action: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""
    memory: list[dict[str, Any]] = Field(default_factory=list)
    scratchpad: str = ""
    goal_embedding: list[float] = Field(default_factory=list)
    state_embedding: list[float] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    diverged: bool = False
    state_hash: str = ""

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _auto_hash(self) -> "AgentState":
        """Auto-compute ``state_hash`` after model initialization if not set."""
        if not self.state_hash:
            self.state_hash = compute_state_hash(
                action=self.action,
                observation=self.observation,
                memory=self.memory,
                scratchpad=self.scratchpad,
                inputs=self.inputs,
                outputs=self.outputs,
            )
        return self

    @property
    def node_id(self) -> str:
        """Canonical node identifier for use in the causal graph."""
        return f"S{self.step_id}"


@runtime_checkable
class StateStore(Protocol):
    """Protocol defining the interface for state persistence backends.

    All state stores must implement these five methods.
    """

    def save(self, state: AgentState) -> None:
        """Persist an agent state snapshot.

        Args:
            state: The state to save.
        """
        ...

    def get(self, step_id: int) -> AgentState | None:
        """Retrieve a state by step ID.

        Args:
            step_id: The step index to look up.

        Returns:
            The matching ``AgentState``, or ``None`` if not found.
        """
        ...

    def get_all(self) -> list[AgentState]:
        """Return all stored states ordered by step_id.

        Returns:
            List of all ``AgentState`` objects, sorted by ``step_id``.
        """
        ...

    def get_last_n(self, n: int) -> list[AgentState]:
        """Return the most recent *n* states.

        Args:
            n: Number of recent states to return.

        Returns:
            Up to *n* most recent ``AgentState`` objects.
        """
        ...

    def clear(self) -> None:
        """Remove all stored states."""
        ...

    def truncate_after(self, step_id: int) -> int:
        """Remove all states with ``step_id`` greater than the given value.

        Args:
            step_id: Keep states up to and including this step.

        Returns:
            Number of states removed.
        """
        ...
