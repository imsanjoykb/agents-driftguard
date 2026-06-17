from __future__ import annotations
import time
from typing import Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field, model_validator
from agents_driftguard.utils.hashing import compute_state_hash

class AgentState(BaseModel):

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
    def save(self, state: AgentState) -> None:
        ...

    def get(self, step_id: int) -> AgentState | None:
        ...

    def get_all(self) -> list[AgentState]:
        ...

    def get_last_n(self, n: int) -> list[AgentState]:
        ...

    def clear(self) -> None:
        ...

    def truncate_after(self, step_id: int) -> int:
        ...
