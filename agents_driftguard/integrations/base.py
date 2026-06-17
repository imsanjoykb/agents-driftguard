from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from agents_driftguard.core.state import AgentState


class BaseExecutorAdapter(ABC):
    @abstractmethod
    def step(self, prompt: str) -> dict[str, Any]:
        ...

    def __call__(self, prompt: str) -> Any:
        """Allow the adapter to be called directly as an executor."""
        return self.step(prompt)

    @abstractmethod
    def parse_step(self, raw: Any) -> Any:
        ...

    @abstractmethod
    def parse_state(
        self,
        raw: dict[str, Any],
        step_id: int,
        goal_embedding: list[float],
        state_embedding: list[float],
        memory: list[dict[str, Any]],
    ) -> AgentState:
        ...

    @abstractmethod
    def is_done(self, raw: dict[str, Any]) -> bool:
        ...

    @abstractmethod
    def get_output(self, raw: dict[str, Any]) -> str:
        ...
