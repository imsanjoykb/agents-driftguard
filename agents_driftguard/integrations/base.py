"""Abstract base executor adapter interface.

Integration adapters translate framework-specific agent internals
(LangChain intermediate steps, OpenAI tool_calls, etc.) into the
uniform ``AgentState`` format that DriftGuard operates on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents_driftguard.core.state import AgentState


class BaseExecutorAdapter(ABC):
    """Abstract adapter that bridges a framework's agent executor with agents_driftguard.

    Subclasses must implement:

    - ``step(prompt)`` — execute one step and return raw results.
    - ``parse_state(raw, step_id, ...)`` — convert raw results to ``AgentState``.
    """

    @abstractmethod
    def step(self, prompt: str) -> dict[str, Any]:
        """Execute a single agent step.

        Args:
            prompt: The current prompt / input for the agent.

        Returns:
            A dict with framework-specific step results.
        """
        ...

    def __call__(self, prompt: str) -> Any:
        """Allow the adapter to be called directly as an executor."""
        return self.step(prompt)

    @abstractmethod
    def parse_step(self, raw: Any) -> Any:
        """Convert raw framework output into a StepResult.
        
        Args:
            raw: Raw step results from ``step()``.
            
        Returns:
            A StepResult object.
        """
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
        """Convert raw framework output into an ``AgentState``.

        Args:
            raw: Raw step results from ``step()``.
            step_id: Current step index.
            goal_embedding: Fixed goal embedding for this run.
            state_embedding: Embedding of the current context.
            memory: Conversation history snapshot.

        Returns:
            A fully-populated ``AgentState``.
        """
        ...

    @abstractmethod
    def is_done(self, raw: dict[str, Any]) -> bool:
        """Check whether the agent has signalled completion.

        Args:
            raw: Raw step results from ``step()``.

        Returns:
            ``True`` if the agent is finished.
        """
        ...

    @abstractmethod
    def get_output(self, raw: dict[str, Any]) -> str:
        """Extract the agent's output text from step results.

        Args:
            raw: Raw step results from ``step()``.

        Returns:
            The output string.
        """
        ...
