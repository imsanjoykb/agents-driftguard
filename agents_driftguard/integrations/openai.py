"""OpenAI function-calling adapter.

Hooks into OpenAI's chat completion API to capture ``tool_calls``
and tool results per step, mapping them to ``AgentState`` objects.

Requires ``openai`` (optional dependency).
"""

from __future__ import annotations

import time
from typing import Any

from agents_driftguard.core.state import AgentState
from agents_driftguard.integrations.base import BaseExecutorAdapter


class OpenAIFunctionCallingAdapter(BaseExecutorAdapter):
    """Adapter for OpenAI function-calling / tool-use agents.

    Manages a single-step tool-call invocation. The caller is responsible
    for the outer loop; this adapter captures one round of tool calls.

    Args:
        client: An ``openai.OpenAI`` client instance.
        model: Model name (default ``"gpt-4o"``).
        tools: List of tool definitions in OpenAI format.
        tool_executor: A callable ``(tool_name, args) -> str`` that executes
            tool calls and returns the result string.

    Example::

        from openai import OpenAI
        from agents_driftguard.integrations.openai import OpenAIFunctionCallingAdapter

        client = OpenAI()
        adapter = OpenAIFunctionCallingAdapter(
            client=client,
            tools=[...],
            tool_executor=my_tool_runner,
        )
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "gpt-4o",
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Any | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.tools = tools or []
        self.tool_executor = tool_executor

    def _get_client(self) -> Any:
        """Lazily initialize the OpenAI client."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai is required for this adapter. "
                "Install it with: pip install agents-driftguard[openai]"
            ) from exc
        self._client = OpenAI()
        return self._client

    def step(self, prompt: str) -> dict[str, Any]:
        """Execute one round of OpenAI chat completion with tool calls.

        Args:
            prompt: The user message to send.

        Returns:
            Dict with ``tool_calls``, ``message``, ``done``, and ``output``.
        """
        client = self._get_client()
        messages = [{"role": "user", "content": prompt}]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.tools:
            kwargs["tools"] = self.tools

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # Check for tool calls
        tool_calls_data: list[dict[str, Any]] = []
        tool_results: list[str] = []

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                import json
                tc_data = {
                    "id": tc.id,
                    "tool_name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
                tool_calls_data.append(tc_data)

                # Execute tool if executor is provided
                if self.tool_executor:
                    result = self.tool_executor(tc.function.name, tc_data["args"])
                    tool_results.append(str(result))
                else:
                    tool_results.append("")

            return {
                "tool_calls": tool_calls_data,
                "tool_results": tool_results,
                "message": message.content or "",
                "done": False,
                "output": "; ".join(tool_results),
            }

        # No tool calls — the model is done
        return {
            "tool_calls": [],
            "tool_results": [],
            "message": message.content or "",
            "done": True,
            "output": message.content or "",
        }

    def parse_state(
        self,
        raw: dict[str, Any],
        step_id: int,
        goal_embedding: list[float],
        state_embedding: list[float],
        memory: list[dict[str, Any]],
    ) -> AgentState:
        """Convert OpenAI step results into an ``AgentState``.

        Args:
            raw: Result dict from ``step()``.
            step_id: Current step index.
            goal_embedding: Fixed goal embedding.
            state_embedding: Current context embedding.
            memory: Conversation history.

        Returns:
            Populated ``AgentState``.
        """
        tool_calls = raw.get("tool_calls", [])
        if tool_calls:
            # Use the first tool call as the primary action
            tc = tool_calls[0]
            action = {
                "tool_name": tc["tool_name"],
                "args": tc.get("args", {}),
                "type": "tool_call",
            }
        else:
            action = {
                "tool_name": "__response__",
                "type": "response",
            }

        observation = raw.get("output", raw.get("message", ""))

        return AgentState(
            step_id=step_id,
            timestamp=time.time(),
            action=action,
            observation=observation,
            memory=memory,
            scratchpad=raw.get("message", ""),
            goal_embedding=goal_embedding,
            state_embedding=state_embedding,
            metadata={
                "framework": "openai",
                "tool_calls": tool_calls,
            },
        )

    def is_done(self, raw: dict[str, Any]) -> bool:
        """Check if the model has finished (no more tool calls).

        Args:
            raw: Result dict from ``step()``.
        """
        return raw.get("done", False)

    def get_output(self, raw: dict[str, Any]) -> str:
        """Extract the output.

        Args:
            raw: Result dict from ``step()``.
        """
        return raw.get("output", raw.get("message", ""))

    def parse_step(self, raw: dict[str, Any]) -> Any:
        from agents_driftguard.__init__ import StepResult

        tool_calls = raw.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            action = {
                "tool_name": tc["tool_name"],
                "args": tc.get("args", {}),
                "type": "tool_call",
            }
        else:
            action = {
                "tool_name": "__response__",
                "type": "response",
            }

        observation = raw.get("output", raw.get("message", ""))
        
        return StepResult(
            action=action,
            observation=observation,
            scratchpad=raw.get("message", ""),
            done=self.is_done(raw),
            raw_output=str(raw),
        )
