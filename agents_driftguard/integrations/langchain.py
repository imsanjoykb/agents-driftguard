from __future__ import annotations
import time
from typing import Any
from agents_driftguard.core.state import AgentState
from agents_driftguard.integrations.base import BaseExecutorAdapter


class LangChainAdapter(BaseExecutorAdapter):
    def __init__(self, agent_executor: Any) -> None:
        self._executor = agent_executor
        self._last_result: dict[str, Any] = {}

    def step(self, prompt: str) -> dict[str, Any]:
        try:
            from langchain_core.agents import AgentAction  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "langchain-core is required for the LangChain adapter. "
                "Install it with: pip install agents-driftguard[langchain]"
            ) from exc

        result = self._executor.invoke({"input": prompt})
        self._last_result = result

        # Extract intermediate steps
        intermediate_steps = result.get("intermediate_steps", [])
        parsed_steps: list[dict[str, Any]] = []

        for action, observation in intermediate_steps:
            step_dict: dict[str, Any] = {
                "tool_name": getattr(action, "tool", "unknown"),
                "args": getattr(action, "tool_input", {}),
                "log": getattr(action, "log", ""),
            }
            parsed_steps.append({
                "action": step_dict,
                "observation": str(observation),
            })

        return {
            "output": result.get("output", ""),
            "intermediate_steps": parsed_steps,
            "done": len(parsed_steps) == 0,
        }

    def parse_state(
        self,
        raw: dict[str, Any],
        step_id: int,
        goal_embedding: list[float],
        state_embedding: list[float],
        memory: list[dict[str, Any]],
    ) -> AgentState:
        steps = raw.get("intermediate_steps", [])
        if steps:
            last_step = steps[-1]
            action = last_step["action"]
            observation = last_step["observation"]
        else:
            action = {"tool_name": "__output__"}
            observation = raw.get("output", "")

        return AgentState(
            step_id=step_id,
            timestamp=time.time(),
            action=action,
            observation=observation,
            memory=memory,
            scratchpad=str(raw.get("intermediate_steps", "")),
            goal_embedding=goal_embedding,
            state_embedding=state_embedding,
            metadata={"framework": "langchain"},
        )

    def is_done(self, raw: dict[str, Any]) -> bool:
        return raw.get("done", True)

    def get_output(self, raw: dict[str, Any]) -> str:
        return raw.get("output", "")

    def parse_step(self, raw: dict[str, Any]) -> Any:
        from agents_driftguard.__init__ import StepResult
        
        steps = raw.get("intermediate_steps", [])
        if steps:
            last_step = steps[-1]
            action = last_step["action"]
            observation = last_step["observation"]
            scratchpad = ""
        else:
            action = {"tool_name": "__output__"}
            observation = raw.get("output", "")
            scratchpad = ""

        return StepResult(
            action=action,
            observation=observation,
            scratchpad=scratchpad,
            done=self.is_done(raw),
            raw_output=str(raw),
        )
