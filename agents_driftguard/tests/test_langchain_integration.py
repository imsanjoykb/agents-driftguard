from typing import Any
from agents_driftguard import AgentUndoWrapper, AgentState
from agents_driftguard.integrations.langchain import LangChainAdapter

def test_langchain_integration_success() -> None:
    class MockAgentExecutor:
        def __init__(self):
            self.call_count = 0
            
        def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "output": "I am thinking...",
                    "intermediate_steps": [
                        (
                            type("MockAction", (), {"tool": "search", "tool_input": "test", "log": "Action: search\n"}),
                            "Search result: found."
                        )
                    ]
                }
            else:
                return {
                    "output": "Final answer: Done.",
                    "intermediate_steps": []
                }

    mock_executor = MockAgentExecutor()
    adapter = LangChainAdapter(agent_executor=mock_executor)

    wrapper = AgentUndoWrapper(
        executor=adapter,
        step_parser=adapter.parse_step,
        detectors=[],
    )
    result = wrapper.run("Do a search")

    assert result.success
    assert result.total_steps == 2
    assert "Final answer: Done." in result.final_output


def test_langchain_integration_tool_failure() -> None:
    class MockAgentExecutor:
        def __init__(self):
            self.call_count = 0
            
        def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "output": "I am thinking...",
                    "intermediate_steps": [
                        (
                            type("MockAction", (), {"tool": "search", "tool_input": "bad", "log": "Action: search\n"}),
                            "Error: timeout"
                        )
                    ]
                }
            elif self.call_count == 2:
                # After rollback
                return {
                    "output": "I am thinking...",
                    "intermediate_steps": [
                        (
                            type("MockAction", (), {"tool": "search", "tool_input": "good", "log": "Action: search\n"}),
                            "Search result: ok."
                        )
                    ]
                }
            else:
                return {
                    "output": "Final answer: Done.",
                    "intermediate_steps": []
                }

    mock_executor = MockAgentExecutor()
    adapter = LangChainAdapter(agent_executor=mock_executor)

    wrapper = AgentUndoWrapper(
        executor=adapter,
        step_parser=adapter.parse_step,
        detectors=["tool_failure"],
    )
    result = wrapper.run("Do a search")

    assert result.success
    assert result.total_steps == 3
    assert result.rollbacks_performed == 1
