from unittest.mock import patch

from agents_driftguard import AgentUndoWrapper
from agents_driftguard.integrations.openai import OpenAIFunctionCallingAdapter


def test_openai_integration_success() -> None:
    # 1. Mock the client
    class MockChoices:
        def __init__(self, message):
            self.message = message

    class MockResponse:
        def __init__(self, message):
            self.choices = [MockChoices(message)]

    class MockMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class MockToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            class MockFunction:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = arguments
            self.function = MockFunction(name, arguments)

    # 2. Set up the adapter
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for info",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    def mock_tools(name, kwargs):
        return f"Result for {kwargs.get('query')}"

    call_count = 0
    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockResponse(MockMessage(tool_calls=[
                MockToolCall("call_1", "search", '{"query": "test"}')
            ]))
        else:
            return MockResponse(MockMessage(content="Final answer: found it."))

    # Note: We pass None for client, since we mock it
    adapter = OpenAIFunctionCallingAdapter(
        client=None,
        model="gpt-4o-mini",
        tools=tools,
        tool_executor=mock_tools,
    )

    # Patch the adapter's client call directly
    with patch.object(adapter, "_get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.side_effect = mock_create
        wrapper = AgentUndoWrapper(
            executor=adapter,
            step_parser=adapter.parse_step,
            detectors=[],
        )
        result = wrapper.run("Do a search")

        assert result.success
        assert result.total_steps == 2
        assert "Final answer: found it." in result.final_output


def test_openai_integration_tool_failure() -> None:
    # Set up similar mocks
    class MockChoices:
        def __init__(self, message):
            self.message = message

    class MockResponse:
        def __init__(self, message):
            self.choices = [MockChoices(message)]

    class MockMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

    class MockToolCall:
        def __init__(self, id, name, arguments):
            self.id = id
            class MockFunction:
                def __init__(self, name, arguments):
                    self.name = name
                    self.arguments = arguments
            self.function = MockFunction(name, arguments)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]

    def mock_tools(name, kwargs):
        if "bad" in kwargs.get("query", ""):
            return "Error: 500 server unavailable"
        return "Success"

    call_count = 0
    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        # First try fails
        if call_count == 1:
            return MockResponse(MockMessage(tool_calls=[
                MockToolCall("call_1", "search", '{"query": "bad"}')
            ]))
        
        # After rollback, it will get the failure context
        # And hopefully try something else (we simulate that here)
        if call_count == 2:
            return MockResponse(MockMessage(tool_calls=[
                MockToolCall("call_2", "search", '{"query": "good"}')
            ]))
            
        return MockResponse(MockMessage(content="Final answer: Done."))

    adapter = OpenAIFunctionCallingAdapter(
        client=None,
        model="gpt-4o-mini",
        tools=tools,
        tool_executor=mock_tools,
    )

    with patch.object(adapter, "_get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.side_effect = mock_create
        wrapper = AgentUndoWrapper(
            executor=adapter,
            step_parser=adapter.parse_step,
            detectors=["tool_failure"],
        )
        result = wrapper.run("Do a search")

        assert result.success
        assert result.total_steps == 3
        assert result.rollbacks_performed == 1
