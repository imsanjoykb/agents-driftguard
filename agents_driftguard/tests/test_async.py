import asyncio

import pytest

from agents_driftguard.async_wrapper import AsyncAgentUndoWrapper
from agents_driftguard.core.detector import DivergenceSignal


@pytest.mark.asyncio
async def test_async_wrapper_success() -> None:
    call_count = 0

    async def dummy_agent(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # Simulate network delay
        await asyncio.sleep(0.01)
        if call_count == 1:
            return "ACTION: search\nOBSERVATION: Found data."
        else:
            return "FINAL ANSWER: Done."

    wrapper = AsyncAgentUndoWrapper(executor=dummy_agent, detectors=[])
    result = await wrapper.run("Do something")

    assert result.success
    assert result.total_steps == 2
    assert result.rollbacks_performed == 0


@pytest.mark.asyncio
async def test_async_wrapper_rollback() -> None:
    call_count = 0

    async def failing_agent(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)

        if "ROLLBACK" in prompt:
            return "FINAL ANSWER: Recovered!"

        if call_count == 1:
            return "ACTION: bad_tool\nOBSERVATION: Error: completely failed"
        
        return "FINAL ANSWER: Unreachable"

    wrapper = AsyncAgentUndoWrapper(executor=failing_agent, detectors=["tool_failure"])
    result = await wrapper.run("Try failing")

    assert result.success
    assert result.total_steps == 2
    assert result.rollbacks_performed == 1
    assert "Recovered!" in result.final_output
