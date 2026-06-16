"""Tests for Network API Config Dependency Rollbacks."""

from __future__ import annotations

import pytest
from agents_driftguard import AgentUndoWrapper, AgentState
from agents_driftguard.core.detector import ToolFailureDetector


def test_network_error_resource_extraction() -> None:
    """Verify ToolFailureDetector extracts credentials/token files from authentication errors."""
    detector = ToolFailureDetector()
    state = AgentState(
        step_id=1,
        action={"tool_name": "authenticate_api"},
        observation="Error: AuthenticationError: Invalid token read from access_token.json",
        memory=[],
    )
    signal = detector.detect(state, [])
    assert signal.detected is True
    assert signal.failed_resource == "access_token.json"


class MockNetworkAgent:
    """Simulates a network config token failure.
    
    1. Search config properties.
    2. Write access_token.json (outputs 'access_token.json').
    3. Check unrelated network latency (unrelated).
    4. Authenticate using access_token.json -> fails with invalid token.
    5. Re-write access_token.json correctly (runs after rollback to S1).
    6. Authenticate succeeds.
    """
    def __init__(self) -> None:
        self.call_count = 0
        self.config_fixed = False

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        
        if "ROLLBACK" in prompt:
            self.config_fixed = True

        if not self.config_fixed:
            if self.call_count == 1:
                return "ACTION: check_net\nOBSERVATION: network properties checked."
            elif self.call_count == 2:
                return "ACTION: write_token\nOBSERVATION: saved to access_token.json successfully"
            elif self.call_count == 3:
                return "ACTION: latency_check\nOBSERVATION: Latency: 15ms"
            else:
                return "ACTION: authenticate_api\nOBSERVATION: Error: AuthenticationError: Invalid token read from access_token.json"
        else:
            if self.call_count == 3:
                return "ACTION: write_token\nOBSERVATION: saved correct token successfully to access_token.json"
            elif self.call_count == 4:
                return "ACTION: authenticate_api\nOBSERVATION: Authentication Successful! API connection established."
            else:
                return "FINAL ANSWER: API authentication completed successfully."


def test_network_dependency_rollback_e2e() -> None:
    """E2E verification of network config token rollback."""
    mock = MockNetworkAgent()
    wrapper = AgentUndoWrapper(
        executor=mock,
        max_rollbacks=3,
        detectors=["tool_failure"],
        embedding_backend="dummy",
        store="memory",
    )
    result = wrapper.run(goal="Authenticate API connection")

    assert result.success is True
    assert result.rollbacks_performed == 1
    assert result.rollback_history[0].rolled_back_to == 1
    assert result.rollback_history[0].steps_undone == 3  # Undoes S2 and S3
