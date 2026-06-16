"""Tests for Compiler Pipeline Dependency Rollbacks."""

from __future__ import annotations

import pytest
from agents_driftguard import AgentUndoWrapper, AgentState
from agents_driftguard.core.detector import ToolFailureDetector


def test_compiler_error_resource_extraction() -> None:
    """Verify ToolFailureDetector extracts module/file names from build/compiler syntax errors."""
    detector = ToolFailureDetector()
    state = AgentState(
        step_id=1,
        action={"tool_name": "compile_build"},
        observation="Error: SyntaxError: unexpected EOF in utils.py",
        memory=[],
    )
    signal = detector.detect(state, [])
    assert signal.detected is True
    assert signal.failed_resource == "utils.py"

    # Testing alternate traceback error string format
    state.observation = "Error: ModuleNotFoundError: No module named 'helpers.py' in config"
    signal = detector.detect(state, [])
    assert signal.detected is True
    assert signal.failed_resource == "helpers.py"


class MockCompilerAgent:
    """Simulates a compiler code syntax failure.
    
    1. Read project requirements.
    2. Write utils.py (outputs 'utils.py' with syntax issue).
    3. Write main.py (outputs 'main.py').
    4. Image assets processing (unrelated).
    5. Compile build -> fails due to syntax error in utils.py.
    6. Re-write utils.py correctly (runs after rollback to S1).
    7. Compile build succeeds.
    """
    def __init__(self) -> None:
        self.call_count = 0
        self.syntax_fixed = False

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        
        if "ROLLBACK" in prompt:
            self.syntax_fixed = True

        if not self.syntax_fixed:
            if self.call_count == 1:
                return "ACTION: read_specs\nOBSERVATION: Spec parameters parsed."
            elif self.call_count == 2:
                return "ACTION: write_file\nOBSERVATION: saved to utils.py successfully"
            elif self.call_count == 3:
                return "ACTION: write_file\nOBSERVATION: saved to main.py successfully"
            elif self.call_count == 4:
                return "ACTION: optimize_images\nOBSERVATION: Resized 3 images."
            else:
                return "ACTION: compile_build\nOBSERVATION: Error compiling build: SyntaxError: unexpected EOF in utils.py"
        else:
            if self.call_count == 3:
                return "ACTION: write_file\nOBSERVATION: saved to utils.py successfully with correct syntax"
            elif self.call_count == 4:
                return "ACTION: compile_build\nOBSERVATION: Build success! Code compiled."
            else:
                return "FINAL ANSWER: Build pipeline completed successfully."


def test_compiler_dependency_rollback_e2e() -> None:
    """E2E verification of compiler pipeline rollback."""
    mock = MockCompilerAgent()
    wrapper = AgentUndoWrapper(
        executor=mock,
        max_rollbacks=3,
        detectors=["tool_failure"],
        embedding_backend="dummy",
        store="memory",
    )
    result = wrapper.run(goal="Compile and build code modules")

    assert result.success is True
    assert result.rollbacks_performed == 1
    assert result.rollback_history[0].rolled_back_to == 1
    assert result.rollback_history[0].steps_undone == 4  # Undoes S2, S3, S4, and S5
