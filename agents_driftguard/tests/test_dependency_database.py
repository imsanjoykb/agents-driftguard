"""Tests for Database Schema Dependency Rollbacks."""

from __future__ import annotations

import pytest
from agents_driftguard import AgentUndoWrapper, AgentState
from agents_driftguard.core.detector import ToolFailureDetector
from agents_driftguard.core.graph import CausalGraph


def test_db_error_resource_extraction() -> None:
    """Verify ToolFailureDetector extracts database table names from SQL error strings."""
    detector = ToolFailureDetector()
    state = AgentState(
        step_id=1,
        action={"tool_name": "execute_query"},
        observation="Error: sqlite3.OperationalError: no such table: customers",
        memory=[],
    )
    signal = detector.detect(state, [])
    assert signal.detected is True
    assert signal.failed_resource == "customers"

    # Testing postgres-style error message
    state.observation = "Error: relation 'orders' does not exist at character 15"
    signal = detector.detect(state, [])
    assert signal.detected is True
    assert signal.failed_resource == "orders"


class MockDBAgent:
    """Simulates a database schema constraint failure.
    
    1. Schema design: creates 'customers' table (outputs 'customers').
    2. Audit configuration (unrelated).
    3. Transaction import (unrelated).
    4. Insert query (fails due to 'customers' schema layout constraints).
    5. Corrected Schema design (runs after rollback to S1).
    6. Insert query succeeds.
    """
    def __init__(self) -> None:
        self.call_count = 0
        self.corrected = False

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        
        if "ROLLBACK" in prompt:
            self.corrected = True

        if not self.corrected:
            if self.call_count == 1:
                return "ACTION: parse_metadata\nOBSERVATION: Metadata analysis done."
            elif self.call_count == 2:
                return "ACTION: create_table\nOBSERVATION: saved to customers successfully"
            elif self.call_count == 3:
                return "ACTION: clear_audit_logs\nOBSERVATION: Logs cleared."
            elif self.call_count == 4:
                return "ACTION: import_transactions\nOBSERVATION: Imported 20 records."
            else:
                return (
                    "ACTION: insert_orders\n"
                    "OBSERVATION: Error: sqlite3.OperationalError: foreign key constraint failed on table: customers"
                )
        else:
            if self.call_count == 3:
                return "ACTION: create_table\nOBSERVATION: saved to customers successfully with proper keys"
            elif self.call_count == 4:
                return "ACTION: insert_orders\nOBSERVATION: Records successfully inserted."
            else:
                return "FINAL ANSWER: Database operations completed successfully."


def test_db_dependency_rollback_e2e() -> None:
    """E2E verification of database table constraint rollback."""
    mock = MockDBAgent()
    wrapper = AgentUndoWrapper(
        executor=mock,
        max_rollbacks=3,
        detectors=["tool_failure"],
        embedding_backend="dummy",
        store="memory",
    )
    result = wrapper.run(goal="Import database records")

    assert result.success is True
    assert result.rollbacks_performed == 1
    assert result.rollback_history[0].rolled_back_to == 1
    assert result.rollback_history[0].steps_undone == 4  # Undoes S2, S3, S4, and S5
