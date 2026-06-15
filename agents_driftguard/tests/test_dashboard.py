import json
import threading
import time
import urllib.request
from unittest.mock import patch

import pytest

from agents_driftguard.dashboard.app import start_dashboard_server
from agents_driftguard.dashboard.tracker import tracker
from agents_driftguard.core.state import AgentState
from agents_driftguard.core.graph import CausalGraph
from agents_driftguard.core.rollback import RollbackResult


@pytest.fixture(scope="module")
def dashboard_server():
    """Start the dashboard server for tests."""
    thread = start_dashboard_server(port=8055, background=True)
    time.sleep(1)  # Give server time to start
    yield
    # No clean shutdown implemented, thread will die with test process


def test_tracker_lifecycle() -> None:
    # 1. Start run
    run_id = tracker.start_run("Test dashboard telemetry")
    run = tracker.get_run(run_id)
    assert run is not None
    assert run.status == "running"
    
    # 2. Add state
    state = AgentState(
        step_id=1,
        action={"tool_name": "test"},
        observation="Testing",
        memory=[],
        scratchpad="Thinking",
        goal_embedding=[],
        state_embedding=[],
    )
    tracker.add_state(run_id, state)
    
    # 3. Add graph
    graph = CausalGraph()
    graph.add_state(state)
    tracker.update_graph(run_id, graph)
    
    # 4. Add rollback
    rollback = RollbackResult(
        success=True,
        steps_undone=1,
        rolled_back_to=0,
    )
    tracker.add_rollback(run_id, rollback)
    
    # 5. Finish
    tracker.finish_run(run_id, True, "Done")
    
    run = tracker.get_run(run_id)
    assert run.status == "success"
    assert len(run.states) == 1
    assert len(run.rollbacks) == 1
    assert run.graph_mermaid != ""


def test_dashboard_api(dashboard_server) -> None:
    run_id = tracker.start_run("API Test")
    
    # Test /api/runs
    req = urllib.request.Request("http://localhost:8055/api/runs")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert "runs" in data
        assert any(r["run_id"] == run_id for r in data["runs"])
        
    # Test /api/runs/<id>
    req = urllib.request.Request(f"http://localhost:8055/api/runs/{run_id}")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode())
        assert "info" in data
        assert "states" in data
        assert "rollbacks" in data
        assert "graph" in data
        assert data["info"]["run_id"] == run_id
