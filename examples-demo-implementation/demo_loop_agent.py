import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# A software engineering agent that gets stuck in a loop.
# It runs tests, fails, "fixes" it, runs tests again, and gets the exact same failure.
call_count = 0
def mock_swe_agent(prompt: str) -> str:
    global call_count
    call_count += 1
    time.sleep(1.5)
    
    if call_count == 1:
        return "ACTION: write_code\nOBSERVATION: Code written to main.py"
    elif call_count == 2:
        return "ACTION: run_tests\nOBSERVATION: 1 test failed: IndexError."
    elif call_count == 3:
        return "ACTION: fix_bug\nOBSERVATION: Applied a +1 to the array index."
    elif call_count == 4:
        # Loop! Exact same observation and action as step 2!
        return "ACTION: run_tests\nOBSERVATION: 1 test failed: IndexError."
    elif call_count == 5:
        # After rollback to step 3, it tries a different fix
        return "ACTION: try_different_fix\nOBSERVATION: Handled the empty array case explicitly."
    elif call_count == 6:
        return "ACTION: run_tests\nOBSERVATION: All tests passed!"
    elif call_count == 7:
        return "ACTION: commit_code\nOBSERVATION: Code committed."
    else:
        return "FINAL ANSWER: Bug resolved and code committed. DriftGuard successfully detected an infinite execution loop at step 4 (IndexError repeating on run_tests). Reverting the agent to step 3 allowed it to replan, handle the empty array case explicitly at step 5, and successfully pass tests and commit the code."

start_dashboard_server(port=8050)

wrapper = AgentUndoWrapper(
    executor=mock_swe_agent,
    detectors=["loop"], # Loop detector looks at state history hashes over a rolling window
    max_rollbacks=3,
)

print("🚀 Starting SWE Agent! Open http://localhost:8050")
result = wrapper.run(goal="Identify and fix the array IndexError in main.py.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Final Output: {result.final_output}")
input("\nPress Enter to exit and close the dashboard...")
