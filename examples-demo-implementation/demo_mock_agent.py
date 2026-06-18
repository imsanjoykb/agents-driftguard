import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# An agent that encounters multiple failures across its run.
# First a tool failure, then an output quality failure.
call_count = 0
def mock_complex_agent(prompt: str) -> str:
    global call_count
    call_count += 1
    time.sleep(1.5)
    
    if call_count == 1:
        return "ACTION: fetch_users\nOBSERVATION: Retrieved 1000 users."
    elif call_count == 2:
        return "ACTION: clean_data\nOBSERVATION: Data cleaned. 50 nulls removed."
    elif call_count == 3:
        # Failure 1: Tool Error
        return "ACTION: export_to_db\nOBSERVATION: Error: Connection timeout to PostgreSQL."
    elif call_count == 4:
        # Rollback 1 (back to step 2). Tries CSV instead.
        return "ACTION: export_to_csv\nOBSERVATION: Data saved to users.csv locally."
    elif call_count == 5:
        # Failure 2: Extremely short / poor output
        return "ACTION: generate_report\nOBSERVATION: done."
    elif call_count == 6:
        # Rollback 2 (back to step 4). Tries a detailed report.
        return "ACTION: generate_report_detailed\nOBSERVATION: Comprehensive 5-page report generated."
    elif call_count == 7:
        return "ACTION: verify_report_formatting\nOBSERVATION: Verification complete. All formatting checks passed."
    else:
        return "FINAL ANSWER: Simulation completed successfully. DriftGuard detected and recovered from 2 trajectory deviations: (1) a database connection timeout at step 3 was resolved by reverting to step 2 and using a local CSV export; (2) a low-quality draft report at step 5 was resolved by reverting to step 4 and generating a detailed summary. The final verified output is now locally archived."

start_dashboard_server(port=8050)

wrapper = AgentUndoWrapper(
    executor=mock_complex_agent,
    detectors=["tool_failure", "output_quality"], # Multiple detectors!
    max_rollbacks=5,
)

print("🚀 Starting Complex Multi-Failure Agent! Open http://localhost:8050")
result = wrapper.run(goal="Clean user data, export records, and generate a formatted summary report.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Final Output: {result.final_output}")
input("\nPress Enter to exit and close the dashboard...")
