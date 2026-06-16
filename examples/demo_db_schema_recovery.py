import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# 1. A simulated database schema migration agent
call_count = 0
def db_agent_executor(prompt: str) -> str:
    global call_count
    call_count += 1
    time.sleep(1.5)  # Pause for presentation visibility
    
    # We check if a rollback context warning is present in the prompt
    if "ROLLBACK" in prompt:
        # Recovery flow: correct table structure
        if call_count == 3:
            return (
                "THOUGHT: Creating the customers table with a correct AUTOINCREMENT PRIMARY KEY.\n"
                "ACTION: execute_query\n"
                "OBSERVATION: saved to customers table successfully with proper key constraint"
            )
        elif call_count == 4:
            return (
                "THOUGHT: Attempting transaction record insertion again.\n"
                "ACTION: insert_orders\n"
                "OBSERVATION: 50 records joined and inserted into customers successfully"
            )
        else:
            return (
                "FINAL ANSWER: Database schema update completed successfully. "
                "DriftGuard detected a foreign key error at step 5 on table 'customers', "
                "rolled back directly to step 1 (bypassing steps 3 and 4), allowing us to fix the schema constraint."
            )
    else:
        # Default failing flow
        if call_count == 1:
            return (
                "THOUGHT: Inspecting database metadata.\n"
                "ACTION: parse_metadata\n"
                "OBSERVATION: Metadata loaded successfully."
            )
        elif call_count == 2:
            return (
                "THOUGHT: Creating customers schema.\n"
                "ACTION: create_table\n"
                "OBSERVATION: saved to customers table successfully (missing PRIMARY KEY key constraint)"
            )
        elif call_count == 3:
            return (
                "THOUGHT: Cleaning up audit logs.\n"
                "ACTION: clear_audit_logs\n"
                "OBSERVATION: Audit logs cleared."
            )
        elif call_count == 4:
            return (
                "THOUGHT: Importing batch transactions.\n"
                "ACTION: import_transactions\n"
                "OBSERVATION: Imported 50 transactions."
            )
        else:
            return (
                "THOUGHT: Linking order transactions to customers table.\n"
                "ACTION: insert_orders\n"
                "OBSERVATION: sqlite3.OperationalError: Error - foreign key constraint failed on table: customers"
            )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=db_agent_executor,
    detectors=["tool_failure"],
    max_rollbacks=3,
)

print("🚀 Starting Database Schema Recovery demo! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Import and link database records securely.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
