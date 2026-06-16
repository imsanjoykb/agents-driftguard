import time
import sys
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# ANSI colors for beautiful console output
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"

# =====================================================================
# SCENARIO 1: Config File Permissions / Paths
# =====================================================================
class ConfigAgent:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.call_count = 0
        self.config_fixed = False

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        time.sleep(1.0)
        
        if "ROLLBACK" in prompt:
            if self.mode == "dependency-aware":
                self.config_fixed = True

        print(f"  {CYAN}[Agent Call #{self.call_count}]{RESET}")
        
        if not self.config_fixed:
            if self.call_count == 1:
                return "ACTION: search_parameters\nOBSERVATION: Found parameters: {key_id: 992}"
            elif self.call_count == 2:
                return "ACTION: write_file\nARGS: {'path': 'config.json'}\nOBSERVATION: saved to config.json successfully"
            elif self.call_count == 3:
                return "ACTION: compute_averages\nOBSERVATION: Average processing time: 12.4ms"
            elif self.call_count == 4:
                return "ACTION: query_weather\nOBSERVATION: Weather conditions: Sunny, 22C"
            else:
                return "ACTION: authenticate_api\nARGS: {'file': 'config.json'}\nOBSERVATION: Error: Permission denied or corrupted format in config.json"
        else:
            if self.call_count == 3:
                return "ACTION: write_file\nARGS: {'path': 'config_fallback.json'}\nOBSERVATION: saved to config_fallback.json successfully"
            elif self.call_count == 4:
                return "ACTION: authenticate_api\nARGS: {'file': 'config_fallback.json'}\nOBSERVATION: Authentication Successful! API Token generated."
            else:
                return (
                    "FINAL ANSWER: Config Scenario Complete. DriftGuard successfully detected the dependency "
                    "error on config.json, bypassed weather/stats, and rolled back to step 1."
                )

# =====================================================================
# SCENARIO 2: Database Table Constraints
# =====================================================================
class DatabaseAgent:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.call_count = 0
        self.schema_fixed = False

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        time.sleep(1.0)
        
        if "ROLLBACK" in prompt:
            if self.mode == "dependency-aware":
                self.schema_fixed = True

        print(f"  {CYAN}[Agent Call #{self.call_count}]{RESET}")
        
        if not self.schema_fixed:
            if self.call_count == 1:
                return "ACTION: parse_metadata\nOBSERVATION: Schema analysis completed."
            elif self.call_count == 2:
                return "ACTION: execute_query\nARGS: {'table': 'customers'}\nOBSERVATION: Error - created customers table without PRIMARY KEY constraint"
            elif self.call_count == 3:
                return "ACTION: fetch_log_audit\nOBSERVATION: Audit logs cleared."
            elif self.call_count == 4:
                return "ACTION: import_transactions\nOBSERVATION: 50 transactions imported successfully."
            else:
                return "ACTION: insert_orders\nARGS: {'table': 'customers'}\nOBSERVATION: sqlite3.OperationalError: Error - foreign key constraint failed on table: customers"
        else:
            if self.call_count == 3:
                return "ACTION: execute_query\nARGS: {'table': 'customers'}\nOBSERVATION: Created customers table with correct AUTOINCREMENT PRIMARY KEY"
            elif self.call_count == 4:
                return "ACTION: insert_orders\nARGS: {'table': 'customers'}\nOBSERVATION: Records joined and inserted successfully."
            else:
                return (
                    "FINAL ANSWER: Database Scenario Complete. DriftGuard detected the schema failure "
                    "on table 'customers', bypassed audit/import steps, and rolled back to correct the schema query."
                )

# =====================================================================
# SCENARIO 3: Compiler Code Build Pipelines
# =====================================================================
class CompilerAgent:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.call_count = 0
        self.syntax_fixed = False

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        time.sleep(1.0)
        
        if "ROLLBACK" in prompt:
            if self.mode == "dependency-aware":
                self.syntax_fixed = True

        print(f"  {CYAN}[Agent Call #{self.call_count}]{RESET}")
        
        if not self.syntax_fixed:
            if self.call_count == 1:
                return "ACTION: read_specs\nOBSERVATION: Spec parameters parsed."
            elif self.call_count == 2:
                return "ACTION: write_file\nARGS: {'path': 'utils.py'}\nOBSERVATION: saved to utils.py with a syntax error: missing closing parenthesis"
            elif self.call_count == 3:
                return "ACTION: write_file\nARGS: {'path': 'main.py'}\nOBSERVATION: saved to main.py successfully"
            elif self.call_count == 4:
                return "ACTION: optimize_images\nOBSERVATION: Resized 3 images."
            else:
                return "ACTION: compile_build\nOBSERVATION: Error compiling build: SyntaxError: unexpected EOF in utils.py"
        else:
            if self.call_count == 3:
                return "ACTION: write_file\nARGS: {'path': 'utils.py'}\nOBSERVATION: saved to utils.py with correct syntax brackets"
            elif self.call_count == 4:
                return "ACTION: compile_build\nOBSERVATION: Build success! Code compiled."
            else:
                return (
                    "FINAL ANSWER: Compiler Scenario Complete. DriftGuard traced compile error to utils.py, "
                    "bypassed image/main file updates, and rolled back directly to correct utils.py."
                )

# =====================================================================
# Execution Runners
# =====================================================================
def run_scenario(scenario_num: int, name: str, agent_class, goal: str):
    print(f"\n========================================================")
    print(f"{BOLD}{MAGENTA}SCENARIO {scenario_num}: {name}{RESET}")
    print(f"========================================================")
    
    # 1. Linear Rollback simulation
    print(f"\n{BOLD}{RED}--> Running with Linear Rollback (Old Policy){RESET}")
    agent_lin = agent_class(mode="linear")
    
    class LinearWrapper(AgentUndoWrapper):
        def run(self, goal: str):
            original_detect = self._detector.detect
            def mock_detect(*args, **kwargs):
                sig = original_detect(*args, **kwargs)
                sig.failed_resource = None  # Force linear
                return sig
            self._detector.detect = mock_detect
            return super().run(goal)

    wrapper_lin = LinearWrapper(
        executor=agent_lin,
        max_rollbacks=2,
        detectors=["tool_failure"],
        embedding_backend="dummy",
        store="memory",
        max_steps=8,
    )
    res_lin = wrapper_lin.run(goal=goal)
    print(f"Linear Policy Result: Success={res_lin.success}, Rollbacks={res_lin.rollbacks_performed}")
    
    # 2. Dependency-Aware Rollback simulation
    print(f"\n{BOLD}{GREEN}--> Running with Dependency-Aware Rollback (DriftGuard V2){RESET}")
    agent_dep = agent_class(mode="dependency-aware")
    wrapper_dep = AgentUndoWrapper(
        executor=agent_dep,
        max_rollbacks=3,
        detectors=["tool_failure"],
        embedding_backend="dummy",
        store="memory",
    )
    res_dep = wrapper_dep.run(goal=goal)
    print(f"Dependency-Aware Policy Result: Success={res_dep.success}, Rollbacks={res_dep.rollbacks_performed}")
    print(f"Final Answer: {res_dep.final_output}")

if __name__ == "__main__":
    print(f"{BOLD}{MAGENTA}=== DriftGuard Causal Dependency-Aware Rollback Demo ==={RESET}")
    print("Starting dashboard server in background on http://localhost:8050...")
    start_dashboard_server(port=8050)
    
    run_scenario(1, "Config File Permissions (File IO)", ConfigAgent, "Authenticate API and query telemetry")
    time.sleep(2)
    
    run_scenario(2, "Database Schema Constraints (Database Table)", DatabaseAgent, "Import and link customers database records")
    time.sleep(2)
    
    run_scenario(3, "Compiler Code Build Pipeline (Code Compiler)", CompilerAgent, "Compile and build code modules")
    
    print("\n--------------------------------------------------------")
    print(f"{BOLD}{GREEN}Demo execution complete.{RESET} Keep this window open and visit")
    print("http://localhost:8050 to see the execution trajectories on the dashboard!")
    print("Press Enter to exit...")
    input()
