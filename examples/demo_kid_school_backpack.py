import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# Simulation state variables tracking the backpack packing steps
call_count = 0
homework_fixed = False
lunch_fixed = False
water_fixed = False

def school_agent_executor(prompt: str) -> str:
    global call_count, homework_fixed, lunch_fixed, water_fixed
    call_count += 1
    time.sleep(1.5)  # Pause for presentation visibility
    
    # Trace the feedback injected in prompt during rollback recovery
    if "ROLLBACK" in prompt:
        if "homework.txt" in prompt:
            homework_fixed = True
        elif "lunchbox" in prompt:
            lunch_fixed = True
        elif "water_bottle" in prompt:
            water_fixed = True

    print(f"  [Student Call #{call_count}] (HomeworkFixed={homework_fixed}, LunchFixed={lunch_fixed}, WaterFixed={water_fixed})")
    
    # -------------------------------------------------------------
    # Step 1: Check calendar
    # -------------------------------------------------------------
    if call_count == 1:
        return (
            "THOUGHT: Checking schedule calendar.\n"
            "ACTION: check_calendar\n"
            "OBSERVATION: Today's schedule: Math class homework due."
        )
        
    # -------------------------------------------------------------
    # Step 2: Print homework
    # -------------------------------------------------------------
    elif call_count == 2:
        if not homework_fixed:
            return (
                "THOUGHT: Printing assignment page.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to homework.txt successfully (oops, printed English assignment instead of Math!)"
            )
        else:
            return (
                "THOUGHT: Printing the correct Math assignment sheet.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to homework.txt successfully"
            )
            
    # -------------------------------------------------------------
    # Step 3: Pack colored pencils (unrelated)
    # -------------------------------------------------------------
    elif call_count == 3:
        return (
            "THOUGHT: Packing my pencil case.\n"
            "ACTION: pack_pencils\n"
            "OBSERVATION: Colored pencils packed in bag."
        )
        
    # -------------------------------------------------------------
    # Step 4: Show parents homework (Fails if English assignment)
    # -------------------------------------------------------------
    elif call_count == 4:
        if not homework_fixed:
            # Divergence 1: Parents refuse to sign
            return (
                "THOUGHT: Showing homework sheet to parents for review signature.\n"
                "ACTION: sign_homework\n"
                "OBSERVATION: Error: Parents refused to sign homework.txt! It contains English stories, but Math class is scheduled."
            )
        else:
            return (
                "THOUGHT: Showing correct homework to parents.\n"
                "ACTION: sign_homework\n"
                "OBSERVATION: Parents reviewed and signed homework.txt successfully."
            )
            
    # -------------------------------------------------------------
    # Step 5: Pack lunch
    # -------------------------------------------------------------
    elif call_count == 5:
        if not lunch_fixed:
            return (
                "THOUGHT: Preparing lunch sandwich from fridge.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to lunchbox successfully (packed an expired, moldy sandwich!)"
            )
        else:
            return (
                "THOUGHT: Preparing lunch sandwich using fresh bread and peanut butter.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to lunchbox successfully"
            )
            
    # -------------------------------------------------------------
    # Step 6: Fill water container
    # -------------------------------------------------------------
    elif call_count == 6:
        if not water_fixed:
            return (
                "THOUGHT: Filling water bottle container from cupboard jug.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to water_bottle successfully (filled with sour vinegar instead of water!)"
            )
        else:
            return (
                "THOUGHT: Filling water bottle container with fresh clean water.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to water_bottle successfully"
            )
            
    # -------------------------------------------------------------
    # Step 7: Eat lunch at recess (Fails if expired sandwich)
    # -------------------------------------------------------------
    elif call_count == 7:
        if not lunch_fixed:
            # Divergence 2: Food rejected
            return (
                "THOUGHT: Opening lunchbox for snack time.\n"
                "ACTION: eat_lunch\n"
                "OBSERVATION: Error: Student refuses to eat! lunchbox sandwich is moldy and expired"
            )
        else:
            return (
                "THOUGHT: Eating fresh lunch sandwich.\n"
                "ACTION: eat_lunch\n"
                "OBSERVATION: Sandwich eaten successfully."
            )
            
    # -------------------------------------------------------------
    # Step 8: Drink at gym class (Fails if vinegar bottle)
    # -------------------------------------------------------------
    elif call_count == 8:
        if not water_fixed:
            # Divergence 3: Vinegar water
            return (
                "THOUGHT: Drinking from water bottle.\n"
                "ACTION: drink_water\n"
                "OBSERVATION: Error: Student coughs! water_bottle contains sour vinegar instead of fresh water"
            )
        else:
            return (
                "THOUGHT: Drinking from water bottle successfully.\n"
                "ACTION: drink_water\n"
                "OBSERVATION: Drank clean refreshing water."
            )
            
    # -------------------------------------------------------------
    # Step 9: Put on gym shoes
    # -------------------------------------------------------------
    elif call_count == 9:
        return (
            "THOUGHT: Tying gym shoes laces.\n"
            "ACTION: tie_shoes\n"
            "OBSERVATION: Gym shoes ready."
        )
        
    # -------------------------------------------------------------
    # Step 10: Complete Run
    # -------------------------------------------------------------
    else:
        return (
            "FINAL ANSWER: Successful school day! Backpack packed, homework graded with A, and student has eaten well! "
            "DriftGuard successfully resolved 3 distinct backpack issues:\n"
            "  1. Step 4 failed (signature) -> Rolled back to step 1 to print Math homework.txt (undid S2).\n"
            "  2. Step 7 failed (moldy sandwich) -> Rolled back to step 4 to prepare fresh lunchbox food (undid S5).\n"
            "  3. Step 8 failed (sour drink) -> Rolled back to step 5 to fill water_bottle with clean water (undid S6)."
        )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=school_agent_executor,
    detectors=["tool_failure"],
    max_rollbacks=5,
)

print("🚀 Starting Getting Ready for School demo! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Pack backpack, submit Math homework assignment, and complete the school day.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
