import time
from agents_driftguard import AgentUndoWrapper
from agents_driftguard.dashboard.app import start_dashboard_server

# Simulation state variables tracking the kid's cake baking steps
call_count = 0
grocery_list_fixed = False
oven_temp_fixed = False
icing_fixed = False

def cooking_agent_executor(prompt: str) -> str:
    global call_count, grocery_list_fixed, oven_temp_fixed, icing_fixed
    call_count += 1
    time.sleep(1.5)  # Pause for presentation visibility
    
    # Trace the feedback injected in prompt during rollback recovery
    if "ROLLBACK" in prompt:
        if "grocery_list.txt" in prompt:
            grocery_list_fixed = True
        elif "oven_config.txt" in prompt:
            oven_temp_fixed = True
        elif "icing_recipe.txt" in prompt:
            icing_fixed = True

    print(f"  [Kid Baker Call #{call_count}] (GroceryListFixed={grocery_list_fixed}, OvenFixed={oven_temp_fixed}, IcingFixed={icing_fixed})")
    
    # -------------------------------------------------------------
    # Step 1: Choose recipe
    # -------------------------------------------------------------
    if call_count == 1:
        return (
            "THOUGHT: Selecting a chocolate cake recipe.\n"
            "ACTION: choose_recipe\n"
            "OBSERVATION: Selected Chocolate Birthday Cake recipe."
        )
        
    # -------------------------------------------------------------
    # Step 2: Write shopping list
    # -------------------------------------------------------------
    elif call_count == 2:
        if not grocery_list_fixed:
            return (
                "THOUGHT: Writing ingredients list: chocolate, flour, milk.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to grocery_list.txt successfully (forgot sugar!)"
            )
        else:
            return (
                "THOUGHT: Re-writing shopping list to include chocolate, flour, milk, AND SUGAR.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to grocery_list.txt successfully"
            )
            
    # -------------------------------------------------------------
    # Step 3: Wipe kitchen table (unrelated)
    # -------------------------------------------------------------
    elif call_count == 3:
        return (
            "THOUGHT: Wiping down the kitchen table so we have a clean workspace.\n"
            "ACTION: wipe_table\n"
            "OBSERVATION: Table is clean."
        )
        
    # -------------------------------------------------------------
    # Step 4: Bake batter (Fails if missing sugar)
    # -------------------------------------------------------------
    elif call_count == 4:
        if not grocery_list_fixed:
            # Divergence 1: Bitter cake
            return (
                "THOUGHT: Mixing ingredients and baking the cake base.\n"
                "ACTION: bake_batter\n"
                "OBSERVATION: Error: The baked batter tastes terrible! It is extremely bitter because we had no sugar in grocery_list.txt"
            )
        else:
            return (
                "THOUGHT: Mixing ingredients and baking cake base.\n"
                "ACTION: bake_batter\n"
                "OBSERVATION: Cake base baked successfully: golden brown and sweet."
            )
            
    # -------------------------------------------------------------
    # Step 5: Configure oven heat
    # -------------------------------------------------------------
    elif call_count == 5:
        if not oven_temp_fixed:
            return (
                "THOUGHT: Adjusting oven temperature configuration for baking top layer.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to oven_config.txt successfully (set to 500 degrees!)"
            )
        else:
            return (
                "THOUGHT: Setting correct baking temperature configuration.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to oven_config.txt successfully"
            )
            
    # -------------------------------------------------------------
    # Step 6: Write icing recipe
    # -------------------------------------------------------------
    elif call_count == 6:
        if not icing_fixed:
            return (
                "THOUGHT: Creating icing steps recipe list: chocolate, cocoa, water.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to icing_recipe.txt successfully (forgot butter!)"
            )
        else:
            return (
                "THOUGHT: Setting correct icing recipe to include cocoa, sugar, AND BUTTER.\n"
                "ACTION: write_file\n"
                "OBSERVATION: saved to icing_recipe.txt successfully"
            )
            
    # -------------------------------------------------------------
    # Step 7: Bake top layer (Fails if oven too hot)
    # -------------------------------------------------------------
    elif call_count == 7:
        if not oven_temp_fixed:
            # Divergence 2: Burnt cake
            return (
                "THOUGHT: Putting the top cake layer in the oven.\n"
                "ACTION: bake_layer\n"
                "OBSERVATION: Error: The cake layer is burnt! Black smoke detected because oven_config.txt temp is 500 degrees"
            )
        else:
            return (
                "THOUGHT: Baking the top cake layer successfully.\n"
                "ACTION: bake_layer\n"
                "OBSERVATION: Top cake layer baked successfully."
            )
            
    # -------------------------------------------------------------
    # Step 8: Apply chocolate icing (Fails if watery)
    # -------------------------------------------------------------
    elif call_count == 8:
        if not icing_fixed:
            # Divergence 3: Watery icing
            return (
                "THOUGHT: Decorating cake with icing.\n"
                "ACTION: apply_icing\n"
                "OBSERVATION: Error: The icing slides off the cake! It is watery because we forgot butter in icing_recipe.txt"
            )
        else:
            return (
                "THOUGHT: Decorating cake with creamy chocolate icing.\n"
                "ACTION: apply_icing\n"
                "OBSERVATION: Icing applied successfully: looks professional."
            )
            
    # -------------------------------------------------------------
    # Step 9: Add birthday candles
    # -------------------------------------------------------------
    elif call_count == 9:
        return (
            "THOUGHT: Placing 10 colorful candles on top.\n"
            "ACTION: add_candles\n"
            "OBSERVATION: Candles placed and lit."
        )
        
    # -------------------------------------------------------------
    # Step 10: Complete Run
    # -------------------------------------------------------------
    else:
        return (
            "FINAL ANSWER: Yummy! The Birthday Cake is sweet, fluffy, and decorated beautifully! "
            "DriftGuard successfully resolved 3 distinct baking issues:\n"
            "  1. Step 4 failed (bitter base) -> Rolled back to step 1 to buy sugar (undid grocery_list.txt at S2).\n"
            "  2. Step 7 failed (burnt layer) -> Rolled back to step 4 to set temp (undid oven_config.txt at S5).\n"
            "  3. Step 8 failed (watery icing) -> Rolled back to step 5 to make creamy icing (undid icing_recipe.txt at S6)."
        )

# 2. Launch the dashboard
start_dashboard_server(port=8050)

# 3. Setup DriftGuard Wrapper
wrapper = AgentUndoWrapper(
    executor=cooking_agent_executor,
    detectors=["tool_failure"],
    max_rollbacks=5,
)

print("🚀 Starting Elaborate Baking Cake demo! Open http://localhost:8050 in your browser to watch.")
result = wrapper.run(goal="Bake and decorate a sweet chocolate birthday cake.")

print("\n=== Agent Finished ===")
print(f"Success: {result.success}")
print(f"Rollbacks Performed: {result.rollbacks_performed}")
print(f"Final Output: {result.final_output}")
