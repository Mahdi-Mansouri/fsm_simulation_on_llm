import re
import time
import random
import json
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from FSM import FSMManager
from database_manager import DatabaseManager

# --- Experiment Configuration ---
# --- CHANGE THE MODEL NAME HERE TO RUN A NEW EXPERIMENT ---
MODEL_NAME = "qwen3-235b-a22b-instruct-2507" 
#MODEL_NAME = "gpt-5" # Example for a second run
#MODEL_NAME = "qwen3-14b" 
SUPPORTS_SYSTEM_PROMPT = True
API_KEY = "YOUR_API_KEY"  # IMPORTANT: REPLACE WITH YOUR KEY
BASE_URL = "https://api.avalai.ir/v1"
TOTAL_INSTANCES = 20
TURNS_PER_INSTANCE = 50 
STEPS_PER_TURN = 1
MAX_WORKERS = 4
# Number of parallel threads
SLEEP_TIME = 1

# --- Helper Functions ---



def decode_response(response: str) -> str | None:
    """Extracts the state name from the <state> tags."""
    if not response: return None
    match = re.search(r"<state>(.*?)</state>", response, re.IGNORECASE)
    return match.group(1).strip() if match else None

def simulate_turn(fsm_manager, start_state, num_steps):
    """Deterministically simulates a sequence of steps to get actions and expected final state."""
    sequence, current_state = [], start_state
    for i in range(num_steps):
        available_actions = list(fsm_manager.transitions.get(current_state, {}).keys())
        if not available_actions: break
        action = random.choice(available_actions)
        sequence.append(action)
        current_state = fsm_manager.transitions[current_state][action]
    return ", ".join(sequence), current_state

# --- Worker Function for Parallel Execution ---

def process_run(instance_id: int):
    """Processes all turns for a single FSM instance for the configured MODEL_NAME."""
    db = DatabaseManager()
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    fsm = FSMManager()
    
    state = db.get_or_create_run_state(instance_id, MODEL_NAME, SUPPORTS_SYSTEM_PROMPT)
    
    # This worker only needs the FSM definition to generate sequences
    fsm_def_cursor = db.cursor.execute("SELECT fsm_definition FROM fsm_definitions WHERE instance_id = ?", (instance_id,))
    fsm_def = json.loads(fsm_def_cursor.fetchone()[0])
    fsm.states, fsm.actions, fsm.transitions, fsm.initial_state = set(fsm_def["states"]), set(fsm_def["actions"]), fsm_def["transitions"], fsm_def["initial_state"]

    # This runs only if the model doesn't support system prompts AND it's the very beginning of the run.
    if not SUPPORTS_SYSTEM_PROMPT and len(state["conversation_history"]) == 1:
        #print(f"Instance {instance_id} ({MODEL_NAME}): Performing one-time priming step.")
        try:
            # Send the FSM definition (which is the first user message) to the model
            completion = client.chat.completions.create(messages=state["conversation_history"], model=MODEL_NAME,temperature = 0.0, extra_body = {"enable_thinking": False})#), extra_body={"enable_thinking": False},temperature=0.7)
            raw_response = completion.choices[0].message.content
            llm_initial_state = decode_response(raw_response)

            # Add the assistant's first response to the history
            state["conversation_history"].append({"role": "assistant", "content": raw_response})
            
            # Check if the model correctly identified the initial state
            if llm_initial_state != fsm.initial_state:
                print(f"WARNING: Instance {instance_id} failed priming. Expected '{fsm.initial_state}', got '{llm_initial_state}'.")
                db.log_error(MODEL_NAME, instance_id, 0, 0, f"<state>{fsm.initial_state}</state>", raw_response, "initialization_failed")
                state["is_task_correct"] = False # Mark the entire task as incorrect from the start

            # Update the LLM's last known state. If it failed, we still proceed using the correct one.
            state["last_llm_state"] = llm_initial_state if llm_initial_state is not None else fsm.initial_state
            
            # Save this priming conversation turn
            db.update_run_state(state)

        except Exception as e:
            db.close()
            print(e)
            return f"ERROR during PRIMING on Instance {instance_id} ({MODEL_NAME}): {e}"
    
    while state["current_turn"] < TURNS_PER_INSTANCE:
        time.sleep(SLEEP_TIME)
        state["current_turn"] += 1
        task_length = state["current_turn"] * STEPS_PER_TURN
        
        action_seq, expected_state_from_llm = simulate_turn(fsm, state["last_llm_state"], STEPS_PER_TURN)
        #action_seq, expected_state_from_llm = fsm.simulate_sequence(state["last_llm_state"],STEPS_PER_TURN,)
        #_, expected_state_abs = simulate_turn(fsm, state["ground_truth_state"], STEPS_PER_TURN)

        state["conversation_history"].append({"role": "user", "content": action_seq})
        try:
            completion = client.chat.completions.create(messages=state["conversation_history"], model=MODEL_NAME,temperature = 0.0, extra_body = {"enable_thinking": False})#, extra_body={"enable_thinking": False},temperature=0.0)
            raw_response = completion.choices[0].message.content
            llm_state = decode_response(raw_response)
            state["conversation_history"].append({"role": "assistant", "content": raw_response})
        except Exception as e:
            db.close()
            print(e)
            return f"ERROR on Instance {instance_id} ({MODEL_NAME}), Turn {state['current_turn']}: {e}"

        turn_correct = (llm_state is not None) and (llm_state == expected_state_from_llm)
        if not turn_correct:
            failure_type = "decode_error" if llm_state is None else "state_mismatch"
            db.log_error(MODEL_NAME, instance_id, state["current_turn"], task_length, f"<state>{expected_state_from_llm}</state>", raw_response, failure_type)
            
        if state["is_task_correct"] and (llm_state != expected_state_from_llm):
            state["is_task_correct"] = False

        state["ground_truth_state"] = expected_state_from_llm
        state["last_llm_state"] = llm_state if llm_state is not None else state["last_llm_state"] ## NEEDS MODIFICATION
        
        db.update_results(MODEL_NAME, task_length, turn_correct, state["is_task_correct"])
        db.update_run_state(state) # Save state after every turn

    state["is_complete"] = True
    db.update_run_state(state)
    db.close()
    return f"Instance {instance_id} ({MODEL_NAME}) completed."

# --- Main Experiment Logic ---

def run_experiment():
    """Main function to orchestrate the FSM evaluation."""
    if API_KEY == "YOUR_API_KEY":
        print("❌ ERROR: Please replace 'YOUR_API_KEY' in the script.")
        return

    db = DatabaseManager()
    
    # Ensure FSM definitions exist up to the configured TOTAL_INSTANCES.
    # If TOTAL_INSTANCES is increased, this will generate and save the new ones.
    db.ensure_fsm_definitions(TOTAL_INSTANCES)
    
    # Prepare runs for potential extension by turn count.
    # This will mark previously completed runs as incomplete if TURNS_PER_INSTANCE is now higher.
    db.prepare_runs_for_extension(MODEL_NAME, TURNS_PER_INSTANCE)
    
    # Get all runs that are not yet complete (new, interrupted, or extended).
    runs_to_process = db.get_runs_to_process(TOTAL_INSTANCES, MODEL_NAME)
    completed_count = TOTAL_INSTANCES - len(runs_to_process)
    db.close()
    
    print(f"--- FSM Evaluation Runner ---")
    print(f"Model under test: {MODEL_NAME}")
    print(f"Target instances: {TOTAL_INSTANCES}")
    print(f"Target turns per instance: {TURNS_PER_INSTANCE}")
    print(f"Found {completed_count}/{TOTAL_INSTANCES} runs already complete for this model at the target configuration.")
    if not runs_to_process:
        print("🎉 All runs for this model are complete.")
        return

    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_run, runs_to_process), total=len(runs_to_process), desc=f"Processing FSMs for {MODEL_NAME}"))
            
    print(f"\n🎉 Experiment for model '{MODEL_NAME}' finished!")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("You can now run plot_results.py to generate the graphs.")

if __name__ == "__main__":
    run_experiment()

