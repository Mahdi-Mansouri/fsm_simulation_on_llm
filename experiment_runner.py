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
#MODEL_NAME = "qwen3-235b-a22b-instruct-2507" 
MODEL_NAME = "qwen3-235b-a22b-thinking-2507"
#MODEL_NAME = "gpt-5" # Example for a second run
#MODEL_NAME = "qwen3-14b" 
SUPPORTS_SYSTEM_PROMPT = True
API_KEY = "YOUR_API_KEY"  
BASE_URL = "https://api.avalai.ir/v1"
TOTAL_INSTANCES = 10
TURNS_PER_INSTANCE = 6
STEPS_PER_TURN = 5
MAX_WORKERS = 5
# Number of parallel threads
SLEEP_TIME = 0
USE_STREAMING = True # Set to True to use streaming API calls

RUN_IDENTIFIER = f"{MODEL_NAME} (Steps: {STEPS_PER_TURN})"
# ---

def decode_response(response: str) -> str | None:
    """Extracts the state name from the <state> tags."""
    if not response: return None
    match = re.search(r"<state>(.*?)</state>", response, re.IGNORECASE)
    return match.group(1).strip() if match else None

def simulate_turn(fsm_manager, start_state, num_steps):
    """Deterministically simulates a sequence of steps to get actions and expected final state."""
    sequence, current_state = [], start_state
    for i in range(num_steps):
        # Gracefully handle states with no outgoing transitions
        if not current_state or not fsm_manager.transitions.get(current_state):
            break
        available_actions = list(fsm_manager.transitions.get(current_state, {}).keys())
        if not available_actions: break
        action = random.choice(available_actions)
        sequence.append(action)
        current_state = fsm_manager.transitions[current_state][action]
    return ", ".join(sequence), current_state

def get_model_response(client, messages, model_name, use_streaming):
    """
    Handles both streaming and non-streaming API calls and returns the full response content.
    Uses the fixed parameters from the experiment.
    """
    raw_response = ""
    try:
        if use_streaming:
            stream = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.0,
                extra_body={"enable_thinking": True},
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    raw_response += chunk.choices[0].delta.content
        else:
            completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.0,
                extra_body={"enable_thinking": False},
                stream=False 
            )
            raw_response = completion.choices[0].message.content
    except Exception as e:
        raise e
    
    return raw_response

def process_run(instance_id: int):
    """Processes all turns for a single FSM instance for the configured RUN_IDENTIFIER."""
    db = DatabaseManager()
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    fsm = FSMManager()
    
    state = db.get_or_create_run_state(instance_id, RUN_IDENTIFIER, SUPPORTS_SYSTEM_PROMPT)
    
    fsm_def_cursor = db.cursor.execute("SELECT fsm_definition FROM fsm_definitions WHERE instance_id = ?", (instance_id,))
    fsm_def = json.loads(fsm_def_cursor.fetchone()[0])
    fsm.states, fsm.actions, fsm.transitions, fsm.initial_state = set(fsm_def["states"]), set(fsm_def["actions"]), fsm_def["transitions"], fsm_def["initial_state"]

    if not SUPPORTS_SYSTEM_PROMPT and len(state["conversation_history"]) == 1:
        try:
            raw_response = get_model_response(
                client, 
                messages=state["conversation_history"], 
                model_name=MODEL_NAME, 
                use_streaming=USE_STREAMING
            )
            llm_initial_state = decode_response(raw_response)

            state["conversation_history"].append({"role": "assistant", "content": raw_response})
            
            if llm_initial_state != fsm.initial_state:
                print(f"WARNING: Instance {instance_id} failed priming. Expected '{fsm.initial_state}', got '{llm_initial_state}'.")
                db.log_error(RUN_IDENTIFIER, instance_id, 0, 0, f"<state>{fsm.initial_state}</state>", raw_response, "initialization_failed")
                state["is_task_correct"] = False

            state["last_llm_state"] = llm_initial_state if llm_initial_state is not None else fsm.initial_state
            
            db.update_run_state(state)

        except Exception as e:
            db.close()
            print(e)
            return f"ERROR during PRIMING on Instance {instance_id} ({RUN_IDENTIFIER}): {e}"
    
    while state["current_turn"] < TURNS_PER_INSTANCE:
        time.sleep(SLEEP_TIME)
        state["current_turn"] += 1
        task_length = state["current_turn"] * STEPS_PER_TURN
        
        action_seq, expected_state_from_llm = simulate_turn(fsm, state["last_llm_state"], STEPS_PER_TURN)

        state["conversation_history"].append({"role": "user", "content": action_seq})
        try:
            raw_response = get_model_response(
                client, 
                messages=state["conversation_history"], 
                model_name=MODEL_NAME, 
                use_streaming=USE_STREAMING
            )
            llm_state = decode_response(raw_response)
            state["conversation_history"].append({"role": "assistant", "content": raw_response})
        except Exception as e:
            db.close()
            print(e)
            return f"ERROR on Instance {instance_id} ({RUN_IDENTIFIER}), Turn {state['current_turn']}: {e}"

        turn_correct = (llm_state is not None) and (llm_state == expected_state_from_llm)
        
        if not turn_correct:
            failure_type = "decode_error" if llm_state is None else "state_mismatch"
            # Log error with RUN_IDENTIFIER
            db.log_error(RUN_IDENTIFIER, instance_id, state["current_turn"], task_length, f"<state>{expected_state_from_llm}</state>", raw_response, failure_type)

        if state["is_task_correct"] and not turn_correct:
            state["is_task_correct"] = False
        
        state["ground_truth_state"] = expected_state_from_llm
        
        if llm_state is not None:
            state["last_llm_state"] = llm_state
        
        db.update_results(RUN_IDENTIFIER, task_length, turn_correct, state["is_task_correct"])
        db.update_run_state(state) 

    state["is_complete"] = True
    db.update_run_state(state)
    db.close()
    return f"Instance {instance_id} ({RUN_IDENTIFIER}) completed."

# --- Main Experiment Logic ---

def run_experiment():
    """Main function to orchestrate the FSM evaluation."""
    if API_KEY == "YOUR_API_KEY":
        print("❌ ERROR: Please replace 'YOUR_API_KEY' in the script.")
        return

    db = DatabaseManager()
    
    print(f"INFO: Verifying sample size. Target is {TOTAL_INSTANCES} instances.")
    db.handle_sample_size_change(
        new_total_instances=TOTAL_INSTANCES,
        model_name=RUN_IDENTIFIER,
        total_turns=TURNS_PER_INSTANCE,
        steps_per_turn=STEPS_PER_TURN
    )
    db.ensure_fsm_definitions(TOTAL_INSTANCES)
    db.prepare_runs_for_extension(RUN_IDENTIFIER, TURNS_PER_INSTANCE)
    
    runs_to_process = db.get_runs_to_process(TOTAL_INSTANCES, RUN_IDENTIFIER)
    completed_count = TOTAL_INSTANCES - len(runs_to_process)
    db.close()
    
    print(f"--- FSM Evaluation Runner ---")
    print(f"Run Identifier: {RUN_IDENTIFIER}")
    print(f"Target instances: {TOTAL_INSTANCES}")
    print(f"Target turns per instance: {TURNS_PER_INSTANCE}")
    print(f"Streaming API: {USE_STREAMING}") # Added streaming status
    print(f"Found {completed_count}/{TOTAL_INSTANCES} runs already complete for this specific configuration.")
    if not runs_to_process:
        print("🎉 All runs for this configuration are complete.")
        return

    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(process_run, runs_to_process), total=len(runs_to_process), desc=f"Processing FSMs for {RUN_IDENTIFIER}"))
            
    print(f"\n🎉 Experiment for '{RUN_IDENTIFIER}' finished!")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("You can now run plot_results.py to generate the graphs.")

if __name__ == "__main__":
    run_experiment()