import sqlite3
import json
from FSM import FSMManager
STATE_NUM = 2
ACTIONS_NUM = 2
TRANSITION_NUM = 4

class DatabaseManager:
    """Handles all SQLite database operations for the multi-model FSM experiment."""

    def __init__(self, db_name="fsm_experiment.db"):
        self.db_name = db_name
        # The check_same_thread=False is important for multi-threaded access
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary tables for a multi-model experiment."""
        # Table to store the 100 base FSM definitions. Populated only once.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fsm_definitions (
                instance_id INTEGER PRIMARY KEY,
                fsm_definition TEXT NOT NULL
            )
        """)

        # Table to store the run state for each model on each FSM instance.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                conversation_history TEXT NOT NULL,
                current_turn INTEGER DEFAULT 0,
                ground_truth_state TEXT NOT NULL,
                last_llm_state TEXT NOT NULL,
                is_task_correct INTEGER DEFAULT 1,
                is_complete INTEGER DEFAULT 0,
                UNIQUE(instance_id, model_name)
            )
        """)

        # Table to store aggregated results, now model-aware.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                task_length INTEGER NOT NULL,
                turn_successes INTEGER DEFAULT 0,
                task_successes INTEGER DEFAULT 0,
                total_runs INTEGER DEFAULT 0,
                UNIQUE(model_name, task_length)
            )
        """)

        # Error log, now model-aware.
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                instance_id INTEGER,
                turn_number INTEGER,
                task_length INTEGER,
                expected_state TEXT,
                llm_raw_response TEXT,
                failure_type TEXT
            )
        """)
        self.conn.commit()

    def ensure_fsm_definitions(self, total_instances):
        """Checks if FSM definitions exist. If not, creates and saves them."""
        self.cursor.execute("SELECT COUNT(*) FROM fsm_definitions")
        count = self.cursor.fetchone()[0]
        if count >= total_instances:
            print("Found existing FSM definitions in the database.")
            return

        print(f"Generating {total_instances} new FSM definitions...")
        for i in range(1, total_instances + 1):
            fsm = FSMManager()
            fsm.create_random_fsm(num_states=STATE_NUM, num_actions=ACTIONS_NUM, num_transitions=TRANSITION_NUM)
            definition = {
                "states": list(fsm.states),
                "actions": list(fsm.actions),
                "transitions": fsm.transitions,
                "initial_state": fsm.initial_state
            }
            self.cursor.execute(
                "INSERT OR IGNORE INTO fsm_definitions (instance_id, fsm_definition) VALUES (?, ?)",
                (i, json.dumps(definition))
            )
        self.conn.commit()
        print("FSM definitions saved successfully.")

    def get_or_create_run_state(self, instance_id, model_name, supports_system_prompt: bool):
        """
        Retrieves the state of a specific model's run on an FSM instance.
        If it doesn't exist, creates a new one using the master FSM definition.
        """
        self.cursor.execute(
            "SELECT * FROM experiment_runs WHERE instance_id = ? AND model_name = ?",
            (instance_id, model_name)
        )
        row = self.cursor.fetchone()

        if row:
            # Run exists, load its state
            return {
                "run_id": row[0], "instance_id": row[1], "model_name": row[2],
                "conversation_history": json.loads(row[3]), "current_turn": row[4],
                "ground_truth_state": row[5], "last_llm_state": row[6],
                "is_task_correct": bool(row[7]), "is_complete": bool(row[8])
            }
        else:
            # Run does not exist, create a new one
            self.cursor.execute("SELECT fsm_definition FROM fsm_definitions WHERE instance_id = ?", (instance_id,))
            def_row = self.cursor.fetchone()
            if not def_row:
                raise Exception(f"FATAL: FSM Definition for instance {instance_id} not found.")

            fsm_def = json.loads(def_row[0])
            fsm = FSMManager() # Use a temporary FSM object to format the prompt
            fsm.states, fsm.actions, fsm.transitions, fsm.initial_state = set(fsm_def["states"]), set(fsm_def["actions"]), fsm_def["transitions"], fsm_def["initial_state"]
            system_prompt = fsm.get_prompt_formatted_fsm()
            
            if supports_system_prompt:
                initial_conversation = [{"role": "system", "content": system_prompt}]
            else:
                # For other models, the "system prompt" is the first user message
                initial_conversation = [{"role": "user", "content": system_prompt}]

            initial_state = {
                "instance_id": instance_id, "model_name": model_name,
                "conversation_history": initial_conversation,
                "current_turn": 0, "ground_truth_state": fsm_def["initial_state"],
                "last_llm_state": fsm_def["initial_state"], "is_task_correct": True,
                "is_complete": False
            }

            self.cursor.execute("""
                INSERT INTO experiment_runs (instance_id, model_name, conversation_history, ground_truth_state, last_llm_state)
                VALUES (?, ?, ?, ?, ?)
            """, (
                instance_id, model_name, json.dumps(initial_state["conversation_history"]),
                initial_state["ground_truth_state"], initial_state["last_llm_state"]
            ))
            self.conn.commit()
            return initial_state

    def update_run_state(self, state):
        """Saves the updated state of an experiment run after a turn."""
        self.cursor.execute("""
            UPDATE experiment_runs
            SET conversation_history = ?, current_turn = ?, ground_truth_state = ?,
                last_llm_state = ?, is_task_correct = ?, is_complete = ?
            WHERE instance_id = ? AND model_name = ?
        """, (
            json.dumps(state["conversation_history"]), state["current_turn"],
            state["ground_truth_state"], state["last_llm_state"],
            int(state["is_task_correct"]), int(state["is_complete"]),
            state["instance_id"], state["model_name"]
        ))
        self.conn.commit()

    def update_results(self, model_name, task_length, turn_was_correct, task_is_correct):
        """Updates the aggregated results table for a specific model."""
        self.cursor.execute(
            "INSERT INTO results (model_name, task_length, turn_successes, task_successes, total_runs) VALUES (?, ?, ?, ?, 1) ON CONFLICT(model_name, task_length) DO UPDATE SET turn_successes = turn_successes + excluded.turn_successes, task_successes = task_successes + excluded.task_successes, total_runs = total_runs + 1",
            (model_name, task_length, int(turn_was_correct), int(task_is_correct))
        )
        self.conn.commit()

    def log_error(self, model_name, instance_id, turn, length, expected, actual, failure_type):
        """Logs the details of a failed turn to the error_log table for a specific model."""
        self.cursor.execute("""
            INSERT INTO error_log (model_name, instance_id, turn_number, task_length, expected_state, llm_raw_response, failure_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model_name, instance_id, turn, length, expected, actual, failure_type))
        self.conn.commit()

    def get_runs_to_process(self, total_instances, model_name):
        """Gets a list of instance_ids that are not yet complete for the given model."""
        self.cursor.execute(
            "SELECT instance_id FROM experiment_runs WHERE model_name = ? AND is_complete = 1",
            (model_name,)
        )
        completed_ids = {row[0] for row in self.cursor.fetchall()}
        all_ids = set(range(1, total_instances + 1))
        return sorted(list(all_ids - completed_ids))

    def get_all_results(self):
        """Retrieves all data from the results table for plotting."""
        self.cursor.execute("SELECT model_name, task_length, turn_successes, task_successes, total_runs FROM results ORDER BY model_name, task_length")
        return self.cursor.fetchall()
        
    def close(self):
        self.conn.close()

