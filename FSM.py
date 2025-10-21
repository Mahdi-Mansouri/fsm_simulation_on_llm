import random
import json

class FSMManager:
    """
    Manages the creation, state, and interaction of a Finite State Machine.
    Uses dedicated lists of English words for state and action names.
    """
    def __init__(self):
        """
        Initializes the FSMManager with empty properties and dedicated word lists.
        """
        self.states = set()
        self.actions = set()
        self.transitions = {}
        self.initial_state = None
        self.current_state = None
        
        self._state_word_list = ['ant',
 'ape',
 'bat',
 'bee',
 'camel',
 'cat',
 'cobra',
 'crow',
 'deer',
 'dog',
 'duck',
 'frog',
 'hamster',
 'horse',
 'lion',
 'monkey',
 'rabbit',
 'apple',
 'water',
 'ice',
 'fire',
 'desk',
 'table',
 'pen',
 'paper',
 'iron',
 'book',
 'class',
 'room',
 'day',
 'worker',
 'year',
 'life',
 'child',
 'family',
 'home',
 'mother',
 'story',
 'eye',
 'game',
 'face',
 'war',
 'health',
 'girl',
 'man',
 'level',
 'city',
 'wife',
 'king',
 'hair',
 'hall',
 'hotel',
 'park',
 'blood',
 'sound',
 'glass',
 'earth',
 'task',
 'radio',
 'peace',
 'image']
        self._action_word_list = ['blue',
 'red',
 'green',
 'yellow',
 'brown',
 'black',
 'white',
 'good',
 'bad',
 'new',
 'old',
 'small',
 'large',
 'short',
 'long',
 'hard',
 'easy',
 'open',
 'clear',
 'hot',
 'cold',
 'dark',
 'light',
 'weak',
 'strong',
 'sad',
 'happy',
 'dry',
 'full',
 'empty',
 'fast',
 'slow',
 'safe',
 'danger',
 'winter',
 'fall',
 'spring',
 'summer',
 'rich',
 'deep',
 'fat',
 'thin',
 'ill',
 'smart',
 'fun',
 'far',
 'live',
 'medium',
 'north',
 'south',
 'west',
 'east']

    def create_random_fsm(self, num_states: int, num_actions: int, num_transitions: int):
        """
        Deletes any existing FSM properties and generates a new, random FSM
        using words from the dedicated word lists.

        Args:
            num_states (int): The number of states the FSM should have.
            num_actions (int): The number of unique actions available in the FSM.
            num_transitions (int): The total number of transitions to create.
        """
        # --- 1. Check if there are enough unique words in each list ---
        if num_states > len(self._state_word_list):
            print(f"❌ Error: Requested {num_states} states, but only "
                  f"{len(self._state_word_list)} unique state names are available.")
            self.initial_state = None
            return
            
        if num_actions > len(self._action_word_list):
            print(f"❌ Error: Requested {num_actions} actions, but only "
                  f"{len(self._action_word_list)} unique action names are available.")
            self.initial_state = None
            return

        # --- 2. Validate the number of transitions ---
        max_possible_transitions = num_states * num_actions
        if not (num_states <= num_transitions <= max_possible_transitions):
            print(f"❌ Error: For {num_states} states and {num_actions} actions, the number of transitions "
                  f"must be between {num_states} (to ensure no dead ends) "
                  f"and {max_possible_transitions} (the max possible).")
            self.initial_state = None # Mark FSM as not created
            return

        # --- 3. Clear existing FSM properties ---
        self.states = set()
        self.actions = set()
        self.transitions = {}
        
        # --- 4. Sample unique names for states and actions from their respective lists ---
        self.states = set(random.sample(self._state_word_list, num_states))
        self.actions = set(random.sample(self._action_word_list, num_actions))
        
        state_list = list(self.states)
        action_list = list(self.actions)

        # --- 5. Define transitions to meet the exact number requested ---
        for state in self.states:
            self.transitions[state] = {}

        # Create a pool of all possible unique (from_state, action) pairs
        available_transition_slots = []
        for s in self.states:
            for a in self.actions:
                available_transition_slots.append((s, a))
        
        # STAGE 1: Ensure every state has at least one outgoing transition.
        states_to_assign = list(self.states)
        random.shuffle(states_to_assign)
        
        for from_state in states_to_assign:
            possible_actions = [action for action in action_list if (from_state, action) in available_transition_slots]
            action = random.choice(possible_actions)
            to_state = random.choice(state_list)
            
            self.transitions[from_state][action] = to_state
            available_transition_slots.remove((from_state, action))

        # STAGE 2: Randomly assign the remaining transitions.
        remaining_to_assign = num_transitions - num_states
        if remaining_to_assign > 0:
            additional_transitions = random.sample(available_transition_slots, remaining_to_assign)
            for from_state, action in additional_transitions:
                to_state = random.choice(state_list)
                self.transitions[from_state][action] = to_state

        # --- 6. Set initial and current state ---
        self.initial_state = random.choice(state_list)
        self.current_state = self.initial_state

    def _generate_example_flow(self) -> str:
        """
        Generates a valid 3-step example conversation flow based on the FSM.
        """
        example_lines = ["Example Conversation Flow:\n"]
        
        # --- Turn 1 ---
        start1 = self.initial_state
        seq1, end1 = self.simulate_sequence(start1, random.randint(1, 2))
        example_lines.append(f"(You begin silently in the {start1} state. The user provides the first prompt.)")
        example_lines.append(f"User: {seq1}")
        example_lines.append(f"Assistant: <state>{end1}</state>")

        # --- Turn 2 ---
        start2 = end1
        seq2, end2 = self.simulate_sequence(start2, random.randint(1, 2))
        example_lines.append(f"(Your internal state is now {start2}. The user provides the second prompt.)")
        example_lines.append(f"User: {seq2}")
        example_lines.append(f"Assistant: <state>{end2}</state>")
        
        # --- Turn 3 ---
        start3 = end2
        seq3, end3 = self.simulate_sequence(start3, random.randint(1, 2))
        example_lines.append(f"(Your internal state is now {start3}. The user provides the third prompt.)")
        example_lines.append(f"User: {seq3}")
        example_lines.append(f"Assistant: <state>{end3}</state>")

        return "\n".join(example_lines)

    def get_prompt_formatted_fsm(self) -> str:
        """
        Returns a string of the FSM definition formatted for the new LLM prompt.
        """
        if not self.initial_state:
            return ""

        # --- 1. Build the FSM Definition block ---
        fsm_def_lines = ["FSM Definition:\n"]
        fsm_def_lines.append(f"States: {', '.join(sorted(list(self.states)))}")
        fsm_def_lines.append(f"Initial State: {self.initial_state}")
        fsm_def_lines.append("Transitions:")
        for from_state, actions in sorted(self.transitions.items()):
            for action, to_state in sorted(actions.items()):
                fsm_def_lines.append(f"From {from_state}, on action {action}, go to {to_state}.")
        
        fsm_definition_str = "\n".join(fsm_def_lines)

        # --- 2. Build the example flow ---
        example_flow_str = self._generate_example_flow()

        # --- 3. Assemble the final prompt using an f-string template ---
        complete_prompt = f"""Role & Goal: You are a meticulous Finite State Machine (FSM) executor. Your sole purpose is to function as a stateful processor based on the FSM definition and rules below. For each user message you receive, you will process it as a sequence of actions, update your internal state, and provide only the final state as your response.

{fsm_definition_str}

Core Operating Rules:

Your initial state at the beginning of this conversation is {self.initial_state}.
Each user prompt will contain a comma-separated string of one or more actions (e.g., action1,action2). All provided actions and sequences are guaranteed to be valid according to the transitions defined above.
You must process the actions sequentially. The resulting state from one action becomes the starting state for the next action in the sequence.
The final state at the end of processing one user prompt becomes your starting state for the next user prompt. You must maintain this state across the entire conversation.

Output Format & Constraints:

Your response must ONLY contain the final state after processing the entire action sequence.
Enclose the final state in <state> tags.
ABSOLUTELY DO NOT provide any other text, explanation, or conversational filler. Your entire response must be, for example, <state>{self.initial_state}</state>.

{example_flow_str}

Your configuration is complete. You will now strictly follow these rules for all subsequent user inputs. Begin.
"""
        return complete_prompt


    def display(self):
        """
        Prints the properties of the current FSM in a readable format to the console.
        """
        if not self.initial_state:
            print("❌ No FSM has been generated yet.")
            return
            
        print("\n--- (Raw) FSM Definition ---")
        print(f"States: {sorted(list(self.states))}")
        print(f"Initial State: {self.initial_state}")
        print(f"Current State: {self.current_state}")
        print("Transitions:")
        print(json.dumps(self.transitions, indent=2))
        print("--------------------------")
        
    def generate_valid_sequence(self, length: int) -> list:
        """
        Generates a sequence of actions that are guaranteed to be valid
        starting from the FSM's current state.
        """
        if not self.current_state:
            return []
            
        sequence = []
        temp_state = self.current_state
        
        for _ in range(length):
            possible_actions = list(self.transitions[temp_state].keys())
            if not possible_actions:
                print(f"Warning: State '{temp_state}' has no outgoing transitions. Sequence terminated early.")
                break
            chosen_action = random.choice(possible_actions)
            sequence.append(chosen_action)
            temp_state = self.transitions[temp_state][chosen_action]
        self.current_state = temp_state
            
        return ', '.join(sequence), self.current_state

    def process_sequence(self, actions: list):
        """
        Processes a given sequence of actions and updates the FSM's actual current state.
        """
        if not self.current_state:
            return

        print(f"\n▶️  (Internal) Processing sequence from state: '{self.current_state}'")
        for action in actions:
            self.current_state = self.transitions[self.current_state][action]
        print(f"🏁 New current state is: '{self.current_state}'")

    def simulate_sequence(self, start_state: str, length: int) -> tuple:
        """
        Simulates a sequence from a given start state without changing the FSM's
        actual current state. Returns the generated sequence and the final state.

        Args:
            start_state (str): The state to start the simulation from.
            length (int): The number of actions in the sequence.

        Returns:
            tuple: A tuple containing (sequence_string, final_state_string).
        """
        sequence = []
        temp_state = start_state
        
        for _ in range(length):
            if not self.transitions.get(temp_state):
                 print(f"Warning: Simulation stopped early because state '{temp_state}' has no outgoing transitions.")
                 break
            possible_actions = list(self.transitions[temp_state].keys())
            if not possible_actions:
                print(f"Warning: Simulation stopped early because state '{temp_state}' has no outgoing transitions.")
                break
            
            chosen_action = random.choice(possible_actions)
            sequence.append(chosen_action)
            temp_state = self.transitions[temp_state][chosen_action]
            
        return ', '.join(sequence), temp_state

# --- Main execution block ---
if __name__ == "__main__":
    fsm_manager = FSMManager()
    print("--- FSM Prompt Helper ---")

    while True:
        print("\nMENU:")
        print("  1          - Generate a new FSM (outputs to FSM.txt)")
        print("  2 <n>      - Generate action sequence (outputs to Transitions.txt)")
        print("  3 <s> <n>  - Simulate sequence from state <s> of length <n>")
        print("  display    - Show the raw FSM structure in the console")
        print("  q          - Quit")
        
        user_input = input("Enter your choice: ").strip().lower()

        if user_input == '1':
            try:
                states_num = int(input("Enter number of states: "))
                actions_num = int(input("Enter number of actions: "))
                transitions_num = int(input("Enter total number of transitions: "))
                if states_num <= 0 or actions_num <= 0 or transitions_num <=0:
                    raise ValueError
                
                fsm_manager.create_random_fsm(
                    num_states=states_num, 
                    num_actions=actions_num,
                    num_transitions=transitions_num
                )
                
                if fsm_manager.initial_state:
                    fsm_output = fsm_manager.get_prompt_formatted_fsm()
                    with open("FSM.txt", "w") as f:
                        f.write(fsm_output)
                    print("\n✅ New prompt successfully written to FSM.txt")

            except ValueError:
                print("❌ Please enter valid positive integers for all inputs.")
        
        elif user_input.startswith('2 '):
            if not fsm_manager.initial_state:
                print("❌ You must generate an FSM first. Use option '1'.")
                continue
            try:
                parts = user_input.split()
                if len(parts) != 2: raise ValueError
                length = int(parts[1])
                if length <= 0: raise ValueError
                
                sequence, final_state = fsm_manager.generate_valid_sequence(length)
                
                with open("Transitions.txt", "w") as f:
                    f.write(sequence)
                
                print(f"\n✅ Action sequence of length {length} written to Transitions.txt")
                print(f"  Predicted final state: '{final_state}'")

            except (ValueError, IndexError):
                print("❌ Invalid format. Please use '2 <n>', where n is a positive number.")

        elif user_input.startswith('3 '):
            if not fsm_manager.initial_state:
                print("❌ You must generate an FSM first. Use option '1'.")
                continue
            try:
                parts = user_input.split()
                if len(parts) != 3: raise ValueError
                
                start_state_name = parts[1]
                sequence_length = int(parts[2])

                if start_state_name not in fsm_manager.states:
                    print(f"❌ Error: State '{start_state_name}' does not exist in the FSM.")
                    continue
                
                if sequence_length <= 0: raise ValueError

                sequence, final_state = fsm_manager.simulate_sequence(start_state_name, sequence_length)
                
                if final_state:
                    print(f"\n--- Simulation Result ---")
                    print(f"  Starting from state: '{start_state_name}'")
                    with open("Transitions.txt", "w") as f:
                        f.write(sequence)
                    print(f"  Predicted final state: '{final_state}'")
                    print(f"  (Note: The FSM's actual current state is still '{fsm_manager.current_state}')")

            except (ValueError, IndexError):
                print("❌ Invalid format. Please use '3 <statename> <n>', where n is a positive integer.")


        elif user_input == 'display':
            fsm_manager.display()

        elif user_input == 'q':
            print("Exiting.")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")