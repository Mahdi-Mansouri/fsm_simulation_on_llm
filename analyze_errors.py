import sqlite3
import sys
from collections import Counter

def analyze_errors(db_name="fsm_experiment.db", model_filter=None):
    """
    Reads the error_log table and prints a summary of the failures.
    Can optionally filter by a specific model name.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        query = "SELECT model_name, failure_type, llm_raw_response, expected_state FROM error_log"
        params = []
        if model_filter:
            query += " WHERE model_name = ?"
            params.append(model_filter)
            
        cursor.execute(query, params)
        errors = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError:
        print(f"Error: Database file '{db_name}' not found. Please run experiment_runner.py first.")
        return

    if not errors:
        model_str = f" for model '{model_filter}'" if model_filter else ""
        print(f"✅ No errors found in the log{model_str}. Congratulations!")
        return

    title_str = f" for model '{model_filter}'" if model_filter else " (All Models)"
    print(f"\n--- Error Analysis --- {len(errors)} Total Errors{title_str} ---\n")

    failure_types = [row[1] for row in errors]
    type_counts = Counter(failure_types)

    print("Error Breakdown by Type:")
    for f_type, count in type_counts.items():
        percentage = (count / len(errors)) * 100
        print(f"  - {f_type}: {count} errors ({percentage:.2f}%)")
    
    print("\n--- Examples of Failed Responses ---\n")
    
    decode_errors = [err for err in errors if err[1] == 'decode_error']
    if decode_errors:
        print("--- Examples of 'decode_error' (Response did not contain '<state>...</state>') ---\n")
        for i, error in enumerate(decode_errors[:5]): # Show up to 5 examples
            print(f"Example {i+1} (Model: {error[0]}):")
            print(f"  Expected: {error[3]}")
            print(f"  Actual Raw Response: \"{error[2]}\"\n")
            
    mismatch_errors = [err for err in errors if err[1] == 'state_mismatch']
    if mismatch_errors:
        print("--- Examples of 'state_mismatch' (LLM returned the wrong state) ---\n")
        for i, error in enumerate(mismatch_errors[:5]): # Show up to 5 examples
            print(f"Example {i+1} (Model: {error[0]}):")
            print(f"  Expected: {error[3]}")
            print(f"  Actual Raw Response: \"{error[2]}\"\n")

if __name__ == "__main__":
    # You can run this script with a model name as a command-line argument
    # e.g., python analyze_errors.py qwen3-0.6b
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_errors(model_filter=model_arg)

