import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(db_name="fsm_experiment.db"):
    """Reads all results from the database, plots accuracy curves, and prints a token usage summary."""
    try:
        conn = sqlite3.connect(db_name)
        # Use pandas to easily read and group data
        df = pd.read_sql_query("SELECT * FROM results", conn)
        conn.close()
    except Exception as e:
        print(f"Error reading database: {e}. Please run experiment_runner.py first.")
        return

    if df.empty:
        print("No results found in the database.")
        return

    models = df['model_name'].unique()
    print(f"Found results for models: {', '.join(models)}")

    # --- Plot 1: Task Accuracy vs. Task Length ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    for model in models:
        model_df = df[df['model_name'] == model].copy()
        model_df['task_accuracy'] = model_df['task_successes'] / model_df['total_runs']
        ax1.plot(model_df['task_length'], model_df['task_accuracy'], marker='.', linestyle='-', label=model, markersize=4)

    ax1.set_title('Task Accuracy vs. Task Length')
    ax1.set_xlabel('Task Length (Total Steps)')
    ax1.set_ylabel('Task Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    fig1.savefig("task_accuracy.png")
    print("Saved task_accuracy.png")

    # --- Plot 2: Turn Accuracy vs. Task Length ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for model in models:
        model_df = df[df['model_name'] == model].copy()
        model_df['turn_accuracy'] = model_df['turn_successes'] / model_df['total_runs']
        ax2.plot(model_df['task_length'], model_df['turn_accuracy'], marker='.', linestyle='-', label=model, markersize=4)

    ax2.set_title('Turn Accuracy vs. Task Length')
    ax2.set_xlabel('Task Length (Total Steps)')
    ax2.set_ylabel('Turn Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    fig2.savefig("turn_accuracy.png")
    print("Saved turn_accuracy.png")

    plt.show()

    # --- Print Token Usage Summary ---
    print("\n--- Token Usage Summary ---")
    try:
        conn = sqlite3.connect(db_name)
        token_df = pd.read_sql_query("SELECT model_name, total_prompt_tokens, total_completion_tokens FROM experiment_runs WHERE is_complete = 1", conn)
        conn.close()

        if not token_df.empty:
            summary = token_df.groupby('model_name').agg(
                total_prompt_tokens=('total_prompt_tokens', 'sum'),
                total_completion_tokens=('total_completion_tokens', 'sum'),
                instances_completed=('model_name', 'count')
            ).reset_index()

            # Calculate total and average for prompt, completion, and overall tokens
            summary['avg_prompt_tokens'] = summary['total_prompt_tokens'] / summary['instances_completed']
            summary['avg_completion_tokens'] = summary['total_completion_tokens'] / summary['instances_completed']
            summary['total_tokens'] = summary['total_prompt_tokens'] + summary['total_completion_tokens']
            summary['avg_total_tokens'] = summary['total_tokens'] / summary['instances_completed']
            
            # Reorder columns for a clearer presentation
            summary = summary[[
                'model_name', 'instances_completed', 
                'total_prompt_tokens', 'avg_prompt_tokens',
                'total_completion_tokens', 'avg_completion_tokens',
                'total_tokens', 'avg_total_tokens'
            ]]
            
            # Format total numbers with commas for readability
            summary['total_prompt_tokens'] = summary['total_prompt_tokens'].map('{:,.0f}'.format)
            summary['total_completion_tokens'] = summary['total_completion_tokens'].map('{:,.0f}'.format)
            summary['total_tokens'] = summary['total_tokens'].map('{:,.0f}'.format)

            print(summary.to_string(index=False, formatters={
                'avg_prompt_tokens': '{:,.2f}'.format,
                'avg_completion_tokens': '{:,.2f}'.format,
                'avg_total_tokens': '{:,.2f}'.format
            }))
        else:
            print("No completed runs with token data found.")

    except Exception as e:
        print(f"Could not retrieve token usage data: {e}")


if __name__ == "__main__":
    plot_results()

