import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(db_name="fsm_experiment.db"):
    """Reads all results from the database and plots accuracy curves for each model."""
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

if __name__ == "__main__":
    plot_results()

