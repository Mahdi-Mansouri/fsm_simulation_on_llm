import sqlite3
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(db_name="fsm_experiment.db"):
    """Reads all results from the database and plots accuracy curves for each model."""
    try:
        conn = sqlite3.connect(db_name)
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

    plt.style.use('seaborn-v0_8-whitegrid') 
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'grid.color': 'lightgray',
        'grid.linestyle': '--',
        'grid.linewidth': 0.7,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0
    })
    
    # --- Plot 1: Task Accuracy vs. Task Length ---
    fig1, ax1 = plt.subplots(figsize=(10, 7)) # Adjusted size

    for model in models:
        model_df = df[df['model_name'] == model].copy()
        model_df['task_accuracy'] = model_df['task_successes'] / model_df['total_runs']
        ax1.plot(
            model_df['task_length'], 
            model_df['task_accuracy'], 
            linestyle='-', 
            label=model
        )

    ax1.set_title('Task Accuracy vs. Task Length')
    ax1.set_xlabel('Task Length')
    ax1.set_ylabel('Task Accuracy')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(left=0)
    ax1.legend()
    ax1.grid(True) 
    fig1.tight_layout() 
    fig1.savefig("task_accuracy.png", dpi=300) 
    print("Saved task_accuracy.png (updated style)")

    # --- Plot 2: Turn Accuracy vs. Task Length ---
    fig2, ax2 = plt.subplots(figsize=(10, 7)) 

    for model in models:
        model_df = df[df['model_name'] == model].copy()
        model_df['turn_accuracy'] = model_df['turn_successes'] / model_df['total_runs']
        ax2.plot(
            model_df['task_length'], 
            model_df['turn_accuracy'],
            linestyle='-', 
            label=model
        )

    ax2.set_title('Turn Accuracy vs. Task Length')
    ax2.set_xlabel('Task Length')
    ax2.set_ylabel('Turn Accuracy')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(left=0)
    ax2.legend()
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig("turn_accuracy.png", dpi=300)
    print("Saved turn_accuracy.png (updated style)")

    plt.show()

if __name__ == "__main__":
    plot_results()