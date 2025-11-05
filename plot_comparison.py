import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 


EXPERIMENTS_TO_COMPARE = [
    #("2 States, 40 Actions", "fsm_experiment1.db"),
    #("40 States, 2 Actions", "fsm_experiment2.db"),
    #("2 States, 40 Actions", "fsm_experiment3.db"),
    #("40 States, 2 Actions", "fsm_experiment4.db"),
    #("10 States, 4 Actions, 40 Transitions", "fsm_experiment5.db"),
    #("10 States, 6 Actions, 40 Transitions", "fsm_experiment6.db"),
    #("10 States, 8 Actions, 40 Transitions", "fsm_experiment7.db"),
    #("10 States, 10 Actions, 40 Transitions", "fsm_experiment8.db"),
    ("4 States, 5 Actions, 1 Step Size, No reasoning", "fsm_experiment11.db"),
    ("4 States, 5 Actions, 2 Step Size, No reasoning", "fsm_experiment10.db"),
    ("4 States, 5 Actions, 2 Step Size, Reasoning", "fsm_experiment9.db"),
    ("4 States, 5 Actions, 5 Step Size, Reasoning", "fsm_experiment12.db"),
    # Example: You could add a third one like this:
    # ("10 States, 5 Actions", "fsm_experiment_large.db")
]

# --- Output Filenames ---
TASK_ACCURACY_PLOT_FILE = "comparison_task_accuracy.png"
TURN_ACCURACY_PLOT_FILE = "comparison_turn_accuracy.png"
# --- End Configuration ---

def plot_separate_comparisons(experiments: list):
    """
    Reads results from multiple database files and plots two separate
    images: one for Task Accuracy and one for Turn Accuracy,
    with an updated aesthetic to match the provided example.

    Args:
        experiments (list): A list of tuples, where each tuple contains
                            (label_for_plot, database_file_path).
    """
    
    # --- Set Matplotlib Style Parameters for a cleaner, academic look ---
    plt.style.use('seaborn-v0_8-whitegrid') # Starting with a clean grid style
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
    
    fig_task, ax_task = plt.subplots(figsize=(10, 7))

    fig_turn, ax_turn = plt.subplots(figsize=(10, 7))

    print(f"--- Aggregating {len(experiments)} Experiment(s) ---")

    for label, db_path in experiments:
        print(f"\nProcessing: '{label}' (from {db_path})")
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM results", conn)
            conn.close()
        except Exception as e:
            print(f"  ❌ Error: Could not read {db_path}. {e}")
            continue

        if df.empty:
            print("  ⚠️ Warning: No 'results' table data found in this database.")
            continue

        models_in_db = df['model_name'].unique()
        print(f"  Found {len(models_in_db)} model result(s) in this DB: {', '.join(models_in_db)}")

        for model_name in models_in_db:
            model_df = df[df['model_name'] == model_name].copy()
            
            if model_df.empty:
                continue

            # Calculate accuracies
            model_df['task_accuracy'] = model_df['task_successes'] / model_df['total_runs']
            model_df['turn_accuracy'] = model_df['turn_successes'] / model_df['total_runs']
            
           
            plot_label = f"{label} - {model_name}"
            
            # --- Plot on Task Accuracy (Figure 1) ---
            ax_task.plot(
                model_df['task_length'], 
                model_df['task_accuracy'], 
                linestyle='-', 
                label=plot_label, 
            )
            
            # --- Plot on Turn Accuracy (Figure 2) ---
            ax_turn.plot(
                model_df['task_length'], 
                model_df['turn_accuracy'], 
                linestyle='-', 
                label=plot_label, 
            )

    ax_task.set_title('Task Accuracy vs. Task Length')
    ax_task.set_xlabel('Task Length')
    ax_task.set_ylabel('Task Accuracy')
    ax_task.set_ylim(-0.05, 1.05)
    ax_task.set_xlim(left=0) 
    ax_task.legend()
    ax_task.grid(True) 
    fig_task.tight_layout()
    
    try:
        fig_task.savefig(TASK_ACCURACY_PLOT_FILE, dpi=300) 
        print(f"\n✅ Successfully saved Task Accuracy plot to {TASK_ACCURACY_PLOT_FILE}")
    except Exception as e:
        print(f"\n❌ Error saving Task Accuracy plot: {e}")

    ax_turn.set_title('Turn Accuracy vs. Task Length')
    ax_turn.set_xlabel('Task Length')
    ax_turn.set_ylabel('Turn Accuracy')
    ax_turn.set_ylim(-0.05, 1.05)
    ax_turn.set_xlim(left=0)
    ax_turn.legend()
    ax_turn.grid(True)
    fig_turn.tight_layout()

    try:
        fig_turn.savefig(TURN_ACCURACY_PLOT_FILE, dpi=300) 
        print(f"✅ Successfully saved Turn Accuracy plot to {TURN_ACCURACY_PLOT_FILE}")
        plt.show()
    except Exception as e:
        print(f"\n❌ Error saving Turn Accuracy plot: {e}")


if __name__ == "__main__":
    plot_separate_comparisons(EXPERIMENTS_TO_COMPARE)