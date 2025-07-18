import wandb
import pandas as pd
from pathlib import Path

ENTITY = "" 
PROJECT = ""
DATASET = "Covertype"
METRIC = "Regret"  

ALGO_GROUPS = {
    "NeuralTS": {"group": "Covertype-NeuralTS", "filters": {}},
    "FGNeuralTS": {"group": "Covertype-FGNeuralTS", "filters": {"config.fg_mode": "hard"}},
    "SFGNeuralTS": {"group": "Covertype-SFGNeuralTS", "filters": {"config.fg_mode": "smooth"}},
}

api = wandb.Api()

def fetch_run_series(run):
    df = run.history(pandas=True, samples=1_000_000)
    step_col = "step" if "step" in df.columns else "_step"
    if step_col not in df.columns or METRIC not in df.columns:
        raise KeyError(f"Missing columns in run {run.id}")
    return df.set_index(step_col)[METRIC]

summary = []

for algo, group_info in ALGO_GROUPS.items():

    filters = {"group": group_info["group"]}
    filters.update(group_info["filters"])
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters=filters)
    runs = [r for r in runs if r.state == "finished"]

    final_regrets = []
    simple_regrets = []
    for run in runs:
        try:
            series = fetch_run_series(run)
            if len(series) < 500:
                print(f"Skipping run {run.id} for {algo}: only {len(series)} steps")
                continue
            final_regrets.append(series.iloc[-1])
            simple_regrets.append(series.iloc[-1] - series.iloc[-500])
        except Exception as e:
            print(f"  Skipping run {run.id}: {e}")

    if not final_regrets:
        continue

    final_regrets = pd.Series(final_regrets)
    simple_regrets = pd.Series(simple_regrets)
    final_mean = final_regrets.mean()
    final_std = final_regrets.std()
    simple_mean = simple_regrets.mean()
    simple_std = simple_regrets.std()

    summary.append({
        "algorithm": algo,
        "final_regret_mean": final_mean,
        "final_regret_std": final_std,
        "simple_regret_mean": simple_mean,
        "simple_regret_std": simple_std,
        "n_runs": len(final_regrets)
    })

summary_df = pd.DataFrame(summary)
output_file = f"{DATASET}_regret_summary.csv"
summary_df.to_csv(output_file, index=False)
print(f"Done! Results saved in {output_file}")