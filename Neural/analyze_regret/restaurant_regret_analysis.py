#!/usr/bin/env python3
"""
Regret analysis for CIFAR-10 dataset.
Fetches data from wandb and computes regret statistics for different algorithms.
Filters out crashed runs and selects the 5 best runs (lowest final regret) for each algorithm.
"""

import wandb
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
ENTITY = ""  # Set your wandb entity here
PROJECT = "ContextualBandit-Image"
DATASET = "Cifar10"
METRIC = "Regret"
NUM_BEST_RUNS = 5  # Number of best runs to select

# Algorithm groups based on the CIFAR-10 config files
ALGO_GROUPS = {
    "LMCTS": {
        "group": "Cifar10-LMCTS", 
        "filters": {}
    },
    "NeuralTS": {
        "group": "Cifar10-NeuralTS", 
        "filters": {}
    },
    "NeuralUCB": {
        "group": "Cifar10-NeuralUCB", 
        "filters": {}
    },
    "NeuralEpsGreedy": {
        "group": "Cifar10-NeuralEpsGreedy", 
        "filters": {}
    },
    "LinTS": {
        "group": "Cifar10-LinTS", 
        "filters": {}
    },
    "NeuralLinUCB": {
        "group": "Cifar10-NeuralLinUCB", 
        "filters": {}
    },
    "FGLMCTS": {
        "group": "Cifar10-FGLMCTS", 
        "filters": {"config.fg_mode": "hard"}
    },
    "SFGLMCTS": {
        "group": "Cifar10-SFGLMCTS", 
        "filters": {"config.fg_mode": "smooth"}
    },
    "FGNeuralTS": {
        "group": "Cifar10-FGNeuralTS", 
        "filters": {"config.fg_mode": "hard"}
    },
    "SFGNeuralTS": {
        "group": "Cifar10-SFGNeuralTS", 
        "filters": {"config.fg_mode": "smooth"}
    },
}

api = wandb.Api()

def fetch_run_series(run):
    """Fetch the regret series for a run."""
    df = run.history(pandas=True, samples=1_000_000)
    step_col = "step" if "step" in df.columns else "_step"
    if step_col not in df.columns or METRIC not in df.columns:
        raise KeyError(f"Missing columns in run {run.id}")
    return df.set_index(step_col)[METRIC]

def main():
    summary = []
    
    print(f"Analyzing regret for {DATASET} dataset...")
    print(f"Project: {ENTITY}/{PROJECT}")
    print(f"Algorithms: {list(ALGO_GROUPS.keys())}")
    print("-" * 50)
    
    for algo, group_info in ALGO_GROUPS.items():
        print(f"\nProcessing {algo}...")
        
        # Build filters
        filters = {"group": group_info["group"]}
        filters.update(group_info["filters"])
        
        # Fetch runs
        try:
            runs = api.runs(f"{ENTITY}/{PROJECT}", filters=filters)
            # Filter to only finished runs (no crashed runs)
            runs = [r for r in runs if r.state == "finished"]
            print(f"  Found {len(runs)} finished runs")
        except Exception as e:
            print(f"  Error fetching runs: {e}")
            continue
        
        if not runs:
            print(f"  No finished runs found for {algo}")
            continue
        
        # Collect all valid runs with their final regrets
        run_data = []
        
        for run in runs:
            try:
                series = fetch_run_series(run)
                if len(series) < 500:
                    print(f"  Skipping run {run.id}: only {len(series)} steps")
                    continue
                
                final_regret = series.iloc[-1]
                simple_regret = series.iloc[-1] - series.iloc[-500]
                
                run_data.append({
                    'run_id': run.id,
                    'final_regret': final_regret,
                    'simple_regret': simple_regret
                })
                
            except Exception as e:
                print(f"  Skipping run {run.id}: {e}")
        
        if not run_data:
            print(f"  No valid runs for {algo}")
            continue
        
        # Sort by final regret and select the NUM_BEST_RUNS best runs
        run_data.sort(key=lambda x: x['final_regret'])
        best_runs = run_data[:NUM_BEST_RUNS]
        
        print(f"  Selected {len(best_runs)} best runs out of {len(run_data)} total runs")
        
        # Extract regrets for statistics
        final_regrets = [run['final_regret'] for run in best_runs]
        simple_regrets = [run['simple_regret'] for run in best_runs]
        
        # Compute statistics
        final_regrets = pd.Series(final_regrets)
        simple_regrets = pd.Series(simple_regrets)
        
        final_mean = final_regrets.mean()
        final_std = final_regrets.std()
        simple_mean = simple_regrets.mean()
        simple_std = simple_regrets.std()
        
        print(f"  Final regret: {final_mean:.2f} ± {final_std:.2f}")
        print(f"  Simple regret: {simple_mean:.2f} ± {simple_std:.2f}")
        print(f"  Best runs selected: {len(final_regrets)}")
        
        summary.append({
            "algorithm": algo,
            "final_regret_mean": final_mean,
            "final_regret_std": final_std,
            "simple_regret_mean": simple_mean,
            "simple_regret_std": simple_std,
            "n_best_runs": len(final_regrets),
            "total_runs": len(run_data)
        })
    
    # Save results
    if summary:
        summary_df = pd.DataFrame(summary)
        output_file = f"{DATASET}_regret_summary.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\n{'='*50}")
        print(f"Results saved to {output_file}")
        print(f"{'='*50}")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo valid data found for any algorithm.")

if __name__ == "__main__":
    main()
