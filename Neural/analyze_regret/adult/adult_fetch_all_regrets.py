import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ALGO_RUNS = {
    "LMCTS": ["jkfiaew9", "8kjc38cg", "o7yobetv", "qlr9bs7r", "u9t24z0b"],
    "FGLMCTS": ["9z9czccu", "kluh5an4", "rpvpe6a9", "3unohhud", "yhhhd7ge"],
    "SFGLMCTS": [
        "2c9a5de0-b326-46d0-b21d-2833324e5915",
        "2e7835b9-1102-443e-afd9-174293cb2393",
        "4ef69b94-3642-4874-8aaf-55bd7037ae1b",
        "51ff5811-c2d3-48b2-8bb1-26ca8b164144",
        "e309243b-f567-43e0-98a1-f10fd83574bb",
    ],
    "NeuralEpsGreedy": ["fcnwrp43", "r0eq3ed3", "qttx3ebx", "m710syhm", "iymgqnz1"],
    "NeuralUCB": ["ui8rrl2g", "2f5fe6dt", "snp0qha7", "8brx2beo", "ausdwzfe"],
}

colors = {
    "LMCTS": "tab:blue",
    "FGLMCTS": "tab:orange",
    "SFGLMCTS": "tab:green",
    "NeuralEpsGreedy": "tab:red",
    "NeuralUCB": "tab:purple",
}

linestyles = {
    "LMCTS": "-",
    "FGLMCTS": "--",
    "SFGLMCTS": "-.",
    "NeuralEpsGreedy": ":",
    "NeuralUCB": (0, (3, 5, 1, 5)),
}

ENTITY = "-" # TO INSERT
PROJECT = "ContextualBandit-Neurips"
METRIC = "Regret"
dataset_name = "adult"

all_algo_stats = {}
final_step_stats = {}

api = wandb.Api()
out_dir = Path(f"{dataset_name}_regret_outputs")
out_dir.mkdir(exist_ok=True)

def fetch_series(run_id: str) -> pd.Series:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    df = run.history(pandas=True, samples=1_000_000)
    step_col = "step" if "step" in df.columns else "_step"
    if step_col not in df.columns:
        raise KeyError(f"No step column found in run {run_id}")
    return df.set_index(step_col)[METRIC].rename(run_id)

for algo, run_ids in ALGO_RUNS.items():
    series_list = [fetch_series(rid) for rid in run_ids]
    runs_df = pd.concat(series_list, axis=1)
    mean_s = runs_df.mean(axis=1)
    std_s = runs_df.std(axis=1)
    runs_df.to_csv(out_dir / f"{dataset_name}_{algo}_raw.csv")
    pd.DataFrame({"mean": mean_s, "std": std_s}).to_csv(out_dir / f"{dataset_name}_{algo}_summary_stats.csv")
    plt.figure(figsize=(8, 5))
    plt.plot(mean_s.index, mean_s, label=f"{algo} mean", color=colors[algo], linestyle=linestyles[algo], linewidth=2.5)
    plt.fill_between(mean_s.index, mean_s - std_s, mean_s + std_s, alpha=0.25, color=colors[algo], label="±1 std")
    plt.xlabel("step", fontsize=18)
    plt.ylabel(METRIC, fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset_name}_{algo}_regret_mean_std.png", dpi=400)
    plt.close()
    all_algo_stats[algo] = (mean_s, std_s)
    final_step_stats[algo] = (mean_s.iloc[-1], std_s.iloc[-1])

plt.figure(figsize=(9, 6))
ax = plt.gca()
for algo, (mean_s, std_s) in all_algo_stats.items():
    ax.plot(mean_s.index, mean_s, label=algo, color=colors[algo], linestyle=linestyles[algo], linewidth=2.5)
    ax.fill_between(mean_s.index, mean_s - std_s, mean_s + std_s, alpha=0.15, color=colors[algo])
ax.set_xlabel("Step", fontsize=18)
ax.set_ylabel(METRIC, fontsize=18)
ax.tick_params(labelsize=16)
ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(out_dir / f"{dataset_name}_all_algos_regret_mean_std.png", dpi=400)
plt.close()

final_df = pd.DataFrame.from_dict(final_step_stats, orient='index', columns=['final_mean', 'final_std'])
final_df.index.name = 'algorithm'
final_df.to_csv(out_dir / f"{dataset_name}_final_stats.csv")

simple_out_dir = Path(f"{dataset_name}_simple_regret_outputs")
simple_out_dir.mkdir(exist_ok=True)
simple_regret_stats = {}

for algo, run_ids in ALGO_RUNS.items():
    simple_regrets = []
    for run_id in run_ids:
        series = fetch_series(run_id)
        if len(series) < 500:
            raise ValueError(f"Run {run_id} has less than 500 steps.")
        final_500 = series.iloc[-500:]
        simple_r = final_500.mean()
        simple_regrets.append(simple_r)
    simple_regrets = pd.Series(simple_regrets, name="simple_regret")
    mean_simple = simple_regrets.mean()
    std_simple = simple_regrets.std()
    simple_regret_stats[algo] = (mean_simple, std_simple)
    simple_regrets.to_csv(simple_out_dir / f"{dataset_name}_{algo}_simple_regret_raw.csv", index=False)

summary_df = pd.DataFrame.from_dict(simple_regret_stats, orient='index', columns=['mean', 'std'])
summary_df.index.name = 'algorithm'
summary_df.to_csv(simple_out_dir / f"{dataset_name}_simple_regret_summary_stats.csv")

plt.figure(figsize=(9, 6))
algos = list(simple_regret_stats.keys())
means = [simple_regret_stats[a][0] for a in algos]
stds = [simple_regret_stats[a][1] for a in algos]
plt.bar(algos, means, yerr=stds, capsize=10, color=[colors[a] for a in algos])
plt.ylabel("Simple Regret (mean over final 500 steps)", fontsize=16)
plt.xticks(rotation=20, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(simple_out_dir / f"{dataset_name}_simple_regret_barplot.png", dpi=400)
plt.close()

simple_algo_stats = {}
for algo, run_ids in ALGO_RUNS.items():
    series_list = [fetch_series(rid) for rid in run_ids]
    runs_df = pd.concat(series_list, axis=1)
    mean_s = runs_df.mean(axis=1)
    std_s = runs_df.std(axis=1)
    runs_df.to_csv(simple_out_dir / f"{dataset_name}_{algo}_raw_simple_regret.csv")
    pd.DataFrame({"mean": mean_s, "std": std_s}).to_csv(simple_out_dir / f"{dataset_name}_{algo}_summary_simple_regret.csv")
    plt.figure(figsize=(8, 5))
    plt.plot(mean_s.index, mean_s, label=f"{algo} mean", color=colors[algo], linestyle=linestyles[algo], linewidth=2.5)
    plt.fill_between(mean_s.index, mean_s - std_s, mean_s + std_s, alpha=0.25, color=colors[algo], label="±1 std")
    plt.xlabel("step", fontsize=18)
    plt.ylabel("Simple Regret", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(simple_out_dir / f"{dataset_name}_{algo}_simple_regret_mean_std.png", dpi=400)
    plt.close()
    simple_algo_stats[algo] = (mean_s, std_s)

plt.figure(figsize=(9, 6))
ax = plt.gca()
for algo, (mean_s, std_s) in simple_algo_stats.items():
    ax.plot(mean_s.index, mean_s, label=algo, color=colors[algo], linestyle=linestyles[algo], linewidth=2.5)
    ax.fill_between(mean_s.index, mean_s - std_s, mean_s + std_s, alpha=0.20, color=colors[algo])
ax.set_xlabel("step", fontsize=18)
ax.set_ylabel("Simple Regret", fontsize=18)
ax.tick_params(labelsize=16)
ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(simple_out_dir / f"{dataset_name}_all_algos_simple_regret_mean_std.png", dpi=400)
plt.close()

zoom_out_dir = Path(f"{dataset_name}_regret_outputs_zoomed")
zoom_out_dir.mkdir(exist_ok=True)

plt.figure(figsize=(9, 6))
ax = plt.gca()
for algo, run_ids in ALGO_RUNS.items():
    series_list = [fetch_series(rid) for rid in run_ids]
    runs_df = pd.concat(series_list, axis=1)
    min_len = min(len(s) for s in series_list)
    if min_len < 500:
        raise ValueError(f"{algo} has less than 500 steps.")
    runs_df = runs_df.iloc[-500:]
    mean_s = runs_df.mean(axis=1)
    std_s = runs_df.std(axis=1)
    ax.plot(mean_s.index, mean_s, label=algo, color=colors[algo], linestyle=linestyles[algo], linewidth=2.5)
    ax.fill_between(mean_s.index, mean_s - std_s, mean_s + std_s, alpha=0.20, color=colors[algo])

ax.set_xlabel("Step (last 500 steps)", fontsize=18)
ax.set_ylabel(METRIC, fontsize=18)
ax.tick_params(labelsize=16)
ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(zoom_out_dir / f"{dataset_name}_zoomed_last500_regret_mean_std.png", dpi=400)
plt.close()
