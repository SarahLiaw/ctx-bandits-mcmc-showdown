import wandb
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ALGO_RUNS = {
    "LMCTS": [
        "jkfiaew9",
        "8kjc38cg",
        "o7yobetv",
        "qlr9bs7r",
        "u9t24z0b",
    ],
    "FGLMCTS": [
        "9z9czccu",
        "kluh5an4",
        "rpvpe6a9",
        "3unohhud",
        "yhhhd7ge",
    ],
    "SFGLMCTS": [
        "2c9a5de0-b326-46d0-b21d-2833324e5915",
        "2e7835b9-1102-443e-afd9-174293cb2393",
        "4ef69b94-3642-4874-8aaf-55bd7037ae1b",
        "51ff5811-c2d3-48b2-8bb1-26ca8b164144",
        "e309243b-f567-43e0-98a1-f10fd83574bb",
    ],
    "NeuralEpsGreedy": [
        "fcnwrp43",
        "r0eq3ed3",
        "qttx3ebx",
        "m710syhm",
        "iymgqnz1",
    ],
    "NeuralUCB": [
        "ui8rrl2g",
        "2f5fe6dt",
        "snp0qha7",
        "8brx2beo",
        "ausdwzfe",
    ],
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

api = wandb.Api()
out_dir = Path(f"{dataset_name}_simple_regret")
out_dir.mkdir(exist_ok=True)

summary_stats = {}

for algo, run_ids in ALGO_RUNS.items():
    deltas = []

    for run_id in run_ids:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        df = run.history(pandas=True, samples=1_000_000)
        step_col = "step" if "step" in df.columns else "_step"
        cum = df.set_index(step_col)[METRIC]

        if len(cum) < 500:
            raise ValueError(f"{algo}/{run_id} only has {len(cum)} steps")

        R_T = cum.iloc[-1]
        R_Tm499 = cum.iloc[-500]
        delta = R_T - R_Tm499
        deltas.append((run_id, delta))

    df_raw = pd.DataFrame(deltas, columns=["run_id", "simple_regret"])
    df_raw.insert(0, "algorithm", algo)
    df_raw.to_csv(out_dir / f"{dataset_name}_{algo}_simple_regret_raw.csv", index=False)

    values = df_raw["simple_regret"]
    summary_stats[algo] = (values.mean(), values.std(ddof=0))

df_summary = pd.DataFrame([
    {"algorithm": algo, "mean": m, "std": s}
    for algo, (m, s) in summary_stats.items()
])
df_summary.to_csv(out_dir / f"{dataset_name}_simple_regret_summary_stats.csv", index=False)

plt.figure(figsize=(9, 6))
algos = df_summary["algorithm"]
means = df_summary["mean"]
stds = df_summary["std"]

plt.bar(
    algos,
    means,
    yerr=stds,
    capsize=8,
    color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"][:len(algos)]
)
plt.ylabel("Δ Simple Regret\n(R_T – R_{T-499})", fontsize=14)
plt.xticks(rotation=20, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(out_dir / f"{dataset_name}_simple_regret_barplot.png", dpi=300)
plt.close()
