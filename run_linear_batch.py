#!/usr/bin/env python3
import subprocess, json, os, random, time, argparse, csv
import numpy as np, torch, matplotlib.pyplot as plt, wandb
from pathlib import Path
from datetime import datetime

# ——————————————————————————————————————————————————————
# 1) SETUP: timestamped output directory
# ——————————————————————————————————————————————————————
ROOT    = Path(__file__).resolve().parents[1] / "MCMC_cb"
CFG_DIR = ROOT / "config" / "linear"

# timestamp for folder + CSV
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR   = ROOT / f"linear_results_{ts}"
OUT_DIR.mkdir(parents=True, exist_ok=False)
print(f"[INFO] Output directory: {OUT_DIR}")

DATA_DIR  = OUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RUN_PY    = ROOT / "run.py"
TENSOR_KEY = "cum_regret"

# ——————————————————————————————————————————————————————
# 2) ARGS
# ——————————————————————————————————————————————————————
parser = argparse.ArgumentParser()
parser.add_argument('--d',       type=int, default=None, help='Override d')
parser.add_argument('--n_seeds', type=int, default=1,   help='Number of seeds')
args = parser.parse_args()
N_SEEDS = args.n_seeds

# ——————————————————————————————————————————————————————
# 3) Load all JSON configs + dump to CSV
# ——————————————————————————————————————————————————————
configs = []
for cfg_path in sorted(CFG_DIR.glob("*.json")):
    cfg = json.loads(cfg_path.read_text())
    if args.d is not None:
        cfg["d"] = args.d
    cfg["agent_name"] = cfg["agent"] if isinstance(cfg["agent"], str) else cfg["agent"]["name"]
    cfg["_file"] = cfg_path.name
    configs.append(cfg)

# define a “canonical” column order
preferred = [
    "_file", "agent_name", "task_type", "T", "d", "context_size", "nb_arms",
    "std_reward", "std_prior", "eta", "step_size", "L_leap",
    "K", "K_not_updated", "lambda", "b", "exp_name", "project_name"
]
all_keys = []
for k in preferred:
    if any(k in c for c in configs):
        all_keys.append(k)
# then any others
extras = sorted({
    k for c in configs for k in c.keys()
    if k not in all_keys and k != "_file"
})
all_keys += extras

csv_path = OUT_DIR / f"configs_{ts}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    for c in configs:
        writer.writerow({k: c.get(k, "") for k in all_keys})

print(f"[INFO] Wrote config CSV to {csv_path}")

# ——————————————————————————————————————————————————————
# 4) Helper
# ——————————————————————————————————————————————————————
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ——————————————————————————————————————————————————————
# 5) MAIN LOOP
# ——————————————————————————————————————————————————————
for cfg in configs:
    algo = cfg["agent_name"]
    algo_out = OUT_DIR / algo
    algo_out.mkdir(exist_ok=True)

    print(f"\n=== {algo} ===")
    all_runs = []

    for seed in range(N_SEEDS):
        print(f" seed {seed}")
        set_seed(seed)

        # ---- shared synthetic data ----
        data_file = DATA_DIR / f"data_seed_{seed}.pt"
        if not data_file.exists():
            # linear: context shape (T,1,context_size)
            ctx_shape = (cfg["T"], 1, cfg["context_size"])
            torch_ctx   = torch.randn(*ctx_shape)
            torch_theta = torch.randn(cfg["d"], 1)
            torch.save({"ctx": torch_ctx, "theta": torch_theta}, data_file)

        # ---- launch run.py ----
        env = os.environ.copy()
        env["LINEAR_DATA_FILE"] = str(data_file)
        env["PYTHONHASHSEED"]   = str(seed)
        env["LINEAR_OUT_DIR"]   = str(OUT_DIR)

        subprocess.run(
            ["python", str(RUN_PY), "--config_path", str(CFG_DIR / cfg["_file"])],
            check=True, env=env
        )

        # ---- collect results ----
        rpt = algo_out / f"seed{seed}" / f"{TENSOR_KEY}.pt"
        run_tensor = torch.load(rpt)
        all_runs.append(run_tensor)
        time.sleep(2)

    # — aggregate & plot —
    all_runs = torch.stack(all_runs)               # (seeds, T)
    mean     = all_runs.mean(dim=0)
    std      = all_runs.std(dim=0)
    simple   = all_runs[:, -500:].mean(dim=1)      # last 500 steps
    s_mean   = simple.mean().item()
    s_std    = simple.std().item()

    t = torch.arange(mean.numel())
    plt.figure(figsize=(8,4))
    plt.plot(t, mean, label=f"{algo} mean")
    plt.fill_between(t, mean-std, mean+std, alpha=0.25)
    plt.xlabel("t"); plt.ylabel("cumulative regret")
    plt.title(f"{algo}: mean ±1 std over {N_SEEDS} seeds")
    plt.legend()
    fig_path = algo_out / "agg.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # — WandB aggregate run —
    agg = wandb.init(
        project="Linear_20d",
        name=f"{algo}_aggregate",
        group=f"{algo}-aggregate",
        reinit=True
    )
    for step, (m, s) in enumerate(zip(mean, std)):
        agg.log({"mean_regret": m.item(),
                 "upper":      (m+s).item(),
                 "lower":      (m-s).item()},
                step=step)
    agg.log({
        "final_mean":       mean[-1].item(),
        "final_std":        std[-1].item(),
        "simple_regret":    s_mean,
        "simple_regret_std":s_std,
        "agg_plot":         wandb.Image(str(fig_path))
    })
    agg.finish()

print("\nAll done! 🎉")
