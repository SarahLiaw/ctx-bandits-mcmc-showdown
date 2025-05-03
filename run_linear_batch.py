import csv 
import subprocess, json, shutil, os, random, glob, pathlib, time
import numpy as np, torch, matplotlib.pyplot as plt
import wandb

import argparse


ROOT = pathlib.Path(__file__).resolve().parents[1] / "MCMC_cb"
CFG_DIR = ROOT / "config" / "linear"
i = 1
while (ROOT / f"linear_results_{i}").exists():
    i += 1
OUT_DIR = ROOT / f"linear_results_{i}"


OUT_DIR.mkdir(parents=True)
print(f"[INFO] Output directory created: {OUT_DIR}")
DATA_DIR = OUT_DIR / "data"
RUN_PY = ROOT / "run.py"

TENSOR_KEY = "cum_regret"

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=None, help='Override dimension d')
parser.add_argument('--n_seeds', type=int, default=1, help='Number of seeds')
args = parser.parse_args()
N_SEEDS = 10
N_SEEDS = args.n_seeds

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    configs = []
    for cfg_file in CFG_DIR.glob("*.json"):
        cfg = json.loads(cfg_file.read_text())
        if args.d is not None:
            cfg["d"] = args.d
        # normalize agent name
        cfg["agent_name"] = cfg["agent"] if isinstance(cfg["agent"], str) else cfg["agent"]["name"]
        cfg["_path"] = cfg_file  # remember where it came from
        configs.append(cfg)

    all_keys = sorted({k for cfg in configs for k in cfg.keys() if k != "_path"})
    # write the CSV
    csv_path = OUT_DIR / "configs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for cfg in configs:
            # for any missing key, dictwriter will insert blank
            writer.writerow({k: cfg.get(k, "") for k in all_keys})


    for cfg in configs:
        algo_name = cfg["agent_name"]
        algo_out  = OUT_DIR / algo_name
        algo_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {algo_name} ===")
        all_runs = []
        for seed in range(N_SEEDS):

            print(f"  seed {seed}")
            set_global_seed(seed)

            # ---- generate or load shared synthetic data ---- #
            data_file = DATA_DIR / f"data_seed_{seed}.pt"
            if not data_file.exists():
                torch_ctx = torch.randn(cfg["T"], 1, cfg["context_size"])
                torch_theta= torch.randn(cfg["d"], 1)
                torch.save({"ctx": torch_ctx, "theta": torch_theta}, data_file)

            # ---- run script ---- #
            env = os.environ.copy()
            env["LINEAR_DATA_FILE"] = str(data_file)
            env["PYTHONHASHSEED"]   = str(seed)
            env["LINEAR_OUT_DIR"] = str(OUT_DIR)
            subprocess.run(["python", str(RUN_PY), "--config_path", str(cfg_file)],
                check=True, env=env)

            # ---- collect result ---- #
            run_tensor = torch.load(algo_out / f"seed{seed}" / f"{TENSOR_KEY}.pt")
            all_runs.append(run_tensor)
            time.sleep(2)

        # ---- aggregate + plot ---- #
        all_runs = torch.stack(all_runs)
        mean = all_runs.mean(dim=0)
        std  = all_runs.std(dim=0)

        # Simple regret (last 500 steps)
        simple = all_runs[:, -500:].mean(dim=1)
        simple_mean = simple.mean().item()
        simple_std  = simple.std().item()

        t = torch.arange(mean.numel())
        plt.figure(figsize=(8,4))
        plt.plot(t, mean, label=f"{algo_name} mean")
        plt.fill_between(t, mean-std, mean+std, alpha=0.25)
        plt.xlabel("t"); plt.ylabel("cumulative regret")
        plt.title(f"{algo_name}: mean ± 1 std over {N_SEEDS} seeds")
        plt.legend()
        fig_path = algo_out / "agg.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # ---- wandb log (aggregate only) ---- #
        agg_run = wandb.init(
            project="Linear",
            name=f"{algo_name}_aggregate",
            group=f"{algo_name}-aggregate",
            reinit=True
        )
        for step, (m, s) in enumerate(zip(mean, std)):
            agg_run.log({
                "mean_regret": m.item(),
                "upper": (m+s).item(),
                "lower": (m-s).item()
            }, step=step)
        agg_run.log({
            "final_mean": mean[-1].item(),
            "final_std" : std[-1].item(),
            "simple_regret": simple_mean,
            "simple_regret_std": simple_std,
            "agg_plot"  : wandb.Image(str(fig_path))
        })
        agg_run.finish()

if __name__ == "__main__":
    main()
