# Quick Start Guide

## Setup (One-time)

```bash
# Install dependencies
make install

# Verify installation
make test
```

---

## Common Commands

### Posterior Analysis

```bash
# Run with default algorithms (LinTS, LMCTS, FGLMCTS, MALATS, PLMCTS)
make posterior-analysis

# Or directly
python posterior_analysis.py

# Custom algorithms
python posterior_analysis.py --algorithms LinTS LMCTS MALATS

# Different seed
python posterior_analysis.py --seed 123
```

### Testing

```bash
# Run all tests
make test

# Verbose output
make test-verbose

# Quick tests only
make test-quick
```

### Batch Experiments

```bash
# Run multiple seeds
python run_linear_batch.py --n_seeds 10

# Override dimension
python run_linear_batch.py --d 50 --n_seeds 5
```

---

## Output Locations

| Command | Output Directory |
|---------|------------------|
| `posterior_analysis.py` | `posterior_analysis_YYYYMMDD_HHMMSS/` |
| `run_linear_batch.py` | `linear_results_YYYYMMDD_HHMMSS/` |
| `run.py` | Agent-specific dirs in output location |

---

## Configuration

### Posterior Analysis (`posterior_analysis.py` lines 31-38)

```python
K_ARMS = 6                      # Number of arms
D_DIM = 20                      # Context dimension
T_HORIZON = 2000                # Time horizon
N_POSTERIOR_SAMPLES = 1500      # Samples to visualize
CORRELATED_CONTEXTS = True      # Elliptical posteriors
```

### Algorithm Configs (`config/linear/*.json`)

```json
{
  "task_type": "linear",
  "d": 20,
  "nb_arms": 5,
  "T": 10000,
  "step_size": 0.01,
  "K": 100
}
```

---

## Interpreting Results

### Posterior Analysis

**Visualization** (`*_posterior_comparison.png`):
- Green scatter = True Bayesian posterior
- Red scatter = Algorithm's approximate posterior
- Overlap = Good approximation

**Metrics** (`results.json`):
- Wasserstein distance < 0.3: Excellent ✅
- 0.3 < W < 0.7: Good ✅
- W > 1.0: Poor ❌
- W = NaN: Never played ⚠️

### Batch Experiments

**Plot** (`agg.png`):
- Mean cumulative regret ± 1 std
- Lower is better

**Metrics** (W&B or `results.json`):
- `final_mean`: Final cumulative regret
- `simple_regret`: Avg of last 500 steps

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Run from repository root |
| Tests fail | Check dependencies: `pip install -r requirements.txt` |
| Circular posteriors | Set `CORRELATED_CONTEXTS = True` |
| Slow execution | Reduce `N_POSTERIOR_SAMPLES` or `T_HORIZON` |
| Out of memory | Decrease `K` (MCMC steps) or batch size |

---

## File Overview

| File | Purpose |
|------|---------|
| `posterior_analysis.py` | Main posterior quality analysis |
| `test_posterior_analysis.py` | Unit tests (15 tests) |
| `run_linear_batch.py` | Batch regret experiments |
| `run.py` | Single experiment runner |
| `README.md` | Complete documentation |
| `TESTING.md` | Testing guide |
| `POSTERIOR_ANALYSIS_SUMMARY.md` | Detailed explanation |
| `Makefile` | Convenience commands |

---

## Getting Help

1. **Documentation**: Start with README.md
2. **Examples**: See `config/linear/*.json` for configurations
3. **Testing**: Run `make test` to verify setup
4. **Issues**: Check POSTERIOR_ANALYSIS_SUMMARY.md for detailed explanation

---

## Next Steps

### For Research
1. Run posterior analysis: `make posterior-analysis`
2. Check visualizations in output directory
3. Compare Wasserstein distances across algorithms
4. Use figures in paper

### For Development
1. Add new algorithm to `src/MCMC.py`
2. Create config in `config/linear/`
3. Test with: `python posterior_analysis.py --algorithms MyAlgorithm`
4. Add unit tests if needed

### For Experiments
1. Modify configs in `config/linear/`
2. Run batch: `python run_linear_batch.py --n_seeds 10`
3. Check W&B dashboard for results
4. Aggregate metrics from output CSV

---

**Ready to start?** Run `make posterior-analysis` and check the output directory! 🚀
