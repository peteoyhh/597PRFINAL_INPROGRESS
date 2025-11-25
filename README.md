# Monte Carlo Simulation of Strategic Trade-offs in Beijing Mahjong

**Authors:** Xu Chen, BoHan Shan  
**NetIDs:** xc74, bohans3

## 🀄 Introduction

This project implements a Monte Carlo simulation to study strategic trade-offs in Beijing-style Mahjong. The simulation compares defensive and aggressive playing strategies under various conditions to test three key hypotheses:

- **H1**: Defensive players achieve higher expected long-term monetary profit than aggressive players
- **H2**: Aggressive players achieve higher average utility (accounting for emotional rewards) than defensive players
- **H3**: Strategy performance depends on table composition (proportion of defensive vs aggressive opponents)

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml          # Configuration parameters
├── mahjong_sim/
│   ├── __init__.py
│   ├── variables.py       # Random variable sampling functions
│   ├── scoring.py         # Score computation functions
│   ├── strategies.py      # Strategy definitions (defensive/aggressive)
│   ├── simulation.py       # Core simulation logic
│   └── utils.py           # Statistical analysis utilities
├── experiments/
│   ├── run_experiment_1.py    # Strategy comparison (H1)
│   ├── run_experiment_2.py    # Utility function analysis (H2)
│   ├── run_experiment_3.py    # Table composition analysis (H3)
│   └── run_sensitivity.py     # Sensitivity analysis
├── tests/
│   ├── test_variables.py
│   ├── test_scoring.py
│   └── test_simulation.py
└── main.py                # Main entry point
```

## 🚀 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Run a specific experiment:
if python command throw you an error, try python3
```bash
python main.py --experiment 1  # Run experiment 1
python main.py --experiment 2  # Run experiment 2
python main.py --experiment 3  # Run experiment 3
python3 experiments/run_experiment_3_table.py # four-players table
python main.py --experiment 4  # Run experiment 4 (sensitivity analysis)
```

### Run all experiments:
```bash
python main.py --all
```

### Run quick demo (single trial):
```bash
python main.py --demo
```

### Run experiments directly:
```bash
python experiments/run_experiment_1.py
python experiments/run_experiment_2.py
python experiments/run_experiment_3.py
python3 experiments/run_experiment_3_table.py
python experiments/run_sensitivity.py
python3 main.py --all
```

## ⚙️ Configuration

All tunable parameters live in `configs/base.yaml`. Update that file (and keep everything YAML-valid) whenever you want to explore a different scenario. Each field is documented inline, but the table below summarizes the intent and safe range of values:

| Key | Meaning | Typical / acceptable range |
| --- | --- | --- |
| `base_points` | Base score before fan multipliers | `0.5 – 4`; stay ≤ 5 to keep payouts realistic |
| `fan_min` | Min fan required for the defensive player to declare Hu | `1 – 3`; must be ≥ 1 because Pi Hu is invalid |
| `t_fan_threshold` | Fan threshold for the aggressive player to continue chasing | `2 – 6`; higher values push for bigger hands |
| `alpha` | Legacy utility weight (kept for compatibility) | `0 – 1`; leave at `0.5` unless you extend the old utility model |
| `penalty_deal_in` | Multiplier applied to the loser on deal-in rounds | `1 – 5`; larger values make mistakes very costly |
| `rounds_per_trial` | Rounds per simulation trial | `50 – 400`; higher means smoother averages, longer runtime |
| `trials` | Number of trials when aggregating stats | `200 – 5000`; ≥ 1000 recommended for stable confidence intervals |
| `random_seed` | Seed for NumPy RNG (ensures reproducibility) | Any integer; change only if you want different stochastic runs |
| `initial_bankroll` | Starting bankroll used in ruin-probability metrics | `200 – 5000`; higher bankroll reduces ruin odds |
| `table_trials` | Number of trials per composition in 4-player tables | `200 – 2000`; ≥ 800 recommended for smooth dealer stats |

> Tip: after editing `base.yaml`, rerun the relevant experiment scripts so the new configuration gets picked up automatically.


## 🧪 Running Tests

```bash
cd your folder directionary
python3 -m pytest tests/
```

## 📊 Experiments

### Experiment 1: Strategy Comparison
Compares defensive vs aggressive strategies on profit and utility metrics. Tests H1.

### Experiment 2: Utility Function Analysis
Uses CRRA (Constant Relative Risk Aversion) utility function with emotional rewards for high-fan hands. Tests H2.

### Experiment 3: Table Composition Analysis
Varies the proportion of defensive players (theta: 0, 0.33, 0.67, 1.0) and analyzes its effect on strategy performance. Tests H3.

### Experiment 4: Sensitivity Analysis
Examines robustness by varying key parameters:
- Deal-in penalty (P): 1, 2, 3, 4, 5
- Alpha (utility weight): 0.1, 0.3, 0.5, 0.7, 0.9
- Fan threshold: 1, 2, 3, 4, 5
- Base points: 1, 2, 4

## 📈 Output

Each experiment outputs:
- Mean values with 95% confidence intervals
- t-test statistics and p-values for comparisons
- Win rates and deal-in rates
- Fan distributions
- Regression analysis (for composition experiment)

## 🔬 Methodology

The simulation models each round with random variables:
- **Q**: Hand quality (potential to complete winning hand)
- **F**: Fan potential (expected hand value)
- **R**: Deal-in risk (probability of discarding winning tile)
- **T**: Threat level (pressure from opponents)
- **K**: Kong events (adds bonus fan)

Scoring follows: `Score = B * 2^fan` where B is base points.



## ✅ Latest Results (seed = 42)

`python3 main.py --all` writes the most recent metrics to `output/all_experiments_output.txt`. Highlights from the latest run (summarized):

- **Experiment 1 (4-player table)**  
  - Defensive: profit ≈ `879`, utility ≈ `-53`, win rate `20.2%`, mean fan `2.53`.  
  - Aggressive: profit ≈ `-74`, utility ≈ `-229`, win rate `5.3%`, mean fan `5.63`.  
  - Insight: defensive play steadily earns chips while aggressive runners hit large-fan wins but suffer consistent negative utility.

- **Experiment 2 (utility focus, 2000 trials × 200 rounds)**  
  - Defensive utility mean `172.0` (95% CI `[169.4, 174.6]`).  
  - Aggressive utility mean `74.2` (95% CI `[70.3, 78.1]`).  
  - Difference (Agg − Def) `-97.8`, `t = 41.10`, `p < 1e-6`.  
  - Profit reference: defensive `23,917`, aggressive `18,199`.

- **Experiment 3A (single-player θ sweep)**  
  - Defensive profit grows from `23.9k` (θ = 0) to `26.7k` (θ = 1) with win rate ≈ `33.6%`.  
  - Aggressive profit increases with more defensive opponents (`17.2k` → `22.4k`), confirming the composition hypothesis in the single-player abstraction.

- **Experiment 3B (full 4-player table)**  
  - θ counts defensive opponents (0–4). θ = 2 lands near zero-sum (DEF `-177`, AGG `+177`).  
  - Regression slopes: defensive `-4.9k` per extra DEF opponent (R² `0.96`), aggressive `-11.7k` (R² `0.81`).  
  - Dealer stats remain roughly zero-sum each round, validating the rotation logic.

- **Experiment 4 (sensitivity)**  
  - Deal-in penalty 1 → 5 expands the DEF–AGG profit gap from `≈3.3k` to `≈11.1k`.  
  - Utility weight α changes do not overturn DEF advantage (DEF ≈ `0.55k`, AGG ≈ `0.30k`).  
  - Aggressive fan threshold trades frequency for hand size (threshold 1: win `33.6%` / fan `2.2`; threshold 5: win `2.1%` / fan `8.3`).  
  - Base points scaling stays proportional (e.g., base 4 → DEF ≈ `101k`, AGG ≈ `68.6k`).


## 📚 Reference

Chen, J. C., Tang, S. C., & Wu, I. C. (n.d.). *Monte-Carlo simulation for Mahjong.*  
National Yang Ming Chiao Tung University Academic Hub.  
https://scholar.nycu.edu.tw/en/publications/monte-carlo-simulation-for-mahjong

