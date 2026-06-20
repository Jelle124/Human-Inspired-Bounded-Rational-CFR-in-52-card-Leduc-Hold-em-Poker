# Bluffing by DQN and CFR in Leduc Hold'em Poker

Thesis codebase: trains DQN and CFR agents (including **Bounded-Rational CFR** variants) in 52-card Leduc Hold'em, evaluates win rates, and reports threshold-based bluff statistics.

Built on the replication from *"Analysis of Bluffing by DQN and CFR in Leduc Hold'em Poker"* (Začiragić, Plaat, Batenburg), extended with BR-CFR ablations for bounded rationality.

## Quick start (Docker)

```bash
docker build -t bluffing-leduc .

# Full thesis suite: 5 experiments × 100K train + 100K eval (~many hours)
docker run -v "$(pwd)/results:/app/results" bluffing-leduc \
  python train_parallel_suite.py --all --seed 42

# Quick sanity check (1K episodes)
docker run -v "$(pwd)/results:/app/results" \
  -e TRAIN_EPISODES=1000 -e EVAL_GAMES=1000 \
  bluffing-leduc python train_parallel_suite.py --experiment replication_dqn_vs_cfr --seed 42
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export WANDB_MODE=disabled

python train_parallel_suite.py --all --seed 42
```

## Experiments

| `--experiment` | Description |
|----------------|-------------|
| `replication_dqn_vs_cfr` | DQN vs standard CFR (baseline) |
| `br_cfr_smart_vs_cfr` | Full CFR-style BR agent vs standard CFR |
| `br_cfr_medium_vs_cfr` | Moderate bounded-rational degradations |
| `br_cfr_dumb_vs_cfr` | Heavy degradations + mood/tilt |
| `cfr_vs_cfr_baseline` | Standard CFR self-play control |

Run all: `python train_parallel_suite.py --all --seed 42`

## Results layout

Each experiment writes to `results/<experiment_name>/`:

- `config_*_seed42.json` — hyperparameters
- `training_logs_*_seed42.csv` — training metrics
- `final_evaluation_summary_*_seed42.json` — win rates + bluff stats
- `bluff_statistics_*_seed42.json` / `.csv`
- `models/` — final checkpoints (**local only**; not in git — see below)
- `checkpoints/` — intermediate snapshots (**local only**)
- `tilt_telemetry_*` — mood/tilt logs (dumb variant; `.jsonl` excluded from git, summary `.json` included)

A frozen copy of seed-42 paper outputs lives in `archive/paper_seed42/`.

## What is committed to GitHub

| Included | Excluded (keep locally or upload to Zenodo) |
|----------|---------------------------------------------|
| Source code | `results/**/models/` (`.pkl`, `.pt`) |
| `config_*.json` | `results/**/checkpoints/` |
| `final_evaluation_summary_*.json` | `*.jsonl` (tilt telemetry streams) |
| `bluff_statistics_*.{json,csv}` | `*.zip` / tarball archives |
| `training_logs_*.csv` | |
| `tilt_telemetry_summary_*.json` | |
| `archive/paper_seed42/` summaries (same as above) | `archive/**/models/` |

To publish trained weights, tarball `results/**/models/` and upload to [Zenodo](https://zenodo.org/) or a GitHub Release; link the DOI/URL in your thesis.

## Code layout

| File | Purpose |
|------|---------|
| `train_parallel_suite.py` | Main entry: train, evaluate, bluff analysis |
| `cfr_agent.py` | Standard tabular CFR agent |
| `br_cfr_agent.py` | Bounded-Rational CFR + mood/tilt |
| `br_cfr_variant_specs.py` | Smart / Medium / Dumb parameter table |
| `br_cfr_config.py` | Presets and mood/tilt defaults |
| `config.py` | Paths and episode counts |
| `custom_leduc_rlcard/` | 52-card Leduc Hold'em environment |

See `docs/BR_CFR_deep_research_deltas.md` for implementation notes vs the methods document.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BLUFFING_RESULTS_DIR` | `./results` | Output directory |
| `TRAIN_EPISODES` | `100000` | Training episodes per experiment |
| `EVAL_GAMES` | `100000` | Final evaluation games |

## Reference

- Original paper: *Analysis of Bluffing by DQN and CFR in Leduc Hold'em Poker*
- Upstream replication: [TarikZ03/Bluffing-by-DQN-and-CFR-in-Leduc-Hold-em-Poker-Codebase](https://github.com/TarikZ03/Bluffing-by-DQN-and-CFR-in-Leduc-Hold-em-Poker-Codebase)

## License

See [LICENSE](LICENSE).
