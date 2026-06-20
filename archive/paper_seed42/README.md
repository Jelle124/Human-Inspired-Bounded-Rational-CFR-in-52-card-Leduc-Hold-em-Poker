# Paper results archive (seed 42)

Frozen copy of thesis experiment outputs before repository cleanup.

## Reproduction command

```bash
python train_parallel_suite.py --all --seed 42
```

## Experiments included

| Directory | Setup |
|-----------|--------|
| `replication_dqn_vs_cfr` | DQN vs standard CFR (baseline replication) |
| `br_cfr_smart_vs_cfr` | BR-CFR smart vs standard CFR |
| `br_cfr_medium_vs_cfr` | BR-CFR medium vs standard CFR |
| `br_cfr_dumb_vs_cfr` | BR-CFR dumb vs standard CFR (mood + tilt) |
| `cfr_vs_cfr_baseline` | Standard CFR self-play control |

## Contents per experiment

- `config_*_seed42.json` — hyperparameters
- `training_logs_*_seed42.csv` — training metrics
- `final_evaluation_summary_*_seed42.json` — win rates + bluff stats
- `bluff_statistics_*_seed42.json` / `.csv` — threshold bluff detector
- `models/` — final trained checkpoints (local; not in git)

Intermediate training checkpoints (`checkpoints/`) are not archived here to save space.
Model weights: tarball `results/**/models/` for Zenodo/GitHub Releases.
