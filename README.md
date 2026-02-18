# Bluffing by DQN and CFR in Leduc Hold'em Poker

Replication of the research from *"Analysis of Bluffing by DQN and CFR in Leduc Hold'em Poker"* (Začiragić, Plaat, Batenburg).

This codebase trains DQN and CFR agents simultaneously in 52-card Leduc Hold'em, evaluates them, and analyzes bluffing behavior using threshold-based and statistical bluff detectors.

## Quick Start (Docker — recommended for laptops)

```bash
# Build the image
docker build -t bluffing-leduc .

# Full replication: 100K training + 100K evaluation (~several hours on a laptop)
docker run -v $(pwd)/results:/app/results bluffing-leduc

# Quick sanity check: 1K training + 1K evaluation (~5–10 minutes)
docker run -v $(pwd)/results:/app/results \
  -e TRAIN_EPISODES=1000 -e EVAL_GAMES=1000 \
  bluffing-leduc
```

Results are saved in `./results/`:
- `results/training/` — DQN and CFR model checkpoints
- `results/evaluation/` — game logs (JSONL)
- Plots from bluff analysis appear in the working directory (or configure via scripts)

## Run Without Docker

```bash
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
# CPU-only PyTorch recommended for laptops (saves ~2GB vs CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Optional: disable Weights & Biases (no account needed)
export WANDB_MODE=disabled

# Quick test (1K episodes)
TRAIN_EPISODES=1000 EVAL_GAMES=1000 python run_pipeline.py

# Full replication (100K episodes, ~several hours)
python run_pipeline.py
```

### Run individual steps

```bash
python simultaneous_training.py    # Train DQN vs CFR
python evaluate_simultaneous.py   # Evaluate and log games
python analyze_bluff_ReactionCFR_DQNBluff.py  # CFR reactions
python analyze_bluff_ReactionDQN_CFRBluff.py  # DQN reactions
python statistical_bluff_detection.py         # Statistical bluff analysis
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `BLUFFING_RESULTS_DIR` | `./results` | Base directory for outputs |
| `TRAIN_EPISODES` | 100000 | Training episodes (paper value) |
| `EVAL_GAMES` | 100000 | Evaluation games (paper value) |
| `WANDB_MODE` | `disabled` in Docker | Set to `online` to log to W&B |

## Paper Details

- **Environment**: 52-card Leduc Hold'em (RLCard-based, extended from 6-card)
- **Training**: DQN and CFR train against each other for 100K episodes
- **DQN**: [256,256] MLP, ϵ-greedy, replay buffer 20K
- **CFR**: Tabular CFR with 10 iterations per episode
- **Bluff detection**: Threshold-based (hand score ≤32) and statistics-based

## Reference

- Paper: *Analysis of Bluffing by DQN and CFR in Leduc Hold'em Poker*
- GitHub: [TarikZ03/Bluffing-by-DQN-and-CFR-in-Leduc-Hold-em-Poker-Codebase](https://github.com/TarikZ03/Bluffing-by-DQN-and-CFR-in-Leduc-Hold-em-Poker-Codebase)
