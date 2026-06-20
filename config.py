"""
Shared configuration for thesis experiments (train_parallel_suite.py).

Override via environment variables:
  BLUFFING_RESULTS_DIR - Base directory for outputs (default: ./results)
  TRAIN_EPISODES       - Training episodes (default: 100000)
  EVAL_GAMES           - Final evaluation games (default: 100000)
"""
import os

RESULTS_DIR = os.environ.get(
    "BLUFFING_RESULTS_DIR",
    os.path.join(os.path.dirname(__file__), "results"),
)
TRAIN_EPISODES = int(os.environ.get("TRAIN_EPISODES", "100000"))
EVAL_GAMES = int(os.environ.get("EVAL_GAMES", "100000"))
