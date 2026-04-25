"""
Shared configuration for the Bluffing by DQN and CFR replication.
Override paths via environment variables:
  BLUFFING_RESULTS_DIR - Base directory for all outputs (default: ./results)
  TRAIN_EPISODES - Training episodes (default: 100000, use 1000 for quick tests)
  EVAL_GAMES - Evaluation games (default: 100000, use 1000 for quick tests)
"""
import os

RESULTS_DIR = os.environ.get("BLUFFING_RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))
TRAIN_DIR = os.path.join(RESULTS_DIR, "training")
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")

# Episode/game counts - reduce for quick laptop runs
TRAIN_EPISODES = int(os.environ.get("TRAIN_EPISODES", "100000"))
EVAL_GAMES = int(os.environ.get("EVAL_GAMES", "100000"))

# Model paths (relative to results dir)
DQN_MODEL_PATH = os.path.join(TRAIN_DIR, "dqn_simultaneous_100K.pt")
CFR_MODEL_PATH = os.path.join(TRAIN_DIR, "cfr_simultaneous_100K.pkl")
# Win rates during simultaneous training (eval every eval_interval episodes); see plot_simultaneous_training_win_rates.py
TRAINING_WIN_RATES_JSONL = os.path.join(TRAIN_DIR, "simultaneous_training_win_rates.jsonl")
# CFR vs BR-CFR (smart preset), both trained from scratch — simultaneous_training_cfr_vs_br_smart.py
CFR_VS_BR_SUBDIR = os.path.join(RESULTS_DIR, "cfr_vs_br_smart")
CFR_VS_BR_TRAIN_DIR = os.path.join(CFR_VS_BR_SUBDIR, "training")
CFR_VS_BR_WIN_RATES_JSONL = os.path.join(
    CFR_VS_BR_TRAIN_DIR, "cfr_vs_br_smart_colearn_training_win_rates.jsonl"
)
CFR_VS_BR_COLEARN_CFR_PATH = os.path.join(CFR_VS_BR_TRAIN_DIR, "cfr_colearn_vs_br_smart.pkl")
CFR_VS_BR_COLEARN_BR_PATH = os.path.join(CFR_VS_BR_TRAIN_DIR, "br_cfr_smart_colearn.pkl")
LOG_ALL_PATH = os.path.join(EVAL_DIR, "evaluation_game_logs_all_100K.jsonl")
LOG_CFR_PATH = os.path.join(EVAL_DIR, "evaluation_game_logs_cfr_pov_100K.jsonl")
LOG_DQN_PATH = os.path.join(EVAL_DIR, "evaluation_game_logs_dqn_pov_100K.jsonl")
