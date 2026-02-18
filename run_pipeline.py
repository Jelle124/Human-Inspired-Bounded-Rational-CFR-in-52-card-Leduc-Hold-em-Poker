#!/usr/bin/env python3
"""
Run the full replication pipeline:
  1. Simultaneous training (DQN vs CFR, 100K episodes by default)
  2. Evaluation (100K games, logs actions for bluff analysis)
  3. Bluff analysis (threshold-based and statistical detectors)

Set TRAIN_EPISODES=1000 and EVAL_GAMES=1000 for a quick sanity check (~5–10 min on a laptop).
"""
import os
import sys
import subprocess

def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"STEP: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print('='*70)
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"ERROR: {desc} failed with exit code {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    train_ep = os.environ.get("TRAIN_EPISODES", "100000")
    eval_games = os.environ.get("EVAL_GAMES", "100000")
    os.environ["TRAIN_EPISODES"] = str(train_ep)
    os.environ["EVAL_GAMES"] = str(eval_games)

    run([sys.executable, "simultaneous_training.py"],
        "1. Simultaneous DQN & CFR training")
    run([sys.executable, "evaluate_simultaneous.py"],
        "2. Evaluation and game logging")
    run([sys.executable, "analyze_bluff_ReactionCFR_DQNBluff.py"],
        "3a. CFR reactions to DQN bluffs")
    run([sys.executable, "analyze_bluff_ReactionDQN_CFRBluff.py"],
        "3b. DQN reactions to CFR bluffs")
    run([sys.executable, "statistical_bluff_detection.py"],
        "3c. Statistical bluff detection")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    from config import RESULTS_DIR
    print(f"Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
