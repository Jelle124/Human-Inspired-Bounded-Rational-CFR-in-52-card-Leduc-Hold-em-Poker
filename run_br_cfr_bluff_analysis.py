#!/usr/bin/env python3
"""
Run bluff analysis for all BR-CFR variants.

1. Generates game logs (DQN vs each BR-CFR) with full action traces
2. Runs "CFR reaction to DQN bluffs" analysis on each log
   (How does each BR-CFR variant react when DQN bluffs?)

Output is saved to results/br_cfr/BLUFF_ANALYSIS_OUTPUT.txt

Usage:
  python run_br_cfr_bluff_analysis.py

  # In Docker:
  docker run -v "$(pwd)/results:/app/results" bluffing python run_br_cfr_bluff_analysis.py

  # Fewer games (faster):
  BR_CFR_EVAL_N=5000 python run_br_cfr_bluff_analysis.py
"""

import os
import sys
import subprocess

from br_cfr_config import BR_CFR_EVAL_DIR, BR_CFR_TRAIN_DIR, BR_CFR_BASE

VARIANTS = ["smart", "medium", "dumb"]
NUM_GAMES = int(os.environ.get("BR_CFR_EVAL_N", "10000"))
OUTPUT_FILE = os.path.join(BR_CFR_BASE, "BLUFF_ANALYSIS_OUTPUT.txt")


def _capture_and_print(cmd, env, msg):
    """Run subprocess, capture output, print it, and return (output, returncode)."""
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = (result.stdout or "") + (result.stderr or "")
    print(msg)
    print(out)
    return out, result.returncode


def main():
    os.makedirs(BR_CFR_BASE, exist_ok=True)
    buffer = []

    def log(*args):
        s = " ".join(str(a) for a in args) + "\n"
        buffer.append(s)
        print(s, end="")

    # Step 1: Generate game logs
    log("=" * 70)
    log("Step 1: Generate game logs (DQN vs each BR-CFR)")
    log("=" * 70)
    env = os.environ.copy()
    env["BR_CFR_WRITE_LOGS"] = "1"
    out, ret = _capture_and_print(
        [sys.executable, "run_evaluate_all_br_cfr.py"],
        env, "",
    )
    buffer.append(out)
    if ret != 0:
        log("ERROR: Evaluation failed")
        sys.exit(ret)

    # Step 2: Run bluff analysis for each variant
    log("\n" + "=" * 70)
    log("Step 2: Bluff analysis (BR-CFR reaction to DQN bluffs)")
    log("=" * 70)

    for variant in VARIANTS:
        log_path = os.path.join(
            BR_CFR_EVAL_DIR,
            f"game_logs_dqn_vs_br_cfr_{variant}_{NUM_GAMES}.jsonl",
        )
        if not os.path.exists(log_path):
            log(f"  SKIP {variant}: {log_path} not found")
            continue

        log(f"\n--- Analyzing {variant} ---")
        env = os.environ.copy()
        env["BLUFF_LOG_PATH"] = os.path.abspath(log_path)
        env.setdefault("WANDB_MODE", "disabled")
        out, ret = _capture_and_print(
            [sys.executable, "analyze_bluff_ReactionCFR_DQNBluff.py"],
            env, "",
        )
        buffer.append(out)
        if ret != 0:
            log(f"  WARNING: Analysis failed for {variant}")

    log("\n" + "=" * 70)
    log("BR-CFR bluff analysis complete")
    log("=" * 70)
    log(f"Logs: {BR_CFR_EVAL_DIR}")
    log("Run with WANDB_MODE=online to log to Weights & Biases")

    buffer.append(f"\nFull output saved to {OUTPUT_FILE}\n")
    with open(OUTPUT_FILE, "w") as f:
        f.write("".join(buffer))
    print(f"\nFull output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
