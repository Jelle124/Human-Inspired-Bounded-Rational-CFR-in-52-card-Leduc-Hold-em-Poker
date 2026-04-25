#!/usr/bin/env python3
"""
Evaluate all checkpoints produced by `parallel_br_cfr_train_suite.py` in one run.

For each of the four .pkl files under `results/br_cfr/parallel_suite_output/` by default
(`PARALLEL_SUITE_DIR` overrides),
runs the same matchups as `evaluate_br_cfr.py` per model:
  - DQN vs loaded policy
  - Loaded policy vs replication CFR (if `results/training/cfr_simultaneous_100K.pkl` exists)

So you get **4 models × 2 matchups = 8 headline numbers** (plus draws), not "2 results" total.

By default uses **BR_CFR_EVAL_GAMES** (100_000) games per matchup from `br_cfr_config`, overridable with
**BR_CFR_EVAL_N**. Per-game JSONL logs are **disabled** here so evaluation stays one summary row per matchup
(graph-friendly); set PARALLEL_EVAL_GAME_LOGS=1 to write full logs (large files at 100k).

Usage:
  python evaluate_parallel_suite.py

  BR_CFR_EVAL_N=100000 python evaluate_parallel_suite.py

Docker:
  docker run --shm-size=2g -v "$(pwd):/app" -v "$(pwd)/results:/app/results" \\
    bluffing-leduc python evaluate_parallel_suite.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import torch
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed

from br_cfr_config import BR_CFR_BASE, BR_CFR_EVAL_DIR, BR_CFR_EVAL_GAMES
from config import CFR_MODEL_PATH, DQN_MODEL_PATH
from custom_leduc_rlcard.leducholdem import LeducholdemEnv
from evaluate_br_cfr import (  # noqa: E402 — side effect: registers custom-leduc-holdem
    CFRPolicyWrapper,
    convert_ndarrays,
    evaluate_match,
    load_cfr_policy,
)

SEED = 42

# Same filenames as parallel_br_cfr_train_suite.py
SUITE_FILES = [
    ("smart_table", "br_cfr_smart_table.pkl"),
    ("medium_table", "br_cfr_medium_table.pkl"),
    ("dumb_table", "br_cfr_dumb_table.pkl"),
    ("cfr_selfplay_baseline", "cfr_selfplay_baseline_smart_iters.pkl"),
]


def _suite_dir() -> str:
    return os.environ.get(
        "PARALLEL_SUITE_DIR",
        os.path.join(BR_CFR_BASE, "parallel_suite_output"),
    )


def main():
    set_seed(SEED)
    os.makedirs(BR_CFR_EVAL_DIR, exist_ok=True)

    if not os.path.exists(DQN_MODEL_PATH):
        print(f"ERROR: DQN not found: {DQN_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    num_games = int(os.environ.get("BR_CFR_EVAL_N", str(BR_CFR_EVAL_GAMES)))
    write_game_logs = os.environ.get("PARALLEL_EVAL_GAME_LOGS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    suite = _suite_dir()

    env = LeducholdemEnv(config={"seed": SEED, "allow_step_back": False})
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[256, 256],
        device=device,
    )
    dqn_agent.q_estimator.qnet.load_state_dict(
        torch.load(DQN_MODEL_PATH, map_location=device)
    )
    dqn_agent.q_estimator.qnet.eval()

    repl_cfr_agent = None
    if os.path.exists(CFR_MODEL_PATH):
        ap, bm, bp = load_cfr_policy(CFR_MODEL_PATH)
        repl_cfr_agent = CFRPolicyWrapper(
            ap, env, bucket_mode=bm, br_cfr_params=bp
        )

    print("=" * 70)
    print("PARALLEL SUITE EVALUATION (4 checkpoints)")
    print("=" * 70)
    print(f"Suite dir:      {suite}")
    print(f"Games/match:    {num_games:,}")
    print(f"DQN:            {DQN_MODEL_PATH}")
    print(f"Replication CFR:{CFR_MODEL_PATH} ({'ok' if repl_cfr_agent else 'missing'})")
    print("=" * 70)

    rows = []
    csv_flat: list[dict] = []

    for label, fname in SUITE_FILES:
        path = os.path.join(suite, fname)
        if not os.path.exists(path):
            print(f"\nSKIP {label}: not found → {path}")
            rows.append({"label": label, "path": path, "skipped": True})
            continue

        ap, bm, bp = load_cfr_policy(path)
        agent = CFRPolicyWrapper(ap, env, bucket_mode=bm, br_cfr_params=bp)

        log_dqn = None
        if write_game_logs:
            log_dqn = os.path.join(
                BR_CFR_EVAL_DIR,
                f"parallel_suite_eval_{label}_vs_dqn_{num_games}.jsonl",
            )
        wins1, _, _ = evaluate_match(
            env, dqn_agent, agent, num_games, 0, 1, log_path=log_dqn
        )
        dqn_wr = wins1[0] / num_games
        pol_wr = wins1[1] / num_games
        dr = wins1[2] / num_games
        row = {
            "label": label,
            "path": path,
            "dqn_wr": dqn_wr,
            "policy_wr": pol_wr,
            "draws": dr,
            "num_games": num_games,
            "log_dqn_vs_policy": log_dqn,
        }
        csv_flat.append(
            {
                "label": label,
                "matchup": "dqn_vs_policy",
                "seat0_role": "DQN",
                "seat1_role": "loaded_policy",
                "num_games": num_games,
                "wr_seat0": dqn_wr,
                "wr_seat1": pol_wr,
                "wr_draw": dr,
            }
        )
        print(
            f"\n[{label}] DQN vs policy: DQN {dqn_wr:.4f} | policy {pol_wr:.4f} | draws {dr:.4f}"
        )
        if log_dqn:
            print(f"    log: {log_dqn}")

        if repl_cfr_agent and os.path.abspath(path) != os.path.abspath(CFR_MODEL_PATH):
            log_pc = None
            if write_game_logs:
                log_pc = os.path.join(
                    BR_CFR_EVAL_DIR,
                    f"parallel_suite_eval_{label}_vs_repl_cfr_{num_games}.jsonl",
                )
            wins2, _, _ = evaluate_match(
                env, agent, repl_cfr_agent, num_games, 0, 1, log_path=log_pc
            )
            row["vs_repl_cfr_policy_wr"] = wins2[0] / num_games
            row["vs_repl_cfr_repl_wr"] = wins2[1] / num_games
            row["vs_repl_cfr_draw_wr"] = wins2[2] / num_games
            row["log_policy_vs_repl"] = log_pc
            csv_flat.append(
                {
                    "label": label,
                    "matchup": "policy_vs_repl_cfr",
                    "seat0_role": "loaded_policy",
                    "seat1_role": "replication_cfr",
                    "num_games": num_games,
                    "wr_seat0": row["vs_repl_cfr_policy_wr"],
                    "wr_seat1": row["vs_repl_cfr_repl_wr"],
                    "wr_draw": row["vs_repl_cfr_draw_wr"],
                }
            )
            print(
                f"    vs replication CFR: policy {row['vs_repl_cfr_policy_wr']:.4f} | "
                f"repl CFR {row['vs_repl_cfr_repl_wr']:.4f} | draws {row['vs_repl_cfr_draw_wr']:.4f}"
            )
            if log_pc:
                print(f"    log: {log_pc}")
        rows.append(row)

    out_json = os.path.join(BR_CFR_EVAL_DIR, f"parallel_suite_eval_summary_{num_games}.json")
    payload = convert_ndarrays(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    suite_json = os.path.join(suite, f"parallel_suite_eval_summary_{num_games}.json")
    with open(suite_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    out_csv = os.path.join(BR_CFR_EVAL_DIR, f"parallel_suite_eval_summary_{num_games}.csv")
    if csv_flat:
        fieldnames = list(csv_flat[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_flat)

    suite_csv = os.path.join(suite, f"parallel_suite_eval_summary_{num_games}.csv")
    if csv_flat:
        with open(suite_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_flat)

    print("\n" + "=" * 70)
    print(f"Summary JSON: {out_json}")
    print(f"Suite JSON:   {suite_json}")
    print(f"Summary CSV:  {out_csv}")
    print(f"Suite CSV:    {suite_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
