#!/usr/bin/env python3
"""
Evaluate all 3 BR-CFR variants (smart, medium, dumb) and write a summary.

Run this after training completes. Evaluates each variant vs DQN and vs
replication CFR, then saves a summary to results/br_cfr/EVALUATION_SUMMARY.txt.

Usage:
  python run_evaluate_all_br_cfr.py

  # In Docker:
  docker run -v "$(pwd)/results:/app/results" bluffing python run_evaluate_all_br_cfr.py

  # Fewer games (quick test):
  BR_CFR_EVAL_N=1000 python run_evaluate_all_br_cfr.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed
from rlcard.envs.registration import register

from custom_leduc_rlcard.leducholdem import LeducholdemEnv
from config import DQN_MODEL_PATH, CFR_MODEL_PATH
from br_cfr_config import BR_CFR_TRAIN_DIR, BR_CFR_EVAL_DIR, BR_CFR_BASE
from evaluate_br_cfr import CFRPolicyWrapper, load_cfr_policy

register(
    env_id="custom-leduc-holdem",
    entry_point="custom_leduc_rlcard.leducholdem:LeducholdemEnv",
)

VARIANTS = ["smart", "medium", "dumb"]
SEED = 42
NUM_GAMES = int(os.environ.get("BR_CFR_EVAL_N", "10000"))
WRITE_LOGS = os.environ.get("BR_CFR_WRITE_LOGS", "0").lower() in ("1", "true", "yes")


def convert_ndarrays(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj


def run_match(env, agent_a, agent_b, num_games, a_id=0, b_id=1, log_path=None):
    """Run match, optionally writing game logs for bluff analysis."""
    env.set_agents([agent_a, agent_b])
    wins = [0, 0, 0]
    logs_list = []
    for g in range(num_games):
        env.reset()
        game_logs = []
        while not env.is_over():
            pid = env.get_player_id()
            state = env.get_state(pid)
            agent = agent_a if pid == a_id else agent_b
            action, _ = agent.eval_step(state)
            raw = state.get("raw_obs", {})
            game_logs.append({
                "player_id": pid,
                "hand": raw.get("hand"),
                "public_card": raw.get("public_card"),
                "legal_actions": raw.get("legal_actions"),
                "action_taken": int(action),
            })
            env.step(action)
        pay = env.get_payoffs()
        if hasattr(agent_a, "notify_hand_end"):
            agent_a.notify_hand_end(float(pay[a_id]))
        if hasattr(agent_b, "notify_hand_end"):
            agent_b.notify_hand_end(float(pay[b_id]))
        if pay[a_id] > pay[b_id]:
            wins[0] += 1
        elif pay[b_id] > pay[a_id]:
            wins[1] += 1
        else:
            wins[2] += 1
        if log_path:
            logs_list.append({
                "game": g + 1,
                "log": convert_ndarrays(game_logs),
                "payoffs": convert_ndarrays(list(pay)),
            })
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            for rec in logs_list:
                f.write(json.dumps(rec) + "\n")
    return wins


def main():
    set_seed(SEED)
    os.makedirs(BR_CFR_EVAL_DIR, exist_ok=True)

    if not os.path.exists(DQN_MODEL_PATH):
        print(f"ERROR: DQN not found at {DQN_MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

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
        repl_cfr_agent = CFRPolicyWrapper(ap, env, bucket_mode=bm, br_cfr_params=bp)

    print("=" * 70)
    print("Evaluate all BR-CFR variants (smart, medium, dumb)")
    print("=" * 70)
    print(f"Games per match: {NUM_GAMES:,}")
    print(f"Write game logs for bluff analysis: {WRITE_LOGS}")
    if not WRITE_LOGS:
        print(f"  (Set BR_CFR_WRITE_LOGS=1 to generate logs for bluff analysis)")
    print()

    results = []
    for variant in VARIANTS:
        path = os.path.join(BR_CFR_TRAIN_DIR, f"br_cfr_{variant}.pkl")
        if not os.path.exists(path):
            print(f"  SKIP {variant}: {path} not found")
            continue

        start = time.time()
        ap, bm, bp = load_cfr_policy(path)
        br_agent = CFRPolicyWrapper(ap, env, bucket_mode=bm, br_cfr_params=bp)

        # DQN vs BR-CFR (DQN=0, BR-CFR=1 — for "CFR reaction to DQN bluffs" analysis)
        log_path = None
        if WRITE_LOGS:
            log_path = os.path.join(
                BR_CFR_EVAL_DIR, f"game_logs_dqn_vs_br_cfr_{variant}_{NUM_GAMES}.jsonl"
            )
        wins = run_match(env, dqn_agent, br_agent, NUM_GAMES, 0, 1, log_path=log_path)
        dqn_wr = wins[0] / NUM_GAMES
        br_wr = wins[1] / NUM_GAMES
        elapsed = time.time() - start

        row = {
            "variant": variant,
            "path": path,
            "dqnvbr_dqn_wr": dqn_wr,
            "dqnvbr_br_wr": br_wr,
            "elapsed_min": elapsed / 60,
        }

        # BR-CFR vs replication CFR
        if repl_cfr_agent:
            wins2 = run_match(env, br_agent, repl_cfr_agent, NUM_GAMES, 0, 1)
            row["brvcfr_br_wr"] = wins2[0] / NUM_GAMES
            row["brvcfr_cfr_wr"] = wins2[1] / NUM_GAMES

        if WRITE_LOGS:
            row["log_path"] = log_path
        results.append(row)
        msg = f"  {variant}: DQN vs BR-CFR -> DQN {dqn_wr:.3f} | BR-CFR {br_wr:.3f} ({elapsed/60:.1f} min)"
        if WRITE_LOGS:
            msg += f"\n      Log: {log_path}"
        print(msg)

    # Write summary
    summary_path = os.path.join(BR_CFR_BASE, "EVALUATION_SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("BR-CFR Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Games per match: {NUM_GAMES:,}\n\n")
        for r in results:
            f.write(f"{r['variant']}:\n")
            f.write(f"  DQN vs BR-CFR: DQN WR={r['dqnvbr_dqn_wr']:.3f}, BR-CFR WR={r['dqnvbr_br_wr']:.3f}\n")
            if "brvcfr_br_wr" in r:
                f.write(f"  BR-CFR vs CFR: BR WR={r['brvcfr_br_wr']:.3f}, CFR WR={r['brvcfr_cfr_wr']:.3f}\n")
            f.write("\n")
    print(f"\nSummary saved to {summary_path}")

    # Also save JSON for scripting
    json_path = os.path.join(BR_CFR_BASE, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved to {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
