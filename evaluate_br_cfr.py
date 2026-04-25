#!/usr/bin/env python3
"""
Evaluate BR-CFR (or replication CFR) against pre-trained DQN and CFR.

This script is SEPARATE from evaluate_simultaneous.py. It can evaluate:
  1. BR-CFR vs DQN
  2. BR-CFR vs replication CFR (loaded from replication results)
  3. Replication CFR vs DQN (for baseline comparison)

Outputs go to results/br_cfr/evaluation/ (not results/evaluation/).

Usage:
  # Evaluate the soft_regret BR-CFR (after training):
  python evaluate_br_cfr.py

  # Evaluate a specific BR-CFR model:
  BR_CFR_MODEL=results/br_cfr/training/br_cfr_state_bucketing.pkl python evaluate_br_cfr.py

  # Evaluate replication CFR vs DQN (baseline, no BR-CFR):
  BR_CFR_MODEL= python evaluate_br_cfr.py  # Uses replication CFR
"""

import os
import json
import pickle
import torch
import numpy as np
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed
from rlcard.envs.registration import register

from custom_leduc_rlcard.leducholdem import LeducholdemEnv
from config import DQN_MODEL_PATH, CFR_MODEL_PATH
from br_cfr_config import (
    BR_CFR_EVAL_DIR,
    BR_CFR_EVAL_GAMES,
    BR_CFR_TRAIN_DIR,
    MOOD_TILT_PARAM_DEFAULTS,
)
from br_cfr_agent import obs_to_key, MoodTiltSession

register(
    env_id="custom-leduc-holdem",
    entry_point="custom_leduc_rlcard.leducholdem:LeducholdemEnv",
)

SEED = 42

# Which BR-CFR model to evaluate (empty string = use replication CFR as baseline)
_DEFAULT_BR_CFR_MODEL = os.path.join(BR_CFR_TRAIN_DIR, "br_cfr_soft_regret.pkl")
BR_CFR_MODEL = os.environ.get("BR_CFR_MODEL", _DEFAULT_BR_CFR_MODEL)


def convert_ndarrays(obj):
    """Convert numpy types for JSON serialization."""
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


def _mood_tilt_session_from_saved(br_params):
    kw = dict(MOOD_TILT_PARAM_DEFAULTS)
    for k in kw:
        if k in br_params:
            kw[k] = br_params[k]
    return MoodTiltSession(**kw)


class CFRPolicyWrapper:
    """
    Wrapper for any CFR-like policy (BR-CFR or replication CFR).
    Uses obs_to_key with bucket_mode when policy has br_cfr_params.
    Replays mood/tilt from saved br_cfr_params when present (defaults off).
    """

    def __init__(self, average_policy, env, bucket_mode="none", br_cfr_params=None):
        self.average_policy = average_policy
        self.env = env
        self.use_raw = False
        self.bucket_mode = bucket_mode
        self.mood_tilt = _mood_tilt_session_from_saved(br_cfr_params or {})

    def _obs_key(self, state):
        return obs_to_key(state, self.env, self.bucket_mode)

    def step(self, state):
        obs = self._obs_key(state)
        legal_actions = list(state["legal_actions"].keys())

        if obs not in self.average_policy:
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions
        else:
            raw_probs = self.average_policy[obs].copy()
            action_probs = np.array(
                [raw_probs[a] if a in legal_actions else 0 for a in range(self.env.num_actions)]
            )

        total = np.sum(action_probs)
        if total > 0:
            action_probs = action_probs / total
        else:
            action_probs = np.zeros(self.env.num_actions)
            for a in legal_actions:
                action_probs[a] = 1.0 / len(legal_actions)

        action_probs = self.mood_tilt.modulate_play_probs(
            action_probs, legal_actions, self.env.num_actions
        )

        return int(np.random.choice(self.env.num_actions, p=action_probs))

    def notify_hand_end(self, payoff: float):
        self.mood_tilt.hand_end(float(payoff))

    def eval_step(self, state):
        action = self.step(state)
        return action, {"probs": {}}


def load_cfr_policy(model_path):
    """
    Load CFR/BR-CFR policy. Returns (average_policy, bucket_mode, br_cfr_params).
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    average_policy = data["average_policy"]
    br_params = data.get("br_cfr_params", {})
    bucket_mode = br_params.get("bucket_mode", "none")
    return average_policy, bucket_mode, br_params


def load_agents(env, cfr_model_path):
    """Load DQN and CFR (or BR-CFR) agents."""
    if not os.path.exists(DQN_MODEL_PATH):
        raise FileNotFoundError(f"DQN not found: {DQN_MODEL_PATH}")

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

    if not cfr_model_path or not os.path.exists(cfr_model_path):
        # Fallback to replication CFR
        cfr_model_path = CFR_MODEL_PATH
        if not os.path.exists(cfr_model_path):
            raise FileNotFoundError(f"CFR not found: {CFR_MODEL_PATH}")

    average_policy, bucket_mode, br_params = load_cfr_policy(cfr_model_path)
    cfr_agent = CFRPolicyWrapper(
        average_policy, env, bucket_mode=bucket_mode, br_cfr_params=br_params
    )
    return dqn_agent, cfr_agent, cfr_model_path


def evaluate_match(env, agent_a, agent_b, num_games, a_id=0, b_id=1, log_path=None):
    """
    Run agent_a (id 0) vs agent_b (id 1). Returns wins, payoffs, optional logs.
    """
    env.set_agents([agent_a, agent_b])
    wins = [0, 0, 0]
    payoffs_a = []
    payoffs_b = []
    logs_list = []

    for g in range(num_games):
        env.reset()
        game_logs = [] if log_path else None
        while not env.is_over():
            pid = env.get_player_id()
            state = env.get_state(pid)
            agent = agent_a if pid == a_id else agent_b
            action, _ = agent.eval_step(state)
            if log_path:
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
        payoffs_a.append(pay[a_id])
        payoffs_b.append(pay[b_id])
        if pay[a_id] > pay[b_id]:
            wins[0] += 1
        elif pay[b_id] > pay[a_id]:
            wins[1] += 1
        else:
            wins[2] += 1
        if log_path:
            rec = {
                "game": g + 1,
                "log": convert_ndarrays(game_logs),
                "payoffs": convert_ndarrays(list(pay)),
            }
            logs_list.append(rec)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            for rec in logs_list:
                f.write(json.dumps(rec) + "\n")
    return wins, payoffs_a, payoffs_b


def main():
    set_seed(SEED)
    os.makedirs(BR_CFR_EVAL_DIR, exist_ok=True)

    env = LeducholdemEnv(config={"seed": SEED, "allow_step_back": False})
    env.reset()

    num_games = BR_CFR_EVAL_GAMES
    if num_games > 50000:
        num_games = min(num_games, 10000)  # Cap for quick runs
    num_games = int(os.environ.get("BR_CFR_EVAL_N", num_games))

    cfr_path = BR_CFR_MODEL if (BR_CFR_MODEL and os.path.exists(BR_CFR_MODEL)) else CFR_MODEL_PATH

    print("=" * 70)
    print("BR-CFR EVALUATION")
    print("=" * 70)
    print(f"CFR/BR-CFR model: {cfr_path or CFR_MODEL_PATH}")
    print(f"DQN model:        {DQN_MODEL_PATH}")
    print(f"Games per match:  {num_games:,}")
    print("=" * 70)

    dqn_agent, cfr_agent, cfr_used = load_agents(env, BR_CFR_MODEL)
    label = "BR-CFR" if "br_cfr" in cfr_used and os.path.basename(cfr_used).startswith("br_cfr_") else "CFR"

    # Match 1: DQN vs CFR/BR-CFR
    log_dqn_vs_cfr = os.path.join(
        BR_CFR_EVAL_DIR,
        f"br_cfr_eval_{label.lower().replace('-', '_')}_vs_dqn_{num_games}.jsonl",
    )
    wins1, pay_a1, pay_b1 = evaluate_match(
        env, dqn_agent, cfr_agent, num_games,
        a_id=0, b_id=1,
        log_path=log_dqn_vs_cfr,
    )
    dqn_wr = wins1[0] / num_games
    cfr_wr = wins1[1] / num_games
    print(f"\n[1] DQN vs {label}:")
    print(f"    DQN WR: {dqn_wr:.3f}, {label} WR: {cfr_wr:.3f}, Draws: {wins1[2]/num_games:.3f}")
    print(f"    Log: {log_dqn_vs_cfr}")

    # Match 2: CFR/BR-CFR vs replication CFR (if we have BR-CFR)
    if "br_cfr" in cfr_used and os.path.exists(CFR_MODEL_PATH):
        _, repl_cfr_agent, _ = load_agents(env, CFR_MODEL_PATH)
        log_br_vs_cfr = os.path.join(
            BR_CFR_EVAL_DIR,
            f"br_cfr_eval_br_cfr_vs_cfr_{num_games}.jsonl",
        )
        wins2, pay_br, pay_cfr = evaluate_match(
            env, cfr_agent, repl_cfr_agent, num_games,
            a_id=0, b_id=1,  # BR-CFR is agent 0, replication CFR is agent 1
            log_path=log_br_vs_cfr,
        )
        br_wr = wins2[0] / num_games
        repl_wr = wins2[1] / num_games
        print(f"\n[2] BR-CFR vs replication CFR:")
        print(f"    BR-CFR WR: {br_wr:.3f}, CFR WR: {repl_wr:.3f}")
        print(f"    Log: {log_br_vs_cfr}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
