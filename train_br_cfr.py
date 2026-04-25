#!/usr/bin/env python3
"""
Train Bounded-Rational CFR (BR-CFR) against the pre-trained DQN.

This script is SEPARATE from the replication pipeline (simultaneous_training.py).
It loads the existing trained DQN and trains a BR-CFR agent against it as a
fixed opponent. No DQN training occurs—only CFR iterations.

Outputs go to results/br_cfr/training/ (not results/training/).

Usage:
  python train_br_cfr.py

  # Use a different preset (see br_cfr_config.PRESETS):
  BR_CFR_PRESET=state_bucketing python train_br_cfr.py

  # Quick test (1K episodes):
  BR_CFR_TRAIN_EPISODES=1000 python train_br_cfr.py
"""

import os
import pickle

import numpy as np
import torch
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed
from rlcard.envs.registration import register

from custom_leduc_rlcard.leducholdem import LeducholdemEnv
from config import DQN_MODEL_PATH  # Replication DQN (read-only)
from br_cfr_config import (
    BR_CFR_TRAIN_DIR,
    BR_CFR_TRAIN_EPISODES,
    BR_CFR_EVAL_GAMES,
    PRESETS,
    ACTIVE_PRESET,
    merge_br_cfr_preset,
)
from br_cfr_agent import BoundedRationalCFRAgent, BRCFRWrapper


# Register environment (same as replication)
register(
    env_id="custom-leduc-holdem",
    entry_point="custom_leduc_rlcard.leducholdem:LeducholdemEnv",
)

SEED = 42


def get_br_cfr_params():
    """Get BR-CFR params from active preset (includes mood/tilt defaults)."""
    preset = PRESETS.get(ACTIVE_PRESET)
    if preset is None:
        raise ValueError(
            f"Unknown BR_CFR_PRESET='{ACTIVE_PRESET}'. "
            f"Valid: {list(PRESETS.keys())}"
        )
    return merge_br_cfr_preset(preset)


def train():
    set_seed(SEED)
    os.makedirs(BR_CFR_TRAIN_DIR, exist_ok=True)

    params = get_br_cfr_params()
    model_name = f"br_cfr_{ACTIVE_PRESET}.pkl"
    save_path = os.path.join(BR_CFR_TRAIN_DIR, model_name)

    env = LeducholdemEnv(config={"seed": SEED, "allow_step_back": True})
    env.reset()

    print("=" * 70)
    print("BR-CFR TRAINING (Bounded-Rational CFR)")
    print("=" * 70)
    print(f"Preset: {ACTIVE_PRESET}")
    print(f"Description: {params['description']}")
    print(f"  iterations_per_episode: {params['iterations_per_episode']}")
    print(f"  memory_decay:          {params['memory_decay']}")
    print(f"  soft_regret_tau:       {params['soft_regret_tau']}")
    print(f"  bucket_mode:           {params['bucket_mode']}")
    print(f"  mood_enabled:          {params['mood_enabled']}")
    print(f"  tilt_enabled:          {params['tilt_enabled']}")
    print(f"  qre_normalize_regrets: {params['qre_normalize_regrets']}")
    print(f"Train episodes: {BR_CFR_TRAIN_EPISODES:,}")
    print(f"Opponent: Pre-trained DQN from {DQN_MODEL_PATH}")
    print(f"Output:   {save_path}")
    print("=" * 70)

    if not os.path.exists(DQN_MODEL_PATH):
        raise FileNotFoundError(
            f"DQN model not found at {DQN_MODEL_PATH}. "
            "Run the replication pipeline first (e.g. run_pipeline.py) to train the DQN."
        )

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
    # Freeze DQN — no training
    for p in dqn_agent.q_estimator.qnet.parameters():
        p.requires_grad = False

    agent_kw = {k: v for k, v in params.items() if k != "description"}
    br_cfr_agent = BoundedRationalCFRAgent(
        env,
        player_id=1,
        opponent_agent=dqn_agent,
        model_path=save_path,
        **agent_kw,
    )
    env.set_agents([dqn_agent, BRCFRWrapper(br_cfr_agent)])

    eval_interval = max(5000, BR_CFR_TRAIN_EPISODES // 20)
    eval_games = min(BR_CFR_EVAL_GAMES, 2000)

    for episode in range(BR_CFR_TRAIN_EPISODES):
        for _ in range(params["iterations_per_episode"]):
            env.reset()
            root_u = br_cfr_agent.traverse_tree(1.0 * np.ones(env.num_players))
            br_cfr_agent.register_hand_outcome(float(root_u[br_cfr_agent.player_id]))
            br_cfr_agent.iteration += 1

        if episode % 1000 == 0:
            total_regret = sum(
                np.sum(np.abs(r)) for r in br_cfr_agent.regrets.values()
            )
            print(
                f"[Episode {episode:,}] BR-CFR iterations: {br_cfr_agent.iteration:,}, "
                f"States: {len(br_cfr_agent.average_policy):,}, "
                f"Regret: {total_regret:.1f}"
            )

        if episode > 0 and episode % eval_interval == 0:
            wins = [0, 0, 0]
            for _ in range(eval_games):
                _, payoffs = env.run(is_training=False)
                if payoffs[0] > payoffs[1]:
                    wins[0] += 1
                elif payoffs[1] > payoffs[0]:
                    wins[1] += 1
                else:
                    wins[2] += 1
            dqn_wr = wins[0] / eval_games
            br_wr = wins[1] / eval_games
            print(
                f"[Ep {episode:,}] Eval: DQN WR={dqn_wr:.3f}, BR-CFR WR={br_wr:.3f}"
            )

    br_cfr_agent.save()
    print(f"\n✅ BR-CFR training complete!")
    print(f"   Saved to {save_path}")
    print(f"   Total iterations: {br_cfr_agent.iteration:,}")
    print(f"   States in policy: {len(br_cfr_agent.average_policy):,}")


if __name__ == "__main__":
    train()
