#!/usr/bin/env python3
"""
Train 3 BR-CFR variants (smart, medium, dumb) sequentially in one run.

Designed for overnight runs: start the container, leave it, have all 3 models
ready in the morning.

Output:
  results/br_cfr/training/br_cfr_smart.pkl
  results/br_cfr/training/br_cfr_medium.pkl
  results/br_cfr/training/br_cfr_dumb.pkl

Usage:
  python run_br_cfr_smart_medium_dumb.py

  # In Docker:
  docker run -v "$(pwd)/results:/app/results" bluffing python run_br_cfr_smart_medium_dumb.py

  # Shorter runs (for testing):
  BR_CFR_TRAIN_EPISODES=1000 python run_br_cfr_smart_medium_dumb.py
"""

import os
import sys
import time
import numpy as np
import torch
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed
from rlcard.envs.registration import register

from custom_leduc_rlcard.leducholdem import LeducholdemEnv
from config import DQN_MODEL_PATH
from br_cfr_config import (
    BR_CFR_BASE,
    BR_CFR_TRAIN_DIR,
    BR_CFR_TRAIN_EPISODES,
    PRESETS,
    merge_br_cfr_preset,
)
from br_cfr_agent import BoundedRationalCFRAgent, BRCFRWrapper

register(
    env_id="custom-leduc-holdem",
    entry_point="custom_leduc_rlcard.leducholdem:LeducholdemEnv",
)

VARIANTS = ["smart", "medium", "dumb"]
SEED = 42


def train_one(env, dqn_agent, preset_name):
    """Train a single BR-CFR variant."""
    params = merge_br_cfr_preset(PRESETS[preset_name])
    save_path = os.path.join(BR_CFR_TRAIN_DIR, f"br_cfr_{preset_name}.pkl")
    agent_kw = {k: v for k, v in params.items() if k != "description"}

    br_cfr_agent = BoundedRationalCFRAgent(
        env,
        player_id=1,
        opponent_agent=dqn_agent,
        model_path=save_path,
        **agent_kw,
    )
    env.set_agents([dqn_agent, BRCFRWrapper(br_cfr_agent)])

    for episode in range(BR_CFR_TRAIN_EPISODES):
        for _ in range(params["iterations_per_episode"]):
            env.reset()
            root_u = br_cfr_agent.traverse_tree(1.0 * np.ones(env.num_players))
            br_cfr_agent.register_hand_outcome(float(root_u[br_cfr_agent.player_id]))
            br_cfr_agent.iteration += 1

        if episode % 5000 == 0 and episode > 0:
            print(
                f"  [{preset_name}] Ep {episode:,}/{BR_CFR_TRAIN_EPISODES:,} | "
                f"Iterations: {br_cfr_agent.iteration:,} | "
                f"States: {len(br_cfr_agent.average_policy):,}"
            )

    br_cfr_agent.save()
    return save_path


def main():
    set_seed(SEED)
    os.makedirs(BR_CFR_TRAIN_DIR, exist_ok=True)

    # Make folder structure self-explanatory
    readme = os.path.join(BR_CFR_BASE, "README.txt")
    with open(readme, "w") as f:
        f.write("BR-CFR (Bounded-Rational CFR) results\n")
        f.write("=====================================\n\n")
        f.write("training/   -> br_cfr_smart.pkl, br_cfr_medium.pkl, br_cfr_dumb.pkl\n")
        f.write("evaluation/ -> evaluation logs (if evaluate_br_cfr.py was run)\n")

    if not os.path.exists(DQN_MODEL_PATH):
        print(f"ERROR: DQN not found at {DQN_MODEL_PATH}", file=sys.stderr)
        print("Run the replication pipeline first.", file=sys.stderr)
        sys.exit(1)

    env = LeducholdemEnv(config={"seed": SEED, "allow_step_back": True})
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
    for p in dqn_agent.q_estimator.qnet.parameters():
        p.requires_grad = False

    print("=" * 70)
    print("BR-CFR: Smart, Medium, Dumb (overnight batch)")
    print("=" * 70)
    print(f"Episodes per variant: {BR_CFR_TRAIN_EPISODES:,}")
    print(f"Variants: {VARIANTS}")
    for v in VARIANTS:
        p = PRESETS[v]
        print(f"  - {v}: {p['description']}")
    print("=" * 70)

    results = []
    for i, variant in enumerate(VARIANTS):
        start = time.time()
        print(f"\n[{i+1}/3] Training {variant}...")
        path = train_one(env, dqn_agent, variant)
        elapsed = time.time() - start
        results.append((variant, path, elapsed))
        print(f"  Done in {elapsed/60:.1f} min -> {path}")

    print("\n" + "=" * 70)
    print("ALL 3 VARIANTS COMPLETE")
    print("=" * 70)
    for variant, path, elapsed in results:
        print(f"  {variant}: {path} ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
