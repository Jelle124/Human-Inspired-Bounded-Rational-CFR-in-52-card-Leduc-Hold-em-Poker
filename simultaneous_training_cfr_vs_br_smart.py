#!/usr/bin/env python3
"""
Co-train standard CFR and BR-CFR (**smart** preset) from scratch against each other.

  - Player 0: CFR (full obs key, regret matching) — same class as vs DQN in replication
  - Player 1: BoundedRationalCFRAgent with `PRESETS['smart']` (10 iters/ep metadata, no decay, τ=0, bucket none)

Each training episode runs `iterations_per_episode` tree traversals for CFR (vs current BR policy)
and the same count for BR-CFR (vs current CFR policy), so both learn from tabula rasa with no
frozen checkpoints.

Outputs (under results/cfr_vs_br_smart/training/ by default):
  - cfr_colearn_vs_br_smart.pkl
  - br_cfr_smart_colearn.pkl
  - cfr_vs_br_smart_colearn_training_win_rates.jsonl

Usage:
  python3 simultaneous_training_cfr_vs_br_smart.py

  TRAIN_EPISODES=1000 python3 simultaneous_training_cfr_vs_br_smart.py

Plot win rates:
  python3 plot_cfr_vs_br_smart_win_rates.py

If you see PermissionError under results/ (often after Docker or sudo created files), either:
  sudo chown -R "$USER:$USER" results
  or:  export BLUFFING_RESULTS_DIR="$HOME/bluffing_results"
The script will fall back to ~/.bluffing_leduc_results/cfr_vs_br_smart/training/ if needed.

Do not use ``sudo python3`` — that uses system Python without your venv (e.g. numpy missing).
"""

from __future__ import annotations

import collections
import json
import os
import pickle

import numpy as np
import torch
import wandb
from rlcard.envs.registration import register
from rlcard.utils import set_seed

import config as _cfg

from br_cfr_agent import BoundedRationalCFRAgent, BRCFRWrapper
from br_cfr_config import PRESETS, merge_br_cfr_preset
from config import TRAIN_EPISODES
from custom_leduc_rlcard.leducholdem import LeducholdemEnv

register(
    env_id="custom-leduc-holdem",
    entry_point="custom_leduc_rlcard.leducholdem:LeducholdemEnv",
)

SMART_PRESET = "smart"
config = {
    "train_episodes": TRAIN_EPISODES,
    "eval_interval": 5_000,
    "eval_games": 2_000,
    "iterations_per_episode": 10,
    "seed": 42,
}


def zero_array_4():
    return np.zeros(4)


class CFRWrapper:
    """Gameplay / opponent wrapper for the learning CFR agent (player 0)."""

    def __init__(self, cfr_agent):
        self.cfr = cfr_agent
        self.env = cfr_agent.env
        self.use_raw = False

    def step(self, state):
        obs = state["obs"].tobytes()
        legal_actions = list(state["legal_actions"].keys())
        if obs not in self.cfr.average_policy:
            action_probs = [1 / len(legal_actions)] * self.env.num_actions
        else:
            raw_probs = self.cfr.average_policy[obs]
            action_probs = [
                raw_probs[a] if a in legal_actions else 0 for a in range(self.env.num_actions)
            ]
            total = sum(action_probs)
            action_probs = [
                p / total if total > 0 else 1 / len(legal_actions) for p in action_probs
            ]
        return torch.multinomial(torch.tensor(action_probs), 1).item()

    def eval_step(self, state):
        action = self.step(state)
        return action, {"probs": {}}


class _OpponentSlot:
    """Late-bind opponent so CFR and BR-CFR can reference each other."""

    __slots__ = ("wrapped",)

    def __init__(self):
        self.wrapped = None

    def eval_step(self, state):
        return self.wrapped.eval_step(state)


class CFRAgainstOpponentAgent:
    """Standard CFR traversal against an opponent with eval_step."""

    def __init__(self, env, player_id, opponent_agent, model_path):
        self.env = env
        self.player_id = player_id
        self.opponent_agent = opponent_agent
        self.model_path = model_path
        self.use_raw = False

        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(zero_array_4)
        self.regrets = collections.defaultdict(zero_array_4)
        self.iteration = 0

    def regret_matching(self, obs):
        regret = self.regrets[obs]
        pos_regret = np.maximum(regret, 0)
        total = np.sum(pos_regret)
        return pos_regret / total if total > 0 else np.ones(self.env.num_actions) / self.env.num_actions

    def traverse_tree(self, probs):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        state = self.env.get_state(current_player)
        obs = state["obs"].tobytes()
        legal_actions = list(state["legal_actions"].keys())

        if current_player != self.player_id:
            action, _ = self.opponent_agent.eval_step(state)
            self.env.step(action)
            utility = self.traverse_tree(probs)
            self.env.step_back()
            return utility

        strategy = self.regret_matching(obs)
        action_utils = np.zeros(self.env.num_actions)
        node_util = 0

        for action in legal_actions:
            prob = strategy[action]
            new_probs = probs.copy()
            new_probs[current_player] *= prob
            self.env.step(action)
            utility = self.traverse_tree(new_probs)
            self.env.step_back()
            action_utils[action] = utility[self.player_id]
            node_util += prob * utility[self.player_id]

        cf_prob = np.prod(probs[:current_player]) * np.prod(probs[current_player + 1 :])
        for action in legal_actions:
            regret = cf_prob * (action_utils[action] - node_util)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += probs[current_player] * strategy[action]

        self.policy[obs] = strategy
        return np.array([node_util if i == self.player_id else 0 for i in range(self.env.num_players)])

    def save(self):
        data = {
            "policy": dict(self.policy),
            "average_policy": dict(self.average_policy),
            "regrets": dict(self.regrets),
            "iteration": self.iteration,
            "training_setup": "cfr_br_smart_colearn",
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)


def resolve_writable_output_paths():
    """
    Prefer config CFR_VS_BR_TRAIN_DIR; if mkdir or write test fails, use ~/.bluffing_leduc_results/...
    Returns (train_dir, win_rates_jsonl, cfr_pkl, br_pkl).
    """
    train_dir = _cfg.CFR_VS_BR_TRAIN_DIR
    try:
        os.makedirs(train_dir, exist_ok=True)
        probe = os.path.join(train_dir, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except (PermissionError, OSError):
        home_base = os.path.join(
            os.path.expanduser("~"), ".bluffing_leduc_results", "cfr_vs_br_smart", "training"
        )
        try:
            os.makedirs(home_base, exist_ok=True)
            probe = os.path.join(home_base, ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
        except OSError as e:
            raise SystemExit(
                f"Cannot create a writable output directory.\n"
                f"  Tried: {train_dir}\n"
                f"  Tried: {home_base}\n"
                f"  ({e})\n\n"
                "Fix repo results ownership (common after Docker/sudo):\n"
                '  sudo chown -R "$USER:$USER" results\n'
                "Or redirect all results:\n"
                '  export BLUFFING_RESULTS_DIR="$HOME/bluffing_results"\n'
            ) from e
        print(
            "\nWARNING: Could not write to the default training directory:\n"
            f"  {train_dir}\n\n"
            f"Using fallback (your home directory):\n"
            f"  {home_base}\n\n"
            "To use results/ inside the project instead:\n"
            '  sudo chown -R "$USER:$USER" results\n'
            "Or:  export BLUFFING_RESULTS_DIR=\"$HOME/bluffing_results\"\n"
        )
        train_dir = home_base

    win_jsonl = os.path.join(train_dir, os.path.basename(_cfg.CFR_VS_BR_WIN_RATES_JSONL))
    cfr_pkl = os.path.join(train_dir, os.path.basename(_cfg.CFR_VS_BR_COLEARN_CFR_PATH))
    br_pkl = os.path.join(train_dir, os.path.basename(_cfg.CFR_VS_BR_COLEARN_BR_PATH))
    return train_dir, win_jsonl, cfr_pkl, br_pkl


def train():
    train_dir, win_jsonl, cfr_pkl, br_pkl = resolve_writable_output_paths()
    set_seed(config["seed"])

    preset = PRESETS.get(SMART_PRESET)
    if preset is None:
        raise KeyError(f"Missing PRESETS['{SMART_PRESET}'] in br_cfr_config")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT_CFR_BR", "BNAIC-cfr-vs-br-smart"),
        name=os.environ.get("WANDB_RUN_NAME_CFR_BR", "CFR_BR_smart_colearn_52card"),
    )

    env = LeducholdemEnv(config={"seed": config["seed"], "allow_step_back": True})
    env.reset()

    print("=" * 70)
    print("CFR vs BR-CFR (SMART) — BOTH FROM SCRATCH (CO-LEARNING)")
    print("=" * 70)
    print(f"Environment: {env.name}")
    print(f"BR preset:   {SMART_PRESET} — {preset['description']}")
    print(f"Output dir:  {train_dir}")
    print(f"CFR save:    {cfr_pkl}")
    print(f"BR-CFR save: {br_pkl}")
    print(f"Win-rate log: {win_jsonl}")
    print(
        f"Episodes: {config['train_episodes']:,} | "
        f"Traversals/ep (each agent): {config['iterations_per_episode']} | "
        f"Eval every {config['eval_interval']:,} ep ({config['eval_games']} games)"
    )
    print("=" * 70)

    try:
        with open(win_jsonl, "w", encoding="utf-8"):
            pass
    except OSError as e:
        print(f"Warning: could not init {win_jsonl}: {e}")

    br_slot = _OpponentSlot()
    cfr_slot = _OpponentSlot()

    cfr_agent = CFRAgainstOpponentAgent(
        env,
        player_id=0,
        opponent_agent=br_slot,
        model_path=cfr_pkl,
    )
    br_preset = merge_br_cfr_preset(dict(PRESETS["smart"]))
    br_kw = {k: v for k, v in br_preset.items() if k != "description"}
    br_agent = BoundedRationalCFRAgent(
        env,
        player_id=1,
        opponent_agent=cfr_slot,
        model_path=br_pkl,
        **br_kw,
    )

    br_slot.wrapped = BRCFRWrapper(br_agent)
    cfr_slot.wrapped = CFRWrapper(cfr_agent)
    env.set_agents([CFRWrapper(cfr_agent), BRCFRWrapper(br_agent)])

    for episode in range(config["train_episodes"]):
        for _ in range(config["iterations_per_episode"]):
            env.reset()
            cfr_agent.traverse_tree(np.ones(env.num_players))
            cfr_agent.iteration += 1

            env.reset()
            br_u = br_agent.traverse_tree(np.ones(env.num_players))
            br_agent.register_hand_outcome(float(br_u[br_agent.player_id]))
            br_agent.iteration += 1

        if episode % 1000 == 0:
            cfr_reg = sum(np.sum(np.abs(r)) for r in cfr_agent.regrets.values())
            br_reg = sum(np.sum(np.abs(r)) for r in br_agent.regrets.values())
            print(
                f"[Episode {episode:,}] "
                f"CFR iters={cfr_agent.iteration:,} states={len(cfr_agent.average_policy):,} |reg|={cfr_reg:.1f} | "
                f"BR iters={br_agent.iteration:,} states={len(br_agent.average_policy):,} |reg|={br_reg:.1f}"
            )
            wandb.log(
                {
                    "episode": episode,
                    "cfr_iterations": cfr_agent.iteration,
                    "cfr_states_seen": len(cfr_agent.average_policy),
                    "cfr_total_regret": cfr_reg,
                    "br_iterations": br_agent.iteration,
                    "br_states_seen": len(br_agent.average_policy),
                    "br_total_regret": br_reg,
                }
            )

        if episode % config["eval_interval"] == 0:
            wins = [0, 0, 0]
            cfr_rewards, br_rewards = [], []
            for _ in range(config["eval_games"]):
                _, payoffs = env.run(is_training=False)
                cfr_rewards.append(payoffs[0])
                br_rewards.append(payoffs[1])
                if payoffs[0] > payoffs[1]:
                    wins[0] += 1
                elif payoffs[1] > payoffs[0]:
                    wins[1] += 1
                else:
                    wins[2] += 1

            cfr_wr = wins[0] / config["eval_games"]
            br_wr = wins[1] / config["eval_games"]
            draw_rate = wins[2] / config["eval_games"]

            wandb.log(
                {
                    "episode": episode,
                    "cfr_new_win_rate_vs_br_smart": cfr_wr,
                    "br_smart_win_rate_vs_cfr_new": br_wr,
                    "draw_rate": draw_rate,
                    "cfr_new_reward_eval": float(np.mean(cfr_rewards)),
                    "cfr_new_reward_std": float(np.std(cfr_rewards)),
                    "br_smart_reward_eval": float(np.mean(br_rewards)),
                    "br_smart_reward_std": float(np.std(br_rewards)),
                    "training_progress": episode / config["train_episodes"],
                }
            )

            row = {
                "episode": int(episode),
                "cfr_new_win_rate_vs_br_smart": float(cfr_wr),
                "br_smart_win_rate_vs_cfr_new": float(br_wr),
                "draw_rate": float(draw_rate),
                "eval_games": int(config["eval_games"]),
            }
            try:
                with open(win_jsonl, "a", encoding="utf-8") as wf:
                    wf.write(json.dumps(row) + "\n")
            except OSError as e:
                print(f"Warning: could not append win rates: {e}")

            print(
                f"[Ep {episode:,}/{config['train_episodes']:,}] "
                f"CFR WR: {cfr_wr:.3f} | BR-smart WR: {br_wr:.3f} | "
                f"Draws: {draw_rate:.3f} | Progress: {episode/config['train_episodes']:.1%}"
            )

    cfr_agent.save()
    br_agent.save()
    print(f"\nSaved CFR to    {CFR_VS_BR_COLEARN_CFR_PATH}")
    print(f"Saved BR-CFR to {CFR_VS_BR_COLEARN_BR_PATH}")
    print(f"Win-rate JSONL: {CFR_VS_BR_WIN_RATES_JSONL}")
    print("Plot: python3 plot_cfr_vs_br_smart_win_rates.py")


if __name__ == "__main__":
    train()
