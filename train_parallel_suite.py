#!/usr/bin/env python3
"""
Unified training/evaluation suite matching thesis Methods experiments.

Experiments:
1) replication_dqn_vs_cfr
2) br_cfr_smart_vs_cfr
3) br_cfr_medium_vs_cfr
4) br_cfr_dumb_vs_cfr
5) cfr_vs_cfr_baseline

All runs initialize both agents from scratch and co-evolve (no frozen opponents).

When a BR preset has ``tilt_enabled`` (dumb), the suite also writes per-trajectory
and per-eval-hand mood/tilt telemetry JSONL plus a small aggregate summary JSON
(see ``tilt_telemetry_*`` under the experiment results directory).
"""

from __future__ import annotations

import os

# Docker `docker run --user uid:gid` often has no /etc/passwd line for that UID.
# PyTorch 2.x may call getpass.getuser() → pwd.getpwuid() during import/optimizer init.
def _bootstrap_uid_without_passwd() -> None:
    try:
        import pwd
    except ImportError:
        return
    try:
        pwd.getpwuid(os.getuid())
    except KeyError:
        uid = os.getuid()
        name = f"uid{uid}"
        os.environ.setdefault("USER", name)
        os.environ.setdefault("LOGNAME", name)
        os.environ.setdefault("HOME", os.path.join("/tmp", f"home_{name}"))
        os.makedirs(os.environ["HOME"], exist_ok=True)
        tdir = os.path.join("/tmp", "torch_inductor", name)
        os.makedirs(tdir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tdir)


_bootstrap_uid_without_passwd()

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from rlcard.agents import DQNAgent
from rlcard.utils import reorganize, set_seed

from br_cfr_agent import BRCFRWrapper, MoodTiltSession, obs_to_key, regret_matching
from br_cfr_variant_specs import BR_CFR_VARIANT_PARAM_TABLE
from config import RESULTS_DIR, TRAIN_EPISODES as DEFAULT_TRAIN_EPISODES
from custom_leduc_rlcard.leducholdem import LeducholdemEnv
from cfr_agent import CFRAgainstOpponentAgent, CFRWrapper, _OpponentSlot

RANK_ORDER = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}
SUIT_ORDER = {"C": 0, "D": 1, "H": 2, "S": 3}
RAISE_ACTION = 1
FOLD_ACTION = 2

DEFAULT_EVAL_GAMES = 100_000
DEFAULT_EVAL_INTERVAL = 10_000
DEFAULT_CHECKPOINT_EVAL_GAMES = 2_000


def _zero_array_4() -> np.ndarray:
    return np.zeros(4, dtype=float)


class BRCFRAgainstOpponentAgent(CFRAgainstOpponentAgent):
    """
    BR-CFR implemented as a configurable subclass of the existing CFR agent.
    """

    def __init__(
        self,
        env: LeducholdemEnv,
        player_id: int,
        opponent_agent: Any,
        model_path: str,
        *,
        iterations_per_episode: int,
        memory_decay: float,
        soft_regret_tau: float,
        bucket_mode: str,
        qre_normalize_regrets: bool = False,
        qre_norm_epsilon: float = 1e-8,
        mood_enabled: bool = False,
        mood_rho: float = 0.92,
        mood_behind_raise_coef: float = 0.25,
        mood_ahead_raise_coef: float = 0.20,
        mood_tanh_scale: float = 10.0,
        tilt_enabled: bool = False,
        tilt_loss_streak_k: int = 3,
        tilt_shock_payoff: float = -6.0,
        tilt_duration_hands: int = 5,
        tilt_tau_multiplier: float = 2.0,
        tilt_min_soft_tau: float = 0.45,
        tilt_raise_extra_logit: float = 0.60,
        tilt_regret_flatten_mult: float = 0.55,
        tilt_uniform_mix: float = 0.08,
        tilt_trigger_probability: float = 1.0,
        tilt_mood_threshold: float | None = None,
        tilt_cooldown_hands: int = 0,
        tilt_refresh_while_active: bool = True,
        tilt_reset_loss_streak_on_trigger: bool = False,
        session_telemetry_enabled: bool = False,
    ) -> None:
        super().__init__(env, player_id, opponent_agent, model_path)
        self.iterations_per_episode = int(iterations_per_episode)
        self.memory_decay = float(memory_decay)
        self.soft_regret_tau = float(soft_regret_tau)
        self.bucket_mode = str(bucket_mode)
        self.qre_normalize_regrets = bool(qre_normalize_regrets)
        self.qre_norm_epsilon = float(qre_norm_epsilon)
        self.mood_tilt = MoodTiltSession(
            mood_enabled=mood_enabled,
            mood_rho=mood_rho,
            mood_behind_raise_coef=mood_behind_raise_coef,
            mood_ahead_raise_coef=mood_ahead_raise_coef,
            mood_tanh_scale=mood_tanh_scale,
            tilt_enabled=tilt_enabled,
            tilt_loss_streak_k=tilt_loss_streak_k,
            tilt_shock_payoff=tilt_shock_payoff,
            tilt_duration_hands=tilt_duration_hands,
            tilt_tau_multiplier=tilt_tau_multiplier,
            tilt_min_soft_tau=tilt_min_soft_tau,
            tilt_raise_extra_logit=tilt_raise_extra_logit,
            tilt_regret_flatten_mult=tilt_regret_flatten_mult,
            tilt_uniform_mix=tilt_uniform_mix,
            tilt_trigger_probability=tilt_trigger_probability,
            tilt_mood_threshold=tilt_mood_threshold,
            tilt_cooldown_hands=tilt_cooldown_hands,
            tilt_refresh_while_active=tilt_refresh_while_active,
            tilt_reset_loss_streak_on_trigger=tilt_reset_loss_streak_on_trigger,
            session_telemetry_enabled=session_telemetry_enabled,
        )
        self.br_cfr_params = {
            "iterations_per_episode": self.iterations_per_episode,
            "memory_decay": self.memory_decay,
            "soft_regret_tau": self.soft_regret_tau,
            "bucket_mode": self.bucket_mode,
            "qre_normalize_regrets": self.qre_normalize_regrets,
            "qre_norm_epsilon": self.qre_norm_epsilon,
            **self.mood_tilt.to_param_dict(),
        }

    def _obs_key(self, state: dict) -> Any:
        return obs_to_key(state, self.env, self.bucket_mode)

    def register_hand_outcome(self, payoff: float):
        return self.mood_tilt.hand_end(
            float(payoff), base_soft_regret_tau=self.soft_regret_tau
        )

    def _regret_matching(self, obs_key: Any) -> np.ndarray:
        regret = self.regrets[obs_key].copy()
        tau = self.mood_tilt.effective_tau(self.soft_regret_tau)
        regret_scale = self.mood_tilt.regret_flatten_scale()
        return regret_matching(
            regret,
            self.env.num_actions,
            tau=tau,
            regret_scale=regret_scale,
            qre_normalize_regrets=self.qre_normalize_regrets,
            qre_norm_epsilon=self.qre_norm_epsilon,
        )

    def _apply_decay(self, obs_key: Any) -> None:
        if self.memory_decay < 1.0:
            self.regrets[obs_key] *= self.memory_decay
            self.average_policy[obs_key] *= self.memory_decay

    def _apply_mood_tilt(self, strategy: np.ndarray, legal_actions: list[int]) -> np.ndarray:
        if not self.mood_tilt.any_enabled():
            return strategy
        return self.mood_tilt.modulate_play_probs(
            strategy,
            legal_actions,
            self.env.num_actions,
        )

    def traverse_tree(self, probs: np.ndarray) -> np.ndarray:
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        state = self.env.get_state(current_player)
        legal_actions = list(state["legal_actions"].keys())

        if current_player != self.player_id:
            action, _ = self.opponent_agent.eval_step(state)
            self.env.step(action)
            utility = self.traverse_tree(probs)
            self.env.step_back()
            return utility

        obs_key = self._obs_key(state)
        self._apply_decay(obs_key)
        strategy = self._regret_matching(obs_key)
        strategy = self._apply_mood_tilt(strategy, legal_actions)

        action_utils = np.zeros(self.env.num_actions)
        node_util = 0.0

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
            regret_val = cf_prob * (action_utils[action] - node_util)
            self.regrets[obs_key][action] += regret_val
            self.average_policy[obs_key][action] += probs[current_player] * strategy[action]

        self.policy[obs_key] = strategy
        return np.array(
            [node_util if i == self.player_id else 0.0 for i in range(self.env.num_players)],
            dtype=float,
        )

    def save(self) -> None:
        data = {
            "policy": dict(self.policy),
            "average_policy": dict(self.average_policy),
            "regrets": dict(self.regrets),
            "iteration": self.iteration,
            "br_cfr_params": dict(self.br_cfr_params),
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)


@dataclass
class ExperimentSpec:
    name: str
    out_dir_name: str
    setup: str  # dqn_vs_cfr | br_vs_cfr | cfr_vs_cfr
    br_variant: str | None = None
    # Merged on top of BR_CFR_VARIANT_PARAM_TABLE[br_variant] for br_vs_cfr runs.
    br_param_overrides: dict[str, Any] | None = None
    # Optional label in training CSV / eval JSON (default: br_cfr_<variant>).
    br_agent_display_name: str | None = None


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "replication_dqn_vs_cfr": ExperimentSpec(
        name="replication_dqn_vs_cfr",
        out_dir_name="replication_dqn_vs_cfr",
        setup="dqn_vs_cfr",
    ),
    "br_cfr_smart_vs_cfr": ExperimentSpec(
        name="br_cfr_smart_vs_cfr",
        out_dir_name="br_cfr_smart_vs_cfr",
        setup="br_vs_cfr",
        br_variant="smart",
    ),
    "br_cfr_medium_vs_cfr": ExperimentSpec(
        name="br_cfr_medium_vs_cfr",
        out_dir_name="br_cfr_medium_vs_cfr",
        setup="br_vs_cfr",
        br_variant="medium",
    ),
    "br_cfr_dumb_vs_cfr": ExperimentSpec(
        name="br_cfr_dumb_vs_cfr",
        out_dir_name="br_cfr_dumb_vs_cfr",
        setup="br_vs_cfr",
        br_variant="dumb",
    ),
    "cfr_vs_cfr_baseline": ExperimentSpec(
        name="cfr_vs_cfr_baseline",
        out_dir_name="cfr_vs_cfr_baseline",
        setup="cfr_vs_cfr",
    ),
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _state_dim(env: LeducholdemEnv) -> int:
    shape0 = env.state_shape[0]
    if isinstance(shape0, (list, tuple, np.ndarray)):
        return int(shape0[0])
    return int(shape0)


def _card_score(card: str | None) -> int:
    if not card or len(card) < 2:
        return 0
    suit = card[0]
    rank = card[1]
    return RANK_ORDER.get(rank, 0) * 4 + SUIT_ORDER.get(suit, 0)


def _hand_score(hand: str | None, public_card: str | None) -> int:
    if not hand:
        return 0
    score = _card_score(hand)
    if public_card and len(public_card) >= 2 and len(hand) >= 2 and hand[1] == public_card[1]:
        return score + 1000
    return score


def _is_threshold_bluff_attempt(hand: str | None, public_card: str | None, action: int) -> bool:
    if action != RAISE_ACTION:
        return False
    return _hand_score(hand, public_card) <= 32


def _evaluate_with_bluff_stats(
    env: LeducholdemEnv,
    agents: list[Any],
    agent_names: list[str],
    num_games: int,
    checkpoint_interval_games: int = 10_000,
    eval_tilt_telemetry_jsonl: str | None = None,
) -> dict[str, Any]:
    env.set_agents(agents)
    wins = [0, 0, 0]
    total_payoffs = [0.0, 0.0]
    action_counts = [0, 0]
    bluff_attempts_by_agent = [0, 0]
    bluff_successes_by_agent = [0, 0]
    checkpoint_rows: list[dict[str, Any]] = []

    tilt_out = None
    if eval_tilt_telemetry_jsonl:
        tilt_out = open(eval_tilt_telemetry_jsonl, "w", encoding="utf-8")

    try:
        for game_idx in range(1, num_games + 1):
            env.reset()
            hand_log: list[dict[str, Any]] = []

            while not env.is_over():
                player_id = env.get_player_id()
                state = env.get_state(player_id)
                action, _ = agents[player_id].eval_step(state)

                raw = state.get("raw_obs", {})
                hand_log.append(
                    {
                        "player_id": int(player_id),
                        "action": int(action),
                        "hand": raw.get("hand"),
                        "public_card": raw.get("public_card"),
                    }
                )
                action_counts[player_id] += 1
                env.step(action)

            payoffs = env.get_payoffs()
            total_payoffs[0] += float(payoffs[0])
            total_payoffs[1] += float(payoffs[1])

            for pid, ag in enumerate(agents):
                if hasattr(ag, "notify_hand_end"):
                    rec = ag.notify_hand_end(float(payoffs[pid]))
                    if tilt_out is not None and pid == 0 and rec:
                        row = {"phase": "eval", "game_index": game_idx, **rec}
                        tilt_out.write(json.dumps(_jsonable(row)) + "\n")

            if payoffs[0] > payoffs[1]:
                wins[0] += 1
            elif payoffs[1] > payoffs[0]:
                wins[1] += 1
            else:
                wins[2] += 1

            for i, row in enumerate(hand_log):
                pid = row["player_id"]
                if _is_threshold_bluff_attempt(row["hand"], row["public_card"], row["action"]):
                    bluff_attempts_by_agent[pid] += 1
                    if i + 1 < len(hand_log):
                        nxt = hand_log[i + 1]
                        if nxt["player_id"] != pid and nxt["action"] == FOLD_ACTION:
                            bluff_successes_by_agent[pid] += 1

            if game_idx % checkpoint_interval_games == 0:
                ck = {
                    "games": game_idx,
                    "agent0_win_rate": wins[0] / game_idx,
                    "agent1_win_rate": wins[1] / game_idx,
                    "draw_rate": wins[2] / game_idx,
                    "agent0_avg_payoff": total_payoffs[0] / game_idx,
                    "agent1_avg_payoff": total_payoffs[1] / game_idx,
                    "agent0_bluff_attempts": bluff_attempts_by_agent[0],
                    "agent1_bluff_attempts": bluff_attempts_by_agent[1],
                    "agent0_bluff_successes": bluff_successes_by_agent[0],
                    "agent1_bluff_successes": bluff_successes_by_agent[1],
                }
                checkpoint_rows.append(ck)
    finally:
        if tilt_out is not None:
            tilt_out.close()

    summary = {
        "num_games": int(num_games),
        "wins": {
            agent_names[0]: int(wins[0]),
            agent_names[1]: int(wins[1]),
            "draws": int(wins[2]),
        },
        "win_rates": {
            agent_names[0]: wins[0] / num_games,
            agent_names[1]: wins[1] / num_games,
            "draws": wins[2] / num_games,
        },
        "average_payoff": {
            agent_names[0]: total_payoffs[0] / num_games,
            agent_names[1]: total_payoffs[1] / num_games,
        },
        "bluff_stats_threshold_detector": {
            "definition": {
                "hand_score": "R_pc*4 + S_pc (+1000 if pair)",
                "attempt_rule": "raise with HandScore <= 32",
                "success_rule": "opponent folds immediately after the raise",
            },
            "total_bluff_attempts": int(sum(bluff_attempts_by_agent)),
            "successful_bluffs": int(sum(bluff_successes_by_agent)),
            "bluff_counts_by_agent": {
                agent_names[0]: {
                    "attempts": int(bluff_attempts_by_agent[0]),
                    "successes": int(bluff_successes_by_agent[0]),
                    "attempt_rate": (
                        bluff_attempts_by_agent[0] / action_counts[0] if action_counts[0] else 0.0
                    ),
                    "success_rate": (
                        bluff_successes_by_agent[0] / bluff_attempts_by_agent[0]
                        if bluff_attempts_by_agent[0]
                        else 0.0
                    ),
                },
                agent_names[1]: {
                    "attempts": int(bluff_attempts_by_agent[1]),
                    "successes": int(bluff_successes_by_agent[1]),
                    "attempt_rate": (
                        bluff_attempts_by_agent[1] / action_counts[1] if action_counts[1] else 0.0
                    ),
                    "success_rate": (
                        bluff_successes_by_agent[1] / bluff_attempts_by_agent[1]
                        if bluff_attempts_by_agent[1]
                        else 0.0
                    ),
                },
            },
            "overall_bluff_attempt_rate": (
                sum(bluff_attempts_by_agent) / sum(action_counts) if sum(action_counts) else 0.0
            ),
            "overall_bluff_success_rate": (
                sum(bluff_successes_by_agent) / sum(bluff_attempts_by_agent)
                if sum(bluff_attempts_by_agent)
                else 0.0
            ),
            "by_evaluation_checkpoint": checkpoint_rows,
        },
    }
    return summary


def _ensure_dirs(experiment_dir: str) -> dict[str, str]:
    models_dir = os.path.join(experiment_dir, "models")
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    return {"root": experiment_dir, "models": models_dir, "checkpoints": checkpoints_dir}


def _save_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode"])
        return
    # Rows may gain keys on later episodes (e.g. checkpoint eval columns); union all keys.
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    rest = sorted(k for k in all_keys if k != "episode")
    fieldnames = (["episode"] + rest) if "episode" in all_keys else sorted(all_keys)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)


def _build_dqn_agent(env: LeducholdemEnv) -> DQNAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep replication hyperparameters unchanged.
    return DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[256, 256],
        learning_rate=0.00005,
        batch_size=64,
        epsilon_end=0.05,
        epsilon_decay_steps=10_000,
        replay_memory_init_size=500,
        replay_memory_size=20_000,
        update_target_estimator_every=1_000,
        device=device,
    )


def _checkpoint_eval(
    env_eval: LeducholdemEnv,
    agent_a: Any,
    agent_b: Any,
    eval_games: int,
) -> dict[str, float]:
    env_eval.set_agents([agent_a, agent_b])
    wins = [0, 0, 0]
    sum_pay = [0.0, 0.0]
    for _ in range(eval_games):
        _, pay = env_eval.run(is_training=False)
        sum_pay[0] += float(pay[0])
        sum_pay[1] += float(pay[1])
        if pay[0] > pay[1]:
            wins[0] += 1
        elif pay[1] > pay[0]:
            wins[1] += 1
        else:
            wins[2] += 1
    return {
        "agent0_win_rate": wins[0] / eval_games,
        "agent1_win_rate": wins[1] / eval_games,
        "draw_rate": wins[2] / eval_games,
        "agent0_avg_payoff": sum_pay[0] / eval_games,
        "agent1_avg_payoff": sum_pay[1] / eval_games,
    }


def _aggregate_tilt_telemetry_jsonl(path: str) -> dict[str, Any]:
    """Scan a tilt telemetry JSONL for coarse counters (works for large files)."""
    agg: dict[str, Any] = {
        "lines": 0,
        "tilt_active_before_true": 0,
        "triggered_refresh_true": 0,
        "trigger_loss_streak_true": 0,
        "trigger_shock_true": 0,
        "sum_abs_payoff": 0.0,
        "sum_effective_tau_used": 0.0,
    }
    if not os.path.isfile(path):
        return agg
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            agg["lines"] += 1
            if row.get("tilt_active_before"):
                agg["tilt_active_before_true"] += 1
            if row.get("triggered_refresh"):
                agg["triggered_refresh_true"] += 1
            if row.get("trigger_loss_streak"):
                agg["trigger_loss_streak_true"] += 1
            if row.get("trigger_shock"):
                agg["trigger_shock_true"] += 1
            agg["sum_abs_payoff"] += abs(float(row.get("payoff", 0.0)))
            agg["sum_effective_tau_used"] += float(row.get("effective_tau_used", 0.0))
    n = max(1, agg["lines"])
    agg["fraction_tilt_active_before"] = agg["tilt_active_before_true"] / n
    agg["mean_abs_payoff"] = agg["sum_abs_payoff"] / n
    agg["mean_effective_tau_used"] = agg["sum_effective_tau_used"] / n
    return agg


def run_experiment(
    spec: ExperimentSpec,
    *,
    seed: int,
    train_episodes: int,
    eval_games: int,
    eval_interval: int,
    checkpoint_eval_games: int,
) -> dict[str, Any]:
    set_seed(seed)

    experiment_dir = os.path.join(RESULTS_DIR, spec.out_dir_name)
    dirs = _ensure_dirs(experiment_dir)

    env_train = LeducholdemEnv(config={"seed": seed, "allow_step_back": True})
    env_train.reset()
    env_eval = LeducholdemEnv(config={"seed": seed, "allow_step_back": False})
    env_eval.reset()

    print("=" * 80)
    print(f"EXPERIMENT: {spec.name}")
    if spec.br_variant:
        _preset_line = spec.br_variant
        if spec.br_param_overrides:
            _preset_line = f"{spec.br_variant} + {spec.br_param_overrides}"
        print(f"Preset: {_preset_line}")
    else:
        print("Preset: none")
    print(f"Deck size: {len(env_train.game.dealer.deck)}")
    print(f"State dimension: {_state_dim(env_train)}")
    print(f"Episodes: {train_episodes:,} | Eval games: {eval_games:,} | Seed: {seed}")
    print("=" * 80)

    training_rows: list[dict[str, Any]] = []
    tilt_train_jsonl_path: str | None = None
    tilt_eval_jsonl_path: str | None = None
    tilt_summary_path: str | None = None
    tilt_train_agg: dict[str, Any] = {}
    cfg_payload = {
        "experiment": spec.name,
        "setup": spec.setup,
        "seed": seed,
        "train_episodes": train_episodes,
        "eval_games": eval_games,
        "eval_interval": eval_interval,
        "checkpoint_eval_games": checkpoint_eval_games,
        "deck_size": len(env_train.game.dealer.deck),
        "state_dim": _state_dim(env_train),
    }
    if spec.br_variant:
        _br_eff = dict(BR_CFR_VARIANT_PARAM_TABLE[spec.br_variant])
        if spec.br_param_overrides:
            cfg_payload["br_preset_table_row"] = dict(BR_CFR_VARIANT_PARAM_TABLE[spec.br_variant])
            cfg_payload["br_preset_overrides"] = dict(spec.br_param_overrides)
            _br_eff.update(spec.br_param_overrides)
        cfg_payload["br_preset"] = _br_eff

    cfg_path = os.path.join(dirs["root"], f"config_{spec.name}_seed{seed}.json")
    _save_json(cfg_path, cfg_payload)

    model_paths: dict[str, str] = {}

    if spec.setup == "dqn_vs_cfr":
        dqn = _build_dqn_agent(env_train)
        cfr_model_path = os.path.join(dirs["models"], f"{spec.name}_seed{seed}_cfr.pkl")
        cfr = CFRAgainstOpponentAgent(env_train, player_id=1, opponent_agent=dqn, model_path=cfr_model_path)
        env_train.set_agents([dqn, CFRWrapper(cfr)])

        for episode in range(1, train_episodes + 1):
            for _ in range(10):
                env_train.reset()
                cfr.traverse_tree(np.ones(env_train.num_players))
                cfr.iteration += 1

            trajectories, payoffs = env_train.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)
            for ts in trajectories[0]:
                if ts:
                    dqn.feed(ts)

            row = {
                "episode": episode,
                "agent0_name": "dqn",
                "agent1_name": "standard_cfr",
                "cfr_states_seen": len(cfr.average_policy),
                "cfr_iterations": cfr.iteration,
                "dqn_train_payoff": float(payoffs[0]),
                "cfr_train_payoff": float(payoffs[1]),
            }

            if episode % 1000 == 0:
                print(
                    f"[{spec.name}] Episode {episode:,}: CFR states={len(cfr.average_policy):,}, "
                    f"CFR iters={cfr.iteration:,}"
                )

            if episode % eval_interval == 0:
                ck_metrics = _checkpoint_eval(env_eval, dqn, CFRWrapper(cfr), checkpoint_eval_games)
                row.update(ck_metrics)

                dqn_ck = os.path.join(
                    dirs["checkpoints"],
                    f"{spec.name}_seed{seed}_ep{episode}_dqn.pt",
                )
                cfr_ck = os.path.join(
                    dirs["checkpoints"],
                    f"{spec.name}_seed{seed}_ep{episode}_cfr.pkl",
                )
                torch.save(dqn.q_estimator.qnet.state_dict(), dqn_ck)
                cfr_ck_data = {
                    "policy": dict(cfr.policy),
                    "average_policy": dict(cfr.average_policy),
                    "regrets": dict(cfr.regrets),
                    "iteration": cfr.iteration,
                }
                with open(cfr_ck, "wb") as f:
                    pickle.dump(cfr_ck_data, f)

            training_rows.append(row)

        dqn_path = os.path.join(dirs["models"], f"{spec.name}_seed{seed}_dqn.pt")
        torch.save(dqn.q_estimator.qnet.state_dict(), dqn_path)
        cfr.save()
        model_paths = {"agent0_dqn": dqn_path, "agent1_cfr": cfr_model_path}
        eval_agents = [dqn, CFRWrapper(cfr)]
        eval_names = ["dqn", "standard_cfr"]

    elif spec.setup == "br_vs_cfr":
        assert spec.br_variant is not None
        br_params = dict(BR_CFR_VARIANT_PARAM_TABLE[spec.br_variant])
        if spec.br_param_overrides:
            br_params.update(spec.br_param_overrides)
        br_display = spec.br_agent_display_name or f"br_cfr_{spec.br_variant}"

        cfr_model_path = os.path.join(dirs["models"], f"{spec.name}_seed{seed}_standard_cfr.pkl")
        br_model_path = os.path.join(dirs["models"], f"{spec.name}_seed{seed}_br_cfr_{spec.br_variant}.pkl")

        cfr_slot = _OpponentSlot()
        br_slot = _OpponentSlot()
        cfr = CFRAgainstOpponentAgent(env_train, player_id=1, opponent_agent=br_slot, model_path=cfr_model_path)
        br_kwargs = dict(br_params)
        if br_kwargs.get("tilt_enabled"):
            br_kwargs["session_telemetry_enabled"] = True
        br = BRCFRAgainstOpponentAgent(
            env_train,
            player_id=0,
            opponent_agent=cfr_slot,
            model_path=br_model_path,
            **br_kwargs,
        )
        cfr_slot.wrapped = CFRWrapper(cfr)
        br_slot.wrapped = BRCFRWrapper(br)
        env_train.set_agents([BRCFRWrapper(br), CFRWrapper(cfr)])

        tilt_train_file = None
        if br_params.get("tilt_enabled"):
            tilt_train_jsonl_path = os.path.join(
                dirs["root"], f"tilt_telemetry_training_{spec.name}_seed{seed}.jsonl"
            )
            tilt_eval_jsonl_path = os.path.join(
                dirs["root"], f"tilt_telemetry_eval_{spec.name}_seed{seed}.jsonl"
            )
            tilt_summary_path = os.path.join(
                dirs["root"], f"tilt_telemetry_summary_{spec.name}_seed{seed}.json"
            )
            tilt_train_file = open(tilt_train_jsonl_path, "w", encoding="utf-8")
            tilt_train_agg = {
                "spec": spec.name,
                "seed": seed,
                "train_episodes": train_episodes,
                "iterations_per_episode": int(br_params["iterations_per_episode"]),
                "mood_tilt_params": {
                    k: br_params[k]
                    for k in sorted(br_params)
                    if k.startswith("tilt_") or k.startswith("mood_")
                },
                "training_traversals_logged": 0,
                "tilt_active_before_true": 0,
                "triggered_refresh_true": 0,
                "trigger_loss_streak_true": 0,
                "trigger_shock_true": 0,
                "sum_abs_payoff": 0.0,
                "sum_effective_tau_used": 0.0,
            }

        for episode in range(1, train_episodes + 1):
            for _ in range(int(br_params["iterations_per_episode"])):
                env_train.reset()
                u_br = br.traverse_tree(np.ones(env_train.num_players))
                rec = br.register_hand_outcome(float(u_br[br.player_id]))
                br.iteration += 1
                if tilt_train_file is not None and rec:
                    tilt_train_agg["training_traversals_logged"] += 1
                    if rec["tilt_active_before"]:
                        tilt_train_agg["tilt_active_before_true"] += 1
                    if rec["triggered_refresh"]:
                        tilt_train_agg["triggered_refresh_true"] += 1
                    if rec["trigger_loss_streak"]:
                        tilt_train_agg["trigger_loss_streak_true"] += 1
                    if rec["trigger_shock"]:
                        tilt_train_agg["trigger_shock_true"] += 1
                    tilt_train_agg["sum_abs_payoff"] += abs(float(rec["payoff"]))
                    tilt_train_agg["sum_effective_tau_used"] += float(rec["effective_tau_used"])
                    row = {
                        "phase": "training",
                        "train_episode": episode,
                        "br_traversal_index": int(br.iteration),
                        **rec,
                    }
                    tilt_train_file.write(json.dumps(_jsonable(row)) + "\n")

                env_train.reset()
                cfr.traverse_tree(np.ones(env_train.num_players))
                cfr.iteration += 1

            row = {
                "episode": episode,
                "agent0_name": br_display,
                "agent1_name": "standard_cfr",
                "agent0_states_seen": len(br.average_policy),
                "agent1_states_seen": len(cfr.average_policy),
                "agent0_iterations": br.iteration,
                "agent1_iterations": cfr.iteration,
            }

            if episode % 1000 == 0:
                print(
                    f"[{spec.name}] Episode {episode:,}: "
                    f"BR states={len(br.average_policy):,}, CFR states={len(cfr.average_policy):,}"
                )

            if episode % eval_interval == 0:
                ck_metrics = _checkpoint_eval(
                    env_eval,
                    BRCFRWrapper(br),
                    CFRWrapper(cfr),
                    checkpoint_eval_games,
                )
                row.update(ck_metrics)

                br_ck = os.path.join(
                    dirs["checkpoints"],
                    f"{spec.name}_seed{seed}_ep{episode}_br.pkl",
                )
                cfr_ck = os.path.join(
                    dirs["checkpoints"],
                    f"{spec.name}_seed{seed}_ep{episode}_cfr.pkl",
                )
                br_prev = br.model_path
                cfr_prev = cfr.model_path
                br.model_path = br_ck
                cfr.model_path = cfr_ck
                br.save()
                cfr.save()
                br.model_path = br_prev
                cfr.model_path = cfr_prev

            training_rows.append(row)

        if tilt_train_file is not None:
            tilt_train_file.close()

        br.save()
        cfr.save()
        model_paths = {"agent0_br_cfr": br_model_path, "agent1_cfr": cfr_model_path}
        eval_agents = [BRCFRWrapper(br), CFRWrapper(cfr)]
        eval_names = [br_display, "standard_cfr"]

    elif spec.setup == "cfr_vs_cfr":
        p0_path = os.path.join(dirs["models"], f"{spec.name}_seed{seed}_cfr_player0.pkl")
        p1_path = os.path.join(dirs["models"], f"{spec.name}_seed{seed}_cfr_player1.pkl")
        slot0 = _OpponentSlot()
        slot1 = _OpponentSlot()
        cfr0 = CFRAgainstOpponentAgent(env_train, player_id=0, opponent_agent=slot1, model_path=p0_path)
        cfr1 = CFRAgainstOpponentAgent(env_train, player_id=1, opponent_agent=slot0, model_path=p1_path)
        slot0.wrapped = CFRWrapper(cfr0)
        slot1.wrapped = CFRWrapper(cfr1)
        env_train.set_agents([CFRWrapper(cfr0), CFRWrapper(cfr1)])

        for episode in range(1, train_episodes + 1):
            for _ in range(10):
                env_train.reset()
                cfr0.traverse_tree(np.ones(env_train.num_players))
                cfr0.iteration += 1
                env_train.reset()
                cfr1.traverse_tree(np.ones(env_train.num_players))
                cfr1.iteration += 1

            row = {
                "episode": episode,
                "agent0_name": "standard_cfr_player0",
                "agent1_name": "standard_cfr_player1",
                "agent0_states_seen": len(cfr0.average_policy),
                "agent1_states_seen": len(cfr1.average_policy),
                "agent0_iterations": cfr0.iteration,
                "agent1_iterations": cfr1.iteration,
            }

            if episode % 1000 == 0:
                print(
                    f"[{spec.name}] Episode {episode:,}: "
                    f"P0 states={len(cfr0.average_policy):,}, P1 states={len(cfr1.average_policy):,}"
                )

            if episode % eval_interval == 0:
                ck_metrics = _checkpoint_eval(
                    env_eval,
                    CFRWrapper(cfr0),
                    CFRWrapper(cfr1),
                    checkpoint_eval_games,
                )
                row.update(ck_metrics)

                p0_ck = os.path.join(
                    dirs["checkpoints"],
                    f"{spec.name}_seed{seed}_ep{episode}_p0.pkl",
                )
                p1_ck = os.path.join(
                    dirs["checkpoints"],
                    f"{spec.name}_seed{seed}_ep{episode}_p1.pkl",
                )
                prev0 = cfr0.model_path
                prev1 = cfr1.model_path
                cfr0.model_path = p0_ck
                cfr1.model_path = p1_ck
                cfr0.save()
                cfr1.save()
                cfr0.model_path = prev0
                cfr1.model_path = prev1

            training_rows.append(row)

        cfr0.save()
        cfr1.save()
        model_paths = {"agent0_cfr": p0_path, "agent1_cfr": p1_path}
        eval_agents = [CFRWrapper(cfr0), CFRWrapper(cfr1)]
        eval_names = ["standard_cfr_player0", "standard_cfr_player1"]
    else:
        raise ValueError(f"Unsupported experiment setup: {spec.setup}")

    train_csv = os.path.join(dirs["root"], f"training_logs_{spec.name}_seed{seed}.csv")
    _save_csv(train_csv, training_rows)

    if spec.setup == "br_vs_cfr" and tilt_eval_jsonl_path:
        br.mood_tilt.reset()

    final_eval = _evaluate_with_bluff_stats(
        env_eval,
        eval_agents,
        eval_names,
        num_games=eval_games,
        checkpoint_interval_games=10_000,
        eval_tilt_telemetry_jsonl=tilt_eval_jsonl_path,
    )
    final_eval["experiment"] = spec.name
    final_eval["seed"] = seed
    final_eval["model_paths"] = model_paths

    summary_json = os.path.join(dirs["root"], f"final_evaluation_summary_{spec.name}_seed{seed}.json")
    _save_json(summary_json, final_eval)

    bluff_json = os.path.join(dirs["root"], f"bluff_statistics_{spec.name}_seed{seed}.json")
    _save_json(bluff_json, final_eval["bluff_stats_threshold_detector"])

    bluff_csv_rows = []
    by_agent = final_eval["bluff_stats_threshold_detector"]["bluff_counts_by_agent"]
    for agent_name, s in by_agent.items():
        bluff_csv_rows.append(
            {
                "agent": agent_name,
                "attempts": s["attempts"],
                "successes": s["successes"],
                "attempt_rate": s["attempt_rate"],
                "success_rate": s["success_rate"],
            }
        )
    bluff_csv = os.path.join(dirs["root"], f"bluff_statistics_{spec.name}_seed{seed}.csv")
    _save_csv(bluff_csv, bluff_csv_rows)

    if tilt_summary_path:
        nlog = int(tilt_train_agg.get("training_traversals_logged", 0))
        if nlog > 0:
            tilt_train_agg["fraction_tilt_active_before"] = (
                tilt_train_agg["tilt_active_before_true"] / nlog
            )
            tilt_train_agg["mean_abs_payoff"] = tilt_train_agg["sum_abs_payoff"] / nlog
            tilt_train_agg["mean_effective_tau_used"] = (
                tilt_train_agg["sum_effective_tau_used"] / nlog
            )
        eval_agg = _aggregate_tilt_telemetry_jsonl(tilt_eval_jsonl_path or "")
        _save_json(
            tilt_summary_path,
            {
                "training_aggregate": tilt_train_agg,
                "eval_aggregate": eval_agg,
                "training_jsonl": tilt_train_jsonl_path,
                "eval_jsonl": tilt_eval_jsonl_path,
                "note": (
                    "Each training line is one BR-CFR root traversal after register_hand_outcome; "
                    "tilt_active_before / *_used describe modulation for that traversal. "
                    "Each eval line is one completed game for seat 0 (BR-CFR). "
                    "Mood/tilt counters were reset to neutral immediately before final evaluation."
                ),
            },
        )

    out: dict[str, Any] = {
        "experiment": spec.name,
        "result_dir": dirs["root"],
        "config_json": cfg_path,
        "training_csv": train_csv,
        "summary_json": summary_json,
        "bluff_json": bluff_json,
        "bluff_csv": bluff_csv,
        "model_paths": model_paths,
    }
    if tilt_train_jsonl_path:
        out["tilt_telemetry_training_jsonl"] = tilt_train_jsonl_path
    if tilt_eval_jsonl_path:
        out["tilt_telemetry_eval_jsonl"] = tilt_eval_jsonl_path
    if tilt_summary_path:
        out["tilt_telemetry_summary_json"] = tilt_summary_path
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis parallel training suite.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_SPECS.keys()),
        help="Run a single experiment.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments sequentially.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=DEFAULT_TRAIN_EPISODES,
        help="Training episodes per experiment.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=DEFAULT_EVAL_GAMES,
        help="Final evaluation games per experiment (learning disabled).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=DEFAULT_EVAL_INTERVAL,
        help="Episodes between intermediate checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-eval-games",
        type=int,
        default=DEFAULT_CHECKPOINT_EVAL_GAMES,
        help="Evaluation games at checkpoint intervals during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and not args.experiment:
        raise SystemExit("Use --all or --experiment <name>.")
    if args.all and args.experiment:
        raise SystemExit("Use either --all or --experiment, not both.")

    todo = (
        [EXPERIMENT_SPECS[args.experiment]]
        if args.experiment
        else [EXPERIMENT_SPECS[k] for k in EXPERIMENT_SPECS]
    )

    print("Running training suite with fixed-limit custom 52-card Leduc Hold'em.")
    print(
        f"Seed={args.seed} | train_episodes={args.train_episodes:,} | "
        f"eval_games={args.eval_games:,}"
    )

    summaries = []
    for spec in todo:
        summaries.append(
            run_experiment(
                spec,
                seed=args.seed,
                train_episodes=args.train_episodes,
                eval_games=args.eval_games,
                eval_interval=args.eval_interval,
                checkpoint_eval_games=args.checkpoint_eval_games,
            )
        )

    print("\nSuite complete.")
    for s in summaries:
        print(f"- {s['experiment']}: {s['result_dir']}")


if __name__ == "__main__":
    main()
