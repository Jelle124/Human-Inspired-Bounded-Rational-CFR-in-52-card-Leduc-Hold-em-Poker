"""
Standard tabular CFR agent for Leduc Hold'em (full observation keys).

Used by train_parallel_suite.py for DQN-vs-CFR replication, BR-CFR co-training,
and CFR self-play baselines.
"""

from __future__ import annotations

import collections
import pickle

import numpy as np
import torch


def zero_array_4() -> np.ndarray:
    return np.zeros(4, dtype=float)


class CFRWrapper:
    """Gameplay wrapper: sample actions from CFR average policy."""

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
    """Late-bind opponent so two CFR agents can reference each other."""

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
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)
