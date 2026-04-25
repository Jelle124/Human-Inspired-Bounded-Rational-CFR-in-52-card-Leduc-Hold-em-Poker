"""
Bounded-Rational CFR (BR-CFR) Agent.

This module implements "dumber" CFR variants by deliberately restricting
information, computation, or memory. It is SEPARATE from the replication
code (simultaneous_training.py, CFRAgainstDQNAgent).

Implementations:
  A) Fewer iterations - limited computation
  B) Forgetting decay - limited memory / recency bias
  C) Soft regret matching - quantal response / noisy choice
  D) State bucketing - coarse perception
  E) Mood + tilt — session reference dependence and episodic loss-of-control
     (see deep-research-report (1).md)

See br_cfr_config.py for parameter motivations and README_BR_CFR.md for
full documentation.
"""

import pickle
import collections
import numpy as np


def zero_array_4():
    """Factory for zero-initialized action arrays (Leduc has 4 actions)."""
    return np.zeros(4)


# LeducholdemEnv.actions == ['call', 'raise', 'fold', 'check'] → raise index 1
RAISE_ACTION_INDEX = 1


# =============================================================================
# STATE BUCKETING (Coarse Perception)
# =============================================================================

def _rank_from_card(card_str):
    """
    Extract rank from card string (e.g. 'HA' -> 'A', 'S2' -> '2').
    Returns '?' if invalid/unknown.
    """
    if card_str is None or not isinstance(card_str, str) or len(card_str) < 2:
        return "?"
    return card_str[1]


def _pot_bucket(all_chips, num_buckets=3):
    """
    Bucket total pot size. Leduc: small blind 1, big blind 2, raises 2.
    Typical pot: 2--12 chips.
    """
    total = sum(all_chips) if isinstance(all_chips, (list, tuple)) else int(all_chips)
    if num_buckets == 1:
        return 0
    if num_buckets == 2:
        return 0 if total < 5 else 1
    # 3 buckets: small (<4), medium (4-8), large (>8)
    if total < 4:
        return 0
    if total < 8:
        return 1
    return 2


def obs_to_key(state, env, bucket_mode):
    """
    Map state to an infoset key for CFR lookup.

    Args:
        state: Dict from env.get_state() with 'obs', 'raw_obs', etc.
        env: LeducholdemEnv (for round_counter)
        bucket_mode: "none" | "coarse" | "medium" | "very_coarse"

    Returns:
        Hashable key (bytes or tuple) for regrets/average_policy lookup.
    """
    if bucket_mode == "none":
        return state["obs"].tobytes()

    raw = state.get("raw_obs", {})
    hand = raw.get("hand")
    public_card = raw.get("public_card")
    all_chips = raw.get("all_chips", [0, 0])
    hand_rank = _rank_from_card(hand)
    public_rank = _rank_from_card(public_card) if public_card else "none"

    round_idx = getattr(env.game, "round_counter", 0)

    if bucket_mode == "very_coarse":
        is_pair = (
            hand and public_card
            and len(str(hand)) >= 2
            and len(str(public_card)) >= 2
            and str(hand)[1] == str(public_card)[1]
        )
        pb = _pot_bucket(all_chips)
        return (round_idx, "pair" if is_pair else "nopair", pb)

    if bucket_mode == "coarse":
        return (round_idx, hand_rank, public_rank)

    if bucket_mode == "medium":
        pb = _pot_bucket(all_chips)
        return (round_idx, hand_rank, public_rank, pb)

    # Fallback to full resolution
    return state["obs"].tobytes()


# =============================================================================
# MOOD AND TILT (session dynamics; deep-research-report (1).md)
# =============================================================================


class MoodTiltSession:
    """
    Reference-dependent mood m_t <- rho * m_t + payoff_t; episodic tilt after
    loss streak k or large negative payoff, with extra noise / raise pressure.
    """

    def __init__(
        self,
        *,
        mood_enabled=False,
        mood_rho=0.92,
        mood_behind_raise_coef=0.25,
        mood_ahead_raise_coef=0.20,
        mood_tanh_scale=10.0,
        tilt_enabled=False,
        tilt_loss_streak_k=3,
        tilt_shock_payoff=-6.0,
        tilt_duration_hands=5,
        tilt_tau_multiplier=2.0,
        tilt_min_soft_tau=0.45,
        tilt_raise_extra_logit=0.60,
        tilt_regret_flatten_mult=0.55,
        tilt_uniform_mix=0.08,
    ):
        self.mood_enabled = bool(mood_enabled)
        self.mood_rho = float(mood_rho)
        self.mood_behind_raise_coef = float(mood_behind_raise_coef)
        self.mood_ahead_raise_coef = float(mood_ahead_raise_coef)
        self.mood_tanh_scale = float(mood_tanh_scale)
        self.tilt_enabled = bool(tilt_enabled)
        self.tilt_loss_streak_k = int(tilt_loss_streak_k)
        self.tilt_shock_payoff = float(tilt_shock_payoff)
        self.tilt_duration_hands = int(tilt_duration_hands)
        self.tilt_tau_multiplier = float(tilt_tau_multiplier)
        self.tilt_min_soft_tau = float(tilt_min_soft_tau)
        self.tilt_raise_extra_logit = float(tilt_raise_extra_logit)
        self.tilt_regret_flatten_mult = float(tilt_regret_flatten_mult)
        self.tilt_uniform_mix = float(tilt_uniform_mix)

        self.mood_m = 0.0
        self.loss_streak = 0
        self.tilt_hands_remaining = 0

    def any_enabled(self):
        return self.mood_enabled or self.tilt_enabled

    def tilt_active(self):
        return bool(self.tilt_enabled and self.tilt_hands_remaining > 0)

    def hand_end(self, payoff: float):
        """Call after each completed hand with this agent's chip payoff."""
        had_tilt = self.tilt_hands_remaining > 0
        if self.mood_enabled:
            self.mood_m = self.mood_rho * self.mood_m + float(payoff)
        trigger = False
        if self.tilt_enabled:
            if payoff < 0:
                self.loss_streak += 1
            else:
                self.loss_streak = 0
            if (
                self.loss_streak >= self.tilt_loss_streak_k
                or float(payoff) <= self.tilt_shock_payoff
            ):
                trigger = True
        if trigger:
            self.tilt_hands_remaining = self.tilt_duration_hands
        elif had_tilt:
            self.tilt_hands_remaining = max(0, self.tilt_hands_remaining - 1)

    def effective_tau(self, base_tau: float) -> float:
        if not self.tilt_active():
            return float(base_tau)
        bt = float(base_tau)
        if bt <= 0:
            return float(self.tilt_min_soft_tau)
        return bt * float(self.tilt_tau_multiplier)

    def regret_flatten_scale(self) -> float:
        if self.tilt_active():
            return float(self.tilt_regret_flatten_mult)
        return 1.0

    def raise_logit_bonus(self) -> float:
        bonus = 0.0
        if self.mood_enabled:
            scale = max(self.mood_tanh_scale, 1e-6)
            if self.mood_m < 0:
                bonus += self.mood_behind_raise_coef * np.tanh(-self.mood_m / scale)
            elif self.mood_m > 0:
                bonus -= self.mood_ahead_raise_coef * np.tanh(self.mood_m / scale)
        if self.tilt_active():
            bonus += float(self.tilt_raise_extra_logit)
        return float(bonus)

    def modulate_play_probs(self, probs, legal_actions, num_actions):
        """Mix with uniform under tilt; apply raise logit bonus; renormalize legal."""
        p = np.asarray(probs, dtype=float).copy()
        li = list(legal_actions)
        if not li:
            return p
        tm = float(self.tilt_uniform_mix) if self.tilt_active() else 0.0
        if tm > 0:
            uni = np.zeros(num_actions)
            for a in li:
                uni[a] = 1.0 / len(li)
            p = (1.0 - tm) * p + tm * uni
        logits = np.zeros(num_actions)
        for a in li:
            logits[a] = np.log(max(p[a], 1e-30))
        rb = self.raise_logit_bonus()
        if rb != 0 and RAISE_ACTION_INDEX in li:
            logits[RAISE_ACTION_INDEX] += rb
        sub = np.array([logits[a] for a in li])
        sub = sub - np.max(sub)
        ex = np.exp(np.clip(sub, -50, 50))
        ex /= np.sum(ex)
        out = np.zeros(num_actions)
        for i, a in enumerate(li):
            out[a] = ex[i]
        return out

    def to_param_dict(self):
        """Serialize static knobs for pickle br_cfr_params."""
        return {
            "mood_enabled": self.mood_enabled,
            "mood_rho": self.mood_rho,
            "mood_behind_raise_coef": self.mood_behind_raise_coef,
            "mood_ahead_raise_coef": self.mood_ahead_raise_coef,
            "mood_tanh_scale": self.mood_tanh_scale,
            "tilt_enabled": self.tilt_enabled,
            "tilt_loss_streak_k": self.tilt_loss_streak_k,
            "tilt_shock_payoff": self.tilt_shock_payoff,
            "tilt_duration_hands": self.tilt_duration_hands,
            "tilt_tau_multiplier": self.tilt_tau_multiplier,
            "tilt_min_soft_tau": self.tilt_min_soft_tau,
            "tilt_raise_extra_logit": self.tilt_raise_extra_logit,
            "tilt_regret_flatten_mult": self.tilt_regret_flatten_mult,
            "tilt_uniform_mix": self.tilt_uniform_mix,
        }


# =============================================================================
# SOFT REGRET MATCHING (Quantal Response)
# =============================================================================

def regret_matching(
    regret,
    num_actions,
    tau=0.0,
    regret_scale=1.0,
    *,
    qre_normalize_regrets=False,
    qre_norm_epsilon=1e-8,
):
    """
    Regret matching, with optional softmax (quantal response) when tau > 0.

    Standard CFR: strategy[a] ∝ max(regret[a], 0), normalized.
    Soft CFR (legacy): strategy ∝ exp(positive_regret / tau), normalized.
    Soft CFR (thesis / QRE): R̃+_a = R+_a / (sum_b R+_b + ε), then
        strategy ∝ exp(η R̃+) with η = 1/tau so existing τ still acts like a
        temperature knob (larger τ ⇒ softer).

    tau=0 -> standard regret matching (deterministic toward best)
    tau>0 -> softer, more randomization, more "mistakes"

    regret_scale: multiplies regrets before ReLU (tilt "flattening" < 1).
    """
    pos_regret = np.maximum(regret * float(regret_scale), 0)
    total = np.sum(pos_regret)

    if total <= 0:
        return np.ones(num_actions) / num_actions

    if tau <= 0:
        return pos_regret / total

    if qre_normalize_regrets:
        denom = total + float(qre_norm_epsilon)
        tilde = pos_regret / denom
        eta = 1.0 / float(tau)
        logits = eta * tilde
    else:
        logits = pos_regret / float(tau)

    logits = np.clip(logits, -50, 50)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


# =============================================================================
# BR-CFR AGENT
# =============================================================================

class BoundedRationalCFRAgent:
    """
    CFR agent with bounded-rationality degradations.

    Trains against a fixed opponent (e.g. pre-trained DQN). Supports:
      - Fewer iterations per episode (limited computation)
      - Forgetting decay on regrets and average policy (limited memory)
      - Soft regret matching via temperature τ (noisy choice)
      - State bucketing for coarse perception
      - Mood + tilt session dynamics (optional)
    """

    def __init__(
        self,
        env,
        player_id,
        opponent_agent,
        model_path,
        *,
        iterations_per_episode=10,
        memory_decay=1.0,
        soft_regret_tau=0.0,
        bucket_mode="none",
        mood_enabled=False,
        mood_rho=0.92,
        mood_behind_raise_coef=0.25,
        mood_ahead_raise_coef=0.20,
        mood_tanh_scale=10.0,
        tilt_enabled=False,
        tilt_loss_streak_k=3,
        tilt_shock_payoff=-6.0,
        tilt_duration_hands=5,
        tilt_tau_multiplier=2.0,
        tilt_min_soft_tau=0.45,
        tilt_raise_extra_logit=0.60,
        tilt_regret_flatten_mult=0.55,
        tilt_uniform_mix=0.08,
        qre_normalize_regrets=False,
        qre_norm_epsilon=1e-8,
    ):
        self.env = env
        self.player_id = player_id
        self.opponent_agent = opponent_agent
        self.model_path = model_path
        self.use_raw = False

        self.iterations_per_episode = iterations_per_episode
        self.memory_decay = memory_decay
        self.soft_regret_tau = soft_regret_tau
        self.bucket_mode = bucket_mode
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
        )

        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(zero_array_4)
        self.regrets = collections.defaultdict(zero_array_4)
        self.iteration = 0

    def _obs_key(self, state):
        return obs_to_key(state, self.env, self.bucket_mode)

    def register_hand_outcome(self, payoff: float):
        """Update mood and tilt counters after one hand (payoff for this player_id)."""
        self.mood_tilt.hand_end(float(payoff))

    def _regret_matching(self, obs):
        regret = self.regrets[obs].copy()
        tau = self.mood_tilt.effective_tau(self.soft_regret_tau)
        rscale = self.mood_tilt.regret_flatten_scale()
        return regret_matching(
            regret,
            self.env.num_actions,
            tau=tau,
            regret_scale=rscale,
            qre_normalize_regrets=self.qre_normalize_regrets,
            qre_norm_epsilon=self.qre_norm_epsilon,
        )

    def _apply_mood_tilt_to_strategy(self, strategy, legal_actions):
        """Renormalize strategy with same raise / uniform mix as play."""
        if not self.mood_tilt.any_enabled():
            return strategy
        return self.mood_tilt.modulate_play_probs(
            strategy, legal_actions, self.env.num_actions
        )

    def _apply_decay(self, obs):
        """Apply forgetting decay to regrets and average policy for this infoset."""
        if self.memory_decay >= 1.0:
            return
        self.regrets[obs] *= self.memory_decay
        self.average_policy[obs] *= self.memory_decay

    def traverse_tree(self, probs):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        state = self.env.get_state(current_player)
        obs = self._obs_key(state)
        legal_actions = list(state["legal_actions"].keys())

        if current_player != self.player_id:
            action, _ = self.opponent_agent.eval_step(state)
            self.env.step(action)
            utility = self.traverse_tree(probs)
            self.env.step_back()
            return utility

        # Apply forgetting before update
        self._apply_decay(obs)

        strategy = self._regret_matching(obs)
        strategy = self._apply_mood_tilt_to_strategy(strategy, legal_actions)
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

        cf_prob = np.prod(probs[:current_player]) * np.prod(probs[current_player + 1:])
        for action in legal_actions:
            regret = cf_prob * (action_utils[action] - node_util)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += probs[current_player] * strategy[action]

        self.policy[obs] = strategy
        return np.array(
            [node_util if i == self.player_id else 0 for i in range(self.env.num_players)]
        )

    def save(self):
        """Save in same format as replication CFR (for compatibility)."""
        data = {
            "policy": dict(self.policy),
            "average_policy": dict(self.average_policy),
            "regrets": dict(self.regrets),
            "iteration": self.iteration,
            "br_cfr_params": {
                "iterations_per_episode": self.iterations_per_episode,
                "memory_decay": self.memory_decay,
                "soft_regret_tau": self.soft_regret_tau,
                "bucket_mode": self.bucket_mode,
                "qre_normalize_regrets": self.qre_normalize_regrets,
                "qre_norm_epsilon": self.qre_norm_epsilon,
                **self.mood_tilt.to_param_dict(),
            },
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)


# =============================================================================
# WRAPPER FOR GAMEPLAY (compatible with evaluation)
# =============================================================================

class BRCFRWrapper:
    """
    Wrapper for BR-CFR to work with env.run() and evaluation.

    Uses the same obs_to_key logic as the agent so lookups match.
    """

    def __init__(self, br_cfr_agent):
        self.cfr = br_cfr_agent
        self.env = br_cfr_agent.env
        self.use_raw = False
        self._bucket_mode = br_cfr_agent.bucket_mode

    def _obs_key(self, state):
        return obs_to_key(state, self.env, self._bucket_mode)

    def step(self, state):
        obs = self._obs_key(state)
        legal_actions = list(state["legal_actions"].keys())

        if obs not in self.cfr.average_policy:
            action_probs = np.ones(self.env.num_actions) / self.env.num_actions
        else:
            raw_probs = self.cfr.average_policy[obs].copy()
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

        action_probs = self.cfr.mood_tilt.modulate_play_probs(
            action_probs, legal_actions, self.env.num_actions
        )

        # int(...) — NumPy 2.x may return a plain int (no .item()); int() is safe for all versions
        return int(np.random.choice(self.env.num_actions, p=action_probs))

    def notify_hand_end(self, payoff: float):
        """Forward session update after a full hand (same seat as this wrapper)."""
        self.cfr.register_hand_outcome(float(payoff))

    def eval_step(self, state):
        action = self.step(state)
        return action, {"probs": {}}
