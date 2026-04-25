"""
Configuration for Bounded-Rational CFR (BR-CFR) variants.

This module is SEPARATE from the replication config (config.py). All BR-CFR
outputs go to results/br_cfr/ to avoid polluting the original replication results.

Parameter choices are motivated by bounded-rationality literature and the goal
of creating "dumber" / more human-like CFR agents for ablation studies.
"""

import os

# =============================================================================
# PATH CONFIGURATION (all under results/br_cfr/ - separate from replication)
# =============================================================================

_BASE = os.environ.get(
    "BLUFFING_RESULTS_DIR",
    os.path.join(os.path.dirname(__file__), "results")
)
BR_CFR_BASE = os.path.join(_BASE, "br_cfr")
BR_CFR_TRAIN_DIR = os.path.join(BR_CFR_BASE, "training")
BR_CFR_EVAL_DIR = os.path.join(BR_CFR_BASE, "evaluation")

# Training episodes for BR-CFR (can use fewer for quick ablation, same as replication for comparison)
BR_CFR_TRAIN_EPISODES = int(os.environ.get("BR_CFR_TRAIN_EPISODES", "100000"))

# Evaluation games when training BR-CFR
BR_CFR_EVAL_GAMES = int(os.environ.get("BR_CFR_EVAL_GAMES", "100000"))

# =============================================================================
# BOUNDED-RATIONALITY PARAMETERS (with motivation)
# =============================================================================


# --- A) LIMITED COMPUTATION (fewer iterations) ---
#
# Motivation: Humans cannot traverse the full game tree every time they act.
# Fewer iterations per episode = less "thinking" per game.
#
# Original replication: 10 iterations_per_episode (Paper Table 2)
# "Dumber" values: 1 = minimal computation, 3 = moderate limit
#
ITERATIONS_PER_EPISODE = 1
# 1  = "Very dumb" - one traversal per episode (10x less than replication)
# 3  = "Moderately dumb"
# 10 = Full replication baseline (no degradation)


# --- B) FORGETTING / RECENCY BIAS (memory decay) ---
#
# Motivation: Humans don't accumulate perfect long-term regret statistics.
# Older experience is weighted less (recency bias). Decay applied each time
# we visit a state during traversal.
#
# decay=1.0  = No forgetting (standard CFR)
# decay=0.999 = Mild forgetting - ~0.37 weight after 1000 visits
# decay=0.99  = Strong forgetting - ~0.37 weight after 100 visits ("dumber")
# decay=0.95  = Very strong - useful for extreme bounded-rationality
#
MEMORY_DECAY = 0.99
# 1.0   = No forgetting (baseline)
# 0.999 = Mild - human-like recency
# 0.99  = Strong - "dumber" agent forgets quickly


# --- C) SOFT REGRET MATCHING (quantal response / temperature) ---
#
# Motivation: Humans don't always pick the best action; they probabilistically
# lean toward it (quantal response / logit choice). Temperature τ controls
# how "sharp" vs "soft" the policy is.
#
# tau -> 0: Deterministic best response (standard regret matching)
# tau = 0.5: Moderately soft - some mistakes, more randomization
# tau = 1.0: Softer - more human-like mistakes
# tau = 2.0: Very soft - heavy randomization ("dumber")
#
# Formula: strategy = softmax(positive_regrets / tau)
# Higher tau => more uniform over actions, more "mistakes"
#
SOFT_REGRET_TAU = 1.0
# 0.0   = Hard regret matching (baseline)
# 0.5   = Slightly soft
# 1.0   = Moderately soft - good for "human-like" bluffs
# 2.0   = Very soft - "dumber"


# --- D) STATE BUCKETING (coarse perception) ---
#
# Motivation: Humans bucket situations; they don't treat every exact state as
# unique. Instead of obs.tobytes() (156-dim unique key), we map to coarse
# buckets: round + hand_rank + public_rank + pot_bucket.
#
# BUCKET_MODE options:
#   "none"     = Full resolution (obs.tobytes()) - standard CFR
#   "coarse"   = Round + hand rank + public rank only (ignore suits, pot)
#   "medium"   = Coarse + 3 pot buckets (small/med/large)
#   "very_coarse" = Round + is_pair + pot_bucket (pair vs non-pair only)
#
BUCKET_MODE = "coarse"
# "none"        = Full state (baseline)
# "coarse"      = round, hand_rank, public_rank (13*14*2 ~ 364 buckets vs thousands)
# "medium"      = coarse + pot_bucket
# "very_coarse" = Minimal - pair/non-pair, round, pot (very "dumb")


# --- E) MONTE CARLO CFR (sampling - optional) ---
#
# Motivation: Humans don't enumerate all outcomes; they sample a few lines.
# Not implemented in initial scope; config reserved for future work.
#
MC_CFR_ENABLED = False
MC_CFR_SAMPLE_ACTIONS = 1  # Number of actions to sample per decision (if enabled)


# --- F) MOOD + TILT (reference-dependent session + episodic loss of control) ---
#
# Motivation: deep-research-report (1).md — mood m_{t+1} = rho*m_t + payoff_t;
# tilt after loss streak k or large negative payoff; higher noise + raise bias.
#
# QRE softmax: thesis-style normalized positive regrets (optional per preset).
BR_CFR_MISC_DEFAULTS = {
    "qre_normalize_regrets": False,
    "qre_norm_epsilon": 1e-8,
}

MOOD_TILT_PARAM_DEFAULTS = {
    "mood_enabled": False,
    "mood_rho": 0.92,
    "mood_behind_raise_coef": 0.25,
    "mood_ahead_raise_coef": 0.20,
    "mood_tanh_scale": 10.0,
    "tilt_enabled": False,
    "tilt_loss_streak_k": 3,
    "tilt_shock_payoff": -6.0,
    "tilt_duration_hands": 5,
    "tilt_tau_multiplier": 2.0,
    "tilt_min_soft_tau": 0.45,
    "tilt_raise_extra_logit": 0.60,
    "tilt_regret_flatten_mult": 0.55,
    "tilt_uniform_mix": 0.08,
}


def merge_br_cfr_preset(preset_dict):
    """Merge mood/tilt + QRE defaults into a preset (single source for training kwargs)."""
    out = dict(MOOD_TILT_PARAM_DEFAULTS)
    out.update(BR_CFR_MISC_DEFAULTS)
    out.update(preset_dict)
    return out


# =============================================================================
# PRESET VARIANTS (for easy ablation)
# =============================================================================

PRESETS = {
    "baseline": {
        "iterations_per_episode": 10,
        "memory_decay": 1.0,
        "soft_regret_tau": 0.0,
        "bucket_mode": "none",
        "description": "Full CFR (replication-equivalent, no degradation)",
    },
    "fewer_iterations": {
        "iterations_per_episode": 1,
        "memory_decay": 1.0,
        "soft_regret_tau": 0.0,
        "bucket_mode": "none",
        "description": "Limited computation only",
    },
    "forgetting": {
        "iterations_per_episode": 10,
        "memory_decay": 0.99,
        "soft_regret_tau": 0.0,
        "bucket_mode": "none",
        "description": "Recency bias / limited memory",
    },
    "soft_regret": {
        "iterations_per_episode": 10,
        "memory_decay": 1.0,
        "soft_regret_tau": 1.0,
        "bucket_mode": "none",
        "qre_normalize_regrets": True,
        "description": "Quantal response / noisy choice (normalized R+ QRE)",
    },
    "state_bucketing": {
        "iterations_per_episode": 10,
        "memory_decay": 1.0,
        "soft_regret_tau": 0.0,
        "bucket_mode": "coarse",
        "description": "Coarse state perception",
    },
    "dumb_all": {
        "iterations_per_episode": 1,
        "memory_decay": 0.99,
        "soft_regret_tau": 1.0,
        "bucket_mode": "coarse",
        "description": "All degradations combined (maximally bounded-rational)",
    },
    # --- Smart / Medium / Dumb (for overnight batch run) ---
    "smart": {
        "iterations_per_episode": 10,
        "memory_decay": 1.0,
        "soft_regret_tau": 0.0,
        "bucket_mode": "none",
        "description": "Full CFR - no degradation (smart)",
    },
    "medium": {
        "iterations_per_episode": 5,
        "memory_decay": 0.999,
        "soft_regret_tau": 0.5,
        "bucket_mode": "none",
        "description": "Moderate degradations (medium)",
    },
    "dumb": {
        "iterations_per_episode": 1,
        "memory_decay": 0.99,
        "soft_regret_tau": 1.0,
        "bucket_mode": "coarse",
        "description": "Heavy degradations (dumb)",
    },
    "mood_tilt": {
        "iterations_per_episode": 10,
        "memory_decay": 1.0,
        "soft_regret_tau": 0.5,
        "bucket_mode": "none",
        "mood_enabled": True,
        "tilt_enabled": True,
        "qre_normalize_regrets": True,
        "description": "Mood + tilt session dynamics (thesis spec)",
    },
}

# Which preset to use (overrides individual params when running train_br_cfr.py)
ACTIVE_PRESET = os.environ.get("BR_CFR_PRESET", "soft_regret")
