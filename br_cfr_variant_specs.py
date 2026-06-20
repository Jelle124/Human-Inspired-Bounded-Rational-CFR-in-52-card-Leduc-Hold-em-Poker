"""
BR-CFR Smart / Medium / Dumb hyperparameters (single source for the parallel suite).

Values match the thesis parameter table (screenshot). Use `get_merged_variant_kwargs`
to obtain a dict suitable for `BoundedRationalCFRAgent(..., **kwargs)` after
`merge_br_cfr_preset` fills defaults.
"""

from br_cfr_config import merge_br_cfr_preset

# Exact table columns (Smart | Medium | Dumb). bucket_mode uses code strings.
BR_CFR_VARIANT_PARAM_TABLE = {
    "smart": {
        "iterations_per_episode": 10,
        "memory_decay": 1.0,
        "soft_regret_tau": 0.0,
        "bucket_mode": "none",
        "qre_normalize_regrets": False,
        "qre_norm_epsilon": 1e-8,
        "mood_enabled": False,
        "tilt_enabled": False,
    },
    "medium": {
        "iterations_per_episode": 5,
        "memory_decay": 0.999,
        "soft_regret_tau": 0.5,
        "bucket_mode": "coarse",
        "qre_normalize_regrets": True,
        "qre_norm_epsilon": 1e-8,
        "mood_enabled": True,
        "mood_rho": 0.95,
        "mood_behind_raise_coef": 0.20,
        "mood_ahead_raise_coef": 0.10,
        "mood_tanh_scale": 4.0,
        "tilt_enabled": False,
    },
    "dumb": {
        "iterations_per_episode": 1,
        "memory_decay": 0.99,
        "soft_regret_tau": 1.0,
        "bucket_mode": "very_coarse",
        "qre_normalize_regrets": True,
        "qre_norm_epsilon": 1e-8,
        "mood_enabled": True,
        "mood_rho": 0.90,
        "mood_behind_raise_coef": 0.35,
        "mood_ahead_raise_coef": 0.20,
        "mood_tanh_scale": 2.5,
        "tilt_enabled": True,
        "tilt_loss_streak_k": 6,
        "tilt_shock_payoff": -10.0,
        "tilt_duration_hands": 2,
        "tilt_trigger_probability": 0.25,
        "tilt_mood_threshold": -3.0,
        "tilt_cooldown_hands": 10,
        "tilt_refresh_while_active": False,
        "tilt_reset_loss_streak_on_trigger": True,
        "tilt_tau_multiplier": 1.5,
        "tilt_min_soft_tau": 1.2,
        "tilt_raise_extra_logit": 0.25,
        "tilt_regret_flatten_mult": 0.85,
        "tilt_uniform_mix": 0.05,
    },
}


def get_merged_variant_kwargs(variant: str) -> dict:
    """
    Return kwargs for BoundedRationalCFRAgent (excludes 'description').

    Merges table row with `merge_br_cfr_preset` so unset mood/tilt fields keep
    safe defaults where the table uses "-".
    """
    if variant not in BR_CFR_VARIANT_PARAM_TABLE:
        raise KeyError(f"Unknown variant {variant!r}; use one of {list(BR_CFR_VARIANT_PARAM_TABLE)}")
    row = dict(BR_CFR_VARIANT_PARAM_TABLE[variant])
    row["description"] = f"table_{variant}"
    merged = merge_br_cfr_preset(row)
    return {k: v for k, v in merged.items() if k != "description"}
