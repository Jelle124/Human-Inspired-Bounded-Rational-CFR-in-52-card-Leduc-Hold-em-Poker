# BR-CFR implementation vs `deep-research-report (1).md`

This file summarizes **agreements** and **intentional or structural differences** between the thesis methods document and the repository after adding mood + tilt.

## Mood and tilt (newly implemented)

| Report item | This codebase |
|-------------|----------------|
| \(m_{t+1} = \rho m_t + \text{payoff}_t\) | `MoodTiltSession.hand_end`: `mood_m = mood_rho * mood_m + payoff` when `mood_enabled`. |
| Behind → more raise / ahead → less | `raise_logit_bonus()`: adds `mood_behind_raise_coef * tanh(-m/scale)` when \(m<0\), subtracts `mood_ahead_raise_coef * tanh(m/scale)` when \(m>0\), applied to **raise** (action index 1). |
| Tilt triggers: loss streak \(k\) or large negative shock | `tilt_loss_streak_k`, `tilt_shock_payoff`; on trigger sets `tilt_hands_remaining = tilt_duration_hands`. |
| Tilt: higher noise, stronger raise bias, flattened regret response | **Noise:** `effective_tau` multiplies base `soft_regret_tau` when tilted; if base \(\tau=0\), uses `tilt_min_soft_tau`. **Raise:** extra `tilt_raise_extra_logit` on raise. **Flatten:** `tilt_regret_flatten_mult` scales regrets before ReLU in `regret_matching`. **Uniform mix:** `tilt_uniform_mix` blends policy toward uniform over legal actions during tilt (extra exploration). |
| Gameplay overlay in wrapper | `BRCFRWrapper.step` and `evaluate_br_cfr.CFRPolicyWrapper` call `modulate_play_probs` after building probabilities from `average_policy`. |
| Training overlay | `traverse_tree` applies the same modulation to the **current-iteration** strategy after regret matching (so learning uses the same distorted policy as play when mood/tilt are on). |

**Hyperparameters** live in `br_cfr_config.MOOD_TILT_PARAM_DEFAULTS` and are saved under `br_cfr_params` in pickles. Preset `mood_tilt` enables both modules with moderate \(\tau\).

## Differences from the report (unchanged or structural)

1. **Soft regret / QRE formula**  
   The report writes \(\pi \propto \exp(\eta \cdot \tilde{R}^+)\) with **normalized** positive regrets \(\tilde{R}^+_a = R^+_a / (\sum_b R^+_b + \varepsilon)\). When `qre_normalize_regrets=True` (defaults **off** in `BR_CFR_MISC_DEFAULTS`; **on** in presets `soft_regret` and `mood_tilt`), `regret_matching` uses that \(\tilde{R}^+\) and sets **\(\eta = 1/\tau\)** so the existing temperature knob still means “larger \(\tau\) ⇒ softer”. When the flag is **False**, the legacy logits \(R^+/\tau\) are kept for backward compatibility with older ablations.

2. **BR-CFR training setting**  
   The report describes **self-play BR vs BR** for training. This repository’s main BR-CFR path (`train_br_cfr.py`) still trains **against the frozen replication DQN** (as before). Co-learning `simultaneous_training_cfr_vs_br_smart.py` is CFR vs BR-CFR, not DQN-free self-play BR only.

3. **What counts as `payoff_t` during training**  
   After each CFR **traversal**, `register_hand_outcome` uses the **root return** `traverse_tree(...)[player_id]` (expected utility for the rooted deal under the traversal), not the payoff of a separate single **played** hand. That matches tree search but differs from a literal “one hand played per update” story.

4. **State abstraction / bucketing**  
   Per request, bucketing was **not** changed. The report’s optional “last aggressive action” bucket is still **not** in `obs_to_key`.

5. **Monte Carlo CFR**  
   The report reserves MC-CFR; `MC_CFR_ENABLED` remains unused / not implemented.

6. **Tilt duration semantics**  
   On trigger, `tilt_hands_remaining` is set to `tilt_duration_hands` and is decremented after each subsequent hand that does not re-trigger; no separate “cooldown” variable beyond that counter.

## Files touched

- `br_cfr_agent.py` — `MoodTiltSession`, `regret_matching(..., regret_scale=, qre_normalize_regrets=)`, agent + wrapper wiring.  
- `br_cfr_config.py` — `MOOD_TILT_PARAM_DEFAULTS`, `BR_CFR_MISC_DEFAULTS`, `merge_br_cfr_preset`, presets `mood_tilt` / `soft_regret` (QRE normalization on).  
- `train_br_cfr.py`, `run_br_cfr_smart_medium_dumb.py`, `simultaneous_training_cfr_vs_br_smart.py` — merged kwargs, `register_hand_outcome` after traversals.  
- `evaluate_br_cfr.py`, `run_evaluate_all_br_cfr.py` — `load_cfr_policy` returns `br_cfr_params`; wrappers apply mood/tilt; `notify_hand_end` after each game.
