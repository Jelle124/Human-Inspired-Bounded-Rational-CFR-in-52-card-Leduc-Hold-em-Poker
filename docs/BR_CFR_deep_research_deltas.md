# BR-CFR implementation vs thesis methods document

See `docs/thesis/deep-research-report.md` for the full methods spec.

## Mood and tilt

| Report item | This codebase |
|-------------|----------------|
| \(m_{t+1} = \rho m_t + \text{payoff}_t\) | `MoodTiltSession.hand_end` in `br_cfr_agent.py` |
| Behind → more raise / ahead → less | `raise_logit_bonus()` applied to raise action |
| Tilt triggers: loss streak or large negative shock | `tilt_loss_streak_k`, `tilt_shock_payoff` |
| Tilt: higher noise, raise bias, flattened regrets | `effective_tau`, `tilt_raise_extra_logit`, `tilt_regret_flatten_mult`, `tilt_uniform_mix` |
| Training + play | `train_parallel_suite.BRCFRAgainstOpponentAgent.traverse_tree` and `BRCFRWrapper` |

## Differences from the methods document

1. **Soft regret / QRE** — Normalized positive regrets when `qre_normalize_regrets=True` (Medium/Dumb presets in `br_cfr_variant_specs.py`).

2. **Training setting** — `train_parallel_suite.py` co-trains BR-CFR vs standard CFR (or DQN vs CFR for replication). Not BR-only self-play.

3. **Payoff signal during training** — `register_hand_outcome` uses root return from each CFR traversal, not a separately played hand.

4. **State bucketing** — Optional `last aggressive action` bucket from the report is not implemented.

5. **Monte Carlo CFR** — Not implemented (`MC_CFR_ENABLED` reserved in `br_cfr_config.py`).

## Key files

- `train_parallel_suite.py` — experiments, training, evaluation, bluff stats
- `cfr_agent.py` — standard tabular CFR
- `br_cfr_agent.py` — BR-CFR + mood/tilt
- `br_cfr_variant_specs.py` — Smart / Medium / Dumb table
