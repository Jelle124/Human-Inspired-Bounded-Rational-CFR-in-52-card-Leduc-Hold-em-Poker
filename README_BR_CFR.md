# Bounded-Rational CFR (BR-CFR)

This document describes the **Bounded-Rational CFR** implementation added for thesis ablation studies. It is **completely separate** from the original replication work (`simultaneous_training.py`, `evaluate_simultaneous.py`, `config.py`). No existing replication files are modified.

## Purpose

CFR assumes full traversal, perfect recall, exact regrets, and infinite iterations. By deliberately restricting these, we obtain "dumber" / more human-like variants suitable for:

- Ablation studies on bounded rationality
- Comparison with DQN bluffing behavior
- Modeling human-like limitations in poker agents

## File Layout

| File | Purpose |
|------|---------|
| `br_cfr_config.py` | Configuration, parameter values, and **documented motivations** |
| `br_cfr_agent.py` | `BoundedRationalCFRAgent` and `BRCFRWrapper` |
| `train_br_cfr.py` | Train BR-CFR against pre-trained DQN |
| `evaluate_br_cfr.py` | Evaluate BR-CFR vs DQN and vs replication CFR |
| `simultaneous_training_cfr_vs_br_smart.py` | Co-train standard CFR and BR-CFR (smart) from scratch against each other |
| `plot_cfr_vs_br_smart_win_rates.py` | Plot eval win rates from that run (JSONL) |
| `README_BR_CFR.md` | This documentation |

All outputs go to `results/br_cfr/` (not `results/`). CFR–BR co-training writes under `results/cfr_vs_br_smart/` (see below).

### CFR vs BR-CFR smart — co-training (Docker)

The main [README.md](README.md) recommends Docker for the replication environment (`docker build -t bluffing-leduc .`). The same pattern runs co-training: mount `results` so checkpoints and JSONL land on the host (avoids permission issues if the host `results/` tree was never root-owned).

The image **bakes in whatever code existed at `docker build` time**. If you see `can't open file '/app/simultaneous_training_cfr_vs_br_smart.py'`, rebuild from a checkout that contains that script:

```bash
docker build -t bluffing-leduc .
```

**Option A — rebuild (normal):** after `git pull` or adding new scripts, run `docker build` again, then:

```bash
# Use the image tag you built (README examples use bluffing-leduc; some sections below use bluffing)
docker run -v "$(pwd)/results:/app/results" bluffing-leduc python simultaneous_training_cfr_vs_br_smart.py

# Shorter run
docker run -v "$(pwd)/results:/app/results" -e TRAIN_EPISODES=10000 bluffing-leduc python simultaneous_training_cfr_vs_br_smart.py
```

**Option B — mount current repo (no rebuild):** use the host’s working tree as `/app` so new files are visible immediately (dependencies still come from the image):

```bash
docker run -v "$(pwd):/app" -v "$(pwd)/results:/app/results" bluffing-leduc \
  python simultaneous_training_cfr_vs_br_smart.py
```

Plot on the host after the run (paths under `./results/...`):

```bash
python3 plot_cfr_vs_br_smart_win_rates.py -i results/cfr_vs_br_smart/training/cfr_vs_br_smart_colearn_training_win_rates.jsonl
```

## Bounded-Rationality Degradations

### A) Fewer Iterations (Limited Computation)

**What:** Reduce `iterations_per_episode` from 10 (replication) to 1 or 3.

**Why:** Humans cannot re-solve the whole game tree every decision.

**Effect:** Fewer traversals per episode → more exploitable, less equilibrium-like policy.

**Preset:** `fewer_iterations` uses 1 iteration.

---

### B) Forgetting Decay (Limited Memory)

**What:** Multiply regrets and average policy by `decay ∈ (0, 1)` each time we visit an infoset.

**Why:** Humans do not accumulate perfect long-term regret; recency bias.

**Effect:** Older experience fades; agent becomes less stable and more "human-like".

**Params:**
- `decay=1.0` → no forgetting (standard CFR)
- `decay=0.99` → strong forgetting
- `decay=0.999` → mild forgetting

**Preset:** `forgetting` uses 0.99.

---

### C) Soft Regret Matching (Quantal Response)

**What:** Replace deterministic regret matching with a softmax over positive regrets. Two modes (see `br_cfr_agent.regret_matching`):

- **Legacy (`qre_normalize_regrets=False`):** `strategy ∝ exp(R^+ / τ)` (unnormalized positives).
- **Thesis-style QRE (`qre_normalize_regrets=True`):**  
  \(\tilde{R}^+_a = R^+_a / (\sum_b R^+_b + \varepsilon)\), then `strategy ∝ exp(η · R̃+)` with **η = 1/τ** so larger τ still means a softer policy.

**Why:** Humans probabilistically lean toward the best action rather than picking it deterministically; normalization matches the methods doc and stabilizes logit scale across infosets.

**Effect:** More randomization, more "mistakes", sometimes atypical bluff frequencies.

**Params:**
- `τ=0` → standard regret matching
- `τ=1.0` → moderately soft
- `τ=2.0` → very soft
- `qre_norm_epsilon` — denominator stabilization when normalizing (default `1e-8`).

**Preset:** `soft_regret` uses τ=1.0 with **normalized** R+ (`qre_normalize_regrets=True`). Other presets default to **False** unless you set the flag in `br_cfr_config.PRESETS`.

---

### D) State Bucketing (Coarse Perception)

**What:** Instead of unique `obs.tobytes()` keys, map states to coarse buckets.

**Why:** Humans group situations (e.g. "pair vs non-pair") instead of treating every hand uniquely.

**Modes:**
- `none` → full resolution (standard CFR)
- `coarse` → (round, hand_rank, public_rank)
- `medium` → coarse + pot bucket
- `very_coarse` → (round, pair/nopair, pot_bucket)

**Effect:** Fewer infosets → generalization but systematic mistakes.

**Preset:** `state_bucketing` uses `coarse`.

---

### E) Mood + tilt (session dynamics)

**What:** Reference-dependent **mood** \(m_{t+1}=\rho m_t+\text{payoff}\) biases raise logits when behind vs ahead; **tilt** is an episodic mode after a loss streak or a large negative payoff, with higher temperature (or a floor \(\tau\) when cold), regret flattening, extra raise logit, and optional uniform mixing over legal actions.

**Why:** Loss-chasing / conservatism when ahead (field evidence) and tilt as transient loss of control (Moreau et al.–style operationalization in the thesis doc).

**Where:** `br_cfr_agent.MoodTiltSession`, training hooks in `train_br_cfr.py` (and related scripts), play path in `BRCFRWrapper` / `evaluate_br_cfr.CFRPolicyWrapper`.

**Preset:** `mood_tilt` enables mood + tilt with defaults from `br_cfr_config.MOOD_TILT_PARAM_DEFAULTS`. Other presets merge these defaults with mood/tilt **off** unless overridden.

**Spec vs code:** See `docs/BR_CFR_deep_research_deltas.md` for differences from `deep-research-report (1).md` (e.g. training still vs DQN in `train_br_cfr.py`, payoff signal from traversals).

---

## Presets

Defined in `br_cfr_config.PRESETS`:

| Preset | Description |
|--------|-------------|
| `smart` | Full CFR (no degradation) — for overnight batch |
| `medium` | Moderate degradations (5 iter, decay 0.999, τ=0.5) — for overnight batch |
| `dumb` | Heavy degradations (1 iter, decay 0.99, τ=1, coarse) — for overnight batch |
| `baseline` | Same as smart |
| `fewer_iterations` | Limited computation only |
| `forgetting` | Recency bias / limited memory |
| `soft_regret` | Quantal response |
| `state_bucketing` | Coarse perception |
| `dumb_all` | All degradations combined |
| `mood_tilt` | Mood + tilt enabled (thesis-style session dynamics) |

## Parallel suite (Smart / Medium / Dumb + CFR self-play baseline)

`parallel_br_cfr_train_suite.py` runs **four processes at once**: three BR-CFR trainers (vs frozen DQN) using the exact hyperparameters in `br_cfr_variant_specs.BR_CFR_VARIANT_PARAM_TABLE`, plus one **tabular CFR vs CFR** self-play job (Smart’s `iterations_per_episode`, same outer-episode budget). Outputs default to `results/br_cfr/parallel_suite_output/` (override with `PARALLEL_SUITE_DIR`).

Mount the **repo** onto `/app` so the image sees your current code (including `parallel_br_cfr_train_suite.py`) without rebuilding, and mount `results` for outputs. Use **`PARALLEL_SEQUENTIAL=1`** to run the four jobs **one after another** (same process memory as a single trainer—recommended on tight RAM):

```bash
docker run --shm-size=2g \
  -v "$(pwd):/app" -v "$(pwd)/results:/app/results" \
  -e PARALLEL_SEQUENTIAL=1 \
  bluffing-leduc \
  python parallel_br_cfr_train_suite.py
```

If you prefer not to mount the repo, **`docker build -t bluffing-leduc .`** after pulling/adding scripts so `COPY . .` includes `parallel_br_cfr_train_suite.py`.

After training, evaluate **all four** checkpoints in one go (8 win rates: DQN vs each policy, each policy vs replication CFR):

```bash
BR_CFR_EVAL_N=100000 python evaluate_parallel_suite.py
```

Single-model eval (only **two** matchups per run) remains: `BR_CFR_MODEL=... python evaluate_br_cfr.py`.

See the script docstring for `PARALLEL_SUITE_DIR`, `PARALLEL_TRAIN_EPISODES`, `PARALLEL_SEQUENTIAL`, and `PARALLEL_MAX_WORKERS`.

## Usage

### Bluff analysis (how BR-CFR reacts to DQN bluffs)

Run the full pipeline: generate game logs, then run bluff analysis for each variant:

```bash
# Docker
docker run -v "$(pwd)/results:/app/results" bluffing python run_br_cfr_bluff_analysis.py

# Or locally
python run_br_cfr_bluff_analysis.py
```

This produces the same bluff metrics as the replication (bluff attempts, success rate, opponent reactions) for smart, medium, and dumb BR-CFR. Full output is saved to `results/br_cfr/BLUFF_ANALYSIS_OUTPUT.txt`.

**Manual workflow:**
1. Generate logs: `BR_CFR_WRITE_LOGS=1 python run_evaluate_all_br_cfr.py`
2. Run analysis for a specific variant:
   `BLUFF_LOG_PATH=results/br_cfr/evaluation/game_logs_dqn_vs_br_cfr_smart_10000.jsonl python analyze_bluff_ReactionCFR_DQNBluff.py`

### Next steps after training (evaluate & summarize)

Run evaluation for all 3 variants and generate a summary:

```bash
# Docker
docker run -v "$(pwd)/results:/app/results" bluffing python run_evaluate_all_br_cfr.py

# Or locally
python run_evaluate_all_br_cfr.py
```

Output:
- `results/br_cfr/EVALUATION_SUMMARY.txt` — Win rates (DQN vs each BR-CFR, BR-CFR vs CFR)
- `results/br_cfr/evaluation_results.json` — Same data in JSON

### 0. Overnight run: Smart, Medium, Dumb (all 3 in one container)

Train all 3 variants sequentially so you can leave it running overnight:

```bash
# Docker (recommended)
docker run -v "$(pwd)/results:/app/results" bluffing python run_br_cfr_smart_medium_dumb.py

# Or locally
python run_br_cfr_smart_medium_dumb.py
```

Output in the morning:
- `results/br_cfr/training/br_cfr_smart.pkl`  — full CFR, no degradation
- `results/br_cfr/training/br_cfr_medium.pkl` — moderate degradations
- `results/br_cfr/training/br_cfr_dumb.pkl`   — heavy degradations

### 1. Train BR-CFR (single variant)

Requires the replication DQN to exist (`results/training/dqn_simultaneous_100K.pt`).

```bash
# Default preset: soft_regret
python train_br_cfr.py

# Different preset
BR_CFR_PRESET=state_bucketing python train_br_cfr.py

# Quick test (1K episodes)
BR_CFR_TRAIN_EPISODES=1000 python train_br_cfr.py
```

Output: `results/br_cfr/training/br_cfr_<preset>.pkl`

### 2. Evaluate

```bash
# Evaluate default BR-CFR (soft_regret)
python evaluate_br_cfr.py

# Evaluate a specific model
BR_CFR_MODEL=results/br_cfr/training/br_cfr_state_bucketing.pkl python evaluate_br_cfr.py

# Evaluate replication CFR (baseline, no BR-CFR model)
BR_CFR_MODEL= python evaluate_br_cfr.py
```

Output: `results/br_cfr/evaluation/br_cfr_eval_*.jsonl`

### 3. Run All Presets (Training)

```bash
for p in baseline fewer_iterations forgetting soft_regret state_bucketing dumb_all; do
  BR_CFR_PRESET=$p python train_br_cfr.py
done
```

## Parameter Motivation Summary

| Parameter | Rationale | "Dumber" Direction |
|-----------|-----------|---------------------|
| `iterations_per_episode` | Less computation per game | Lower (e.g. 1) |
| `memory_decay` | Recency bias | &lt; 1 (e.g. 0.99) |
| `soft_regret_tau` | Noisy / quantal choice | &gt; 0 (e.g. 1.0) |
| `bucket_mode` | Coarse perception | `coarse` or `very_coarse` |

## Academic Wording

When describing these agents in a thesis:

> "We model human-like limitations by restricting computation (fewer iterations), memory (decay), and perception (state abstraction), producing a bounded-rational CFR agent (BR-CFR)."

Terms: *Bounded-rational CFR*, *Approximate CFR with limited computation and perception*, *Human-inspired bounded-rational agent*.

## Dependencies

Same as the main replication: `rlcard`, `torch`, `numpy`, etc. No new dependencies.
