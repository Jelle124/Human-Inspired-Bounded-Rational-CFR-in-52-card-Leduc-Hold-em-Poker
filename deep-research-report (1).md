# Methods

## Executive summary

This thesis uses **52-card Leduc Hold’em** as an imperfect-information poker testbed to study bluffing in (i) **DQN vs CFR** (replication of the original experiment) and (ii) a **Human-Inspired Bounded-Rational CFR (BR‑CFR)** variant. The full pipeline has four parts: (A) replicate the original **simultaneous DQN–CFR co-training** experiment in RLCard with the same custom 52-card Leduc environment and training loop described by the authors and reflected in their released code; (B) train a **self-play CFR baseline** whose frozen **average policy** serves as a stable evaluation opponent; (C) define **BR‑CFR** by implementing bounded-rational mechanisms grounded in behavioral game theory and poker psychology—**soft regret-matching** (Quantal Response), **limited memory** (regret decay), **state abstraction/bucketing** (limited thinking), and **loss-frame / tilt modulation** (reference dependence and loss-of-control episodes); (D) evaluate all agents via **round-robin tournaments** with bluff metrics computed using the same **threshold-based** and **statistics-based** bluff detectors as the original paper.

The methodological choices are justified by: RLCard’s standardized card-game interface and ability to support tree traversal via `step_back` citeturn15view0turn7view7; the original DQN/CFR bluffing study and its 52-card environment modifications and bluff detectors citeturn6view0turn6view1turn6view2turn6view3; the foundational CFR convergence result that **self-play regret minimization yields approximate Nash equilibria** through the **average strategy** citeturn15view1; and behavioral models of bounded rationality and poker-specific behavior (QRE, Cognitive Hierarchy, Prospect Theory, poker field evidence on loss-chasing, and tilt studies). citeturn10view1turn11view0turn12view0turn15view3turn21view0

## Replication of DQN versus CFR in 52-card Leduc Hold’em

### Environment and RLCard-based implementation

All experiments are conducted in an extended version of **Leduc Hold’em** implemented on top of the **RLCard** framework, an open-source toolkit for reinforcement learning research in card games. citeturn15view0turn6view0 RLCard provides standardized multi-agent APIs, including observation dictionaries containing `obs` and `legal_actions`, and supports **game-tree traversal** via `step`/`step_back`, which is essential for CFR-style recursive traversal. citeturn7view7turn3search1

Following the original bluffing study, the RLCard Leduc implementation is modified to use a **full 52-card deck** (13 ranks × 4 suits) and a deterministic judger with suits as tie-breakers to avoid ties. citeturn6view0turn6view1 Consistent with the original paper’s Methods section, the environment modification targets are:

- `LeducholdemGame`: expanded deck initialization and state space citeturn6view0  
- `Dealer`: full 52-card deck dealing citeturn6view0  
- `Judger`: deterministic evaluation with suit tie-breakers citeturn6view0turn6view1  
while the `Player` and `Round` components preserve RLCard’s normal Leduc structure (two betting rounds, fixed-limit actions with a raise cap). citeturn6view0turn6view1

### Fixed game parameters

We retain the same fixed-limit game parameters reported by the original study (Table 1 in that paper): two players; blinds 1/2; raise sizes 2 pre-flop and 4 post-flop; and at most two raises per player per round. citeturn6view1

| Property | Value |
|---|---|
| Players | 2 |
| Deck | 52 cards (4 suits × 13 ranks) |
| Blinds | Small blind = 1, Big blind = 2 |
| Raise size | Pre-flop = 2, Post-flop = 4 |
| Raise cap | Max 2 raises per player per round |

*(Environment parameters as specified in the original DQN/CFR bluffing study.)* citeturn6view1

### DQN implementation and configuration in replication

The DQN agent is instantiated using RLCard’s `DQNAgent` (PyTorch-based). The conceptual basis is the DQN algorithm introduced by Mnih et al., which combines Q-learning with neural function approximation and stabilizing mechanisms such as target networks and replay memory. citeturn15view2turn6view1

The original paper explicitly notes RLCard’s card-game adaptation: illegal actions are masked (in their description, Q-values for illegal actions are set to −∞ to prevent selection). citeturn6view1

For our replication, we follow the authors’ released training script structure and hyperparameterization as reflected in their open-source `simultaneous_training.py` (custom environment registration, DQN initialization, and training loop). citeturn19view0turn17search0 The replication configuration used in the provided code is:

| Hyperparameter | Value (replication code) |
|---|---|
| MLP hidden layers | [256, 256] |
| Learning rate | 5e-5 |
| Batch size | 64 |
| Epsilon end | 0.05 |
| Epsilon decay steps | 50,000 |
| Replay memory init size | 1,000 |
| Replay memory size | 100,000 |
| Discount factor (γ) | RLCard default unless overridden in agent |
| Train episodes | 100,000 |
| Eval interval | 5,000 episodes |
| Eval games per checkpoint | 2,000 |
| Random seed | 42 |
| State dimension (reported) | 156 |
| Deck size | 52 |

*(Values as specified in the provided training script.)* citeturn19view0

> Note: the published paper reports a slightly different DQN hyperparameter table (e.g., ε-decay steps and replay sizes) in its Table 2; for strict reproducibility of *implementation behavior*, this thesis prioritizes the authors’ released code configuration while keeping the paper’s game rules and detection definitions fixed. citeturn6view2turn19view0

### Custom CFR implementation used in replication

The original paper reports that they implement a custom CFR variant (based on RLCard’s CFR) that can train against an externally provided opponent policy, and that evaluation uses the **average policy** accumulated during training. citeturn6view1 This aligns with the CFR principle that average strategies are the stable outputs. citeturn15view1

In the provided replication code, CFR is implemented as `CFRAgainstDQNAgent`, which:

- stores cumulative `regrets` and `average_policy` keyed by an information-state representation (`obs = state['obs'].tobytes()`), citeturn19view0  
- computes strategies via regret matching (positive regrets normalized, else uniform), citeturn19view0  
- traverses the game tree recursively using `env.step(...)` and `env.step_back()` (enabled via `allow_step_back=True`), consistent with RLCard’s CFR traversal support. citeturn19view0turn7view7  

For action selection during *actual games* (training episodes and evaluation games), the code wraps the CFR agent in `CFRWrapper` and samples from the learned `average_policy`, defaulting to uniform when an information state has not been seen. citeturn19view0 This matches the original paper’s emphasis that average-policy play is used for stability. citeturn6view1turn15view1

### Simultaneous training loop and intermediate evaluation

The replication uses **simultaneous co-training** where CFR and DQN adapt to each other, consistent with the original study’s Method description. citeturn6view1turn6view2 Specifically, each episode consists of:

1. CFR performs `iterations_per_episode` recursive traversals (10 in the provided code) to update regrets and the average policy. citeturn19view0turn6view1  
2. One full game is played via `env.run(is_training=True)`, producing trajectories and payoffs. citeturn19view0  
3. DQN updates by feeding replay transitions produced from reorganized trajectories (`reorganize`), matching RLCard’s multi-agent trajectory handling. citeturn19view0turn15view0  

Intermediate evaluation is performed every `eval_interval` episodes (5,000) by running `eval_games` matches (2,000) with `is_training=False` and logging win counts and reward statistics. citeturn19view0turn6view2 Model artifacts are saved at the end of training: DQN weights as a PyTorch state dict and CFR tables via pickle (policy, average policy, regrets, iteration). citeturn19view0

## Self-play CFR baseline for stable evaluation

### Rationale: why a self-play baseline is needed

A CFR policy trained against a simultaneously learning DQN is an opponent-conditioned, non-stationary training outcome. For a stable benchmark, we train a **self-play CFR** baseline in which both players follow CFR updates against each other, yielding an approximation to equilibrium play.

This is justified by the foundational result that in **zero-sum extensive-form games**, if both players’ average overall regret is small, the **average strategy profile** is an approximate Nash equilibrium (ε-equilibrium); and regret-minimizing algorithms in self-play can thus compute approximate Nash equilibria. citeturn15view1

### Training protocol for baseline CFR

We train two identical CFR agents (Player 0 and Player 1) in the same 52-card Leduc environment used in replication, with `allow_step_back=True` to enable recursive traversals. citeturn7view7turn6view1 At each CFR iteration, each player updates regrets at visited information sets and accumulates an average-strategy numerator (strategy sum). The baseline policy used for evaluation is the **average policy** (not the last-iterate regret-matching policy), consistent with CFR theory and with the original paper’s evaluation practice. citeturn15view1turn6view1

### Checkpointing and convergence monitoring

Because exact exploitability computation can be expensive in custom environments, we use a practical convergence criterion based on policy stability and performance:

- **Checkpoint schedule:** save average-policy checkpoints every fixed number of traversals (e.g., every 50k–200k iterations, depending on runtime). citeturn15view1turn7view7  
- **Stability metric:** evaluate each checkpoint against the previous checkpoint and a small fixed opponent suite (e.g., random agent, replication CFR, frozen DQN snapshot) to detect plateaus (changes within sampling noise).  
- **Stopping rule:** stop once win rate and average payoff differences across consecutive checkpoints fall below a small threshold (e.g., <1% absolute win rate change over two checkpoints), indicating diminishing strategy change.

The final chosen baseline opponent is a **frozen checkpoint** of the self-play CFR average policy, used identically across all later evaluation matchups.

## Human-Inspired Bounded-Rational CFR

### Behavioral grounding and design objectives

BR‑CFR is designed to be a **CFR-family agent that remains structurally game-theoretic**, but exhibits systematic bounded-rational deviations that are empirically plausible for humans. The design draws from:

- **Quantal Response Equilibrium (QRE):** players choose higher-payoff actions with higher probability, modeled via logit choice; the error level is controlled by parameter λ, where λ=0 implies fully random behavior and λ→∞ implies negligible error. citeturn10view1  
- **Cognitive Hierarchy (CH):** empirical strategic behavior often reflects limited steps of reasoning; CH defines step‑0 as random and higher steps as best responses to lower steps, with a typical average of ~1.5 steps across many games. citeturn11view0  
- **Prospect Theory:** the certainty effect supports risk aversion in sure gains and risk seeking in sure losses; value is reference-dependent and losses loom larger than gains. citeturn12view0  
- **Poker field evidence on “break-even” loss chasing:** experienced poker players increase risk-taking after losses and become more conservative when ahead; effects can be transitory. citeturn15view3turn9view1  
- **Tilt as loss-of-control episodes:** tilt is described as a period where rational control breaks down, producing impaired decisions, emotional dysregulation, cognitive distortion (“I’ll win my money back”), and financial loss; tilt frequency is empirically linked to gambling risk. citeturn21view0  

Additionally, to emphasize that bluffing is not “pure psychology” but can be equilibrium-rational mixing, we ground the conceptual motivation in Kuhn’s simplified poker analysis showing that optimal play includes bluffing with positive probability. citeturn15view4turn2search3

### BR‑CFR mechanisms

BR‑CFR extends tabular CFR with five mechanisms. The first three are the core bounded-rational CFR modifications; the last two are “human-session dynamics” modules motivated by poker behavior research.

#### Soft regret-matching (QRE-inspired)

Instead of standard regret-matching based on normalized positive regrets, BR‑CFR uses a **softmax (logit) transformation** of regret signals to generate stochastic policies. This implements the same principle as logit QRE—better actions are more likely, but suboptimal actions can still occur, with noise controlled by a single parameter. citeturn10view1turn10view0

Operationally, for information set \(I\) and action \(a\), let \(R^+(I,a)=\max(R(I,a),0)\). BR‑CFR computes

\[
\pi(a\mid I)\propto \exp(\eta \cdot \tilde{R}^+(I,a)),
\]

where \(\eta\) is an inverse-temperature (“precision”) parameter and \(\tilde{R}^+\) is a normalized regret signal (e.g., dividing by \(\sum_a R^+(I,a)+\epsilon\) to stabilize scale). The λ interpretation from QRE motivates treating \(\eta\approx 0\) as highly noisy and larger values as increasingly deterministic. citeturn10view1

#### Regret decay (bounded memory / recency bias)

To model limited memory and recency bias, BR‑CFR applies a decay factor \(\lambda_{\text{decay}}\in(0,1]\) to cumulative regrets (and optionally to strategy sums) at each update step:

\[
R(I,a)\leftarrow \lambda_{\text{decay}}\,R(I,a) + \Delta R(I,a).
\]

This causes older experiences to matter less, producing non-equilibrium but human-plausible instability.

#### State abstraction and bucketing (limited thinking)

To reflect limited cognitive processing and the CH notion that people do not condition behavior on full equilibrium reasoning depth, BR‑CFR reduces information-set granularity through **state abstraction** (bucketing). citeturn11view0turn6view3

Instead of identifying information sets by `obs = state['obs'].tobytes()` (full RLCard feature vector), BR‑CFR maps states into discrete buckets based on a subset of features intended to be “human-salient,” such as:

- betting round (pre-flop / post-flop),
- public card presence and coarse rank bucket,
- private card coarse rank bucket,
- pot size bucket or commitment bucket (e.g., “no money,” “some money,” “high commitment”),
- last aggressive action indicator.

This merges many exact RL states into fewer “cognitive categories,” leading to systematic generalization errors.

#### Loss-frame “mood” modulation (reference dependence + poker field evidence)

BR‑CFR maintains a session-level **reference-dependent mood variable** \(m_t\) updated from recent payoffs:

\[
m_{t+1}=\rho m_t + \text{payoff}_t
\]

where \(\rho\in[0,1)\) controls how quickly effects “wear off,” reflecting empirical findings that much of the loss/gain effect decays relatively quickly. citeturn9view1turn15view3

When \(m_t<0\) (behind), the agent becomes more risk-seeking by increasing raise probability (or adding a positive bias to raise logits). When \(m_t>0\) (ahead), it becomes more conservative by reducing aggression. This specification is motivated by Prospect Theory’s gain/loss asymmetry and by the observed poker “break-even” effect and conservatism when ahead. citeturn12view0turn15view3

#### Tilt mode (episodic loss of control)

Tilt is operationalized as a transient mode where decision noise and aggression increase after triggers such as:

- a loss streak of length \(k\), or
- a large negative payoff shock.

This follows the empirical definition of tilt as an episode where rational control fails, with increased emotional dysregulation and cognitive distortions. citeturn21view0turn14view0

In tilt mode, BR‑CFR applies:
- higher action noise (lower \(\eta\) / higher temperature),
- stronger raise bias,
- optionally a brief reduction in sensitivity to regret magnitudes (“flattened” responses).

### Mapping literature findings to BR‑CFR knobs

| Literature source | Key finding used | BR‑CFR knob | Implementation location |
|---|---|---|---|
| McKelvey & Palfrey (1995) | Logit choice: actions chosen probabilistically by payoff; λ controls error (λ=0 random; λ→∞ near-perfect). citeturn10view1 | Soft regret-matching precision/temperature (\(\eta\) or τ) | `CFRAgainstDQNAgent.regret_matching` (replace ratio normalization with softmax); plus gameplay sampling in `CFRWrapper.step` |
| Camerer, Ho & Chong (2004) | Step‑0 random; higher levels best respond to lower levels; average ~1.5 steps fits many games. citeturn11view0 | Limited reasoning via abstraction + fewer traversals | Info-set bucketing in `obs_key(state)`; training budget settings |
| Kahneman & Tversky (1979) | Certainty effect: risk aversion in sure gains, risk seeking in sure losses; steeper value for losses. citeturn12view0 | Mood-based gain/loss modulation | Action-probability bias based on \(m_t\) |
| Eil & Lien (2014) | “Break-even effect”: losses increase risk-taking; gains make players more conservative; effect decays. citeturn15view3turn9view1 | Mood decay parameter \(\rho\) + loss-chasing raise bias | Update after each hand; apply biases in `CFRWrapper.step` |
| Moreau et al. (2020) | Tilt: episode of loss of rational control; linked to cognitive distortion; predicts harmful gambling indicators. citeturn21view0 | Tilt mode trigger + duration; increased noise and aggression | Gameplay wrapper + session state |
| Kuhn (1951) | Even simplified poker yields optimal bluffing with nonzero probability; bluffing is structural. citeturn15view4turn2search3 | Sanity constraint: BR‑CFR should still bluff sometimes | Used to justify nonzero bluff rate in model goals |

### BR‑CFR pipeline overview

```mermaid
flowchart TD
  A[Train/eval environment: 52-card Leduc in RLCard] --> B[State abstraction: bucketed info-set key]
  B --> C[Regret table + strategy sums (tabular CFR)]
  C --> D[Soft regret matching (logit / QRE-inspired)]
  D --> E[Mood modulation (loss-chasing / conservatism)]
  E --> F{Tilt trigger?}
  F -- no --> G[Sample legal action]
  F -- yes --> H[Tilt mode: higher noise + aggression]
  H --> G
  G --> I[Execute action in env]
  I --> J[Update regrets & average policy]
  J --> C
  I --> K[Bluff detection + logging]
```

## Training procedures for BR‑CFR

### Training setting

BR‑CFR is trained in **self-play** (BR‑CFR vs BR‑CFR) in the same 52-card Leduc environment used for replication, with step-back enabled for recursive traversal consistency. citeturn7view7turn6view1 Self-play is chosen to avoid training BR‑CFR as a narrow “exploiter” of one particular opponent and to isolate which behaviors emerge from bounded-rational mechanisms rather than opponent idiosyncrasies. This aligns with CFR’s theoretical framing as self-play regret minimization for equilibrium approximation. citeturn15view1

### Required code modifications (where the BR‑CFR changes are implemented)

The provided replication code contains two natural hooks for implementing BR‑CFR:

1. **Training-time policy computation:** modify `CFRAgainstDQNAgent.regret_matching` and the way information states are keyed (`obs = state['obs'].tobytes()`). citeturn19view0turn6view3  
2. **Gameplay-time action sampling:** modify `CFRWrapper.step`, which currently samples from `average_policy` with legal-action masking and renormalization. citeturn19view0turn6view1  

A standard implementation pattern is:
- keep CFR’s regret updates intact (tabular), but
- implement “human-like execution” (noise, mood, tilt) primarily in `CFRWrapper.step`, so the learned policy can still be meaningfully interpreted as “training output,” and bounded rationality is an explicit overlay.

### BR‑CFR hyperparameters and recommended ranges

The thesis treats BR‑CFR as a parameterized family (not one fixed agent). We report results for three representative regimes (e.g., *dumb / medium / smart bounded rationality*) by varying a small number of interpretable knobs:

- **QRE precision \(\eta\):** low \(\eta\) produces near-uniform behavior; high \(\eta\) approaches deterministic regret matching, consistent with QRE’s λ semantics. citeturn10view1  
- **Regret decay \(\lambda_{\text{decay}}\):** closer to 1.0 means long memory; lower values emphasize recency.
- **Abstraction level:** number of rank buckets and pot/betting buckets.
- **Mood decay \(\rho\):** controls how quickly “being behind/ahead” effects wear off, consistent with the transitory nature of the field effects. citeturn9view1turn15view3  
- **Tilt trigger and duration:** consistent with tilt as episodic loss of control. citeturn21view0turn14view0  

## Evaluation protocol and bluff detection

### Tournament evaluation protocol

We evaluate agents using round-robin tournaments with **10,000 games per matchup** (fixed opponents; no learning during evaluation). Tournament size is chosen to reduce Monte Carlo noise in win-rate and bluff-rate estimates. Final evaluation includes at least:

- DQN (replication trained) citeturn6view2turn19view0  
- CFR trained against DQN (replication CFR) citeturn6view2turn19view0  
- CFR self-play baseline (frozen average policy) citeturn15view1  
- BR‑CFR variants (e.g., dumb/medium/smart)  

Primary strength metrics are:
- **Win rate**
- **Average payoff per hand**

### Bluff detection methods

To remain directly comparable to the original bluffing study, we implement the same two definitions: a **threshold-based detector** and a **statistics-based detector**. citeturn6view2turn6view3

#### Threshold-based detector

The paper defines a hand-strength score:

\[
\text{HandScore} =
\begin{cases}
(R_{pc}\times 4)+S_{pc}, & \text{if no pair},\\
(R_{pc}\times 4)+S_{pc}+1000, & \text{if there is a pair}.
\end{cases}
\]

Here \(R_{pc}\) is the rank index of the private card and \(S_{pc}\) is its suit index. citeturn6view2turn6view3

A **bluff attempt** is recorded when an agent takes a **raise** action while holding a weak hand (HandScore ≤ 32, corresponding to “less than 10s and no pairs” under their encoding). citeturn6view2 A **successful bluff** is recorded if the opponent folds immediately in response. citeturn6view2

We report:
- bluff attempt rate (attempts / decision opportunities),
- bluff success rate (successful bluffs / bluff attempts),
- bluff distribution by private rank bucket.

#### Statistics-based detector

The paper also formalizes bluffing as a combination of **misrepresentation** and **EV preference**. Let \(s(h)\) be a hand-strength function and \(u(h,a)\) denote expected utility (EV) of acting with hand \(h\) at public context \(pc\). Let \(\mu(h' \mid a, pc)\) be the observer’s belief distribution over the actor’s private hand after seeing action \(a\). A bluff occurs if the action suggests strength greater than the actual hand and if the action is utility-preferred:

\[
s(h) < \mathbb{E}_{h'\sim \mu(\cdot \mid a,pc)}[s(h')]
\quad \text{and} \quad
u(h,a) > u(h,a_{\text{passive}}).
\]

citeturn6view3

Because computing full belief distributions is expensive, the original study approximates this by summarizing the empirical distribution of strengths observed for the same action/context using means and standard deviations, and by using payoffs as EV; we follow their described approximation procedure to maintain comparability. citeturn6view3

### Additional behavioral metrics

Beyond raw counts, we compute:

- **Hand-strength distribution of bluffs:** frequency of bluff attempts conditioned on (bucketed) private rank and pair/non-pair status. citeturn6view3  
- **Phase-conditional bluffing:** pre-flop vs post-flop bluff rates to capture timing effects.
- **Classifier detectability:** train a lightweight classifier on sequences of public context + actions to predict “AI-equilibrium-like” (CFR baseline) vs “human-like” (BR‑CFR), treating above-chance accuracy as a proxy for behavioral distinguishability.

## Reproducibility, seeds, checkpoints, and artifacts

To ensure reproducibility:

- **Randomness control:** we fix seeds at the environment and framework level using RLCard’s seed setter in all training and evaluation runs (replication code uses seed 42). citeturn19view0turn7view7  
- **Deterministic judging:** suits are used as tie-breakers to eliminate ties in winner determination, matching the original 52-card Leduc modifications. citeturn6view0turn6view1  
- **Checkpointing:** DQN weights are saved as model state dicts; CFR/BR‑CFR are saved as serialized tables containing regrets and average policy. citeturn19view0turn6view1  
- **Logged artifacts:** training curves (win rate checkpoints, regret totals, states seen) and evaluation metrics are logged (e.g., via experiment tracking tools), mirroring the replication script’s logging structure. citeturn19view0  
- **Open-source reference implementation:** the replication baseline is grounded in the authors’ publicly released code repository referenced by the paper. citeturn17search0turn19view0turn6view0  

### Prioritized citations used in this Methods chapter

The sources below are the “load-bearing” citations for this Methods design:

- Original study (environment modifications, training design, bluff detectors): citeturn6view0turn6view1turn6view2turn6view3  
- Replication code artifact (exact training loop and parameterization used): citeturn19view0turn17search0  
- RLCard toolkit and traversal support (`step_back`): citeturn15view0turn7view7  
- CFR regret minimization and average-strategy equilibrium guarantee: citeturn15view1  
- QRE (logit, λ controls noise): citeturn10view1turn10view0  
- Cognitive Hierarchy (step‑0 random; limited depth): citeturn11view0  
- Prospect Theory (gains vs losses; risk seeking in losses): citeturn12view0  
- Poker field evidence for loss-chasing and “ahead” conservatism: citeturn15view3turn9view1  
- Tilt definition and correlates: citeturn21view0turn14view0  
- Bluffing as equilibrium mixing (conceptual anchor): citeturn15view4turn2search3