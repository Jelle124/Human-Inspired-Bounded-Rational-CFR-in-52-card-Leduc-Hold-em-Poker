#!/usr/bin/env python3
"""
Run four training jobs in parallel (multiprocessing):

  1. BR-CFR **smart**  (vs frozen replication DQN) — params from `br_cfr_variant_specs`
  2. BR-CFR **medium** — same
  3. BR-CFR **dumb** — same
  4. **Baseline CFR vs CFR** self-play (tabular CFR, `obs.tobytes()`, hard regret matching)

The baseline is classical CFR (no bounded-rational extras). It uses **Smart’s**
`iterations_per_episode` (10) and the same outer-episode count as the BR jobs
(`PARALLEL_TRAIN_EPISODES` / `BR_CFR_TRAIN_EPISODES`); BR-specific knobs do not exist in that agent.

Requires a trained DQN at `config.DQN_MODEL_PATH` (same as `train_br_cfr.py`).

Docker (mount repo so `/app/parallel_br_cfr_train_suite.py` exists without rebuilding;
use PARALLEL_SEQUENTIAL=1 on tight memory—runs four jobs sequentially, not four processes):

  docker run --shm-size=2g \\
    -v "$(pwd):/app" -v "$(pwd)/results:/app/results" \\
    -e PARALLEL_SEQUENTIAL=1 \\
    bluffing-leduc python parallel_br_cfr_train_suite.py

Alternatively rebuild after code changes: `docker build -t bluffing-leduc .`

Quick test:

  PARALLEL_TRAIN_EPISODES=500 python parallel_br_cfr_train_suite.py

Optional:

  PARALLEL_SUITE_DIR=/app/results/br_cfr/my_suite  # override output folder
  PARALLEL_MAX_WORKERS=4
  PARALLEL_SEQUENTIAL=1   # run jobs one after another (low RAM / debug)

Win-rate logging (graph-friendly):

  results/.../parallel_suite_output/training_win_rates/*.jsonl — one JSON object per line
    every PARALLEL_TRAIN_EVAL_INTERVAL episodes (default 10_000), plus a final row
    at the end of training. Mid-run sample size: PARALLEL_TRAIN_EVAL_GAMES (default 10_000).

  results/.../parallel_suite_output/parallel_suite_training_win_rates_wide.csv — merged long
    table from all JSONL files (written when main() finishes).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import torch
from rlcard.agents import DQNAgent
from rlcard.utils import set_seed

from br_cfr_agent import BoundedRationalCFRAgent, BRCFRWrapper
from br_cfr_config import BR_CFR_BASE, BR_CFR_EVAL_GAMES, BR_CFR_TRAIN_EPISODES
from br_cfr_variant_specs import BR_CFR_VARIANT_PARAM_TABLE, get_merged_variant_kwargs
from config import DQN_MODEL_PATH
from custom_leduc_rlcard.leducholdem import LeducholdemEnv
# Side effect: registers `custom-leduc-holdem` with RLCard (do not register again here).
from simultaneous_training_cfr_vs_br_smart import (
    CFRAgainstOpponentAgent,
    CFRWrapper,
    _OpponentSlot,
)

DEFAULT_SEED = int(os.environ.get("PARALLEL_SEED", "42"))

# Mid-training eval: episode checkpoints and games per checkpoint (final paper eval is separate).
_DEFAULT_TRAIN_EVAL_INTERVAL = 10_000
_DEFAULT_TRAIN_EVAL_GAMES = 10_000


def _train_eval_interval() -> int:
    return int(os.environ.get("PARALLEL_TRAIN_EVAL_INTERVAL", str(_DEFAULT_TRAIN_EVAL_INTERVAL)))


def _train_eval_games_mid() -> int:
    return int(os.environ.get("PARALLEL_TRAIN_EVAL_GAMES", str(_DEFAULT_TRAIN_EVAL_GAMES)))


def _append_win_rate_jsonl(log_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _suite_output_dir() -> str:
    # Default ``parallel_suite_output`` avoids clashing with Docker root-owned ``parallel_suite/``.
    d = os.environ.get(
        "PARALLEL_SUITE_DIR",
        os.path.join(BR_CFR_BASE, "parallel_suite_output"),
    )
    os.makedirs(d, exist_ok=True)
    probe = os.path.join(d, ".write_probe")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("1")
        os.remove(probe)
    except OSError as e:
        raise SystemExit(
            f"Output dir {d!r} is not writable ({e}). "
            "If a previous Docker run created root-owned files, run:\n"
            f"  sudo chown -R \"$USER:$(id -gn)\" {BR_CFR_BASE!r}\n"
            "or set PARALLEL_SUITE_DIR to a writable directory."
        ) from e
    return d


def train_br_cfr_table_variant(
    variant: str,
    save_path: str,
    train_episodes: int,
    eval_games: int,
    seed: int,
) -> str:
    """Train one BR-CFR agent vs frozen DQN; save pickle to save_path."""
    # eval_games is kept for API compatibility; mid-training sample size is PARALLEL_TRAIN_EVAL_GAMES.
    _ = eval_games
    set_seed(seed)
    if not os.path.exists(DQN_MODEL_PATH):
        raise FileNotFoundError(
            f"DQN not found at {DQN_MODEL_PATH}. Train replication DQN first."
        )

    agent_kw = get_merged_variant_kwargs(variant)
    iterations = int(agent_kw["iterations_per_episode"])

    env = LeducholdemEnv(config={"seed": seed, "allow_step_back": True})
    env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[256, 256],
        device=device,
    )
    dqn_agent.q_estimator.qnet.load_state_dict(
        torch.load(DQN_MODEL_PATH, map_location=device)
    )
    dqn_agent.q_estimator.qnet.eval()
    for p in dqn_agent.q_estimator.qnet.parameters():
        p.requires_grad = False

    br = BoundedRationalCFRAgent(
        env,
        player_id=1,
        opponent_agent=dqn_agent,
        model_path=save_path,
        **agent_kw,
    )
    env.set_agents([dqn_agent, BRCFRWrapper(br)])

    eval_interval = max(1, _train_eval_interval())
    eval_games_mid = max(1, _train_eval_games_mid())
    out_dir = os.path.dirname(save_path)
    curve_dir = os.path.join(out_dir, "training_win_rates")
    curve_path = os.path.join(curve_dir, f"br_cfr_{variant}.jsonl")

    def _log_eval(episode_done: int) -> None:
        wins = [0, 0, 0]
        for _ in range(eval_games_mid):
            _, payoffs = env.run(is_training=False)
            if payoffs[0] > payoffs[1]:
                wins[0] += 1
            elif payoffs[1] > payoffs[0]:
                wins[1] += 1
            else:
                wins[2] += 1
        n = float(eval_games_mid)
        row = {
            "kind": "br_cfr",
            "variant": variant,
            "episode": episode_done,
            "eval_games": eval_games_mid,
            "dqn_wr": wins[0] / n,
            "policy_wr": wins[1] / n,
            "draw_wr": wins[2] / n,
            "seed": seed,
        }
        _append_win_rate_jsonl(curve_path, row)
        print(
            f"[{variant}] eval ep={episode_done:,} "
            f"DQN_WR={row['dqn_wr']:.4f} BR_WR={row['policy_wr']:.4f} "
            f"(n={eval_games_mid}) → {curve_path}",
            flush=True,
        )

    t0 = time.time()
    print(f"[{variant}] start → {save_path}", flush=True)
    for episode in range(train_episodes):
        for _ in range(iterations):
            env.reset()
            root_u = br.traverse_tree(1.0 * np.ones(env.num_players))
            br.register_hand_outcome(float(root_u[br.player_id]))
            br.iteration += 1

        if episode % 1000 == 0:
            print(
                f"[{variant}] ep={episode:,} iters={br.iteration:,} "
                f"states={len(br.average_policy):,}",
                flush=True,
            )

        if episode > 0 and episode % eval_interval == 0:
            _log_eval(episode)

    _log_eval(train_episodes)

    br.save()
    elapsed = time.time() - t0
    print(f"[{variant}] done in {elapsed/60:.1f} min → {save_path}", flush=True)
    return save_path


def train_cfr_selfplay_baseline(
    save_path: str,
    train_episodes: int,
    iterations_per_episode: int,
    seed: int,
) -> str:
    """
    Symmetric tabular CFR self-play (player 0 / player 1). Saves player-0 tables
    to `save_path` (same pickle shape as replication CFR for evaluation tools).
    """
    set_seed(seed)
    env = LeducholdemEnv(config={"seed": seed, "allow_step_back": True})
    env.reset()

    partner_path = save_path + ".partner_tmp.pkl"
    slot0 = _OpponentSlot()
    slot1 = _OpponentSlot()
    cfr0 = CFRAgainstOpponentAgent(env, 0, slot1, save_path)
    cfr1 = CFRAgainstOpponentAgent(env, 1, slot0, partner_path)
    slot0.wrapped = CFRWrapper(cfr1)
    slot1.wrapped = CFRWrapper(cfr0)
    env.set_agents([CFRWrapper(cfr0), CFRWrapper(cfr1)])

    eval_interval = max(1, _train_eval_interval())
    eval_games_mid = max(1, _train_eval_games_mid())
    out_dir = os.path.dirname(save_path)
    curve_dir = os.path.join(out_dir, "training_win_rates")
    curve_path = os.path.join(curve_dir, "cfr_selfplay_baseline.jsonl")

    def _log_eval(episode_done: int) -> None:
        wins = [0, 0, 0]
        for _ in range(eval_games_mid):
            _, payoffs = env.run(is_training=False)
            if payoffs[0] > payoffs[1]:
                wins[0] += 1
            elif payoffs[1] > payoffs[0]:
                wins[1] += 1
            else:
                wins[2] += 1
        n = float(eval_games_mid)
        row = {
            "kind": "cfr_baseline",
            "variant": None,
            "episode": episode_done,
            "eval_games": eval_games_mid,
            "player0_wr": wins[0] / n,
            "player1_wr": wins[1] / n,
            "draw_wr": wins[2] / n,
            "seed": seed,
        }
        _append_win_rate_jsonl(curve_path, row)
        print(
            f"[cfr_baseline] eval ep={episode_done:,} "
            f"P0_WR={row['player0_wr']:.4f} P1_WR={row['player1_wr']:.4f} "
            f"(n={eval_games_mid}) → {curve_path}",
            flush=True,
        )

    t0 = time.time()
    print(f"[cfr_baseline] CFR vs CFR self-play → {save_path}", flush=True)
    for episode in range(train_episodes):
        for _ in range(iterations_per_episode):
            env.reset()
            cfr0.traverse_tree(np.ones(env.num_players))
            cfr0.iteration += 1
            env.reset()
            cfr1.traverse_tree(np.ones(env.num_players))
            cfr1.iteration += 1
        if episode % 1000 == 0:
            print(
                f"[cfr_baseline] ep={episode:,} "
                f"p0_iters={cfr0.iteration:,} p1_iters={cfr1.iteration:,} "
                f"states0={len(cfr0.average_policy):,}",
                flush=True,
            )

        if episode > 0 and episode % eval_interval == 0:
            _log_eval(episode)

    _log_eval(train_episodes)

    cfr0.save()
    if os.path.isfile(partner_path):
        try:
            os.remove(partner_path)
        except OSError:
            pass
    elapsed = time.time() - t0
    print(f"[cfr_baseline] done in {elapsed/60:.1f} min → {save_path}", flush=True)
    return save_path


def _merge_training_win_rates_csv(out_dir: str) -> str | None:
    """Combine all training_win_rates/*.jsonl into one sorted CSV for plotting."""
    pattern = os.path.join(out_dir, "training_win_rates", "*.jsonl")
    files = sorted(glob(pattern))
    if not files:
        return None
    rows: list[dict] = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        return None
    rows.sort(
        key=lambda r: (
            str(r.get("variant") or r.get("kind", "")),
            int(r.get("episode", 0)),
        )
    )
    outp = os.path.join(out_dir, "parallel_suite_training_win_rates_wide.csv")
    fieldnames = sorted({k for row in rows for k in row})
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Merged training curves: {outp}", flush=True)
    return outp


def _job_entry(payload: dict) -> dict:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    kind = payload["kind"]
    try:
        if kind == "br_cfr":
            path = train_br_cfr_table_variant(
                payload["variant"],
                payload["save_path"],
                payload["train_episodes"],
                payload["eval_games"],
                payload["seed"],
            )
        elif kind == "cfr_baseline":
            path = train_cfr_selfplay_baseline(
                payload["save_path"],
                payload["train_episodes"],
                payload["iterations_per_episode"],
                payload["seed"],
            )
        else:
            raise ValueError(f"unknown job kind {kind!r}")
        return {"ok": True, "path": path, "kind": kind, "variant": payload.get("variant")}
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "kind": kind,
            "variant": payload.get("variant"),
        }


def main():
    # One episode budget for all four jobs (override with PARALLEL_TRAIN_EPISODES)
    te = int(
        os.environ.get(
            "PARALLEL_TRAIN_EPISODES",
            os.environ.get("BR_CFR_TRAIN_EPISODES", str(BR_CFR_TRAIN_EPISODES)),
        )
    )

    eval_games = int(os.environ.get("BR_CFR_EVAL_GAMES", BR_CFR_EVAL_GAMES))
    seed = DEFAULT_SEED
    out_dir = _suite_output_dir()

    smart_iters = BR_CFR_VARIANT_PARAM_TABLE["smart"]["iterations_per_episode"]

    jobs = [
        {
            "kind": "br_cfr",
            "variant": "smart",
            "save_path": os.path.join(out_dir, "br_cfr_smart_table.pkl"),
            "train_episodes": te,
            "eval_games": eval_games,
            "seed": seed,
        },
        {
            "kind": "br_cfr",
            "variant": "medium",
            "save_path": os.path.join(out_dir, "br_cfr_medium_table.pkl"),
            "train_episodes": te,
            "eval_games": eval_games,
            "seed": seed,
        },
        {
            "kind": "br_cfr",
            "variant": "dumb",
            "save_path": os.path.join(out_dir, "br_cfr_dumb_table.pkl"),
            "train_episodes": te,
            "eval_games": eval_games,
            "seed": seed,
        },
        {
            "kind": "cfr_baseline",
            "variant": None,
            "save_path": os.path.join(out_dir, "cfr_selfplay_baseline_smart_iters.pkl"),
            "train_episodes": te,
            "iterations_per_episode": smart_iters,
            "seed": seed,
        },
    ]

    print("=" * 70)
    print("PARALLEL BR-CFR SUITE + CFR SELF-PLAY BASELINE")
    print("=" * 70)
    print(f"Output directory: {out_dir}")
    print(f"Train episodes (each job): {te:,}")
    print(f"Baseline CFR iters/episode: {smart_iters} (matches Smart table row)")
    print(f"Seed: {seed}")
    print(
        f"Mid-train eval: every {_train_eval_interval():,} ep, "
        f"{_train_eval_games_mid():,} games/sample"
    )
    print("=" * 70)

    manifest = {
        "train_episodes_each_job": te,
        "parallel_train_eval_interval": _train_eval_interval(),
        "parallel_train_eval_games": _train_eval_games_mid(),
        "br_cfr_eval_games_config": eval_games,
        "seed": seed,
        "out_dir": out_dir,
        "smart_iters_per_episode": smart_iters,
    }
    with open(os.path.join(out_dir, "parallel_suite_run_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    sequential = os.environ.get("PARALLEL_SEQUENTIAL", "").lower() in ("1", "true", "yes")
    max_workers = int(os.environ.get("PARALLEL_MAX_WORKERS", "4"))

    if sequential:
        results = [_job_entry(j) for j in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_job_entry, j): j for j in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())

    _merge_training_win_rates_csv(out_dir)

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print("\n" + "=" * 70)
    print(f"Finished: {len(ok)} ok, {len(bad)} failed")
    for r in results:
        if r.get("ok"):
            print(f"  OK  {r.get('kind')} {r.get('variant') or ''} → {r.get('path')}")
        else:
            print(f"  ERR {r.get('kind')} {r.get('variant') or ''}: {r.get('error')}")
    print("=" * 70)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
