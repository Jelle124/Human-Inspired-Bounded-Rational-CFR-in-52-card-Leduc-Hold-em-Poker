#!/usr/bin/env python3
"""
Figure 1 style: DQN vs CFR win rate during simultaneous training (eval games).

Data sources (first match wins):
  1) JSONL from disk (written by simultaneous_training.py since this helper was added)
  2) Optional: Weights & Biases run history (original 100K experiment if logged there)

Examples:
  python plot_simultaneous_training_win_rates.py
  python plot_simultaneous_training_win_rates.py -i results/training/simultaneous_training_win_rates.jsonl
  WANDB_API_KEY=... python plot_simultaneous_training_win_rates.py \\
    --wandb-entity YOUR_ENTITY --wandb-run-name Simultaneous_DQN_CFR_52card_100K
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from config import TRAINING_WIN_RATES_JSONL, TRAIN_EPISODES


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = []
    dqn_wr = []
    cfr_wr = []
    for r in rows:
        if "dqn_win_rate_vs_cfr" not in r or "cfr_win_rate_vs_dqn" not in r:
            continue
        if r["dqn_win_rate_vs_cfr"] is None or r["cfr_win_rate_vs_dqn"] is None:
            continue
        episodes.append(int(r["episode"]))
        dqn_wr.append(float(r["dqn_win_rate_vs_cfr"]))
        cfr_wr.append(float(r["cfr_win_rate_vs_dqn"]))
    if not episodes:
        raise ValueError("No valid win-rate rows found.")
    # Last row wins if the same episode was logged twice
    merged: dict[int, tuple[float, float]] = {}
    for ep, dw, cw in zip(episodes, dqn_wr, cfr_wr):
        merged[int(ep)] = (float(dw), float(cw))
    keys = sorted(merged)
    e = np.array(keys, dtype=float)
    d = np.array([merged[k][0] for k in keys])
    c = np.array([merged[k][1] for k in keys])
    return e, d, c


def _load_from_wandb(entity: str, project: str, run_name: str) -> list[dict[str, Any]]:
    import wandb

    api = wandb.Api(timeout=120)
    run = None
    for r in api.runs(f"{entity}/{project}"):
        if r.name == run_name:
            run = r
            break
    if run is None:
        raise FileNotFoundError(
            f"No run named {run_name!r} in {entity}/{project}. "
            "Check the run name in your W&B dashboard."
        )
    hist = run.history(
        keys=["episode", "dqn_win_rate_vs_cfr", "cfr_win_rate_vs_dqn"],
        pandas=True,
        samples=50_000,
    )
    if hist is None or hist.empty:
        raise ValueError("W&B run history is empty for the requested keys.")
    hist = hist.dropna(subset=["dqn_win_rate_vs_cfr", "cfr_win_rate_vs_dqn"])
    if hist.empty:
        raise ValueError("No rows with both win-rate metrics in W&B history.")
    rows = []
    for _, row in hist.iterrows():
        rows.append({
            "episode": int(row["episode"]),
            "dqn_win_rate_vs_cfr": float(row["dqn_win_rate_vs_cfr"]),
            "cfr_win_rate_vs_dqn": float(row["cfr_win_rate_vs_dqn"]),
        })
    return rows


def plot_figure(
    episodes: np.ndarray,
    dqn_wr: np.ndarray,
    cfr_wr: np.ndarray,
    out_path: str,
    max_episode: int | None,
) -> None:
    if max_episode is None:
        max_episode = int(max(episodes.max(), TRAIN_EPISODES))

    def _ep_fmt(x, _pos):
        x = float(x)
        if abs(x) < 1:
            return "0"
        if x % 1000 == 0 and x >= 1000:
            return f"{int(x // 1000)}k"
        return str(int(x))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    ax0.plot(episodes, dqn_wr, color="tab:blue", linewidth=1.2)
    ax0.set_title("DQN win rate against CFR during Training")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Win Rate")
    ax0.set_xlim(0, max_episode)
    ax0.set_ylim(0.44, 0.56)
    ax0.xaxis.set_major_formatter(FuncFormatter(_ep_fmt))
    ax0.yaxis.grid(True, color="lightgray", linestyle="-", linewidth=0.8)
    ax0.set_axisbelow(True)
    ax0.text(0.5, -0.22, "(a) DQN win rate vs. CFR", transform=ax0.transAxes, ha="center")

    ax1.plot(episodes, cfr_wr, color="tab:red", linewidth=1.2)
    ax1.set_title("CFR win rate against DQN during Training")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Win Rate")
    ax1.set_xlim(0, max_episode)
    ax1.set_ylim(0.44, 0.56)
    ax1.xaxis.set_major_formatter(FuncFormatter(_ep_fmt))
    ax1.yaxis.grid(True, color="lightgray", linestyle="-", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.text(0.5, -0.22, "(b) CFR win rate vs. DQN", transform=ax1.transAxes, ha="center")

    fig.suptitle(
        "Win rate during simultaneous training of DQN and CFR (eval win rate, 2K games per checkpoint)",
        fontsize=10,
        y=1.02,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot simultaneous-training win rates (Fig. 1 style).")
    p.add_argument(
        "-i", "--input",
        default=TRAINING_WIN_RATES_JSONL,
        help=f"JSONL path (default: {TRAINING_WIN_RATES_JSONL})",
    )
    p.add_argument(
        "-o", "--output",
        default=os.path.join(os.path.dirname(TRAINING_WIN_RATES_JSONL), "fig1_simultaneous_training_win_rates.png"),
        help="Output PNG path",
    )
    p.add_argument(
        "--max-episode",
        type=int,
        default=None,
        help="X-axis upper limit (default: max(episodes) or TRAIN_EPISODES from config)",
    )
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"), help="W&B entity / username")
    p.add_argument(
        "--wandb-project",
        default="BNAIC-simultaneous-training-100K",
        help="W&B project (default matches simultaneous_training.py)",
    )
    p.add_argument(
        "--wandb-run-name",
        default="Simultaneous_DQN_CFR_52card_100K",
        help="W&B run display name (default matches simultaneous_training.py)",
    )
    p.add_argument(
        "--save-jsonl",
        default=None,
        help="If set, write merged W&B rows to this JSONL path (e.g. results/training/...)",
    )
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    used_wandb = False

    if os.path.isfile(args.input) and os.path.getsize(args.input) > 0:
        rows = _load_jsonl(args.input)
    elif args.wandb_entity:
        rows = _load_from_wandb(args.wandb_entity, args.wandb_project, args.wandb_run_name)
        used_wandb = True
    else:
        print(
            "No local win-rate JSONL found (or file is empty), and --wandb-entity was not set.\n\n"
            "Options:\n"
            "  • Re-run training: simultaneous_training.py now writes\n"
            f"    {TRAINING_WIN_RATES_JSONL}\n"
            "  • Or pull from W&B (after `wandb login`):\n"
            "    WANDB_ENTITY=your_username python plot_simultaneous_training_win_rates.py \\\n"
            "      --wandb-run-name Simultaneous_DQN_CFR_52card_100K\n",
            file=sys.stderr,
        )
        return 1

    if args.save_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_jsonl)) or ".", exist_ok=True)
        with open(args.save_jsonl, "w", encoding="utf-8") as wf:
            for r in sorted(rows, key=lambda x: int(x["episode"])):
                wf.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows)} rows to {args.save_jsonl}")

    ep, dqn, cfr = _rows_to_arrays(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    plot_figure(ep, dqn, cfr, args.output, args.max_episode)

    src = "W&B" if used_wandb else args.input
    print(f"Plotted {len(ep)} checkpoints from {src} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
