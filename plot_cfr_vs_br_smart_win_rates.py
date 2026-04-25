#!/usr/bin/env python3
"""
Plot eval win rates from simultaneous_training_cfr_vs_br_smart.py (JSONL).

Same two-panel layout as plot_simultaneous_training_win_rates.py, for co-trained
CFR vs BR-CFR (smart preset). Default JSONL path is set in config.py
(`cfr_vs_br_smart_colearn_training_win_rates.jsonl`).

Usage:
  python plot_cfr_vs_br_smart_win_rates.py
  python plot_cfr_vs_br_smart_win_rates.py -i results/cfr_vs_br_smart/training/cfr_vs_br_smart_training_win_rates.jsonl
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

from config import CFR_VS_BR_WIN_RATES_JSONL, TRAIN_EPISODES


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rows_to_arrays(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = []
    cfr_wr = []
    br_wr = []
    for r in rows:
        if "cfr_new_win_rate_vs_br_smart" not in r or "br_smart_win_rate_vs_cfr_new" not in r:
            continue
        episodes.append(int(r["episode"]))
        cfr_wr.append(float(r["cfr_new_win_rate_vs_br_smart"]))
        br_wr.append(float(r["br_smart_win_rate_vs_cfr_new"]))
    if not episodes:
        raise ValueError("No valid win-rate rows (expected keys from cfr_vs_br_smart JSONL).")
    merged: dict[int, tuple[float, float]] = {}
    for ep, cw, bw in zip(episodes, cfr_wr, br_wr):
        merged[int(ep)] = (float(cw), float(bw))
    keys = sorted(merged)
    e = np.array(keys, dtype=float)
    c = np.array([merged[k][0] for k in keys])
    b = np.array([merged[k][1] for k in keys])
    return e, c, b


def plot_figure(
    episodes: np.ndarray,
    cfr_wr: np.ndarray,
    br_wr: np.ndarray,
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

    ax0.plot(episodes, cfr_wr, color="tab:blue", linewidth=1.2)
    ax0.set_title("CFR win rate vs BR-CFR (smart) during co-training")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("Win Rate")
    ax0.set_xlim(0, max_episode)
    ax0.set_ylim(0.44, 0.56)
    ax0.xaxis.set_major_formatter(FuncFormatter(_ep_fmt))
    ax0.yaxis.grid(True, color="lightgray", linestyle="-", linewidth=0.8)
    ax0.set_axisbelow(True)
    ax0.text(0.5, -0.22, "(a) CFR vs BR-smart", transform=ax0.transAxes, ha="center")

    ax1.plot(episodes, br_wr, color="tab:red", linewidth=1.2)
    ax1.set_title("BR-CFR (smart) win rate vs CFR during co-training")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Win Rate")
    ax1.set_xlim(0, max_episode)
    ax1.set_ylim(0.44, 0.56)
    ax1.xaxis.set_major_formatter(FuncFormatter(_ep_fmt))
    ax1.yaxis.grid(True, color="lightgray", linestyle="-", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.text(0.5, -0.22, "(b) BR-smart vs CFR", transform=ax1.transAxes, ha="center")

    fig.suptitle(
        "Win rate while co-training CFR and BR-CFR smart from scratch (eval, 2K games per checkpoint)",
        fontsize=10,
        y=1.02,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot CFR vs BR-smart training win rates.")
    p.add_argument(
        "-i",
        "--input",
        default=CFR_VS_BR_WIN_RATES_JSONL,
        help=f"JSONL path (default: {CFR_VS_BR_WIN_RATES_JSONL})",
    )
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(
            os.path.dirname(CFR_VS_BR_WIN_RATES_JSONL),
            "fig_cfr_vs_br_smart_win_rates.png",
        ),
        help="Output PNG path",
    )
    p.add_argument(
        "--max-episode",
        type=int,
        default=None,
        help="X-axis upper limit",
    )
    args = p.parse_args()

    if not os.path.isfile(args.input) or os.path.getsize(args.input) == 0:
        print(f"No data at {args.input}. Run simultaneous_training_cfr_vs_br_smart.py first.", file=sys.stderr)
        return 1

    rows = _load_jsonl(args.input)
    ep, cfr_w, br_w = _rows_to_arrays(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    plot_figure(ep, cfr_w, br_w, args.output, args.max_episode)
    print(f"Plotted {len(ep)} checkpoints from {args.input} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
