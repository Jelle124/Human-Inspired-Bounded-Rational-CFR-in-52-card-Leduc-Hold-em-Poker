#!/usr/bin/env python3
"""
Remove prior experiment outputs while keeping replication DQN + CFR checkpoints.

Does **not** rely on deleting the whole ``results/`` tree (Docker may leave root-owned
files under ``parallel_suite``). Instead it removes known output subtrees with
``ignore_errors=True``, clears ``results/training`` except the two checkpoints we
restore from a temp copy, and recreates BR-CFR layout.

Run before a clean parallel-suite run:
  python3 reset_results_for_parallel_suite.py
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile

from br_cfr_config import BR_CFR_BASE, BR_CFR_EVAL_DIR, BR_CFR_TRAIN_DIR
from config import CFR_MODEL_PATH, CFR_VS_BR_SUBDIR, DQN_MODEL_PATH, EVAL_DIR, RESULTS_DIR, TRAIN_DIR


def _rmtree_quiet(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    keep_paths = [p for p in (DQN_MODEL_PATH, CFR_MODEL_PATH) if os.path.isfile(p)]
    if not keep_paths:
        print(
            "WARNING: No replication DQN/CFR found to preserve. "
            "Train replication models before parallel_br_cfr_train_suite.py."
        )

    td = tempfile.mkdtemp(prefix="bluffing_reset_")
    try:
        backups = []
        for path in keep_paths:
            b = os.path.join(td, os.path.basename(path))
            shutil.copy2(path, b)
            backups.append((path, b))

        _rmtree_quiet(EVAL_DIR)
        _rmtree_quiet(CFR_VS_BR_SUBDIR)
        _rmtree_quiet(BR_CFR_EVAL_DIR)
        _rmtree_quiet(BR_CFR_TRAIN_DIR)
        _rmtree_quiet(os.path.join(BR_CFR_BASE, "parallel_suite"))
        _rmtree_quiet(os.path.join(BR_CFR_BASE, "parallel_suite_output"))

        for stray in glob.glob(os.path.join(BR_CFR_BASE, "*.txt")) + glob.glob(
            os.path.join(BR_CFR_BASE, "*.json")
        ):
            try:
                os.remove(stray)
            except OSError:
                pass

        _rmtree_quiet(TRAIN_DIR)
        os.makedirs(TRAIN_DIR, exist_ok=True)
        for orig, b in backups:
            shutil.copy2(b, orig)

        os.makedirs(BR_CFR_EVAL_DIR, exist_ok=True)
        os.makedirs(os.path.join(BR_CFR_BASE, "parallel_suite_output"), exist_ok=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    print(f"Cleaned outputs under {RESULTS_DIR!r}; preserved {len(keep_paths)} checkpoint(s).")
    print(
        "Note: root-owned files under old parallel_suite may remain; "
        "new runs default to parallel_suite_output/ (see parallel_br_cfr_train_suite.py)."
    )


if __name__ == "__main__":
    main()
