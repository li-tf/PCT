#!/usr/bin/env python3
"""Freeze and evaluate a 2-D pCT experiment without changing source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from common import load_experiment  # noqa: E402
from evaluation_core import (  # noqa: E402
    evaluate_metrics,
    freeze_baseline,
    make_split,
    verify_baseline,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Freeze results0716 and calculate common RSP/validation-WEPL metrics"
    )
    value.add_argument("--experiment", default="0716")
    value.add_argument(
        "--action", choices=("freeze", "split", "metrics", "verify", "all"), default="all"
    )
    value.add_argument("--force", action="store_true", help="replace evaluation-owned outputs only")
    value.add_argument("--batch-size", type=int, default=4096, help="GPU validation batch size")
    value.add_argument("--device", type=int, default=0, help="CUDA device index")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    experiment = load_experiment(args.experiment)
    config = json.loads((HERE / "evaluation_config.json").read_text(encoding="utf-8"))
    baseline_dir = HERE / "baselines" / f"results{args.experiment}"
    qc_dir = HERE / "qc" / f"results{args.experiment}"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    actions = ("freeze", "split", "metrics", "verify") if args.action == "all" else (args.action,)
    completed: list[str] = []
    for action in actions:
        print(f"\n=== evaluation action: {action} ===", flush=True)
        if action == "freeze":
            freeze_baseline(experiment, baseline_dir, args.force)
        elif action == "split":
            make_split(experiment, baseline_dir, config, args.force)
        elif action == "metrics":
            evaluate_metrics(
                experiment, baseline_dir, config, args.force, args.batch_size, args.device
            )
        elif action == "verify":
            verify_baseline(experiment, baseline_dir, qc_dir)
        completed.append(action)
    print(
        json.dumps(
            {
                "status": "PASS",
                "experiment": args.experiment,
                "action": args.action,
                "completed": completed,
                "baseline_dir": str(baseline_dir),
                "qc_dir": str(qc_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
