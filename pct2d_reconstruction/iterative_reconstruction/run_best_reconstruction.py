#!/usr/bin/env python3
"""Run the frozen Stage-4 best configuration on another 2-D pCT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "best_reconstruction_config.json"
RUNNER = HERE / "run_iterative_reconstruction.py"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="safe name used for code-side QC")
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--initial-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        default="0716",
        help="experiment metadata used for paths/optional truth metrics",
    )
    parser.add_argument("--runs", type=int, default=720)
    parser.add_argument("--angle-step-deg", type=float, default=0.5)
    parser.add_argument("--phantom-radius-mm", type=float, default=100.0)
    parser.add_argument(
        "--air-wepl-slope",
        type=float,
        default=0.0,
        help="calibrated external Air correction; Stage-1 value is 0.00114710",
    )
    parser.add_argument(
        "--wepl-model",
        choices=["bb78", "g4_water_calibrated"],
        default="bb78",
    )
    parser.add_argument(
        "--wepl-calibration",
        type=Path,
        help="frozen Stage-6B range table for g4_water_calibrated",
    )
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--grid-spacing-mm", type=float)
    parser.add_argument("--path-step-mm", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--with-truth-metrics",
        action="store_true",
        help="use experiment-specific truth/ROI files after reconstruction",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the delegated command only"
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_name):
        raise SystemExit("--run-name must contain only letters, digits, _, -, or .")
    if (
        args.runs < 1
        or args.angle_step_deg <= 0
        or args.phantom_radius_mm <= 0
        or args.air_wepl_slope < 0
    ):
        raise SystemExit("runs, angle step, and support radius must be positive")
    if not args.pairs_dir.is_dir():
        raise FileNotFoundError(args.pairs_dir)
    if not args.initial_image.is_file():
        raise FileNotFoundError(args.initial_image)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = config["reconstruction"]
    overrides = {
        "--grid-size": args.grid_size,
        "--grid-spacing-mm": args.grid_spacing_mm,
        "--path-step-mm": args.path_step_mm,
        "--batch-size": args.batch_size,
    }
    invalid = [name for name, value in overrides.items() if value is not None and value <= 0]
    if invalid:
        raise SystemExit(f"{', '.join(invalid)} must be positive when specified")
    if args.runs < int(frozen["subsets"]):
        raise SystemExit(
            f"--runs must be at least the frozen subset count ({frozen['subsets']})"
        )
    qc_dir = HERE / "qc" / "best_runs" / args.run_name
    command = [
        sys.executable,
        str(RUNNER),
        "--experiment",
        args.experiment,
        "--pairs-dir",
        str(args.pairs_dir),
        "--initial-image",
        str(args.initial_image),
        "--output-dir",
        str(args.output_dir),
        "--qc-dir",
        str(qc_dir),
        "--runs",
        str(args.runs),
        "--angle-step-deg",
        str(args.angle_step_deg),
        "--phantom-radius-mm",
        str(args.phantom_radius_mm),
        "--air-wepl-slope",
        str(args.air_wepl_slope),
        "--wepl-model",
        args.wepl_model,
        "--epochs",
        str(frozen["epochs"]),
        "--sample-fraction",
        str(frozen["sample_fraction"]),
        "--grid-size",
        str(args.grid_size or frozen["grid_size"]),
        "--grid-spacing-mm",
        str(args.grid_spacing_mm or frozen["grid_spacing_mm"]),
        "--path-step-mm",
        str(args.path_step_mm or frozen["path_step_mm"]),
        "--batch-size",
        str(args.batch_size or frozen["batch_size"]),
        "--subsets",
        str(frozen["subsets"]),
        "--relaxation",
        str(frozen["relaxation"]),
        "--relaxation-decay",
        str(frozen["relaxation_decay"]),
        "--initialization",
        "fdk_nohann",
        "--device",
        str(args.device),
        "--regularizer",
        str(frozen["regularizer"]),
        "--regularization-weight",
        str(frozen["regularization_weight"]),
        "--regularization-iterations",
        str(frozen["regularization_iterations"]),
        "--regularization-every-epochs",
        str(frozen["regularization_every_epochs"]),
        "--huber-delta",
        str(frozen["huber_delta"]),
        "--primal-step",
        str(frozen["primal_step"]),
        "--dual-step",
        str(frozen["dual_step"]),
    ]
    if args.wepl_calibration is not None:
        command.extend(["--wepl-calibration", str(args.wepl_calibration)])
    if not args.with_truth_metrics:
        command.append("--skip-truth-metrics")
    if args.force:
        command.append("--force")
    print("Frozen best configuration:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("\nDelegated command:")
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
