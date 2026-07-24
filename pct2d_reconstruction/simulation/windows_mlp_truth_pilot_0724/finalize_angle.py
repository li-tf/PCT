#!/usr/bin/env python3
"""Merge and validate one completed MLP-truth angle in a fresh process."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from run_angle import (
    config_sha256,
    inspect_reference,
    inspect_trajectory,
    load_config,
    merge_trajectory_parts,
    validate_config,
)


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "simulation_config.json")
    parser.add_argument("--angle", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    parser.add_argument("--protons-per-projection", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clock = time.perf_counter()
    config_path = args.config.resolve()
    config = load_config(config_path)
    protons = int(
        args.protons_per_projection or config["protons_per_projection"]
    )
    validate_config(config, args.angle, protons)
    output_dir = args.output_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    metadata_path = qc_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("simulation metadata is missing")
    record = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if record.get("status") != "simulation_completed":
        raise RuntimeError(
            f"simulation phase is not complete: {record.get('status')}"
        )
    if (
        record.get("config_sha256") != config_sha256(config_path)
        or int(record.get("protons_per_projection", -1)) != protons
        or int(record.get("angle_index", -1)) != args.angle
    ):
        raise RuntimeError("metadata does not match this finalization request")

    summaries = {}
    for name in ("PhaseSpaceIn", "PhaseSpaceOut"):
        path = output_dir / f"{name}.root"
        if not path.is_file():
            raise RuntimeError(f"missing output: {path.name}")
        summaries[path.name] = inspect_reference(path, name)

    parts = [
        (str(item["tree"]), Path(item["path"]))
        for item in record["trajectory_part_files"]
    ]
    trajectory_name = str(config["trajectory_actor_name"])
    trajectory_path = output_dir / f"{trajectory_name}.root"
    parts_exist = [path.is_file() for _, path in parts]
    if all(parts_exist):
        merge_qc = merge_trajectory_parts(
            parts, trajectory_path, trajectory_name
        )
    elif trajectory_path.is_file() and not any(parts_exist):
        merge_qc = {
            "part_count": len(parts),
            "recovered_existing_merged_file": True,
        }
    else:
        raise RuntimeError(
            "trajectory finalization is incomplete: only some part files exist"
        )
    summaries[trajectory_path.name] = inspect_trajectory(
        trajectory_path,
        trajectory_name,
        float(config["phantom_max_step_mm"]),
    )

    in_events = summaries["PhaseSpaceIn.root"]["unique_primary_events"]
    trajectory_events = summaries[trajectory_path.name][
        "unique_primary_events"
    ]
    out_events = summaries["PhaseSpaceOut.root"]["unique_primary_events"]
    if not (
        0 < trajectory_events <= in_events and 0 < out_events <= in_events
    ):
        raise RuntimeError(
            "unexpected event counts: trajectory and exit must be "
            "nonempty subsets of entrance histories"
        )
    if not (qc_dir / "protonct.txt").is_file():
        raise RuntimeError("missing SimulationStatisticsActor output")

    finalize_seconds = time.perf_counter() - clock
    record.update(
        status="completed",
        completed_at=datetime.now().isoformat(timespec="seconds"),
        finalization_elapsed_seconds=finalize_seconds,
        elapsed_seconds=(
            float(record["simulation_elapsed_seconds"]) + finalize_seconds
        ),
        root_qc=summaries,
        trajectory_merge=merge_qc,
        event_counts={
            "entrance_primary_events": in_events,
            "trajectory_primary_events": trajectory_events,
            "exit_primary_events": out_events,
        },
        output_bytes=sum(item["bytes"] for item in summaries.values()),
    )
    metadata_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    (qc_dir / "completed.flag").write_text(
        f"scenario={config['scenario_id']}\n"
        f"angle={args.angle}\n"
        f"seed={record['random_seed']}\n"
        f"config_sha256={record['config_sha256']}\n",
        encoding="ascii",
    )
    print(
        f"Finalized MLP truth angle {args.angle:03d}; "
        f"total elapsed={record['elapsed_seconds']:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
