#!/usr/bin/env python3
"""Run sharded ROOT pairing, 3-sigma filtering, and Schulte MLP DDB projection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPOSITORY = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(REPOSITORY))

from common import load_experiment, path_for  # noqa: E402
from preprocessing import paircuts, projection  # noqa: E402


RUNS = 720


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(directory: Path, force: bool, patterns: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    existing = [path for pattern in patterns for path in directory.glob(pattern)]
    if existing and not force:
        raise FileExistsError(
            f"refusing to overwrite {len(existing)} files in {directory}; use --force"
        )
    if force:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def pair_one(run_id: int, simulation_dir: str, pairs_dir: str) -> dict:
    from applications.pctpairprotons import pctpairprotons

    source = Path(simulation_dir) / f"run_{run_id:03d}"
    output = Path(pairs_dir)
    # pctpairprotons inserts the RunID with a legacy string replacement on
    # every dot in the output path, so the temporary directory must not begin
    # with a dot (or contain any dot at all).
    temporary = output / f"pairing_tmp_{run_id:04d}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        argv = [
            "--input-in", str(source / "PhaseSpaceIn.root"),
            "--input-out", str(source / "PhaseSpaceOut.root"),
            "--output", str(temporary / "pairs.mhd"),
            "--plane-in", "-110",
            "--plane-out", "110",
            "--psin", "PhaseSpaceIn",
            "--psout", "PhaseSpaceOut",
            "--min-run", "0",
            "--max-run", "1",
            "--stream-by-run",
            "--no-nuclear",
        ]
        args = pctpairprotons.build_parser().parse_args(argv)
        metrics = pctpairprotons.process(args)
        if len(metrics) != 1 or int(metrics[0]["run_id"]) != 0:
            raise RuntimeError(f"angle {run_id}: unexpected local RunID metrics {metrics}")
        for suffix in ("mhd", "raw"):
            source_path = temporary / f"pairs0000.{suffix}"
            target_path = output / f"pairs{run_id:04d}.{suffix}"
            source_path.replace(target_path)
        header_path = output / f"pairs{run_id:04d}.mhd"
        header = header_path.read_text(encoding="utf-8")
        header = header.replace(
            "ElementDataFile = pairs0000.raw",
            f"ElementDataFile = pairs{run_id:04d}.raw",
        )
        header_path.write_text(header, encoding="utf-8")

        # ``pctpairprotons --no-nuclear`` matches the same TrackID at the
        # entrance and exit.  It does not itself restrict TrackID to one.
        # This distinction is invisible in a Vacuum world, but Air can create
        # upstream secondary tracks that cross both reference planes.  The
        # established pCT chain is primary-only, so enforce that invariant
        # explicitly before exposing the list-mode pairs to later stages.
        paired = paircuts.read_mhd(header_path)
        primary = paired[:, 4, 2] == 1.0
        primary_pairs = paired[primary]
        paircuts.write_mhd(header_path, primary_pairs)

        row = dict(metrics[0])
        row["local_run_id"] = row.pop("run_id")
        row["run_id"] = run_id
        row["matched_same_track_pairs"] = int(row["pairs"])
        row["secondary_pairs_removed"] = int(np.count_nonzero(~primary))
        row["pairs"] = int(len(primary_pairs))
        row["primary_pairs"] = int(len(primary_pairs))
        row["secondary_pairs"] = 0
        row["input_in"] = str(source / "PhaseSpaceIn.root")
        row["input_out"] = str(source / "PhaseSpaceOut.root")
        return row
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def run_pairing(simulation: Path, pairs: Path, qc: Path, jobs: int, force: bool) -> None:
    prepare_output(pairs, force, ("pairs*.mhd", "pairs*.raw", "pairing_tmp_*"))
    missing = [
        path
        for run_id in range(RUNS)
        for path in (
            simulation / f"run_{run_id:03d}" / "PhaseSpaceIn.root",
            simulation / f"run_{run_id:03d}" / "PhaseSpaceOut.root",
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} ROOT inputs; first: {missing[0]}")

    started = datetime.now()
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(pair_one, run_id, str(simulation), str(pairs)): run_id
            for run_id in range(RUNS)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"pairing {len(rows):03d}/{RUNS}: angle={row['run_id']:03d}, "
                f"pairs={row['pairs']:,}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["run_id"]))
    write_csv(qc / "pairing_runs.csv", rows)
    total_pairs = sum(int(row["pairs"]) for row in rows)
    summary = {
        "status": "PASS",
        "algorithm": (
            "pctpairprotons --stream-by-run --no-nuclear followed by explicit "
            "TrackID=1 selection"
        ),
        "input_layout": "720 independent native-Windows ROOT shard pairs",
        "global_run_id_source": "run directory index",
        "runs": len(rows),
        "total_primary_pairs": total_pairs,
        "elapsed_seconds": time.perf_counter() - start,
        "started": started.isoformat(timespec="seconds"),
        "stopped": datetime.now().isoformat(timespec="seconds"),
        "output_mhd": len(list(pairs.glob("pairs*.mhd"))),
        "output_raw": len(list(pairs.glob("pairs*.raw"))),
    }
    if summary["output_mhd"] != RUNS or summary["output_raw"] != RUNS:
        summary["status"] = "FAIL"
    (qc / "pairing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if summary["status"] != "PASS":
        raise RuntimeError("pairing output completeness check failed")


def filter_one(run_id: int, pairs_dir: str, filtered_dir: str) -> dict:
    pairs = paircuts.read_mhd(Path(pairs_dir) / f"pairs{run_id:04d}.mhd")
    filtered, diagnostics = paircuts.filter_pairs(pairs)
    paircuts.write_mhd(Path(filtered_dir) / f"pairs{run_id:04d}.mhd", filtered)
    diagnostics["run_id"] = run_id
    diagnostics["retained_fraction"] = len(filtered) / len(pairs)
    return diagnostics


def run_filtering(pairs: Path, filtered: Path, qc: Path, jobs: int, force: bool) -> None:
    prepare_output(filtered, force, ("pairs*.mhd", "pairs*.raw"))
    missing = [pairs / f"pairs{run_id:04d}.mhd" for run_id in range(RUNS) if not (pairs / f"pairs{run_id:04d}.mhd").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} pair files; first: {missing[0]}")
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(filter_one, run_id, str(pairs), str(filtered)): run_id
            for run_id in range(RUNS)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"filtering {len(rows):03d}/{RUNS}: angle={row['run_id']:03d}, "
                f"{row['input']:,}->{row['output']:,}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["run_id"]))
    write_csv(qc / "filtering_runs.csv", rows)
    total_input = sum(int(row["input"]) for row in rows)
    total_output = sum(int(row["output"]) for row in rows)
    summary = {
        "status": "PASS",
        "algorithm": "test0713 local 125x2 grid energy/angle joint 3-sigma",
        "runs": len(rows),
        "input_pairs": total_input,
        "output_pairs": total_output,
        "retained_fraction": total_output / total_input,
        "all_output_primary": all(int(row["output_primary"]) == int(row["output"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - start,
        "output_mhd": len(list(filtered.glob("pairs*.mhd"))),
        "output_raw": len(list(filtered.glob("pairs*.raw"))),
    }
    if not summary["all_output_primary"] or summary["output_mhd"] != RUNS or summary["output_raw"] != RUNS:
        summary["status"] = "FAIL"
    (qc / "filtering_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["status"] != "PASS":
        raise RuntimeError("filtering QC failed")


def run_projection(filtered: Path, root: Path, qc: Path, jobs: int, force: bool) -> None:
    ddb = root / "projections_ddb"
    prepare_output(ddb, force, ("proj*.mhd", "proj*.raw"))
    missing = [filtered / f"pairs{run_id:04d}.mhd" for run_id in range(RUNS) if not (filtered / f"pairs{run_id:04d}.mhd").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} filtered files; first: {missing[0]}")
    start = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(projection.process_run, run_id, str(filtered), str(root), False): run_id
            for run_id in range(RUNS)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"projection {len(rows):03d}/{RUNS}: angle={row['run_id']:03d}", flush=True)
    rows.sort(key=lambda row: int(row["run_id"]))
    write_csv(qc / "projection_runs.csv", rows)
    summary = {
        "status": "PASS",
        "algorithm": "Schulte MLP DDB",
        "ionization_potential_ev": projection.IONIZATION_POTENTIAL_EV,
        "runs": len(rows),
        "size": list(projection.SIZE),
        "spacing_mm": list(projection.SPACING_MM),
        "object_zero_count": sum(int(row["object_zero_count"]) for row in rows),
        "variance_nonfinite": sum(int(row["variance_nonfinite"]) for row in rows),
        "variance_negative": sum(int(row["variance_negative"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - start,
        "output_mhd": len(list(ddb.glob("proj*.mhd"))),
        "output_raw": len(list(ddb.glob("proj*.raw"))),
    }
    if summary["variance_nonfinite"] or summary["variance_negative"] or summary["output_mhd"] != RUNS or summary["output_raw"] != RUNS:
        summary["status"] = "FAIL"
    (qc / "projection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["status"] != "PASS":
        raise RuntimeError("projection QC failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="0716")
    parser.add_argument("--stage", choices=("pairing", "filtering", "projection", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    config = load_experiment(args.experiment)
    simulation = path_for(config, "simulation_data")
    root = path_for(config, "preprocessing_data")
    pairs = root / "pairs"
    filtered = root / "pairs_filtered"
    qc = HERE / "qc" / f"results{args.experiment}"
    qc.mkdir(parents=True, exist_ok=True)

    if args.stage in ("pairing", "all"):
        run_pairing(simulation, pairs, qc, args.jobs, args.force)
    if args.stage in ("filtering", "all"):
        run_filtering(pairs, filtered, qc, args.jobs, args.force)
    if args.stage in ("projection", "all"):
        run_projection(filtered, root, qc, args.jobs, args.force)
    print(f"Completed preprocessing stage={args.stage} for experiment {args.experiment}")


if __name__ == "__main__":
    main()
