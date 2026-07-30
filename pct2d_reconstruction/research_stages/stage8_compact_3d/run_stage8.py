#!/usr/bin/env python3
"""Stage-8 compact-3D readiness, storage and calibration guard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CONFIG = HERE / "stage8_config.json"
QC = HERE / "qc"


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def locate(root, run_id):
    for name in (f"run_{run_id:03d}", f"run_{run_id:04d}", f"angle_{run_id:03d}", f"{run_id:03d}"):
        path = root / name
        if path.is_dir():
            return path
    return None


def preflight(config):
    root = resolve(config["simulation_data"])
    gate_path = resolve(config["wepl_gate"])
    gate = load(gate_path) if gate_path.is_file() else {"status": "MISSING"}
    complete, missing, total_bytes = 0, [], 0
    if root.is_dir():
        for run_id in range(int(config["runs"])):
            directory = locate(root, run_id)
            absent = (
                list(config["required_root"]) if directory is None else
                [name for name in config["required_root"] if not (directory/name).is_file()]
            )
            if absent:
                missing.append({"run_id": run_id, "files": absent})
            else:
                complete += 1
                total_bytes += sum((directory/name).stat().st_size for name in config["required_root"])
    else:
        missing = [
            {"run_id": run_id, "files": list(config["required_root"])}
            for run_id in range(int(config["runs"]))
        ]
    ready = (
        complete == int(config["runs"])
        and gate.get("status") == "PASS"
        and resolve(config["wepl_model"]).is_file()
    )
    voxels = int(np_prod(config["grid"]["size"]))
    result = {
        "status": "READY" if ready else "BLOCKED",
        "wepl_calibration_status": gate.get("status"),
        "data_root": str(root),
        "complete_runs": complete,
        "expected_runs": config["runs"],
        "missing_run_count": len(missing),
        "first_missing": missing[:5],
        "root_bytes": total_bytes,
        "grid_voxels": voxels,
        "float32_volume_bytes": 4 * voxels,
        "next_action": (
            "run 3-D pairing/operator smoke tests"
            if ready else
            (
                "attach compact-3D data"
                if gate.get("status") == "PASS"
                and resolve(config["wepl_model"]).is_file()
                else "complete Stage 6B calibration and attach compact-3D data"
            )
        ),
    }
    QC.mkdir(parents=True, exist_ok=True)
    temporary = QC / "preflight.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(QC / "preflight.json")
    print(json.dumps(result, indent=2))
    return result


def np_prod(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["preflight", "status", "all"], required=True)
    args = parser.parse_args()
    config = load(CONFIG)
    if args.action == "status":
        path = QC / "preflight.json"
        print(path.read_text(encoding="utf-8") if path.is_file() else '{"status":"PENDING_PREFLIGHT"}')
        return
    result = preflight(config)
    if args.action == "all":
        if result["status"] != "READY":
            raise SystemExit("Stage 8 is guarded by Stage 6B and compact-3D input completeness")
        raise SystemExit(
            "Stage 8 inputs are READY. The trilinear CUDA operator and formal "
            "3-D reconstruction remain a separate implementation step."
        )


if __name__ == "__main__":
    main()
