#!/usr/bin/env python
import argparse
from itertools import zip_longest
import json
from pathlib import Path

import itk
from itk import PCT as pct
import numpy as np
import numpy.lib.recfunctions as rfn


def build_parser():
    parser = pct.PCTArgumentParser(
        description="Pair corresponding protons from GATE ROOT files"
    )
    parser.add_argument(
        "-i",
        "--input-in",
        help="Root phase space file of particles before object",
        required=True,
    )
    parser.add_argument(
        "-j",
        "--input-out",
        help="Root phase space file of particles after object",
        required=True,
    )
    parser.add_argument("-o", "--output", help="Output file name", required=True)
    parser.add_argument(
        "--plane-in",
        help="Plane position of incoming protons",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--plane-out",
        help="Plane position of outgoing protons",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--min-run", help="Minimum run (inclusive)", default=0, type=int
    )
    parser.add_argument(
        "--max-run", help="Maximum run (exclusive)", default=1e6, type=int
    )
    parser.add_argument(
        "--no-nuclear",
        help="Remove inelastic nuclear collisions",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fit", help="Fit file used to convert from energy loss or TOF to WEPL"
    )
    parser.add_argument(
        "--fit-kind",
        help="Whether to convert to WEPL using energy loss or TOF",
        choices=["tof", "energy"],
    )
    parser.add_argument(
        "--store-time",
        help="Store time instead of energy in the output list-mode",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--stream-by-run",
        help=(
            "Read ordered ROOT trees incrementally and write each run before "
            "loading the next one. The default whole-tree behavior is unchanged."
        ),
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", default=False, action="store_true"
    )
    parser.add_argument(
        "--psin", help="Name of tree in input phase space", default="PhaseSpace"
    )
    parser.add_argument(
        "--psout", help="Name of tree in output phase space", default="PhaseSpace"
    )

    return parser


def _open_tree(uproot, root_file, tree_name):
    # The default uproot MultithreadedFileSource can block indefinitely while
    # waiting for its basket-reading worker in restricted execution
    # environments. Local phase-space files are seekable, so a memory map is
    # deterministic and avoids that worker queue. Keep uproot's default source
    # for non-local inputs such as URLs.
    open_options = {}
    if Path(root_file).is_file():
        open_options["handler"] = uproot.source.file.MemmapSource
    return uproot.open(root_file, **open_options)[tree_name]


def _branches_to_recarray(branches):
    # Some uproot versions return a dictionary and others a structured ndarray.
    if isinstance(branches, dict):
        dtype = [(name, branch.dtype) for name, branch in branches.items()]
    elif isinstance(branches, np.ndarray):
        dtype = branches.dtype.descr
    else:
        raise NotImplementedError(f"Unsupported uproot array type: {type(branches)}")

    ps = np.rec.recarray((len(branches["RunID"]),), dtype=dtype)
    for branch_name, *_ in dtype:
        ps[branch_name] = branches[branch_name]
    return ps


def _rename_phase_space_fields(ps):
    ps = rfn.rename_fields(
        ps,
        {"Position_X": "u", "Position_Y": "v", "Position_Z": "w"},
    )
    return rfn.rename_fields(
        ps,
        {"Direction_X": "du", "Direction_Y": "dv", "Direction_Z": "dw"},
    )


def _load_tree_as_df(uproot, root_file, tree_name, min_run, max_run):
    tree = _open_tree(uproot, root_file, tree_name)
    ps = _rename_phase_space_fields(_branches_to_recarray(tree.arrays(library="np")))
    return ps[(ps["RunID"] >= min_run) & (ps["RunID"] < max_run)]


def _iter_tree_runs(uproot, root_file, tree_name, min_run, max_run):
    """Yield ``(RunID, recarray)`` while reading an ordered ROOT tree once."""

    tree = _open_tree(uproot, root_file, tree_name)
    pending_run = None
    pending_parts = []
    previous_run = -1

    for branches in tree.iterate(step_size="128 MB", library="np"):
        chunk = _branches_to_recarray(branches)
        if len(chunk) == 0:
            continue
        run_ids = chunk["RunID"]
        if np.any(run_ids[1:] < run_ids[:-1]) or int(run_ids[0]) < previous_run:
            raise RuntimeError(f"{tree_name} is not ordered by RunID")
        previous_run = int(run_ids[-1])
        starts = np.flatnonzero(np.r_[True, run_ids[1:] != run_ids[:-1]])
        ends = np.r_[starts[1:], len(chunk)]

        for start, end in zip(starts, ends):
            run_id = int(run_ids[start])
            part = chunk[start:end].copy()
            if pending_run is None:
                pending_run = run_id
            if run_id != pending_run:
                if min_run <= pending_run < max_run:
                    combined = np.concatenate(pending_parts).view(np.recarray)
                    yield pending_run, _rename_phase_space_fields(combined)
                pending_run = run_id
                pending_parts = []
            pending_parts.append(part)

    if pending_run is not None and min_run <= pending_run < max_run:
        combined = np.concatenate(pending_parts).view(np.recarray)
        yield pending_run, _rename_phase_space_fields(combined)


def _pair_phase_spaces(args_info, ps_in, ps_out):
    ps_in["w"] = args_info.plane_in
    ps_out["w"] = args_info.plane_out

    input_rows = len(ps_in)
    output_rows = len(ps_out)
    merge_columns = ["RunID", "EventID"]
    if args_info.no_nuclear:
        merge_columns.append("TrackID")

    output_key_columns = ["RunID", "EventID", "TrackID"]
    output_unique_keys = len(np.unique(ps_out[output_key_columns]))

    # Preserve the legacy behavior: remove duplicate entrance merge keys and
    # keep the first occurrence.
    _, unique_index = np.unique(ps_in[merge_columns], return_index=True)
    ps_in = ps_in[unique_index]
    input_unique_keys = len(ps_in)
    matched_input_keys = len(
        np.intersect1d(ps_in[merge_columns], np.unique(ps_out[merge_columns]))
    )

    ps_in.dtype.names = [
        name if name in merge_columns else name + "_in" for name in ps_in.dtype.names
    ]
    ps_out.dtype.names = [
        name if name in merge_columns else name + "_out"
        for name in ps_out.dtype.names
    ]
    ps_in_uniques = [name for name in ps_in.dtype.names if name not in merge_columns]
    ps_out_uniques = [name for name in ps_out.dtype.names if name not in merge_columns]

    if args_info.no_nuclear:
        intersect, intersect_in, intersect_out = np.intersect1d(
            ps_in[merge_columns], ps_out[merge_columns], return_indices=True
        )
        pairs = rfn.merge_arrays(
            (
                intersect,
                ps_in[ps_in_uniques][intersect_in],
                ps_out[ps_out_uniques][intersect_out],
            ),
            asrecarray=True,
            flatten=True,
        )
    else:
        track_max = int(ps_out["TrackID_out"].max())
        pairs_list = []
        for track_id in range(track_max + 1):
            ps_out_t = ps_out[ps_out["TrackID_out"] == track_id]
            intersect, intersect_in, intersect_out = np.intersect1d(
                ps_in[merge_columns], ps_out_t[merge_columns], return_indices=True
            )
            pairs_t = rfn.merge_arrays(
                (
                    intersect,
                    ps_in[ps_in_uniques][intersect_in],
                    ps_out_t[ps_out_uniques][intersect_out],
                ),
                asrecarray=True,
                flatten=True,
            )
            if len(pairs_t) > 0:
                pairs_list.append(pairs_t)
        if not pairs_list:
            raise RuntimeError("No corresponding proton pairs were found")
        pairs = rfn.stack_arrays(pairs_list, asrecarray=True)
        np.recarray.sort(
            pairs, order=["RunID", "EventID", "TrackID_in", "TrackID_out"]
        )

    metrics = {
        "input_rows": input_rows,
        "input_unique_merge_keys": input_unique_keys,
        "input_duplicates_removed": input_rows - input_unique_keys,
        "output_rows": output_rows,
        "output_unique_run_event_track_keys": output_unique_keys,
        "output_duplicate_run_event_track_rows": output_rows - output_unique_keys,
        "matched_input_merge_keys": matched_input_keys,
        "unmatched_input_merge_keys": input_unique_keys - matched_input_keys,
        "pairs": len(pairs),
        "primary_pairs": int(
            np.count_nonzero(
                pairs["TrackID"] == 1
                if args_info.no_nuclear
                else pairs["TrackID_out"] == 1
            )
        ),
    }
    metrics["secondary_pairs"] = metrics["pairs"] - metrics["primary_pairs"]
    return pairs, metrics


def _apply_fit(args_info, pairs, verbose):
    if args_info.fit is None:
        return
    verbose("Converting energy loss or TOF to WEPL…")
    with open(args_info.fit, encoding="utf-8") as fit_stream:
        parameters = json.load(fit_stream)
    if args_info.fit_kind == "tof":
        xs = pairs["PreGlobalTime_out"] - pairs["PreGlobalTime_in"]
    elif args_info.fit_kind == "energy":
        xs = pairs["KineticEnergy_in"] - pairs["KineticEnergy_out"]
    else:
        raise NotImplementedError
    wepls = np.polyval(parameters, xs)
    pairs["KineticEnergy_in"] = 0.0
    pairs["KineticEnergy_out"] = wepls


def _write_run(args_info, pairs, run_id, measurement_column, verbose):
    ps_np = np.empty(shape=(len(pairs), 5, 3), dtype=np.float32)
    ps_np[:, 0, 0] = pairs["u_in"]
    ps_np[:, 0, 1] = pairs["v_in"]
    ps_np[:, 0, 2] = pairs["w_in"]
    ps_np[:, 1, 0] = pairs["u_out"]
    ps_np[:, 1, 1] = pairs["v_out"]
    ps_np[:, 1, 2] = pairs["w_out"]
    ps_np[:, 2, 0] = pairs["du_in"]
    ps_np[:, 2, 1] = pairs["dv_in"]
    ps_np[:, 2, 2] = pairs["dw_in"]
    ps_np[:, 3, 0] = pairs["du_out"]
    ps_np[:, 3, 1] = pairs["dv_out"]
    ps_np[:, 3, 2] = pairs["dw_out"]
    ps_np[:, 4, 0] = pairs[measurement_column + "_in"]
    ps_np[:, 4, 1] = pairs[measurement_column + "_out"]
    ps_np[:, 4, 2] = (
        pairs["TrackID"] if args_info.no_nuclear else pairs["TrackID_out"]
    )

    component_type = itk.ctype("float")
    pixel_type = itk.Vector[component_type, 3]
    image_type = itk.Image[pixel_type, 2]
    pairs_image = itk.GetImageFromArray(ps_np, ttype=image_type)
    output_file = args_info.output.replace(".", f"{run_id:04d}.")
    itk.imwrite(pairs_image, output_file)
    verbose(f"Wrote file {output_file}.")


def _process_whole_tree(args_info, uproot, measurement_column, verbose):
    ps_in = _load_tree_as_df(
        uproot,
        args_info.input_in,
        args_info.psin,
        args_info.min_run,
        args_info.max_run,
    )
    verbose("Read input phase space:\n" + str(ps_in))
    ps_out = _load_tree_as_df(
        uproot,
        args_info.input_out,
        args_info.psout,
        args_info.min_run,
        args_info.max_run,
    )
    verbose("Read output phase space:\n" + str(ps_out))
    pairs, metrics = _pair_phase_spaces(args_info, ps_in, ps_out)
    _apply_fit(args_info, pairs, verbose)

    number_of_runs = int(pairs["RunID"].max()) + 1
    run_range = range(args_info.min_run, min(number_of_runs, args_info.max_run))
    for run_id in run_range:
        pairs_run = pairs[pairs["RunID"] == run_id]
        if len(pairs_run):
            _write_run(args_info, pairs_run, run_id, measurement_column, verbose)
    return [metrics]


def _process_stream_by_run(args_info, uproot, measurement_column, verbose):
    input_runs = _iter_tree_runs(
        uproot,
        args_info.input_in,
        args_info.psin,
        args_info.min_run,
        args_info.max_run,
    )
    output_runs = _iter_tree_runs(
        uproot,
        args_info.input_out,
        args_info.psout,
        args_info.min_run,
        args_info.max_run,
    )
    metrics_by_run = []
    for item_in, item_out in zip_longest(input_runs, output_runs):
        if item_in is None or item_out is None:
            raise RuntimeError("Entrance and exit ROOT trees contain different RunIDs")
        run_in, ps_in = item_in
        run_out, ps_out = item_out
        if run_in != run_out:
            raise RuntimeError(
                f"Entrance and exit RunIDs differ: {run_in} != {run_out}"
            )
        pairs, metrics = _pair_phase_spaces(args_info, ps_in, ps_out)
        _apply_fit(args_info, pairs, verbose)
        _write_run(args_info, pairs, run_in, measurement_column, verbose)
        metrics["run_id"] = run_in
        metrics_by_run.append(metrics)
        verbose("Run QC: " + json.dumps(metrics, sort_keys=True))
    return metrics_by_run


def process(args_info: argparse.Namespace):
    import uproot

    verbose = print if args_info.verbose else lambda _message: None
    measurement_column = "PreGlobalTime" if args_info.store_time else "KineticEnergy"
    if args_info.stream_by_run:
        return _process_stream_by_run(
            args_info, uproot, measurement_column, verbose
        )
    return _process_whole_tree(args_info, uproot, measurement_column, verbose)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
