#!/usr/bin/env python3
"""Run DracoGS, MesonGS, and VideoGS baselines for QUEEN.

Note: QUEEN baseline pipelines currently use range-based frame arguments
(`--frame_start/--frame_end/--interval`). To support sparse selected frame
lists, this runner executes the full continuous range that spans each list,
then downstream collection filters to the selected frame IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


BASELINE_ENVS: dict[str, str] = {
    "dracogs": "queen",
    "mesongs": "mesongs",
    "videogs": "queen",
}
BASELINES = list(BASELINE_ENVS.keys())
EVALUATION_ENV = "queen"
CUDA_DEVICE = "1"
SKIP_EXISTING = True

DATASET_NAME = "Neural_3D_Video"
DATA_PATH = "/synology/rajrup/Queen"
SH_DEGREE = 2
RESOLUTION = 2

DRACOGS_EG = 16
DRACOGS_EO = 16
DRACOGS_ET = 16
DRACOGS_ES = 16
DRACOGS_CL = 10

VIDEOGS_QPS = [25]
VIDEOGS_GROUP_SIZE = 20

BASELINE_FRAME_IDS: dict[str, list[int]] = {
    "dracogs": list(range(1, 201, 10)),
    "mesongs": list(range(1, 201, 10)),
    "videogs": list(range(1, 201, 20)),
}
SEQUENCES: list[str] = [
    "cook_spinach",
    "coffee_martini",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
QUEEN_ROOT = SCRIPTS_DIR.parent
MESONGS_ROOT = QUEEN_ROOT / "MesonGS"


@dataclass(frozen=True)
class ExperimentPaths:
    dataset_path: str
    gt_model_path: str
    output_folder: str


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_header(message: str) -> None:
    print("")
    print("=" * 70)
    print(f"  {message}")
    print("=" * 70)


def log_step(message: str) -> None:
    print(f"--- {message}")


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
    if dry_run:
        print(
            f"[DRY RUN] cwd={cwd} | CUDA_VISIBLE_DEVICES={CUDA_DEVICE} | "
            f"{shlex.join(cmd)}"
        )
        return
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def conda_python_cmd(env_name: str, script_path: Path, args: list[str]) -> list[str]:
    return ["conda", "run", "-n", env_name, "python", str(script_path), *args]


def _output_complete(output_folder: str) -> bool:
    """Return True if the evaluation directory already exists (run completed)."""
    return Path(output_folder, "evaluation").is_dir()


def _cleanup_partial(output_folder: str) -> None:
    """Remove a partially-generated output folder so it won't block future reruns."""
    p = Path(output_folder)
    if p.exists():
        log_step(f"Cleaning up partial output: {output_folder}")
        shutil.rmtree(p)


def get_available_conda_envs() -> set[str]:
    proc = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    env_paths = payload.get("envs", [])
    return {Path(p).name for p in env_paths}


def ensure_required_envs(baselines: list[str]) -> None:
    required = {EVALUATION_ENV}
    required.update(BASELINE_ENVS[baseline] for baseline in baselines)

    available = get_available_conda_envs()
    missing = sorted(env for env in required if env not in available)
    if missing:
        raise RuntimeError(
            "Missing required conda environment(s): "
            + ", ".join(missing)
            + ". Please create/install them before running experiments."
        )


def selected_to_span(frame_ids: list[int]) -> tuple[int, int, int]:
    if not frame_ids:
        raise ValueError("Frame list must not be empty")
    sorted_ids = sorted(set(int(v) for v in frame_ids))
    return sorted_ids[0], sorted_ids[-1], 1


def frame_span_tag(frame_start: int, frame_end: int, interval: int) -> str:
    return f"frames_{frame_start}_{frame_end}_int_{interval}"


def get_output_folder(
    baseline: str,
    sequence: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    videogs_qp: int | None = None,
) -> str:
    model_root = Path(DATA_PATH) / "pretrained_output" / DATASET_NAME / f"queen_compressed_{sequence}"
    run_tag = frame_span_tag(frame_start, frame_end, interval)

    if baseline == "dracogs":
        tag = (
            f"eg_{DRACOGS_EG}_eo_{DRACOGS_EO}_"
            f"et_{DRACOGS_ET}_es_{DRACOGS_ES}_cl_{DRACOGS_CL}"
        )
        return str(model_root / "compression" / "dracogs" / tag / run_tag)
    if baseline == "mesongs":
        return str(model_root / "compression" / "mesongs" / "params_default" / run_tag)
    if baseline == "videogs":
        if videogs_qp is None:
            raise ValueError("videogs_qp must be provided for videogs output folder")
        return str(model_root / "compression" / "videogs" / f"qp_{videogs_qp}" / run_tag)

    raise ValueError(f"Unknown baseline: {baseline}")


def get_paths(
    sequence: str,
    baseline: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    videogs_qp: int | None = None,
) -> ExperimentPaths:
    dataset_path = str(Path(DATA_PATH) / DATASET_NAME / sequence)
    gt_model_path = str(
        Path(DATA_PATH) / "pretrained_output" / DATASET_NAME / f"queen_compressed_{sequence}"
    )
    output_folder = get_output_folder(
        baseline,
        sequence,
        frame_start,
        frame_end,
        interval,
        videogs_qp=videogs_qp,
    )
    return ExperimentPaths(dataset_path, gt_model_path, output_folder)


def run_evaluation(
    paths: ExperimentPaths,
    sequence: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    dry_run: bool,
) -> None:
    cmd = conda_python_cmd(
        EVALUATION_ENV,
        QUEEN_ROOT / "scripts" / "evaluate_decompress.py",
        [
            "--config",
            "configs/dynerf.yaml",
            "-s",
            paths.dataset_path,
            "-m",
            paths.gt_model_path,
            "--decompressed_ply_path",
            f"{paths.output_folder}/decompressed_ply",
            "--output_render_path",
            f"{paths.output_folder}/evaluation",
            "--frame_start",
            str(frame_start),
            "--frame_end",
            str(frame_end),
            "--interval",
            str(interval),
        ],
    )
    run_cmd(cmd, cwd=QUEEN_ROOT, dry_run=dry_run)


def run_dracogs(sequence: str, selected_frames: list[int], dry_run: bool, skip_existing: bool) -> None:
    frame_start, frame_end, interval = selected_to_span(selected_frames)
    paths = get_paths(sequence, "dracogs", frame_start, frame_end, interval)
    if skip_existing and _output_complete(paths.output_folder):
        log_step(f"SKIP (exists) DracoGS | {sequence} | frames: {frame_start}-{frame_end}:{interval}")
        return
    log_step(
        f"DracoGS | {sequence} | frames: {frame_start}-{frame_end}:{interval} | {timestamp()}"
    )

    cmd = conda_python_cmd(
        BASELINE_ENVS["dracogs"],
        QUEEN_ROOT / "scripts" / "dracogs_baseline" / "compress_decompress_pipeline.py",
        [
            "--ply_path",
            paths.gt_model_path,
            "--output_folder",
            paths.output_folder,
            "--output_ply_folder",
            f"{paths.output_folder}/decompressed_ply",
            "--frame_start",
            str(frame_start),
            "--frame_end",
            str(frame_end),
            "--interval",
            str(interval),
            "--sh_degree",
            str(SH_DEGREE),
            "--scene_name",
            sequence,
            "--eg",
            str(DRACOGS_EG),
            "--eo",
            str(DRACOGS_EO),
            "--et",
            str(DRACOGS_ET),
            "--es",
            str(DRACOGS_ES),
            "--cl",
            str(DRACOGS_CL),
        ],
    )
    try:
        run_cmd(cmd, cwd=QUEEN_ROOT, dry_run=dry_run)
        run_evaluation(paths, sequence, frame_start, frame_end, interval, dry_run)
    except subprocess.CalledProcessError:
        _cleanup_partial(paths.output_folder)
        raise


def run_mesongs(sequence: str, selected_frames: list[int], dry_run: bool, skip_existing: bool) -> None:
    frame_start, frame_end, interval = selected_to_span(selected_frames)
    paths = get_paths(sequence, "mesongs", frame_start, frame_end, interval)
    if skip_existing and _output_complete(paths.output_folder):
        log_step(f"SKIP (exists) MesonGS | {sequence} | frames: {frame_start}-{frame_end}:{interval}")
        return
    log_step(
        f"MesonGS | {sequence} | frames: {frame_start}-{frame_end}:{interval} | {timestamp()}"
    )

    cmd = conda_python_cmd(
        BASELINE_ENVS["mesongs"],
        QUEEN_ROOT / "scripts" / "mesongs_baseline" / "compression_decompress_pipeline.py",
        [
            "--ply_path",
            paths.gt_model_path,
            "--dataset_path",
            paths.dataset_path,
            "--output_folder",
            paths.output_folder,
            "--output_ply_folder",
            f"{paths.output_folder}/decompressed_ply",
            "--frame_start",
            str(frame_start),
            "--frame_end",
            str(frame_end),
            "--interval",
            str(interval),
            "--sh_degree",
            str(SH_DEGREE),
            "--scene_name",
            sequence,
        ],
    )
    try:
        run_cmd(cmd, cwd=MESONGS_ROOT, dry_run=dry_run)
        run_evaluation(paths, sequence, frame_start, frame_end, interval, dry_run)
    except subprocess.CalledProcessError:
        _cleanup_partial(paths.output_folder)
        raise


def run_videogs(sequence: str, selected_frames: list[int], dry_run: bool, skip_existing: bool) -> None:
    for anchor_frame in sorted(set(int(v) for v in selected_frames)):
        gop_start = anchor_frame
        gop_end = anchor_frame + VIDEOGS_GROUP_SIZE - 1
        for qp in VIDEOGS_QPS:
            paths = get_paths(
                sequence,
                "videogs",
                gop_start,
                gop_end,
                1,
                videogs_qp=qp,
            )
            if skip_existing and _output_complete(paths.output_folder):
                log_step(
                    f"SKIP (exists) VideoGS | {sequence} | QP: {qp} | "
                    f"anchor: {anchor_frame} | frames: {gop_start}-{gop_end}:1"
                )
                continue
            log_step(
                f"VideoGS | {sequence} | QP: {qp} | "
                f"anchor: {anchor_frame} | frames: {gop_start}-{gop_end}:1 | {timestamp()}"
            )

            cmd = conda_python_cmd(
                BASELINE_ENVS["videogs"],
                QUEEN_ROOT / "scripts" / "videogs_baseline" / "compress_decompress_pipeline.py",
                [
                    "--ply_path",
                    paths.gt_model_path,
                    "--output_folder",
                    paths.output_folder,
                    "--output_ply_folder",
                    f"{paths.output_folder}/decompressed_ply",
                    "--frame_start",
                    str(gop_start),
                    "--frame_end",
                    str(gop_end),
                    "--interval",
                    "1",
                    "--group_size",
                    str(VIDEOGS_GROUP_SIZE),
                    "--sh_degree",
                    str(SH_DEGREE),
                    "--qp",
                    str(qp),
                ],
            )
            try:
                run_cmd(cmd, cwd=QUEEN_ROOT, dry_run=dry_run)
                run_evaluation(paths, sequence, gop_start, gop_end, 1, dry_run)
            except subprocess.CalledProcessError:
                _cleanup_partial(paths.output_folder)
                raise


def get_expected_output_folders(
    baseline: str,
    sequence: str,
    selected_frames: list[int],
) -> list[str]:
    if baseline == "videogs":
        output_folders: list[str] = []
        for anchor_frame in sorted(set(int(v) for v in selected_frames)):
            gop_start = anchor_frame
            gop_end = anchor_frame + VIDEOGS_GROUP_SIZE - 1
            output_folders.extend(
                get_output_folder(
                    baseline,
                    sequence,
                    gop_start,
                    gop_end,
                    1,
                    videogs_qp=qp,
                )
                for qp in VIDEOGS_QPS
            )
        return output_folders

    frame_start, frame_end, interval = selected_to_span(selected_frames)
    return [get_output_folder(baseline, sequence, frame_start, frame_end, interval)]


BASELINE_RUNNERS: dict[str, Callable[[str, list[int], bool, bool], None]] = {
    "dracogs": run_dracogs,
    "mesongs": run_mesongs,
    "videogs": run_videogs,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected baseline experiments for QUEEN")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=SKIP_EXISTING, help="Skip runs whose evaluation output already exists (default: %(default)s)")
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINES,
        default=BASELINES,
        help="Baselines to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_baselines = args.baselines

    unknown = [name for name in selected_baselines if name not in BASELINE_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown baseline(s): {unknown}")

    ensure_required_envs(selected_baselines)

    run_start = time.time()
    log_header("Selected Baseline Experiments Runner (QUEEN)")
    print(f"  Started:    {timestamp()}")
    print(f"  Dataset:    {DATASET_NAME}")
    print(f"  Baselines:  {', '.join(selected_baselines)}")
    print(f"  Sequences:  {len(SEQUENCES)}")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  SH degree:  {SH_DEGREE}")
    print(f"  CUDA:       {CUDA_DEVICE}")
    print(f"  Data path:  {DATA_PATH}")
    if args.dry_run:
        print("  Mode:       DRY RUN")
    if args.skip_existing:
        print("  Skip:       existing outputs")
    print("=" * 70)

    failed_runs: list[tuple[str, str]] = []

    for sequence in SEQUENCES:
        log_header(f"Sequence: {sequence}")

        for baseline in selected_baselines:
            selected_frames = BASELINE_FRAME_IDS[baseline]
            runner = BASELINE_RUNNERS[baseline]
            selected_str = ",".join(str(v) for v in selected_frames)
            log_header(f"{baseline.upper()} | {sequence} | Frames: {selected_str}")
            step_start = time.time()
            try:
                runner(sequence, selected_frames, args.dry_run, args.skip_existing)
            except subprocess.CalledProcessError as exc:
                print(
                    f"WARNING: {baseline} failed for {sequence} "
                    f"(exit {exc.returncode})"
                )
                failed_runs.append((baseline, sequence))
            elapsed = int(time.time() - step_start)
            print(f"  {baseline.upper()} | {sequence} completed in {elapsed}s")

    total_sec = int(time.time() - run_start)
    log_header("All experiments complete!")
    print(f"  Finished:      {timestamp()}")
    print(f"  Total time:    {total_sec // 3600}h {(total_sec % 3600) // 60}m {total_sec % 60}s")

    if failed_runs:
        print("")
        print(f"  FAILED RUNS ({len(failed_runs)}):")
        for baseline, sequence in failed_runs:
            print(f"    - {baseline} | {sequence}")
    else:
        print("  All runs succeeded.")

    print("")
    print("  Output locations:")
    for sequence in SEQUENCES:
        for baseline in selected_baselines:
            selected_frames = BASELINE_FRAME_IDS[baseline]
            for out in get_expected_output_folders(
                baseline,
                sequence,
                selected_frames,
            ):
                print(f"    {baseline} | {sequence}: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
