#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""High-level orchestrator for the LiVoGS per-frame RD experiment pipeline (QUEEN)."""

import csv
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, TypedDict

from numpy import arange

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUEEN_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if QUEEN_ROOT not in sys.path:
    sys.path.insert(0, QUEEN_ROOT)
from scripts.livogs_baseline.rd_pipeline import config
from scripts.livogs_baseline.rd_pipeline.config import SequenceCfg

from scripts.livogs_baseline.rd_pipeline import qp as _qp
from scripts.livogs_baseline.rd_pipeline import plot as _plot


DATA_PATH = config.DATA_PATH
PRETRAINED_ROOT = os.path.join(DATA_PATH, "pretrained_output", "Neural_3D_Video")
RAW_DATA_ROOT = os.path.join(DATA_PATH, "Neural_3D_Video")

SEQUENCES: list[SequenceCfg] = [
    # {
    #     "dataset_name": "Neural_3D_Video",
    #     "sequence_name": "coffee_martini",
    #     "qp_dir_name": "DyNeRF_coffee_martini",
    # },
    # {
    #     "dataset_name": "Neural_3D_Video",
    #     "sequence_name": "cook_spinach",
    #     "qp_dir_name": "DyNeRF_cook_spinach",
    # },
    # {
    #     "dataset_name": "Neural_3D_Video",
    #     "sequence_name": "cut_roasted_beef",
    #     "qp_dir_name": "DyNeRF_cut_roasted_beef",
    # },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "flame_salmon_1",
        "qp_dir_name": "DyNeRF_flame_salmon_1",
    },
    # {
    #     "dataset_name": "Neural_3D_Video",
    #     "sequence_name": "flame_steak",
    #     "qp_dir_name": "DyNeRF_flame_steak",
    # },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "sear_steak",
        "qp_dir_name": "DyNeRF_sear_steak",
    },
]

FRAME_IDS = [1]
RESOLUTION = config.RESOLUTION
SH_DEGREE = config.SH_DEGREE
SH_COLOR_SPACE = config.SH_COLOR_SPACE
RLGR_BLOCK_SIZE = config.RLGR_BLOCK_SIZE
DEVICE = config.DEVICE

STAGE2_GPUS = [0, 1]
STAGE2_WORKERS_PER_GPU = 3
STAGE2_DISABLE_IMAGE_AND_PLY_SAVING = True
SKIP_SAVED_EXPERIEMNTS = True
RUN_EVALUATE = True
RUN_PLOT = False

EXPERIMENT_BETA_VALUES = [0.0]
EXPERIMENT_BASELINE_QPS = [v / 255.0 for v in [0.01, 0.1, 0.5, 1, 2, 4, 8, 16]]
# Stage-2 evaluates this depth list. Plot aggregation may use only a subset
# selected from PLOTS (see _depths_needed_for_plotting).
EXPERIMENT_DEPTHS = [12, 13, 14, 15, 16, 17, 18]
EXPERIMENT_QP_QUATS: list[float] = [0.0001, 0.001, 0.01, 0.1]
EXPERIMENT_QP_SCALES: list[float] = [0.0001, 0.001, 0.01, 0.1]
EXPERIMENT_QP_OPACITY: list[float] = [0.0001, 0.001, 0.01, 0.1]

PLOT_PSNR_RANGE: Optional[tuple[float, float]] = None

# Plot specs drive two things:
# 1) Plot filtering in stage_plot:
#    - "curve_var" is the sweep variable for separate RD curves.
#    - "fixed" contains exact-match filters applied before plotting.
# 2) Aggregation scope in main():
#    - If curve_var == "depth", aggregate all EXPERIMENT_DEPTHS.
#    - Otherwise, if fixed contains "depth", aggregate only that depth.
#    - If fixed omits "depth", aggregate all EXPERIMENT_DEPTHS.
#
# Important: any knob not fixed in a spec can still vary in that plot. For
# example, omitting baseline_qp means all available baseline_qp rows are kept.
PLOTS: list[dict[str, Any]] = [
    {
        "curve_var": "qp_opacity",
        "fixed": {
            "beta": 0.0,
            "qp_quats": 0.00005,
            "qp_scales": 0.0001,
            "qp_opacity": 0.0001,
        },
    },
]

QP_CONFIGS_ROOT = config.QP_CONFIGS_ROOT


class Stage2Job(TypedDict):
    idx: int
    label: str
    depth: int
    gpu_id: int
    cmd: list[str]
    env: dict[str, str]


def _run_subprocess(
    label: str,
    cmd: list[str],
    cwd: str = config.QUEEN_ROOT,
    env: Optional[dict[str, str]] = None,
) -> bool:
    """Run a subprocess; return True on success."""
    sep = "=" * 70
    print(f"\n{sep}\n{label}\n{sep}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] '{label}' failed (exit {result.returncode})")
        return False
    return True


def find_qp_jsons(seq: SequenceCfg, frame_id: int) -> list[str]:
    """Return sorted QP config JSON paths for one sequence+frame."""
    pattern = config.qp_json_pattern(QP_CONFIGS_ROOT, seq["qp_dir_name"], frame_id)
    return sorted(glob.glob(pattern))


def _normalize_float_set(values: Optional[list[float]]) -> Optional[set[float]]:
    if values is None:
        return None
    return {round(float(v), 6) for v in values}


ConfigKey = tuple[float, float, float, float, float]


def _expected_config_keys(
    baseline_qps: list[float],
    betas: list[float],
    qp_quats_list: list[float],
    qp_scales_list: list[float],
    qp_opacity_list: list[float],
) -> set[ConfigKey]:
    qp_set = _normalize_float_set(baseline_qps) or set()
    beta_set = _normalize_float_set(betas) or set()
    quats_set = _normalize_float_set(qp_quats_list) or set()
    scales_set = _normalize_float_set(qp_scales_list) or set()
    opacity_set = _normalize_float_set(qp_opacity_list) or set()
    return {
        (qp, beta, q, s, o)
        for qp in qp_set
        for beta in beta_set
        for q in quats_set
        for s in scales_set
        for o in opacity_set
    }


def _existing_config_keys(json_files: list[str]) -> tuple[set[ConfigKey], int]:
    keys: set[ConfigKey] = set()
    invalid = 0
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                qp_data = json.load(f)
            baseline_qp = round(float(qp_data["baseline_qp"]), 6)
            beta = round(float(qp_data["beta"]), 6)
            quantize_cfg = qp_data.get("quantize_config", {})
            qp_quats = round(float(qp_data.get("qp_quats", quantize_cfg.get("quats", 0))), 6)
            qp_scales = round(float(qp_data.get("qp_scales", quantize_cfg.get("scales", 0))), 6)
            qp_opacity = round(float(qp_data.get("qp_opacity", quantize_cfg.get("opacity", 0))), 6)
            keys.add((baseline_qp, beta, qp_quats, qp_scales, qp_opacity))
        except Exception:
            invalid += 1
    return keys, invalid


def _find_missing_config_keys(
    seq: SequenceCfg,
    frame_id: int,
    baseline_qps: list[float],
    betas: list[float],
    qp_quats_list: list[float],
    qp_scales_list: list[float],
    qp_opacity_list: list[float],
) -> tuple[set[ConfigKey], int, int, int]:
    expected_keys = _expected_config_keys(
        baseline_qps, betas, qp_quats_list, qp_scales_list, qp_opacity_list,
    )
    json_files = find_qp_jsons(seq, frame_id)
    existing_keys, invalid = _existing_config_keys(json_files)
    missing_keys = expected_keys - existing_keys
    return missing_keys, len(expected_keys), len(existing_keys), invalid


def _format_config_key_list(keys: set[ConfigKey], max_items: int = 6) -> str:
    if not keys:
        return ""
    ordered = sorted(keys)
    shown = ordered[:max_items]
    summary = ", ".join([f"(qp={q}, beta={b}, q={qv}, s={sv}, o={ov})" for q, b, qv, sv, ov in shown])
    if len(ordered) > max_items:
        summary += f", ... (+{len(ordered) - max_items} more)"
    return summary


def _format_value_list(values: set[float], max_items: int = 20) -> str:
    if not values:
        return "[]"
    ordered = sorted(values)
    shown = ordered[:max_items]
    summary = "[" + ", ".join([str(v) for v in shown]) + "]"
    if len(ordered) > max_items:
        summary += f" ... (+{len(ordered) - max_items} more)"
    return summary


def _to_float_list(values: list[float]) -> list[float]:
    return [float(v) for v in values]


def _normalize_stage2_gpus(gpus: list[int]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()

    for raw_id in gpus:
        gpu_id = int(raw_id)
        if gpu_id < 0:
            raise ValueError(f"Invalid GPU id {gpu_id}. GPU ids must be >= 0.")
        if gpu_id in seen:
            continue
        seen.add(gpu_id)
        normalized.append(gpu_id)

    if not normalized:
        raise ValueError("Stage-2 GPU list is empty after normalization.")

    return normalized


def filter_qp_jsons_by_selection(
    json_files: list[str],
    selected_baseline_qps: Optional[list[float]],
    selected_betas: Optional[list[float]],
) -> list[str]:
    qp_set = _normalize_float_set(selected_baseline_qps)
    beta_set = _normalize_float_set(selected_betas)

    if qp_set is None and beta_set is None:
        return json_files

    filtered: list[str] = []
    skipped_missing = 0
    skipped_filtered = 0

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                qp_data = json.load(f)
            baseline_qp = round(float(qp_data["baseline_qp"]), 6)
            beta = round(float(qp_data["beta"]), 6)
        except Exception:
            skipped_missing += 1
            continue

        if qp_set is not None and baseline_qp not in qp_set:
            skipped_filtered += 1
            continue
        if beta_set is not None and beta not in beta_set:
            skipped_filtered += 1
            continue

        filtered.append(json_path)

    if skipped_missing > 0:
        print(f"  [WARN] Skipped {skipped_missing} QP JSONs with invalid/missing baseline_qp or beta fields.")
    if skipped_filtered > 0:
        print(f"  [INFO] Filtered out {skipped_filtered} QP JSONs by selected baseline_qp/beta.")

    return filtered


def stage_generate(
    generate_baseline_qps: list[float],
    generate_betas: list[float],
    generate_qp_quats: list[float],
    generate_qp_scales: list[float],
    generate_qp_opacity: list[float],
    selected_qp_dir_names: Optional[list[str]] = None,
    frame_ids: Optional[list[int]] = None,
) -> bool:
    """Stage 1: generate QP config JSONs via direct library call."""
    sep = "=" * 70
    print(f"\n{sep}\nStage 1: Generate QP configs\n{sep}")
    try:
        _qp.generate(
            sequences=SEQUENCES,
            frame_ids=frame_ids or FRAME_IDS,
            baseline_qps=generate_baseline_qps,
            beta_values=generate_betas,
            output_root=QP_CONFIGS_ROOT,
            data_path=DATA_PATH,
            selected_qp_dir_names=selected_qp_dir_names,
            qp_quats_list=generate_qp_quats,
            qp_scales_list=generate_qp_scales,
            qp_opacity_list=generate_qp_opacity,
        )
        return True
    except Exception as exc:
        print(f"[ERROR] Stage 1 generation failed: {exc}")
        return False


def ensure_experiment_configs() -> bool:
    missing_entries: list[tuple[SequenceCfg, int, set[ConfigKey], int, int, int]] = []
    for seq in SEQUENCES:
        for frame_id in FRAME_IDS:
            missing, expected_count, existing_count, invalid = _find_missing_config_keys(
                seq,
                frame_id,
                EXPERIMENT_BASELINE_QPS,
                EXPERIMENT_BETA_VALUES,
                EXPERIMENT_QP_QUATS,
                EXPERIMENT_QP_SCALES,
                EXPERIMENT_QP_OPACITY,
            )
            if missing:
                missing_entries.append((seq, frame_id, missing, expected_count, existing_count, invalid))

    if not missing_entries:
        print("[INFO] All experiment QP configs already exist.")
        return True

    print("[INFO] Missing experiment QP configs detected; running Stage 1 generation.")
    for seq, frame_id, missing, expected_count, existing_count, invalid in missing_entries:
        print(
            f"  - {seq['qp_dir_name']} frame {frame_id}: missing {len(missing)}/{expected_count} "
            f"(existing={existing_count}, invalid={invalid})"
        )

    missing_qp_dir_names = sorted({seq["qp_dir_name"] for seq, _, _, _, _, _ in missing_entries})
    missing_frame_ids = sorted({frame_id for _, frame_id, _, _, _, _ in missing_entries})

    if not stage_generate(
        EXPERIMENT_BASELINE_QPS,
        EXPERIMENT_BETA_VALUES,
        EXPERIMENT_QP_QUATS,
        EXPERIMENT_QP_SCALES,
        EXPERIMENT_QP_OPACITY,
        selected_qp_dir_names=missing_qp_dir_names,
        frame_ids=missing_frame_ids,
    ):
        print("[ERROR] Stage 1 generation failed while filling missing experiment configs.")
        return False

    unresolved = []
    for seq in SEQUENCES:
        for frame_id in FRAME_IDS:
            missing, expected_count, existing_count, invalid = _find_missing_config_keys(
                seq,
                frame_id,
                EXPERIMENT_BASELINE_QPS,
                EXPERIMENT_BETA_VALUES,
                EXPERIMENT_QP_QUATS,
                EXPERIMENT_QP_SCALES,
                EXPERIMENT_QP_OPACITY,
            )
            if missing:
                unresolved.append((seq, frame_id, missing, expected_count, existing_count, invalid))

    if unresolved:
        print("[ERROR] Experiment QP configs are still missing after generation:")
        for seq, frame_id, missing, expected_count, existing_count, invalid in unresolved:
            print(
                f"  - {seq['qp_dir_name']} frame {frame_id}: missing {len(missing)}/{expected_count} "
                f"(existing={existing_count}, invalid={invalid})"
            )
            print(f"    Missing keys: {_format_config_key_list(missing)}")
        return False

    print("[INFO] Missing experiment QP configs generated successfully.")
    return True


def _is_saved_experiment_complete(
    seq: SequenceCfg,
    frame_id: int,
    depth: int,
    label: str,
    require_evaluation: bool,
) -> bool:
    exp_dir = config.experiment_dir(
        DATA_PATH,
        seq["dataset_name"],
        seq["sequence_name"],
        frame_id,
        depth,
        label,
    )
    if not os.path.isdir(exp_dir):
        return False

    required_paths = [
        os.path.join(exp_dir, "qp_config.json"),
        os.path.join(exp_dir, "benchmark_livogs.csv"),
        os.path.join(exp_dir, "livogs_config.json"),
    ]
    if require_evaluation:
        required_paths.append(os.path.join(exp_dir, "evaluation", "evaluation_results.json"))

    return all(os.path.exists(path) for path in required_paths)


def stage_evaluate(seq: SequenceCfg, frame_id: int, depths: list[int]) -> list[str]:
    """Stage 2: compress + evaluate for every selected QP config JSON."""
    json_files = find_qp_jsons(seq, frame_id)
    json_files = filter_qp_jsons_by_selection(
        json_files,
        selected_baseline_qps=EXPERIMENT_BASELINE_QPS,
        selected_betas=EXPERIMENT_BETA_VALUES,
    )
    if not depths:
        print("  [WARN] No experiment depths selected; skipping Stage 2.")
        return []
    if not json_files:
        print(f"  [WARN] No selected QP config JSONs found for {seq['qp_dir_name']} frame {frame_id}")
        print(f"         Expected pattern: {QP_CONFIGS_ROOT}/{seq['qp_dir_name']}/frame_{frame_id}/qp_*.json")
        print(f"         Selection: baseline_qps={EXPERIMENT_BASELINE_QPS}, beta_values={EXPERIMENT_BETA_VALUES}")
        return []

    print(f"\n  Found {len(json_files)} QP configs for {seq['sequence_name']} frame {frame_id}")

    gpus = _normalize_stage2_gpus(STAGE2_GPUS) if STAGE2_GPUS else [0]
    workers_per_gpu = max(1, STAGE2_WORKERS_PER_GPU)
    total_candidates = len(json_files) * len(depths)
    print(f"  Stage-2 GPUs: {gpus}  |  workers/GPU: {workers_per_gpu}")
    print("  Stage-2 worker device: cuda:0 (mapped by CUDA_VISIBLE_DEVICES)")
    print(f"  Stage-2 depths: {depths}")

    jobs: list[Stage2Job] = []
    skipped_saved = 0
    skipped_frame_mismatch = 0
    require_evaluation = not STAGE2_DISABLE_IMAGE_AND_PLY_SAVING

    for json_path in json_files:
        label = os.path.splitext(os.path.basename(json_path))[0]
        qp_frame_id: Optional[int] = None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                qp_data = json.load(f)
            label = qp_data.get("label", label)
            raw_qp_frame_id = qp_data.get("frame_id")
            if raw_qp_frame_id is not None:
                try:
                    qp_frame_id = int(raw_qp_frame_id)
                except (TypeError, ValueError):
                    print(
                        f"  [WARN] Skipping QP config with invalid frame_id={raw_qp_frame_id!r}: "
                        f"{json_path}"
                    )
                    skipped_frame_mismatch += 1
                    continue
        except Exception:
            pass

        if qp_frame_id is not None and qp_frame_id != frame_id:
            print(
                f"  [WARN] Skipping QP config from frame_id={qp_frame_id} while running "
                f"frame_id={frame_id}: {json_path}"
            )
            skipped_frame_mismatch += 1
            continue

        for depth in depths:
            if SKIP_SAVED_EXPERIEMNTS and _is_saved_experiment_complete(
                seq=seq,
                frame_id=frame_id,
                depth=depth,
                label=label,
                require_evaluation=require_evaluation,
            ):
                skipped_saved += 1
                continue
            idx = len(jobs)
            gpu_id = gpus[idx % len(gpus)]
            cmd = [
                sys.executable,
                os.path.join(config.THIS_DIR, "worker.py"),
                "--data_path",
                DATA_PATH,
                "--dataset_name",
                seq["dataset_name"],
                "--sequence_name",
                seq["sequence_name"],
                "--frame_id",
                str(frame_id),
                "--j",
                str(depth),
                "--sh_color_space",
                SH_COLOR_SPACE,
                "--rlgr_block_size",
                str(RLGR_BLOCK_SIZE),
                "--resolution",
                str(RESOLUTION),
                "--sh_degree",
                str(SH_DEGREE),
                "--qp_config_json",
                json_path,
                "--device",
                "cuda:0",
            ]
            if STAGE2_DISABLE_IMAGE_AND_PLY_SAVING:
                cmd.append("--disable_image_and_ply_saving")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            jobs.append(
                {
                    "idx": idx,
                    "label": label,
                    "depth": depth,
                    "gpu_id": gpu_id,
                    "cmd": cmd,
                    "env": env,
                }
            )

    if skipped_saved > 0:
        print(f"  [INFO] Skipping {skipped_saved}/{total_candidates} saved experiments.")
    if skipped_frame_mismatch > 0:
        print(f"  [INFO] Skipped {skipped_frame_mismatch} QP configs due to frame-id mismatch/invalid value.")

    if not jobs:
        print("  [INFO] No pending Stage-2 experiments to run.")
        return []

    total_workers = min(len(jobs), len(gpus) * workers_per_gpu)
    print(f"  Stage-2 queued jobs: {len(jobs)}  |  total workers: {total_workers}")

    failed: list[str] = []
    if total_workers <= 1:
        for job in jobs:
            ok = _run_subprocess(
                f"  [{job['idx']+1}/{len(jobs)}] Evaluate: {job['label']} (J={job['depth']}, GPU {job['gpu_id']})",
                job["cmd"],
                env=job["env"],
            )
            if not ok:
                failed.append(f"{job['label']}/J_{job['depth']}")
        return failed

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        future_to_job = {
            executor.submit(
                _run_subprocess,
                f"  [{job['idx']+1}/{len(jobs)}] Evaluate: {job['label']} (J={job['depth']}, GPU {job['gpu_id']})",
                job["cmd"],
                config.QUEEN_ROOT,
                job["env"],
            ): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            ok = future.result()
            if not ok:
                failed.append(f"{job['label']}/J_{job['depth']}")

    return failed


def _normalize_plot_fixed_filters(
    plot_specs: list[dict[str, Any]],
) -> Optional[list[dict[str, float]]]:
    normalized: list[dict[str, float]] = []

    for idx, spec in enumerate(plot_specs):
        fixed = spec.get("fixed", {})
        if not isinstance(fixed, dict):
            continue
        if any(key not in config.KNOB_NAMES for key in fixed.keys()):
            continue

        normalized_fixed: dict[str, float] = {}
        for key, raw_value in fixed.items():
            try:
                normalized_fixed[key] = round(float(raw_value), 6)
            except (TypeError, ValueError):
                print(
                    f"[WARN] Plot spec #{idx} has non-numeric fixed value for '{key}': "
                    f"{raw_value!r}. Skipping aggregation prefilter."
                )
                return None
        normalized.append(normalized_fixed)

    return normalized


def _row_matches_fixed_constraints(row: dict[str, Any], fixed: dict[str, float]) -> bool:
    for key, target_value in fixed.items():
        row_value = row.get(key)
        if row_value is None:
            return False
        try:
            if round(float(row_value), 6) != target_value:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _filter_aggregate_rows_for_plots(
    rows: list[dict[str, Any]],
    plot_specs: Optional[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not rows or not plot_specs:
        return rows

    normalized_fixed_filters = _normalize_plot_fixed_filters(plot_specs)
    if normalized_fixed_filters is None:
        return rows
    if not normalized_fixed_filters:
        return rows

    return [
        row
        for row in rows
        if any(_row_matches_fixed_constraints(row, fixed) for fixed in normalized_fixed_filters)
    ]


def stage_aggregate(
    seq: SequenceCfg,
    frame_id: int,
    depths: list[int],
    plot_specs: Optional[list[dict[str, Any]]] = None,
) -> str:
    output_root = config.rd_output_root(DATA_PATH, seq["dataset_name"], seq["sequence_name"])
    all_rows: list[dict[str, Any]] = []
    for depth in depths:
        frame_dir = os.path.join(output_root, f"frame_{frame_id}", f"J_{depth}")
        depth_results = _plot.collect_results(frame_dir, frame_id, depth=depth)
        all_rows.extend(depth_results)

    collected_count = len(all_rows)
    all_rows = _filter_aggregate_rows_for_plots(all_rows, plot_specs)
    filtered_count = len(all_rows)

    csv_path = config.all_results_csv(DATA_PATH, seq["dataset_name"], seq["sequence_name"], frame_id)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "depth", "baseline_qp", "beta", "qp_quats", "qp_scales", "qp_opacity",
        "compressed_bytes", "compressed_mb", "decomp_psnr", "gt_psnr", "label",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    if plot_specs:
        print(
            f"[INFO] Aggregated {filtered_count}/{collected_count} rows "
            f"after applying plot fixed-filter union into: {csv_path}"
        )
    else:
        print(f"[INFO] Aggregated {filtered_count} rows into: {csv_path}")
    return csv_path


def _depths_needed_for_plotting(
    plot_specs: list[dict[str, Any]],
    experiment_depths: list[int],
) -> list[int]:
    """Return the minimal depth set required to satisfy all plot specs."""
    base_depths: list[int] = []
    base_seen: set[int] = set()
    for raw_depth in experiment_depths:
        depth = int(raw_depth)
        if depth in base_seen:
            continue
        base_seen.add(depth)
        base_depths.append(depth)

    needs_all_depths = False
    requested_depths: list[int] = []
    requested_seen: set[int] = set()

    for spec in plot_specs:
        curve_var = spec.get("curve_var")
        fixed = spec.get("fixed", {})

        if curve_var == "depth":
            needs_all_depths = True
            continue

        if not isinstance(fixed, dict) or "depth" not in fixed:
            continue

        try:
            depth = int(fixed["depth"])
        except (TypeError, ValueError):
            continue

        if depth in requested_seen:
            continue
        requested_seen.add(depth)
        requested_depths.append(depth)

    if needs_all_depths or not requested_depths:
        return base_depths

    selected = [depth for depth in base_depths if depth in requested_seen]
    for depth in requested_depths:
        if depth not in base_seen:
            selected.append(depth)
    return selected


def stage_plot(seq: SequenceCfg, frame_id: int, plot_spec: dict[str, Any]) -> None:
    csv_path = config.all_results_csv(DATA_PATH, seq["dataset_name"], seq["sequence_name"], frame_id)
    if not os.path.exists(csv_path):
        print(f"[WARN] No aggregated CSV found: {csv_path}")
        return
    plot_dir = config.plot_output_dir(DATA_PATH, seq["dataset_name"], seq["sequence_name"])
    curve_var = plot_spec["curve_var"]
    fixed = plot_spec["fixed"]
    if curve_var not in config.KNOB_NAMES:
        print(f"[WARN] Invalid curve_var '{curve_var}'. Expected one of {sorted(config.KNOB_NAMES)}")
        return
    invalid_keys = [k for k in fixed if k not in config.KNOB_NAMES]
    if invalid_keys:
        print(f"[WARN] Invalid fixed keys {invalid_keys}; skipping.")
        return
    fixed_tag = "_".join(f"{k}{v}" for k, v in sorted(fixed.items()))
    output_path = os.path.join(
        plot_dir,
        f"rd_{seq['sequence_name']}_frame{frame_id}_{curve_var}_sweep_{fixed_tag}.png",
    )
    _plot.plot_rd(
        csv_path=csv_path,
        curve_var=curve_var,
        fixed=fixed,
        output_path=output_path,
        sequence_name=seq["sequence_name"],
        frame_id=frame_id,
        psnr_range=PLOT_PSNR_RANGE,
    )


def main() -> None:
    """Run configured RD stages for all sequences/frames."""
    sep = "=" * 70
    print(sep)
    print("LiVoGS RD Pipeline (QUEEN)")
    print(f"  Sequences:  {[s['sequence_name'] for s in SEQUENCES]}")
    print(f"  Frame IDs:  {FRAME_IDS}")
    print(f"  Stages:     evaluate={RUN_EVALUATE}  plot={RUN_PLOT}")
    print(f"  Stage-2:    gpus={STAGE2_GPUS} workers_per_gpu={STAGE2_WORKERS_PER_GPU} fast_no_save={STAGE2_DISABLE_IMAGE_AND_PLY_SAVING}")
    print(f"  Stage-2:    skip_saved_experiemnts={SKIP_SAVED_EXPERIEMNTS}")
    print(f"  Raw root:   {RAW_DATA_ROOT}")
    print(f"  Train root: {PRETRAINED_ROOT}")
    print(f"  QP configs: {QP_CONFIGS_ROOT}")
    print(
        f"  Experiment: beta_values={EXPERIMENT_BETA_VALUES} baseline_qps={EXPERIMENT_BASELINE_QPS} "
        f"depths={EXPERIMENT_DEPTHS}"
    )
    print(
        f"  Attr QPs:   quats={EXPERIMENT_QP_QUATS} scales={EXPERIMENT_QP_SCALES} "
        f"opacity={EXPERIMENT_QP_OPACITY}"
    )
    print(f"  Plotting:   specs={PLOTS} psnr_range={PLOT_PSNR_RANGE}")
    plot_aggregate_depths = _depths_needed_for_plotting(PLOTS, EXPERIMENT_DEPTHS)
    print(f"  Plotting:   aggregate_depths={plot_aggregate_depths}")
    print(sep)

    if RUN_EVALUATE:
        if not ensure_experiment_configs():
            print("[ERROR] Cannot continue: experiment QP config requirements are not satisfied.")
            raise SystemExit(1)

    all_failures: list[str] = []

    for seq in SEQUENCES:
        for frame_id in FRAME_IDS:
            print(f"\n{sep}")
            print(f"Sequence: {seq['sequence_name']}  |  Frame: {frame_id}")
            print(sep)

            if RUN_EVALUATE:
                failed = stage_evaluate(seq, frame_id, EXPERIMENT_DEPTHS)
                if failed:
                    all_failures += [f"{seq['sequence_name']}/frame_{frame_id}/{f}" for f in failed]

            if RUN_PLOT and not STAGE2_DISABLE_IMAGE_AND_PLY_SAVING:
                stage_aggregate(seq, frame_id, plot_aggregate_depths, plot_specs=PLOTS)
                for plot_spec in PLOTS:
                    stage_plot(seq, frame_id, plot_spec=plot_spec)
            elif RUN_PLOT and STAGE2_DISABLE_IMAGE_AND_PLY_SAVING:
                print("[WARN] Stage 3 plot skipped because Stage-2 fast mode disabled image/PLY saving.")

    print(f"\n{sep}")
    print("Pipeline complete.")
    if all_failures:
        print(f"  Failed experiments ({len(all_failures)}):")
        for failure in all_failures:
            print(f"    {failure}")
        raise SystemExit(1)
    print("  All experiments completed successfully.")
    print(sep)


if __name__ == "__main__":
    main()
