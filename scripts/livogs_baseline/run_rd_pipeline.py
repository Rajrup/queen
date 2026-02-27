#!/usr/bin/env python3
"""High-level orchestrator for the LiVoGS per-frame RD experiment pipeline (QUEEN)."""

import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TypedDict

from numpy import arange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import SequenceCfg

import qp as _qp
import plot as _plot


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

STAGE2_GPUS = [0, 2, 3]
STAGE2_WORKERS_PER_GPU = 1
STAGE2_DISABLE_IMAGE_AND_PLY_SAVING = False
SKIP_SAVED_EXPERIEMNTS = False

RUN_EVALUATE = True
RUN_PLOT = True

EXPERIMENT_BETA_VALUES = [0.0]
EXPERIMENT_BASELINE_QPS = [0.01, 0.1, 0.5] + list(arange(1, 11, 0.5))
EXPERIMENT_DEPTHS = [10, 12, 15, 18]

PLOT_BETA_VALUES = [0.0]
PLOT_BASELINE_QPS = list(EXPERIMENT_BASELINE_QPS)
PLOT_DEPTHS = list(EXPERIMENT_DEPTHS)
PLOT_MODES = ["depth_under_beta"]
PLOT_PSNR_RANGE: Optional[tuple[float, float]] = None

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


def _expected_qp_pairs(baseline_qps: list[float], betas: list[float]) -> set[tuple[float, float]]:
    qp_set = _normalize_float_set(baseline_qps) or set()
    beta_set = _normalize_float_set(betas) or set()
    return {(qp, beta) for qp in qp_set for beta in beta_set}


def _existing_qp_pairs(json_files: list[str]) -> tuple[set[tuple[float, float]], int]:
    pairs: set[tuple[float, float]] = set()
    invalid = 0
    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                qp_data = json.load(f)
            baseline_qp = round(float(qp_data["baseline_qp"]), 6)
            beta = round(float(qp_data["beta"]), 6)
            pairs.add((baseline_qp, beta))
        except Exception:
            invalid += 1
    return pairs, invalid


def _find_missing_qp_pairs(
    seq: SequenceCfg,
    frame_id: int,
    baseline_qps: list[float],
    betas: list[float],
) -> tuple[set[tuple[float, float]], int, int, int]:
    expected_pairs = _expected_qp_pairs(baseline_qps, betas)
    json_files = find_qp_jsons(seq, frame_id)
    existing_pairs, invalid = _existing_qp_pairs(json_files)
    missing_pairs = expected_pairs - existing_pairs
    return missing_pairs, len(expected_pairs), len(existing_pairs), invalid


def _format_pair_list(pairs: set[tuple[float, float]], max_items: int = 6) -> str:
    if not pairs:
        return ""
    ordered = sorted(pairs)
    shown = ordered[:max_items]
    summary = ", ".join([f"(qp={q}, beta={b})" for q, b in shown])
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
        )
        return True
    except Exception as exc:
        print(f"[ERROR] Stage 1 generation failed: {exc}")
        return False


def ensure_experiment_configs() -> bool:
    missing_entries: list[tuple[SequenceCfg, int, set[tuple[float, float]], int, int, int]] = []
    for seq in SEQUENCES:
        for frame_id in FRAME_IDS:
            missing, expected_count, existing_count, invalid = _find_missing_qp_pairs(
                seq,
                frame_id,
                EXPERIMENT_BASELINE_QPS,
                EXPERIMENT_BETA_VALUES,
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
        selected_qp_dir_names=missing_qp_dir_names,
        frame_ids=missing_frame_ids,
    ):
        print("[ERROR] Stage 1 generation failed while filling missing experiment configs.")
        return False

    unresolved = []
    for seq in SEQUENCES:
        for frame_id in FRAME_IDS:
            missing, expected_count, existing_count, invalid = _find_missing_qp_pairs(
                seq,
                frame_id,
                EXPERIMENT_BASELINE_QPS,
                EXPERIMENT_BETA_VALUES,
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
            print(f"    Missing pairs: {_format_pair_list(missing)}")
        return False

    print("[INFO] Missing experiment QP configs generated successfully.")
    return True


def ensure_plot_configs_exist(seq: SequenceCfg, frame_id: int) -> bool:
    expected_pairs = _expected_qp_pairs(PLOT_BASELINE_QPS, PLOT_BETA_VALUES)
    json_files = find_qp_jsons(seq, frame_id)
    existing_pairs, invalid = _existing_qp_pairs(json_files)
    missing = expected_pairs - existing_pairs
    expected_count = len(expected_pairs)
    existing_count = len(existing_pairs)
    if not missing:
        return True

    existing_qps = {q for q, _ in existing_pairs}
    existing_betas = {b for _, b in existing_pairs}

    print(
        f"[ERROR] Missing plotting QP configs for {seq['qp_dir_name']} frame {frame_id}: "
        f"missing {len(missing)}/{expected_count} (existing={existing_count}, invalid={invalid})"
    )
    print(
        f"        Required pairs: baseline_qps={_to_float_list(PLOT_BASELINE_QPS)}, "
        f"beta_values={_to_float_list(PLOT_BETA_VALUES)}"
    )
    print(f"        Missing pairs: {_format_pair_list(missing)}")
    print(f"        Available baseline_qps: {_format_value_list(existing_qps)}")
    print(f"        Available beta_values: {_format_value_list(existing_betas)}")

    print("[INFO] Attempting Stage 1 generation for missing plotting QP configs...")
    if not stage_generate(
        PLOT_BASELINE_QPS,
        PLOT_BETA_VALUES,
        selected_qp_dir_names=[seq["qp_dir_name"]],
        frame_ids=[frame_id],
    ):
        print("[ERROR] Failed to generate missing plotting QP configs.")
        return False

    json_files = find_qp_jsons(seq, frame_id)
    existing_pairs, invalid = _existing_qp_pairs(json_files)
    missing = expected_pairs - existing_pairs
    if not missing:
        print(f"[INFO] Missing plotting QP configs generated for {seq['qp_dir_name']} frame {frame_id}.")
        return True

    print(
        f"[ERROR] Plotting QP configs still missing after generation for {seq['qp_dir_name']} frame {frame_id}: "
        f"missing {len(missing)}/{expected_count}"
    )
    print(f"        Missing pairs: {_format_pair_list(missing)}")
    return False


def ensure_plot_depth_results_exist(seq: SequenceCfg, frame_id: int, depth: int) -> bool:
    depth_dir = config.experiment_dir(
        DATA_PATH,
        seq["dataset_name"],
        seq["sequence_name"],
        frame_id,
        depth,
        "",
    ).rstrip(os.sep)
    depth_dir = os.path.dirname(depth_dir)

    if not os.path.isdir(depth_dir):
        print(
            f"[WARN] Missing plotting results directory for {seq['sequence_name']} "
            f"frame {frame_id} (J={depth}): {depth_dir}"
        )
        return False

    has_experiment_dirs = any(
        os.path.isdir(os.path.join(depth_dir, entry)) for entry in os.listdir(depth_dir)
    )
    if not has_experiment_dirs:
        print(
            f"[WARN] Empty plotting results directory for {seq['sequence_name']} "
            f"frame {frame_id} (J={depth}): {depth_dir}"
        )
        return False

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


def stage_plot(
    seq: SequenceCfg,
    frame_id: int,
    plot_mode: str,
    depth: Optional[int] = None,
    target_beta: Optional[float] = None,
    plot_depths: Optional[list[int]] = None,
) -> None:
    """Stage 3: plot RD curves via direct library call."""
    output_root = config.rd_output_root(DATA_PATH, seq["dataset_name"], seq["sequence_name"])
    plot_dir = config.plot_output_dir(DATA_PATH, seq["dataset_name"], seq["sequence_name"])

    if plot_mode == "beta_under_depth":
        if depth is None:
            print("[WARN] stage_plot called with plot_mode=beta_under_depth but no depth.")
            return
        _plot.main(
            frame_id=frame_id,
            output_root=output_root,
            plot_output_dir=plot_dir,
            sequence_name=seq["sequence_name"],
            beta_values=PLOT_BETA_VALUES,
            baseline_qps=PLOT_BASELINE_QPS,
            plot_mode=plot_mode,
            octree_depth=depth,
            psnr_range=PLOT_PSNR_RANGE,
        )
        return

    if plot_mode == "depth_under_beta":
        if target_beta is None:
            print("[WARN] stage_plot called with plot_mode=depth_under_beta but no target_beta.")
            return
        _plot.main(
            frame_id=frame_id,
            output_root=output_root,
            plot_output_dir=plot_dir,
            sequence_name=seq["sequence_name"],
            beta_values=PLOT_BETA_VALUES,
            baseline_qps=PLOT_BASELINE_QPS,
            plot_mode=plot_mode,
            target_beta=target_beta,
            plot_depths=plot_depths,
            psnr_range=PLOT_PSNR_RANGE,
        )
        return

    print(f"[WARN] Unsupported plot mode '{plot_mode}'.")


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
        f"  Plotting:   beta_values={PLOT_BETA_VALUES} baseline_qps={PLOT_BASELINE_QPS} "
        f"depths={PLOT_DEPTHS} modes={PLOT_MODES} psnr_range={PLOT_PSNR_RANGE}"
    )
    print(sep)

    if RUN_EVALUATE:
        if not ensure_experiment_configs():
            print("[ERROR] Cannot continue: experiment QP config requirements are not satisfied.")
            raise SystemExit(1)

    all_failures: list[str] = []
    had_stage_failures = False

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
                if not ensure_plot_configs_exist(seq, frame_id):
                    had_stage_failures = True
                    all_failures.append(f"{seq['sequence_name']}/frame_{frame_id}/plot_stage")
                    continue
                available_plot_depths = [
                    depth for depth in PLOT_DEPTHS
                    if ensure_plot_depth_results_exist(seq, frame_id, depth)
                ]

                for plot_mode in PLOT_MODES:
                    if plot_mode == "beta_under_depth":
                        for depth in available_plot_depths:
                            stage_plot(
                                seq,
                                frame_id,
                                plot_mode=plot_mode,
                                depth=depth,
                            )
                    elif plot_mode == "depth_under_beta":
                        if not available_plot_depths:
                            print(
                                f"[WARN] Stage 3 plot skipped for {seq['sequence_name']} frame {frame_id}: "
                                "no depth results available for depth_under_beta mode."
                            )
                            had_stage_failures = True
                            all_failures.append(
                                f"{seq['sequence_name']}/frame_{frame_id}/plot_stage/{plot_mode}/no_depths"
                            )
                            continue
                        for beta in PLOT_BETA_VALUES:
                            stage_plot(
                                seq,
                                frame_id,
                                plot_mode=plot_mode,
                                target_beta=beta,
                                plot_depths=available_plot_depths,
                            )
                    else:
                        print(f"[WARN] Unknown plot mode '{plot_mode}' in PLOT_MODES.")
                        had_stage_failures = True
                        all_failures.append(
                            f"{seq['sequence_name']}/frame_{frame_id}/plot_stage/unknown_mode/{plot_mode}"
                        )
            elif RUN_PLOT and STAGE2_DISABLE_IMAGE_AND_PLY_SAVING:
                print("[WARN] Stage 3 plot skipped because Stage-2 fast mode disabled image/PLY saving.")

    print(f"\n{sep}")
    print("Pipeline complete.")
    if all_failures or had_stage_failures:
        print(f"  Failed experiments ({len(all_failures)}):")
        for failure in all_failures:
            print(f"    {failure}")
        raise SystemExit(1)
    print("  All experiments completed successfully.")
    print(sep)


if __name__ == "__main__":
    main()
