#!/usr/bin/env python3
"""High-level orchestrator for the LiVoGS per-frame RD experiment pipeline (QUEEN)."""

import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, TypedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUEEN_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if QUEEN_ROOT not in sys.path:
    sys.path.insert(0, QUEEN_ROOT)
from scripts.livogs_baseline.rd_pipeline import config
from scripts.livogs_baseline.rd_pipeline.config import SequenceCfg

from scripts.livogs_baseline.rd_pipeline import qp as _qp


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
NVCOMP_ALGORITHM = "ANS"

STAGE2_GPUS = [0, 1]
STAGE2_WORKERS_PER_GPU = 24
STAGE2_ENABLE_IMAGE_SAVING = True
STAGE2_ENABLE_PLY_SAVING = True
SKIP_SAVED_EXPERIEMNTS = True
RD_OUTPUT_SUBDIR = "livogs_rd_nvcomp"

EXPERIMENT_BETA_VALUES = [0.0]
EXPERIMENT_QP_SH_VALUES = [v / 255.0 for v in [0.01, 0.1, 0.5, 1, 2, 4, 8, 16]]
EXPERIMENT_DEPTHS = [12, 13, 14, 15, 16, 17, 18]
EXPERIMENT_QP_QUATS: list[float] = [0.0001, 0.001, 0.01, 0.02, 0.04, 0.06]
EXPERIMENT_QP_SCALES: list[float] = [0.0001, 0.001, 0.01, 0.02, 0.04, 0.06]
EXPERIMENT_QP_OPACITY: list[float] = [0.0001, 0.001, 0.01, 0.02, 0.04, 0.06]
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


def _remove_failed_experiment(
    seq: SequenceCfg,
    frame_id: int,
    depth: int,
    label: str,
) -> None:
    """Remove the output directory for a failed/interrupted experiment.

    This prevents partial results from being treated as complete when
    SKIP_SAVED_EXPERIEMNTS is True on the next run.
    """
    exp_dir = config.experiment_dir(
        DATA_PATH,
        seq["dataset_name"],
        seq["sequence_name"],
        frame_id,
        depth,
        label,
        rd_subdir_name=RD_OUTPUT_SUBDIR,
    )
    if os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir, ignore_errors=True)
        print(f"  [CLEANUP] Removed partial results for failed experiment: {exp_dir}")

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
    qp_sh_values: list[float],
    betas: list[float],
    qp_quats_list: list[float],
    qp_scales_list: list[float],
    qp_opacity_list: list[float],
) -> set[ConfigKey]:
    qp_set = _normalize_float_set(qp_sh_values) or set()
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
            qp_sh = round(float(qp_data["qp_sh"]), 6)
            beta = round(float(qp_data["beta"]), 6)
            quantize_cfg = qp_data.get("quantize_config", {})
            qp_quats = round(float(qp_data.get("qp_quats", quantize_cfg.get("quats", 0))), 6)
            qp_scales = round(float(qp_data.get("qp_scales", quantize_cfg.get("scales", 0))), 6)
            qp_opacity = round(float(qp_data.get("qp_opacity", quantize_cfg.get("opacity", 0))), 6)
            keys.add((qp_sh, beta, qp_quats, qp_scales, qp_opacity))
        except Exception:
            invalid += 1
    return keys, invalid


def _find_missing_config_keys(
    seq: SequenceCfg,
    frame_id: int,
    qp_sh_values: list[float],
    betas: list[float],
    qp_quats_list: list[float],
    qp_scales_list: list[float],
    qp_opacity_list: list[float],
) -> tuple[set[ConfigKey], int, int, int]:
    expected_keys = _expected_config_keys(
        qp_sh_values, betas, qp_quats_list, qp_scales_list, qp_opacity_list,
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
    selected_qp_sh_values: Optional[list[float]],
    selected_betas: Optional[list[float]],
) -> list[str]:
    qp_set = _normalize_float_set(selected_qp_sh_values)
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
            qp_sh = round(float(qp_data["qp_sh"]), 6)
            beta = round(float(qp_data["beta"]), 6)
        except Exception:
            skipped_missing += 1
            continue

        if qp_set is not None and qp_sh not in qp_set:
            skipped_filtered += 1
            continue
        if beta_set is not None and beta not in beta_set:
            skipped_filtered += 1
            continue

        filtered.append(json_path)

    if skipped_missing > 0:
        print(f"  [WARN] Skipped {skipped_missing} QP JSONs with invalid/missing qp_sh or beta fields.")
    if skipped_filtered > 0:
        print(f"  [INFO] Filtered out {skipped_filtered} QP JSONs by selected qp_sh/beta.")

    return filtered


def stage_generate(
    generate_qp_sh_values: list[float],
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
            qp_sh_values=generate_qp_sh_values,
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
                EXPERIMENT_QP_SH_VALUES,
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
        EXPERIMENT_QP_SH_VALUES,
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
                EXPERIMENT_QP_SH_VALUES,
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
        rd_subdir_name=RD_OUTPUT_SUBDIR,
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
        selected_qp_sh_values=EXPERIMENT_QP_SH_VALUES,
        selected_betas=EXPERIMENT_BETA_VALUES,
    )
    if not depths:
        print("  [WARN] No experiment depths selected; skipping Stage 2.")
        return []
    if not json_files:
        print(f"  [WARN] No selected QP config JSONs found for {seq['qp_dir_name']} frame {frame_id}")
        print(f"         Expected pattern: {QP_CONFIGS_ROOT}/{seq['qp_dir_name']}/frame_{frame_id}/qp_*.json")
        print(f"         Selection: qp_sh_values={EXPERIMENT_QP_SH_VALUES}, beta_values={EXPERIMENT_BETA_VALUES}")
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
    require_evaluation = STAGE2_ENABLE_IMAGE_SAVING

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
                "--rd_output_subdir",
                RD_OUTPUT_SUBDIR,
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
                "--nvcomp_algorithm",
                str(NVCOMP_ALGORITHM) if NVCOMP_ALGORITHM is not None else "None",
            ]
            if not STAGE2_ENABLE_PLY_SAVING:
                cmd.append("--disable_ply_saving")
            if not STAGE2_ENABLE_IMAGE_SAVING:
                cmd.append("--disable_image_saving")

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
                _remove_failed_experiment(seq, frame_id, job["depth"], job["label"])
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
                _remove_failed_experiment(seq, frame_id, job["depth"], job["label"])

    return failed



def main() -> None:
    """Run configured RD stages for all sequences/frames."""
    sep = "=" * 70
    print(sep)
    print("LiVoGS RD Pipeline (QUEEN)")
    print(f"  Sequences:  {[s['sequence_name'] for s in SEQUENCES]}")
    print(f"  Frame IDs:  {FRAME_IDS}")
    print(f"  Stage-2:    gpus={STAGE2_GPUS} workers_per_gpu={STAGE2_WORKERS_PER_GPU} enable_image_saving={STAGE2_ENABLE_IMAGE_SAVING} enable_ply_saving={STAGE2_ENABLE_PLY_SAVING}")
    print(f"  Stage-2:    skip_saved_experiemnts={SKIP_SAVED_EXPERIEMNTS}")
    print(f"  Raw root:   {RAW_DATA_ROOT}")
    print(f"  Train root: {PRETRAINED_ROOT}")
    print(f"  QP configs: {QP_CONFIGS_ROOT}")
    print(
        f"  Experiment: beta_values={EXPERIMENT_BETA_VALUES} qp_sh_values={EXPERIMENT_QP_SH_VALUES} "
        f"depths={EXPERIMENT_DEPTHS}"
    )
    print(
        f"  Attr QPs:   quats={EXPERIMENT_QP_QUATS} scales={EXPERIMENT_QP_SCALES} "
        f"opacity={EXPERIMENT_QP_OPACITY}"
    )
    print(f"  nvCOMP:     {NVCOMP_ALGORITHM if NVCOMP_ALGORITHM else 'none'}")
    print(sep)

    if not ensure_experiment_configs():
        print("[ERROR] Cannot continue: experiment QP config requirements are not satisfied.")
        raise SystemExit(1)

    all_failures: list[str] = []

    for seq in SEQUENCES:
        for frame_id in FRAME_IDS:
            print(f"\n{sep}")
            print(f"Sequence: {seq['sequence_name']}  |  Frame: {frame_id}")
            print(sep)

            failed = stage_evaluate(seq, frame_id, EXPERIMENT_DEPTHS)
            if failed:
                all_failures += [f"{seq['sequence_name']}/frame_{frame_id}/{f}" for f in failed]


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
