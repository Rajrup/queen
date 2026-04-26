#!/usr/bin/env python3

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.livogs_baseline.collect_rd_results import collect_rd_root
from scripts.livogs_baseline.rd_pipeline import config
from scripts.livogs_baseline.rd_pipeline.config import SequenceCfg


EXPERIMENT_SEQUENCES: list[SequenceCfg] = [
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "coffee_martini",
        "qp_dir_name": "DyNeRF_coffee_martini",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "cook_spinach",
        "qp_dir_name": "DyNeRF_cook_spinach",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "cut_roasted_beef",
        "qp_dir_name": "DyNeRF_cut_roasted_beef",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "flame_salmon_1",
        "qp_dir_name": "DyNeRF_flame_salmon_1",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "flame_steak",
        "qp_dir_name": "DyNeRF_flame_steak",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "sear_steak",
        "qp_dir_name": "DyNeRF_sear_steak",
    },
]
EXPERIMENT_FRAME_IDS: list[int] = [1]

DATA_PATH = config.DATA_PATH
QP_CONFIGS_ROOT = config.QP_CONFIGS_ROOT

SOURCE_RD_SUBDIR = "livogs_rd_nvcomp"
TARGET_RD_SUBDIR = "livogs_rd_new"

SH_COLOR_SPACE = config.SH_COLOR_SPACE
RLGR_BLOCK_SIZE = config.RLGR_BLOCK_SIZE
RESOLUTION = config.RESOLUTION
SH_DEGREE = config.SH_DEGREE
NVCOMP_ALGORITHM: str | None = "ANS"

RUN_STAGE2_EVALUATION = True
WRITE_SUMMARY_CSV = True
STAGE2_GPUS: list[int] = [0]
STAGE2_WORKERS_PER_GPU = 16
STAGE2_ENABLE_EVALUATION = True
STAGE2_SAVE_EVAL_RENDERS = True
STAGE2_SAVE_DECOMPRESSED_PLY = True
SKIP_SAVED_RESULTS = True

SEED_LEFT_INCLUSIVE = True
MAX_SEED_POINTS: int | None = None

DC_QP_SWEEP = [v / 255.0 for v in (1, 2, 4, 8, 16)]
AC_QP_SWEEP = [v / 255.0 for v in (0.1, 1, 4, 8, 16, 32, 64, 100, 128)]
LOSSLESS_QP = DC_QP_SWEEP[0]

MATCH_EPS = 1e-9
NEAR_LOSSLESS_DROP_DB = 0.2
SWEEP_LABEL_PREFIX = "hullleft_gtloss02"


@dataclass
class AnchorPoint:
    depth: int
    qp_sh: float
    qp_quats: float
    qp_scales: float
    qp_opacity: float
    decomp_psnr: float
    compressed_mb: float


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
    sep = "=" * 70
    print(f"\n{sep}\n{label}\n{sep}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] '{label}' failed (exit {result.returncode})")
        return False
    return True


def _normalize_float_set(values: Optional[list[float]]) -> Optional[set[float]]:
    if values is None:
        return None
    return {round(float(v), 6) for v in values}


def _filter_qp_jsons_by_selection(
    json_files: list[str],
    selected_qp_sh_values: Optional[list[float]],
    selected_betas: Optional[list[float]],
    selected_qp_quats: Optional[list[float]] = None,
    selected_qp_scales: Optional[list[float]] = None,
    selected_qp_opacity: Optional[list[float]] = None,
) -> list[str]:
    qp_set = _normalize_float_set(selected_qp_sh_values)
    beta_set = _normalize_float_set(selected_betas)
    quats_set = _normalize_float_set(selected_qp_quats)
    scales_set = _normalize_float_set(selected_qp_scales)
    opacity_set = _normalize_float_set(selected_qp_opacity)

    if (
        qp_set is None
        and beta_set is None
        and quats_set is None
        and scales_set is None
        and opacity_set is None
    ):
        return json_files

    filtered: list[str] = []
    skipped_missing = 0
    skipped_filtered = 0

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                qp_data = json.load(f)
            qp_sh = round(float(qp_data.get("qp_sh", qp_data.get("qp_sh"))), 6)
            beta = round(float(qp_data["beta"]), 6)
            quantize_cfg = qp_data.get("quantize_config", {})
            qp_quats = round(float(qp_data.get("qp_quats", quantize_cfg.get("quats", 0))), 6)
            qp_scales = round(float(qp_data.get("qp_scales", quantize_cfg.get("scales", 0))), 6)
            qp_opacity = round(float(qp_data.get("qp_opacity", quantize_cfg.get("opacity", 0))), 6)
        except Exception:
            skipped_missing += 1
            continue

        if qp_set is not None and qp_sh not in qp_set:
            skipped_filtered += 1
            continue
        if beta_set is not None and beta not in beta_set:
            skipped_filtered += 1
            continue
        if quats_set is not None and qp_quats not in quats_set:
            skipped_filtered += 1
            continue
        if scales_set is not None and qp_scales not in scales_set:
            skipped_filtered += 1
            continue
        if opacity_set is not None and qp_opacity not in opacity_set:
            skipped_filtered += 1
            continue

        filtered.append(json_path)

    if skipped_missing > 0:
        print(
            f"  [WARN] Skipped {skipped_missing} QP JSONs with invalid/missing selection fields "
            "(qp_sh/beta/qp_quats/qp_scales/qp_opacity)."
        )
    if skipped_filtered > 0:
        print(f"  [INFO] Filtered out {skipped_filtered} QP JSONs by selected criteria.")

    return filtered


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


def _resolve_stage2_options(log_messages: bool = True) -> tuple[bool, bool, bool]:
    enable_evaluation = bool(STAGE2_ENABLE_EVALUATION)
    save_eval_renders = bool(STAGE2_SAVE_EVAL_RENDERS)
    save_decompressed_ply = bool(STAGE2_SAVE_DECOMPRESSED_PLY)

    if enable_evaluation and not save_decompressed_ply:
        if log_messages:
            print("  [WARN] Evaluation requires decompressed PLY files; forcing save_decompressed_ply=True.")
        save_decompressed_ply = True
    if not enable_evaluation and save_eval_renders:
        if log_messages:
            print("  [INFO] save_eval_renders ignored because evaluation is disabled.")
        save_eval_renders = False
    return enable_evaluation, save_eval_renders, save_decompressed_ply


def _to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _round6(v: float) -> float:
    return round(float(v), 6)


def _safe_qp_token(v: float) -> str:
    tok = f"{v:.8g}"
    tok = tok.replace("-", "m").replace("+", "")
    tok = tok.replace(".", "p")
    return tok


def _find_hull_csv(seq: SequenceCfg, frame_id: int) -> str:
    rd_root = config.rd_output_root(
        DATA_PATH,
        seq["dataset_name"],
        seq["sequence_name"],
        rd_subdir_name=SOURCE_RD_SUBDIR,
    )
    plot_dir = os.path.join(rd_root, "plots")
    expected = [
        os.path.join(
            plot_dir,
            f"convex_hull_{seq['dataset_name']}_{seq['sequence_name']}_frame{frame_id}.csv",
        ),
        os.path.join(
            plot_dir,
            f"convex_hull_{seq['dataset_name']}_{SOURCE_RD_SUBDIR}_frame{frame_id}.csv",
        ),
    ]
    for path in expected:
        if os.path.exists(path):
            return path

    candidates = sorted(glob.glob(os.path.join(plot_dir, f"convex_hull_*_frame{frame_id}.csv")))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No convex hull CSV found under: {plot_dir}")
    raise FileNotFoundError(f"Multiple hull CSVs matched for frame {frame_id}: {candidates}")


def _load_hull_points(hull_csv: str) -> list[AnchorPoint]:
    points: list[AnchorPoint] = []
    with open(hull_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            points.append(
                AnchorPoint(
                    depth=int(round(_to_float(row, "depth"))),
                    qp_sh=_to_float(row, "qp_sh"),
                    qp_quats=_to_float(row, "qp_quats"),
                    qp_scales=_to_float(row, "qp_scales"),
                    qp_opacity=_to_float(row, "qp_opacity"),
                    decomp_psnr=_to_float(row, "decomp_psnr"),
                    compressed_mb=_to_float(row, "compressed_mb"),
                )
            )
    if not points:
        raise RuntimeError(f"Hull CSV has no rows: {hull_csv}")
    return points


def _load_collected_rows(seq: SequenceCfg) -> list[dict[str, Any]]:
    rd_root = config.rd_output_root(
        DATA_PATH,
        seq["dataset_name"],
        seq["sequence_name"],
        rd_subdir_name=SOURCE_RD_SUBDIR,
    )
    csv_path = os.path.join(rd_root, "collected_rd_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing collected RD CSV: {csv_path}")

    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row: dict[str, Any] = dict(r)
            for k in ("frame_id", "depth"):
                row[k] = int(row[k])
            for k in (
                "qp_sh",
                "beta",
                "qp_quats",
                "qp_scales",
                "qp_opacity",
                "gt_psnr",
                "decomp_psnr",
                "psnr_drop",
                "compressed_mb",
            ):
                row[k] = float(row[k])
            rows.append(row)
    return rows


def _is_close(a: float, b: float, eps: float = MATCH_EPS) -> bool:
    return abs(a - b) <= eps


def _match_anchor_row(anchor: AnchorPoint, rows: list[dict[str, Any]], frame_id: int) -> dict[str, Any]:
    exact = [
        r
        for r in rows
        if r["frame_id"] == frame_id
        and int(r["depth"]) == anchor.depth
        and _is_close(float(r["qp_sh"]), anchor.qp_sh)
        and _is_close(float(r["qp_quats"]), anchor.qp_quats)
        and _is_close(float(r["qp_scales"]), anchor.qp_scales)
        and _is_close(float(r["qp_opacity"]), anchor.qp_opacity)
    ]
    if exact:
        exact.sort(key=lambda r: (-float(r["decomp_psnr"]), float(r["compressed_mb"])))
        return exact[0]

    nearest: tuple[float, dict[str, Any]] | None = None
    for r in rows:
        if r["frame_id"] != frame_id:
            continue
        dist = 0.0
        dist += (float(r["depth"]) - float(anchor.depth)) ** 2
        dist += (float(r["qp_sh"]) - anchor.qp_sh) ** 2
        dist += (float(r["qp_quats"]) - anchor.qp_quats) ** 2
        dist += (float(r["qp_scales"]) - anchor.qp_scales) ** 2
        dist += (float(r["qp_opacity"]) - anchor.qp_opacity) ** 2
        if nearest is None or dist < nearest[0]:
            nearest = (dist, r)

    if nearest is None:
        raise RuntimeError("No candidate rows for anchor matching.")
    return nearest[1]


def _pick_global_near_lossless_anchor(
    hull_points: list[AnchorPoint],
    rows: list[dict[str, Any]],
    frame_id: int,
) -> tuple[AnchorPoint, dict[str, Any]]:
    candidates: list[tuple[AnchorPoint, dict[str, Any]]] = []
    for p in hull_points:
        row = _match_anchor_row(p, rows, frame_id)
        psnr_drop = float(row.get("psnr_drop", row.get("gt_psnr", 0.0) - row.get("decomp_psnr", 0.0)))
        if psnr_drop <= NEAR_LOSSLESS_DROP_DB:
            candidates.append((p, row))

    if not candidates:
        raise RuntimeError(
            f"No hull points satisfy near-lossless criterion psnr_drop <= {NEAR_LOSSLESS_DROP_DB} dB"
        )

    candidates.sort(key=lambda x: float(x[0].compressed_mb))
    return candidates[0]


def _select_seed_hull_points(hull_points: list[AnchorPoint], left_near_lossless: AnchorPoint) -> list[AnchorPoint]:
    anchor_mb = float(left_near_lossless.compressed_mb)
    if SEED_LEFT_INCLUSIVE:
        seeds = [p for p in hull_points if float(p.compressed_mb) <= anchor_mb + MATCH_EPS]
    else:
        seeds = [p for p in hull_points if float(p.compressed_mb) < anchor_mb - MATCH_EPS]
    seeds.sort(key=lambda p: float(p.compressed_mb))
    if MAX_SEED_POINTS is not None:
        seeds = seeds[: max(0, int(MAX_SEED_POINTS))]
    return seeds


def _load_anchor_qp_config(seq: SequenceCfg, frame_id: int, row: dict[str, Any]) -> dict[str, Any]:
    exp_dir = config.experiment_dir(
        DATA_PATH,
        seq["dataset_name"],
        seq["sequence_name"],
        frame_id,
        int(row["depth"]),
        str(row["label"]),
        rd_subdir_name=SOURCE_RD_SUBDIR,
    )
    qp_json = os.path.join(exp_dir, "qp_config.json")
    if not os.path.exists(qp_json):
        raise FileNotFoundError(f"Anchor qp_config.json not found: {qp_json}")
    with open(qp_json, encoding="utf-8") as f:
        return json.load(f)


def _set_scalar_or_repeat(value_or_list: Any, target: float) -> Any:
    if isinstance(value_or_list, list):
        return [float(target)] * len(value_or_list)
    return float(target)


def _build_sweep_payload(
    anchor_cfg: dict[str, Any],
    frame_id: int,
    dc_qp: float,
    ac_qp: float,
    depth: int,
    seed_idx: int,
) -> dict[str, Any]:
    payload = dict(anchor_cfg)
    payload["frame_id"] = int(frame_id)
    payload["octree_depth"] = int(depth)
    payload["qp_sh"] = float(ac_qp)

    qc = dict(anchor_cfg.get("quantize_config", {}))
    sh_dc = qc.get("sh_dc", [anchor_cfg.get("qp_sh", LOSSLESS_QP)] * 3)
    sh_rest = qc.get("sh_rest", [anchor_cfg.get("qp_sh", LOSSLESS_QP)] * 24)
    qc["sh_dc"] = _set_scalar_or_repeat(sh_dc, dc_qp)
    qc["sh_rest"] = _set_scalar_or_repeat(sh_rest, ac_qp)

    payload["quantize_config"] = qc
    payload["acdc_sweep_mode"] = "cartesian_acdc"
    payload["acdc_dc_qp"] = float(dc_qp)
    payload["acdc_ac_qp"] = float(ac_qp)
    payload["acdc_seed_index"] = int(seed_idx)
    payload["acdc_source"] = f"left_near_lossless_convex_hull_psnrdrop_lt_{NEAR_LOSSLESS_DROP_DB}"
    return payload


def _write_sweep_qp_jsons(
    seq: SequenceCfg,
    frame_id: int,
    depth: int,
    anchor_cfg: dict[str, Any],
    seed_idx: int,
) -> tuple[str, list[str]]:
    custom_qp_dir = f"{seq['qp_dir_name']}_{SWEEP_LABEL_PREFIX}_d{depth}_seed{seed_idx}"
    out_dir = config.qp_json_output_dir(QP_CONFIGS_ROOT, custom_qp_dir, frame_id)
    os.makedirs(out_dir, exist_ok=True)
    for stale_json in glob.glob(os.path.join(out_dir, "qp_*.json")):
        os.remove(stale_json)

    written: list[str] = []
    for dc_qp in DC_QP_SWEEP:
        for ac_qp in AC_QP_SWEEP:
            payload = _build_sweep_payload(anchor_cfg, frame_id, dc_qp, ac_qp, depth, seed_idx)
            label = (
                f"{SWEEP_LABEL_PREFIX}_cart_dc_{_safe_qp_token(dc_qp)}"
                f"_ac_{_safe_qp_token(ac_qp)}_d{depth}_seed{seed_idx}"
            )
            payload["label"] = label
            out_path = os.path.join(out_dir, f"qp_{label}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            written.append(out_path)
    return custom_qp_dir, written


def _run_stage2_for_anchor(
    seq: SequenceCfg,
    frame_id: int,
    depth: int,
    custom_qp_dir: str,
    anchor_cfg: dict[str, Any],
) -> list[str]:
    seq_for_eval: SequenceCfg = {
        "dataset_name": seq["dataset_name"],
        "sequence_name": seq["sequence_name"],
        "qp_dir_name": custom_qp_dir,
    }
    quantize_cfg = anchor_cfg.get("quantize_config", {})
    selected_qp_sh_values = sorted({_round6(v) for v in (DC_QP_SWEEP + AC_QP_SWEEP)})
    selected_betas = [float(anchor_cfg.get("beta", 0.0))]
    selected_qp_quats = [float(anchor_cfg.get("qp_quats", quantize_cfg["quats"]))]
    selected_qp_scales = [float(anchor_cfg.get("qp_scales", quantize_cfg["scales"]))]
    selected_qp_opacity = [float(anchor_cfg.get("qp_opacity", quantize_cfg["opacity"]))]

    enable_evaluation, save_eval_renders, save_decompressed_ply = _resolve_stage2_options(
        log_messages=False
    )

    json_pattern = config.qp_json_pattern(QP_CONFIGS_ROOT, seq_for_eval["qp_dir_name"], frame_id)
    json_files = sorted(glob.glob(json_pattern))
    json_files = _filter_qp_jsons_by_selection(
        json_files,
        selected_qp_sh_values=selected_qp_sh_values,
        selected_betas=selected_betas,
        selected_qp_quats=selected_qp_quats,
        selected_qp_scales=selected_qp_scales,
        selected_qp_opacity=selected_qp_opacity,
    )
    if not json_files:
        print(
            f"  [WARN] No selected QP config JSONs found for {seq_for_eval['qp_dir_name']} frame {frame_id}. "
            f"Pattern: {json_pattern}"
        )
        return []

    def is_saved_experiment_complete(exp_label: str, exp_depth: int) -> bool:
        exp_dir = config.experiment_dir(
            DATA_PATH,
            seq["dataset_name"],
            seq["sequence_name"],
            frame_id,
            exp_depth,
            exp_label,
            rd_subdir_name=TARGET_RD_SUBDIR,
        )
        if not os.path.isdir(exp_dir):
            return False
        required_paths = [
            os.path.join(exp_dir, "qp_config.json"),
            os.path.join(exp_dir, "benchmark_livogs.csv"),
            os.path.join(exp_dir, "livogs_config.json"),
        ]
        if enable_evaluation:
            required_paths.append(os.path.join(exp_dir, "evaluation", "evaluation_results.json"))
        return all(os.path.exists(path) for path in required_paths)

    def remove_failed_experiment(exp_label: str, exp_depth: int) -> None:
        exp_dir = config.experiment_dir(
            DATA_PATH,
            seq["dataset_name"],
            seq["sequence_name"],
            frame_id,
            exp_depth,
            exp_label,
            rd_subdir_name=TARGET_RD_SUBDIR,
        )
        if os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir, ignore_errors=True)
            print(f"  [CLEANUP] Removed partial results for failed experiment: {exp_dir}")

    gpus = _normalize_stage2_gpus(STAGE2_GPUS) if STAGE2_GPUS else [0]
    workers_per_gpu = max(1, STAGE2_WORKERS_PER_GPU)

    jobs: list[Stage2Job] = []
    skipped_saved = 0
    skipped_frame_mismatch = 0
    for json_path in json_files:
        label = os.path.splitext(os.path.basename(json_path))[0]
        qp_frame_id: int | None = None
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                qp_data = json.load(f)
            label = str(qp_data.get("label", label))
            raw_qp_frame_id = qp_data.get("frame_id")
            if raw_qp_frame_id is not None:
                try:
                    qp_frame_id = int(raw_qp_frame_id)
                except (TypeError, ValueError):
                    print(f"  [WARN] Skipping QP config with invalid frame_id={raw_qp_frame_id!r}: {json_path}")
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

        if SKIP_SAVED_RESULTS and is_saved_experiment_complete(label, depth):
            skipped_saved += 1
            continue

        idx = len(jobs)
        gpu_id = gpus[idx % len(gpus)]
        cmd = [
            sys.executable,
            os.path.join(config.THIS_DIR, "worker.py"),
            "--data_path", DATA_PATH,
            "--dataset_name", seq["dataset_name"],
            "--sequence_name", seq["sequence_name"],
            "--rd_output_subdir", TARGET_RD_SUBDIR,
            "--frame_id", str(frame_id),
            "--j", str(depth),
            "--sh_color_space", str(SH_COLOR_SPACE),
            "--rlgr_block_size", str(RLGR_BLOCK_SIZE),
            "--resolution", str(RESOLUTION),
            "--sh_degree", str(SH_DEGREE),
            "--qp_config_json", json_path,
            "--device", "cuda:0",
            "--nvcomp_algorithm", str(NVCOMP_ALGORITHM) if NVCOMP_ALGORITHM is not None else "None",
        ]
        if not save_decompressed_ply:
            cmd.append("--disable_ply_saving")
        if not enable_evaluation:
            cmd.append("--disable_evaluation")
        if save_eval_renders:
            cmd.append("--save_renders")

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
        print(f"  [INFO] Skipping {skipped_saved} saved experiments for depth={depth}.")
    if skipped_frame_mismatch > 0:
        print(f"  [INFO] Skipped {skipped_frame_mismatch} QP configs due to frame-id mismatch/invalid value.")
    if not jobs:
        print("  [INFO] No pending Stage-2 experiments to run.")
        return []

    total_workers = min(len(jobs), len(gpus) * workers_per_gpu)
    print(
        f"  Stage-2 manual config: qp_sh={selected_qp_sh_values} beta={selected_betas} "
        f"q/s/o={selected_qp_quats}/{selected_qp_scales}/{selected_qp_opacity}"
    )
    print(f"  Stage-2 queued jobs: {len(jobs)} | workers: {total_workers} | gpus: {gpus}")

    failed: list[str] = []
    if total_workers <= 1:
        for job in jobs:
            ok = _run_subprocess(
                f"  [{job['idx'] + 1}/{len(jobs)}] Evaluate: {job['label']} (J={job['depth']}, GPU {job['gpu_id']})",
                job["cmd"],
                env=job["env"],
            )
            if not ok:
                failed.append(f"{job['label']}/J_{job['depth']}")
                remove_failed_experiment(job["label"], job["depth"])
        return failed

    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        future_to_job = {
            executor.submit(
                _run_subprocess,
                f"  [{job['idx'] + 1}/{len(jobs)}] Evaluate: {job['label']} (J={job['depth']}, GPU {job['gpu_id']})",
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
                remove_failed_experiment(job["label"], job["depth"])

    return failed


def _collect_new_results(seq: SequenceCfg, frame_id: int) -> list[dict[str, Any]]:
    rd_root = config.rd_output_root(
        DATA_PATH,
        seq["dataset_name"],
        seq["sequence_name"],
        rd_subdir_name=TARGET_RD_SUBDIR,
    )
    return collect_rd_root(rd_root, seq["sequence_name"], frame_ids=[frame_id])


def _parse_cartesian_label(label: str) -> tuple[float, float, int] | None:
    m = re.search(r"_dc_([0-9emp]+)_ac_([0-9emp]+)_d\d+_seed(\d+)$", label)
    if m is None:
        return None
    dc_tok, ac_tok, seed_tok = m.groups()
    dc_qp = float(dc_tok.replace("m", "-").replace("p", "."))
    ac_qp = float(ac_tok.replace("m", "-").replace("p", "."))
    return dc_qp, ac_qp, int(seed_tok)


def _analyze_cartesian(rows: list[dict[str, Any]], label_prefix: str) -> list[dict[str, Any]]:
    step_rows: list[dict[str, Any]] = []
    for r in rows:
        label = str(r.get("label", ""))
        if not label.startswith(label_prefix):
            continue
        parsed = _parse_cartesian_label(label)
        if parsed is None:
            continue
        dc_qp, ac_qp, seed_idx = parsed
        step_rows.append(
            {
                "step": "cartesian_acdc",
                "label": label,
                "depth": int(r["depth"]),
                "seed_idx": int(seed_idx),
                "qp_dc": float(dc_qp),
                "qp_ac": float(ac_qp),
                "qp_quats": float(r.get("qp_quats", math.nan)),
                "qp_scales": float(r.get("qp_scales", math.nan)),
                "qp_opacity": float(r.get("qp_opacity", math.nan)),
                "beta": float(r.get("beta", math.nan)),
                "decomp_psnr": float(r["decomp_psnr"]),
                "compressed_mb": float(r["compressed_mb"]),
            }
        )
    step_rows.sort(key=lambda x: (x["seed_idx"], x["depth"], x["qp_dc"], x["qp_ac"]))
    return step_rows


def _write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "step",
        "label",
        "depth",
        "seed_idx",
        "qp_dc",
        "qp_ac",
        "qp_quats",
        "qp_scales",
        "qp_opacity",
        "beta",
        "decomp_psnr",
        "compressed_mb",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    stage2_enable_eval, stage2_save_renders, stage2_save_ply = _resolve_stage2_options(log_messages=False)
    sep = "=" * 72
    print(sep)
    print("LiVoGS AC/DC Hull Sweep Runner (queen)")
    print(f"  Source RD subdir: {SOURCE_RD_SUBDIR}")
    print(f"  Target RD subdir: {TARGET_RD_SUBDIR}")
    print(f"  Sequences: {[s['sequence_name'] for s in EXPERIMENT_SEQUENCES]}")
    print(f"  Frame IDs: {EXPERIMENT_FRAME_IDS}")
    print(
        "  Flags: "
        f"run_stage2={RUN_STAGE2_EVALUATION}, stage2_enable_evaluation={stage2_enable_eval}, stage2_save_renders={stage2_save_renders}, stage2_save_decompressed_ply={stage2_save_ply}, "
        f"stage2_gpus={STAGE2_GPUS}, workers_per_gpu={STAGE2_WORKERS_PER_GPU}, skip_saved_results={SKIP_SAVED_RESULTS}, write_summary={WRITE_SUMMARY_CSV}, "
        f"seed_left_inclusive={SEED_LEFT_INCLUSIVE}, max_seed_points={MAX_SEED_POINTS}"
    )
    print(sep)

    failures: list[str] = []
    skipped_missing_hull: list[str] = []
    for seq in EXPERIMENT_SEQUENCES:
        for frame_id in EXPERIMENT_FRAME_IDS:
            print(f"\n{sep}")
            print(f"Sequence={seq['sequence_name']} Frame={frame_id}")
            print(sep)

            try:
                hull_csv = _find_hull_csv(seq, frame_id)
            except FileNotFoundError as exc:
                skip_tag = f"{seq['sequence_name']}/frame_{frame_id}"
                skipped_missing_hull.append(skip_tag)
                print(f"  [SKIP] {skip_tag}: {exc}")
                continue
            hull_points = _load_hull_points(hull_csv)
            print(f"  Hull CSV: {hull_csv}")
            print(f"  Hull depths (raw): {sorted({p.depth for p in hull_points})}")

            collected_rows = _load_collected_rows(seq)
            global_anchor, global_anchor_row = _pick_global_near_lossless_anchor(hull_points, collected_rows, frame_id)
            print(
                "  Global anchor (left near-lossless hull point): "
                f"depth={global_anchor.depth} q/s/o={global_anchor.qp_quats:.8g}/{global_anchor.qp_scales:.8g}/{global_anchor.qp_opacity:.8g} "
                f"psnr_drop={float(global_anchor_row['psnr_drop']):.6f} size={global_anchor.compressed_mb:.6f}MB"
            )
            print(f"    Anchor source label: {global_anchor_row['label']}")

            seed_points = _select_seed_hull_points(hull_points, global_anchor)
            print(f"  Seed hull points selected: {len(seed_points)}")

            written_total = 0
            for seed_idx, seed_point in enumerate(seed_points):
                seed_row = _match_anchor_row(seed_point, collected_rows, frame_id)
                seed_cfg = _load_anchor_qp_config(seq, frame_id, seed_row)
                depth = seed_point.depth
                print(
                    f"  Seed #{seed_idx}: size={seed_point.compressed_mb:.6f}MB depth={depth} "
                    f"q/s/o={seed_point.qp_quats:.8g}/{seed_point.qp_scales:.8g}/{seed_point.qp_opacity:.8g}"
                )

                custom_qp_dir, written = _write_sweep_qp_jsons(seq, frame_id, depth, seed_cfg, seed_idx)
                written_total += len(written)
                print(f"    Wrote {len(written)} sweep QP JSONs to: {os.path.dirname(written[0])}")

                if RUN_STAGE2_EVALUATION:
                    failed = _run_stage2_for_anchor(seq, frame_id, depth, custom_qp_dir, seed_cfg)
                    if failed:
                        failures.extend(
                            [f"{seq['sequence_name']}/frame_{frame_id}/seed_{seed_idx}/J_{depth}/{x}" for x in failed]
                        )

            print(f"  Total QP JSONs written for frame {frame_id}: {written_total}")

            if WRITE_SUMMARY_CSV:
                new_rows = _collect_new_results(seq, frame_id)
                summary_rows = _analyze_cartesian(new_rows, f"{SWEEP_LABEL_PREFIX}_cart_dc_")
                rd_root_new = config.rd_output_root(
                    DATA_PATH,
                    seq["dataset_name"],
                    seq["sequence_name"],
                    rd_subdir_name=TARGET_RD_SUBDIR,
                )
                summary_csv = os.path.join(rd_root_new, f"acdc_hull_sweep_summary_frame{frame_id}.csv")
                _write_summary_csv(summary_csv, summary_rows)
                print(f"  Summary CSV: {summary_csv}")
                print(
                    "  AC/DC Cartesian range: "
                    f"DC={[v*255 for v in DC_QP_SWEEP]}/255 x AC={[v*255 for v in AC_QP_SWEEP]}/255"
                )

    print(f"\n{sep}")
    if skipped_missing_hull:
        print(f"Skipped due to missing/ambiguous hull CSV ({len(skipped_missing_hull)}):")
        for s in skipped_missing_hull:
            print(f"  - {s}")
    if failures:
        print(f"Finished with failures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("Finished successfully.")
    print(sep)


if __name__ == "__main__":
    main()
