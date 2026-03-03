#!/usr/bin/env python3
"""Collect LiVoGS RD experiment results into a single CSV.

Walks the experiment directory tree and produces one row per
(sequence, frame, depth, qp-combination) with columns:

    sequencename, frameid, depth, baseline_qp, beta, opacity, scales, quats,
    gtpsnr, gtssim, psnr, ssim, psnrdrop, ssimdrop, size

Edit the "Global configuration" section below, then run::

    python scripts/livogs_baseline/collect_rd_results.py
"""

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Global configuration  (edit here)
# ---------------------------------------------------------------------------

# Each entry is a dict describing one livogs_rd/ root to scan.
# Required key:
#   "path"       — absolute path to a livogs_rd/ directory
# Optional keys:
#   "name"       — sequence name override (default: inferred from path)
#   "frame_ids"  — list of frame IDs to collect (default: all frames)
RD_OUTPUT_ROOTS: list[dict[str, Any]] = [
    # --- Queen ---
    {
        "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/compression/livogs_rd",
        "frame_ids": [1],
    },
    # {
    #     "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/livogs_rd",
    #     "frame_ids": [1],
    # },
]

OUTPUT_CSV: Optional[str] = None

# Maximum parallel workers for I/O (tune for your NFS / local disk)
MAX_WORKERS = 32


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "sequence_name",
    "frame_id",
    "depth",
    "baseline_qp",
    "beta",
    "qp_opacity",
    "qp_scales",
    "qp_quats",
    "gt_psnr",
    "gt_ssim",
    "decomp_psnr",
    "decomp_ssim",
    "psnr_drop",
    "ssim_drop",
    "size_bytes",
    "compressed_mb",
    "label",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_frame_id(dirname: str) -> Optional[int]:
    """Extract frame id from a directory name like ``frame_0``."""
    m = re.match(r"^frame_(\d+)$", dirname)
    return int(m.group(1)) if m else None


def _parse_depth(dirname: str) -> Optional[int]:
    """Extract depth (J) from a directory name like ``J_14``."""
    m = re.match(r"^J_(\d+)$", dirname)
    return int(m.group(1)) if m else None


def _frame_id_matches(stored_value: Any, target_frame_id: int) -> bool:
    """Flexible frame-id comparison (handles int, str, zero-padded str)."""
    if stored_value == target_frame_id:
        return True
    try:
        return int(stored_value) == target_frame_id
    except (TypeError, ValueError):
        return str(stored_value) == str(target_frame_id)


def _infer_sequence_name(rd_root: str) -> str:
    """Infer a sequence name from the rd_root path.

    Expects ``…/<something>/compression/livogs_rd``.
    Returns ``<something>`` with a ``queen_compressed_`` prefix stripped.
    """
    parts = os.path.normpath(rd_root).split(os.sep)
    # Walk backwards to find "compression" and take the part before it
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "compression" or parts[i] == "livogs_rd":
            continue
        name = parts[i]
        # Strip queen-specific prefix for cleaner names
        if name.startswith("queen_compressed_"):
            name = name[len("queen_compressed_"):]
        return name
    # Fallback: last non-livogs_rd component
    return os.path.basename(os.path.dirname(rd_root))


def load_experiment(
    exp_dir: str,
    frame_id: int,
) -> Optional[dict[str, Any]]:
    """Load one experiment's QP config, benchmark, and evaluation results.

    Returns a flat dict with the target CSV columns, or None on failure.
    """
    qp_config_path = os.path.join(exp_dir, "qp_config.json")
    benchmark_path = os.path.join(exp_dir, "benchmark_livogs.csv")
    eval_json_path = os.path.join(exp_dir, "evaluation", "evaluation_results.json")

    # --- QP config ---
    try:
        with open(qp_config_path, encoding="utf-8") as f:
            qp_cfg = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    quantize_cfg = qp_cfg.get("quantize_config", {})
    try:
        baseline_qp = float(qp_cfg.get("baseline_qp", qp_cfg.get("sh_qp")))
        beta = float(qp_cfg.get("beta", 0))
        qp_quats = float(qp_cfg.get("qp_quats", quantize_cfg.get("quats", 0)))
        qp_scales = float(qp_cfg.get("qp_scales", quantize_cfg.get("scales", 0)))
        qp_opacity = float(qp_cfg.get("qp_opacity", quantize_cfg.get("opacity", 0)))
    except (TypeError, ValueError):
        return None

    # --- Benchmark (compressed size) ---
    compressed_bytes: Optional[int] = None
    try:
        with open(benchmark_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if _frame_id_matches(row["frame_id"], frame_id):
                    compressed_bytes = int(row["compressed_size_bytes"])
                    break
    except (OSError, KeyError, ValueError):
        return None
    if compressed_bytes is None:
        return None

    # --- Evaluation results ---
    try:
        with open(eval_json_path, encoding="utf-8") as f:
            eval_data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None

    gt_psnr: Optional[float] = None
    gt_ssim: Optional[float] = None
    decomp_psnr: Optional[float] = None
    decomp_ssim: Optional[float] = None

    # Try per-frame first, fall back to summary
    for fr in eval_data.get("per_frame", []):
        if _frame_id_matches(fr["frame"], frame_id):
            gt_psnr = float(fr["gt_psnr"])
            gt_ssim = float(fr["gt_ssim"])
            decomp_psnr = float(fr["decomp_psnr"])
            decomp_ssim = float(fr["decomp_ssim"])
            break

    if decomp_psnr is None:
        summary = eval_data.get("summary", {})
        gt_psnr = summary.get("gt_psnr")
        gt_ssim = summary.get("gt_ssim")
        decomp_psnr = summary.get("decomp_psnr")
        decomp_ssim = summary.get("decomp_ssim")

    if decomp_psnr is None or gt_psnr is None:
        return None
    if decomp_ssim is None or gt_ssim is None:
        return None

    label = qp_cfg.get("label", os.path.basename(exp_dir))

    return {
        "baseline_qp": baseline_qp,
        "beta": beta,
        "qp_opacity": qp_opacity,
        "qp_scales": qp_scales,
        "qp_quats": qp_quats,
        "gt_psnr": gt_psnr,
        "gt_ssim": gt_ssim,
        "decomp_psnr": decomp_psnr,
        "decomp_ssim": decomp_ssim,
        "psnr_drop": gt_psnr - decomp_psnr,
        "ssim_drop": gt_ssim - decomp_ssim,
        "size_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / (1024 * 1024),
        "label": label,
    }


def collect_rd_root(
    rd_root: str,
    sequence_name: str,
    frame_ids: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """Scan all experiment directories under one livogs_rd/ root (parallel I/O)."""
    if not os.path.isdir(rd_root):
        print(f"[WARN] RD output root not found: {rd_root}")
        return []

    # Discover frame directories and sort numerically
    frame_dirs: list[tuple[int, str]] = []
    for entry in os.listdir(rd_root):
        fid = _parse_frame_id(entry)
        if fid is not None:
            frame_dirs.append((fid, os.path.join(rd_root, entry)))
    frame_dirs.sort()

    if frame_ids is not None:
        allowed = set(frame_ids)
        frame_dirs = [(fid, p) for fid, p in frame_dirs if fid in allowed]

    # Build flat list of (exp_dir, frame_id, depth) jobs
    jobs: list[tuple[str, int, int]] = []
    for frame_id, frame_path in frame_dirs:
        for entry in os.listdir(frame_path):
            d = _parse_depth(entry)
            if d is None:
                continue
            depth_path = os.path.join(frame_path, entry)
            for label in os.listdir(depth_path):
                exp_dir = os.path.join(depth_path, label)
                if os.path.isdir(exp_dir):
                    jobs.append((exp_dir, frame_id, d))

    if not jobs:
        return []

    # Parallel load
    total = len(jobs)
    done = 0
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total)) as pool:
        futures = {
            pool.submit(load_experiment, exp_dir, fid): (fid, depth)
            for exp_dir, fid, depth in jobs
        }
        for future in as_completed(futures):
            done += 1
            print(f"\r  {done}/{total} experiments loaded", end="", flush=True)
            fid, depth = futures[future]
            result = future.result()
            if result is None:
                continue
            result["sequence_name"] = sequence_name
            result["frame_id"] = fid
            result["depth"] = depth
            rows.append(result)
    print()

    # Sort for deterministic output: depth descending, then by qp values
    rows.sort(key=lambda r: (-r["depth"], r["baseline_qp"], r["qp_quats"], r["qp_scales"], r["qp_opacity"]))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict[str, Any]] = []

    for entry in RD_OUTPUT_ROOTS:
        rd_root = entry["path"]
        seq_name = entry.get("name") or _infer_sequence_name(rd_root)
        frame_ids = entry.get("frame_ids")

        print(f"Scanning: {seq_name}  ({rd_root})")
        rows = collect_rd_root(rd_root, seq_name, frame_ids=frame_ids)
        all_rows.extend(rows)
        print(f"  Collected {len(rows)} result(s)")

    if not all_rows:
        print("[WARN] No results collected.")
        return

    output_path = OUTPUT_CSV
    if output_path is None:
        first_root = RD_OUTPUT_ROOTS[0]["path"]
        output_path = os.path.join(first_root, "collected_rd_results.csv")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})

    print(f"\nWrote {len(all_rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()
