#!/usr/bin/env python3
"""Collect GPCC RD experiment results into a single CSV.

Walks the experiment directory tree and produces one row per
(experiment, frame) with columns:

    experiment_name, frame_idx, f_rest_qp, f_dc_qp, opacity_qp,
    gt_psnr, gt_ssim, decomp_psnr, decomp_ssim, psnr_drop, ssim_drop,
    total_compressed_bytes, compressed_mb, opacity_bytes, dc_bytes,
    rest_bytes, scale_bytes, rot_bytes

Usage:
    python gpcc_baseline/collect_rd_results.py --experiment_dir <path> [--output_csv <path>] [--frame_idx <int>]
"""

import argparse
import csv
import json
import os
from typing import Any, Optional


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "experiment_name",
    "frame_idx",
    "f_rest_qp",
    "f_dc_qp",
    "opacity_qp",
    "gt_psnr",
    "gt_ssim",
    "decomp_psnr",
    "decomp_ssim",
    "psnr_drop",
    "ssim_drop",
    "total_compressed_bytes",
    "compressed_mb",
    "opacity_bytes",
    "dc_bytes",
    "rest_bytes",
    "scale_bytes",
    "rot_bytes",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_experiment(
    exp_dir: str,
    frame_idx: int,
) -> Optional[dict[str, Any]]:
    """Load one experiment's benchmark and evaluation results.

    Returns a flat dict with the target CSV columns, or None on failure.
    """
    benchmark_path = os.path.join(exp_dir, "benchmark_gpcc.csv")
    eval_json_path = os.path.join(exp_dir, "evaluation", "evaluation_results.json")

    # --- Parse experiment name to get QP values ---
    exp_name = os.path.basename(exp_dir)
    f_rest_qp: Optional[int] = None
    f_dc_qp: Optional[int] = None
    opacity_qp: Optional[int] = None

    try:
        parts = exp_name.split("_")
        for p in parts:
            if p.startswith("rest"):
                f_rest_qp = int(p[4:])
            elif p.startswith("dc"):
                f_dc_qp = int(p[2:])
            elif p.startswith("opacity"):
                opacity_qp = int(p[7:])
    except (ValueError, IndexError):
        pass

    # --- Benchmark (compressed size) ---
    total_bytes: Optional[int] = None
    opacity_bytes: int = 0
    dc_bytes: int = 0
    rest_bytes: int = 0
    scale_bytes: int = 0
    rot_bytes: int = 0

    if not os.path.exists(benchmark_path):
        return None

    try:
        with open(benchmark_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["frame_idx"]) == frame_idx:
                    total_bytes = int(row["total_compressed_bytes"])
                    opacity_bytes = int(row.get("opacity_bytes", 0))
                    dc_bytes = int(row.get("dc_bytes", 0))
                    rest_bytes = int(row.get("rest_bytes", 0))
                    scale_bytes = int(row.get("scale_bytes", 0))
                    rot_bytes = int(row.get("rot_bytes", 0))
                    break
    except (OSError, KeyError, ValueError):
        return None

    if total_bytes is None:
        return None

    # --- Evaluation results (optional) ---
    gt_psnr: Optional[float] = None
    gt_ssim: Optional[float] = None
    decomp_psnr: Optional[float] = None
    decomp_ssim: Optional[float] = None

    if os.path.exists(eval_json_path):
        try:
            with open(eval_json_path, encoding="utf-8") as f:
                eval_data = json.load(f)

            # Try per-frame first, fall back to summary
            for fr in eval_data.get("per_frame", []):
                if int(fr["frame"]) == frame_idx:
                    gt_psnr = float(fr.get("gt_psnr", 0))
                    gt_ssim = float(fr.get("gt_ssim", 0))
                    decomp_psnr = float(fr.get("decomp_psnr", 0))
                    decomp_ssim = float(fr.get("decomp_ssim", 0))
                    break

            if decomp_psnr is None:
                summary = eval_data.get("summary", {})
                gt_psnr = summary.get("gt_psnr")
                gt_ssim = summary.get("gt_ssim")
                decomp_psnr = summary.get("decomp_psnr")
                decomp_ssim = summary.get("decomp_ssim")
        except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass

    return {
        "experiment_name": exp_name,
        "frame_idx": frame_idx,
        "f_rest_qp": f_rest_qp,
        "f_dc_qp": f_dc_qp,
        "opacity_qp": opacity_qp,
        "gt_psnr": gt_psnr,
        "gt_ssim": gt_ssim,
        "decomp_psnr": decomp_psnr,
        "decomp_ssim": decomp_ssim,
        "psnr_drop": (gt_psnr - decomp_psnr) if (gt_psnr and decomp_psnr) else None,
        "ssim_drop": (gt_ssim - decomp_ssim) if (gt_ssim and decomp_ssim) else None,
        "total_compressed_bytes": total_bytes,
        "compressed_mb": total_bytes / (1024 * 1024),
        "opacity_bytes": opacity_bytes,
        "dc_bytes": dc_bytes,
        "rest_bytes": rest_bytes,
        "scale_bytes": scale_bytes,
        "rot_bytes": rot_bytes,
    }


def collect_experiments(
    experiment_dir: str,
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Scan all experiment directories and collect results."""
    if not os.path.isdir(experiment_dir):
        print(f"[WARN] Experiment directory not found: {experiment_dir}")
        return []

    # Discover experiment directories (direct subdirectories)
    exp_dirs: list[str] = []
    try:
        for entry in os.listdir(experiment_dir):
            exp_path = os.path.join(experiment_dir, entry)
            if os.path.isdir(exp_path):
                exp_dirs.append(exp_path)
    except OSError as e:
        print(f"[WARN] Failed to list experiment directory: {e}")
        return []

    if not exp_dirs:
        print(f"[WARN] No experiment directories found in: {experiment_dir}")
        return []

    # Load each experiment
    rows: list[dict[str, Any]] = []
    for exp_path in sorted(exp_dirs):
        result = load_experiment(exp_path, frame_idx)
        if result is not None:
            rows.append(result)

    return rows


def _write_rows_to_csv(rows: list[dict[str, Any]], output_path: str) -> None:
    """Write rows to CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect GPCC RD experiment results into a CSV."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Root directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path (default: experiment_dir/collected_rd_results.csv)",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index to collect (default: 0)",
    )

    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    frame_idx = args.frame_idx
    output_csv = args.output_csv or os.path.join(experiment_dir, "collected_rd_results.csv")

    print(f"Scanning: {experiment_dir}")
    rows = collect_experiments(experiment_dir, frame_idx)
    print(f"  Collected {len(rows)} result(s)")

    _write_rows_to_csv(rows, output_csv)
    print(f"  Wrote {len(rows)} row(s) to: {output_csv}")


if __name__ == "__main__":
    main()
