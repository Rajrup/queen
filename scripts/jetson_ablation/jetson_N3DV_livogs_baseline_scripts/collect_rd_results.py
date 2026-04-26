#!/usr/bin/env python3
"""Collect LiVoGS RD experiment results into a single CSV.

Walks the experiment directory tree and produces one row per
(sequence, frame, depth, qp-combination) with columns:

    sequencename, frameid, depth, qp_sh, beta, opacity, scales, quats,
    gtpsnr, gtssim, psnr, ssim, psnrdrop, ssimdrop, size

Edit the "Global configuration" section below, then run::

    python scripts/livogs_baseline/collect_rd_results.py
"""

import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Optional


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
    "qp_sh",
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
    "position_compressed_bytes",
    "attribute_compressed_bytes",
    # Per-dimension compressed bytes are appended dynamically; see
    # _build_csv_columns() and _detect_per_dim_columns().
    "label",
]


_PER_DIM_RE = re.compile(r"^(quats|scales|opacity|sh)_dim(\d+)_compressed_bytes$")


def _per_dim_sort_key(col: str) -> tuple[int, int]:
    """Sort key for per-dim column names: group order then dim number."""
    m = _PER_DIM_RE.match(col)
    assert m is not None
    group_order = {"quats": 0, "scales": 1, "opacity": 2, "sh": 3}
    return (group_order[m.group(1)], int(m.group(2)))


def _detect_per_dim_columns(keys: Iterable[str]) -> list[str]:
    """Return sorted per-dimension compressed-bytes column names from keys."""
    cols = [k for k in keys if _PER_DIM_RE.match(k)]
    cols.sort(key=_per_dim_sort_key)
    return cols


def _build_csv_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Build full CSV column list by inserting per-dim columns before 'label'."""
    if not rows:
        return list(CSV_COLUMNS)
    # Detect per-dim columns from the first row's keys
    per_dim_cols = _detect_per_dim_columns(rows[0].keys())
    # Insert per-dim columns before 'label' (last element of CSV_COLUMNS)
    return CSV_COLUMNS[:-1] + per_dim_cols + CSV_COLUMNS[-1:]


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


def _discover_jobs(rd_root: str, frame_ids: Optional[list[int]]) -> list[tuple[str, int, int]]:
    frame_dirs: list[tuple[int, str]] = []
    allowed = set(frame_ids) if frame_ids is not None else None

    with os.scandir(rd_root) as frame_entries:
        for frame_entry in frame_entries:
            if not frame_entry.is_dir(follow_symlinks=False):
                continue
            fid = _parse_frame_id(frame_entry.name)
            if fid is None:
                continue
            if allowed is not None and fid not in allowed:
                continue
            frame_dirs.append((fid, frame_entry.path))

    frame_dirs.sort()

    jobs: list[tuple[str, int, int]] = []
    for frame_id, frame_path in frame_dirs:
        with os.scandir(frame_path) as depth_entries:
            for depth_entry in depth_entries:
                if not depth_entry.is_dir(follow_symlinks=False):
                    continue
                depth = _parse_depth(depth_entry.name)
                if depth is None:
                    continue
                with os.scandir(depth_entry.path) as label_entries:
                    for label_entry in label_entries:
                        if label_entry.is_dir(follow_symlinks=False):
                            jobs.append((label_entry.path, frame_id, depth))

    return jobs


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["depth"],
        row["frame_id"],
        row["qp_sh"],
        row["beta"],
        row["qp_quats"],
        row["qp_scales"],
        row["qp_opacity"],
        row["label"],
        row["size_bytes"],
        row["position_compressed_bytes"],
    )


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
        qp_sh = float(qp_cfg.get("qp_sh", qp_cfg.get("sh_qp", qp_cfg.get("baseline_qp"))))
        beta = float(qp_cfg.get("beta", 0))
        qp_quats = float(qp_cfg.get("qp_quats", quantize_cfg.get("quats", 0)))
        qp_scales = float(qp_cfg.get("qp_scales", quantize_cfg.get("scales", 0)))
        qp_opacity = float(qp_cfg.get("qp_opacity", quantize_cfg.get("opacity", 0)))
    except (TypeError, ValueError):
        return None

    # --- Benchmark (compressed size + per-dim breakdown) ---
    compressed_bytes: Optional[int] = None
    position_compressed_bytes: int = 0
    attribute_compressed_bytes: int = 0
    per_dim_bytes: dict[str, int] = {}
    try:
        with open(benchmark_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            per_dim_cols = _detect_per_dim_columns(reader.fieldnames or [])
            for row in reader:
                if _frame_id_matches(row["frame_id"], frame_id):
                    compressed_bytes = int(row["compressed_size_bytes"])
                    position_compressed_bytes = int(row.get("position_compressed_bytes", 0))
                    attribute_compressed_bytes = int(row.get("attribute_compressed_bytes", 0))
                    for col in per_dim_cols:
                        per_dim_bytes[col] = int(row.get(col, 0))
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
        "qp_sh": qp_sh,
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
        "position_compressed_bytes": position_compressed_bytes,
        "attribute_compressed_bytes": attribute_compressed_bytes,
        **per_dim_bytes,
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

    jobs = _discover_jobs(rd_root, frame_ids)

    if not jobs:
        return []

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

    rows.sort(key=_row_sort_key)
    return rows


def _default_output_csv_for_root(rd_root: str) -> str:
    return os.path.join(rd_root, "collected_rd_results.csv")


def _write_rows_to_csv(rows: list[dict[str, Any]], output_path: str) -> None:
    columns = _build_csv_columns(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_rows = 0
    written_files = 0

    if OUTPUT_CSV is not None and len(RD_OUTPUT_ROOTS) > 1:
        print("[WARN] OUTPUT_CSV is ignored when multiple RD_OUTPUT_ROOTS are configured.")
        print("       Writing one CSV per root instead.")

    for entry in RD_OUTPUT_ROOTS:
        rd_root = entry["path"]
        seq_name = entry.get("name") or _infer_sequence_name(rd_root)
        frame_ids = entry.get("frame_ids")
        output_path = entry.get("output_csv")
        if output_path is None:
            if OUTPUT_CSV is not None and len(RD_OUTPUT_ROOTS) == 1:
                output_path = OUTPUT_CSV
            else:
                output_path = _default_output_csv_for_root(rd_root)

        print(f"Scanning: {seq_name}  ({rd_root})")
        rows = collect_rd_root(rd_root, seq_name, frame_ids=frame_ids)
        print(f"  Collected {len(rows)} result(s)")

        _write_rows_to_csv(rows, output_path)
        written_files += 1
        total_rows += len(rows)
        print(f"  Wrote {len(rows)} row(s) to: {output_path}")

    if written_files == 0:
        print("[WARN] No RD roots were processed.")
        return

    if total_rows == 0:
        print("[WARN] No results collected across all roots.")

    print(f"\nWrote {total_rows} total rows across {written_files} file(s).")


if __name__ == "__main__":
    main()
