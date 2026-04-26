#!/usr/bin/env python3
"""Plot compressed-size breakdown for RD operating points from collected CSV.

Generates three plots per sequence/frame:
  1. Stacked bar chart of convex-hull operating points (sorted by total size)
  2. Normalized (100%) stacked bar of hull points (proportion analysis)
  3. All-points RD scatter colored by dominant size component

Uses the same collection pipeline as plot_rd_hull_results.py.
Edit RD_OUTPUT_ROOTS and SCATTER_SPEC below, then run::

    python scripts/livogs_baseline/plot_rd_size_breakdown.py
"""

import csv
import os
import re
import sys
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from collect_rd_results import (
    _build_csv_columns,
    _infer_sequence_name,
    collect_rd_root,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global configuration  (edit here — mirrors plot_rd_hull_results.py)
# ---------------------------------------------------------------------------

# Each entry is a dict describing one livogs_rd/ root to scan.
# Required key:
#   "path"       — absolute path to a livogs_rd/ directory
# Optional keys:
#   "name"       — sequence name override (default: inferred from path)
#   "frame_ids"  — list of frame IDs to collect (default: all frames)
RD_OUTPUT_ROOTS: list[dict[str, Any]] = [
    {
        "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/compression/livogs_rd",
        "frame_ids": [1],
    },
    {
        "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/livogs_rd",
        "frame_ids": [1],
    },
]

SCATTER_SPEC: dict[str, Any] = {
    "name": "default_qps",
    "fixed": {
        "depth": [12, 13, 14, 15, 16, 17, 18],
        "qp_sh": [v / 255.0 for v in (0.01, 0.1, 0.5, 1, 2, 4, 8, 16)],
        "qp_opacity": [0.001, 0.01, 0.02, 0.04, 0.06],
        "qp_scales": [0.001, 0.01, 0.02, 0.04, 0.06],
        "qp_quats": [0.001, 0.01, 0.02, 0.04, 0.06],
    },
}

FORCE_COLLECT: bool = False
PLOT_OUTPUT_DIR: Optional[str] = None
COLLECTED_CSV: Optional[str] = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOB_NAMES = frozenset({"depth", "qp_sh", "beta", "qp_quats", "qp_scales", "qp_opacity"})

_PER_DIM_RE = re.compile(r"^(quats|scales|opacity|sh)_dim(\d+)_compressed_bytes$")

DEDUP_COMBO_KEYS = ("depth", "qp_sh", "qp_opacity", "qp_scales", "qp_quats")

# Component display order and colors (consistent with plot_compressed_size.py)
COMPONENT_ORDER = ["Position", "Quats", "Scales", "Opacity", "SH DC", "SH Rest"]
COMPONENT_COLORS = ["seagreen", "steelblue", "coral", "goldenrod", "mediumpurple", "lightcoral"]

FALLBACK_COMPONENT_ORDER = ["Position", "Attributes"]
FALLBACK_COMPONENT_COLORS = ["seagreen", "steelblue"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round6(value: float) -> float:
    return round(float(value), 6)


def _matches_fixed(row: dict[str, Any], fixed: dict[str, Any]) -> bool:
    for key, target in fixed.items():
        row_value = row.get(key)
        if row_value is None:
            return False
        if isinstance(target, (list, tuple)):
            if not any(_round6(float(row_value)) == _round6(float(tv)) for tv in target):
                return False
        else:
            if _round6(float(row_value)) != _round6(float(target)):
                return False
    return True


def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _upper_convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper convex hull of 2D points (x=rate, y=quality), sorted by x."""
    sorted_pts = sorted(set(points))
    if len(sorted_pts) <= 1:
        return list(sorted_pts)
    upper: list[tuple[float, float]] = []
    for p in sorted_pts:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(p)
    return upper


def _is_better_row(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    cand_psnr = float(candidate.get("decomp_psnr") or -1e18)
    cur_psnr = float(current.get("decomp_psnr") or -1e18)
    if cand_psnr > cur_psnr + 1e-12:
        return True
    if cur_psnr > cand_psnr + 1e-12:
        return False
    cand_size = float(candidate.get("compressed_mb") or 1e18)
    cur_size = float(current.get("compressed_mb") or 1e18)
    return cand_size < cur_size - 1e-12


def _deduplicate_operating_points(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    best_by_combo: dict[tuple[float, ...], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        if any(row.get(k) is None for k in DEDUP_COMBO_KEYS):
            passthrough.append(row)
            continue
        combo = tuple(_round6(float(row[k])) for k in DEDUP_COMBO_KEYS)
        prev = best_by_combo.get(combo)
        if prev is None or _is_better_row(row, prev):
            best_by_combo[combo] = row
    deduped = list(best_by_combo.values()) + passthrough
    removed = len(rows) - len(deduped)
    deduped.sort(key=lambda r: float(r.get("compressed_mb") or 1e18))
    return deduped, removed


def _per_dim_sort_key(col: str) -> tuple[int, int]:
    m = _PER_DIM_RE.match(col)
    assert m is not None
    group_order = {"quats": 0, "scales": 1, "opacity": 2, "sh": 3}
    return (group_order[m.group(1)], int(m.group(2)))


def _detect_per_dim_columns(keys: Any) -> list[str]:
    cols = [k for k in keys if _PER_DIM_RE.match(k)]
    cols.sort(key=_per_dim_sort_key)
    return cols


def _aggregate_components(row: dict[str, Any], per_dim_cols: list[str]) -> dict[str, float]:
    """Aggregate per-dimension compressed bytes into component groups (in MB)."""
    to_mb = 1.0 / (1024 * 1024)

    position = float(row.get("position_compressed_bytes") or 0) * to_mb

    quats_total = (
        sum(float(row.get(c, 0) or 0) for c in per_dim_cols if c.startswith("quats_dim"))
        * to_mb
    )
    scales_total = (
        sum(float(row.get(c, 0) or 0) for c in per_dim_cols if c.startswith("scales_dim"))
        * to_mb
    )
    opacity_total = (
        sum(float(row.get(c, 0) or 0) for c in per_dim_cols if c.startswith("opacity_dim"))
        * to_mb
    )

    # SH dims: 0-2 = DC (3 color channels), 3+ = rest coefficients
    sh_cols = sorted(
        [c for c in per_dim_cols if c.startswith("sh_dim")],
        key=lambda c: int(_PER_DIM_RE.match(c).group(2)),  # type: ignore[union-attr]
    )
    sh_dc_total = sum(float(row.get(c, 0) or 0) for c in sh_cols[:3]) * to_mb
    sh_rest_total = sum(float(row.get(c, 0) or 0) for c in sh_cols[3:]) * to_mb

    return {
        "Position": position,
        "Quats": quats_total,
        "Scales": scales_total,
        "Opacity": opacity_total,
        "SH DC": sh_dc_total,
        "SH Rest": sh_rest_total,
    }


def _aggregate_components_fallback(row: dict[str, Any]) -> dict[str, float]:
    """Fallback aggregation when only coarse position/attribute bytes exist."""
    to_mb = 1.0 / (1024 * 1024)
    position = float(row.get("position_compressed_bytes") or 0) * to_mb
    attributes = float(row.get("attribute_compressed_bytes") or 0) * to_mb
    if attributes <= 0:
        total_mb = float(row.get("compressed_mb") or 0)
        attributes = max(0.0, total_mb - position)
    return {
        "Position": position,
        "Attributes": attributes,
    }


def _dominant_component(components: dict[str, float]) -> str:
    """Return the name of the largest component."""
    return max(components, key=lambda k: components[k])


# ---------------------------------------------------------------------------
# Config resolution (same as plot_rd_hull_results.py)
# ---------------------------------------------------------------------------


def _infer_dataset_name(rd_root: str) -> str:
    parts = os.path.normpath(rd_root).split(os.sep)
    for i, part in enumerate(parts):
        if part == "pretrained_output" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown_dataset"


def _default_collected_csv(rd_root: str) -> str:
    return os.path.join(rd_root, "collected_rd_results.csv")


def _default_plot_dir(rd_root: str) -> str:
    return os.path.join(rd_root, "plots")


def _csv_has_breakdown_columns(csv_path: str) -> bool:
    """Return True if collected CSV has either detailed or coarse breakdown columns."""
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            has_per_dim = len(_detect_per_dim_columns(reader.fieldnames)) > 0
            has_coarse = "attribute_compressed_bytes" in reader.fieldnames
            return has_per_dim or has_coarse
    except OSError:
        return False


def _parse_scatter_spec(spec: dict[str, Any]) -> dict[str, Any]:
    raw_fixed = spec.get("fixed", {})
    if not isinstance(raw_fixed, dict):
        raw_fixed = {}
    invalid_keys = [k for k in raw_fixed if k not in KNOB_NAMES]
    if invalid_keys:
        print(f"[WARN] Invalid fixed keys: {invalid_keys}. Ignoring them.")
        raw_fixed = {k: v for k, v in raw_fixed.items() if k in KNOB_NAMES}
    name = spec.get("name")
    return {"name": str(name) if name is not None else None, "fixed": raw_fixed}


# ---------------------------------------------------------------------------
# Collection (mirrors plot_rd_hull_results.py, but uses _build_csv_columns
# to preserve per-dimension breakdown columns)
# ---------------------------------------------------------------------------


def collect_all() -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []

    if COLLECTED_CSV is not None and len(RD_OUTPUT_ROOTS) > 1:
        print("[WARN] COLLECTED_CSV is ignored when multiple RD_OUTPUT_ROOTS are configured.")
        print("       Writing one CSV per root instead.")

    for entry in RD_OUTPUT_ROOTS:
        rd_root = entry["path"]
        seq_name = entry.get("name") or _infer_sequence_name(rd_root)
        dataset_name = entry.get("dataset") or _infer_dataset_name(rd_root)
        frame_ids = entry.get("frame_ids")
        csv_path = entry.get("output_csv")
        if csv_path is None:
            if COLLECTED_CSV is not None and len(RD_OUTPUT_ROOTS) == 1:
                csv_path = COLLECTED_CSV
            else:
                csv_path = _default_collected_csv(rd_root)

        if not FORCE_COLLECT and os.path.exists(csv_path):
            if _csv_has_breakdown_columns(csv_path):
                print(f"  CSV already exists, skipping collect: {csv_path}")
                collected.append({
                    "rd_root": rd_root,
                    "sequence_name": seq_name,
                    "dataset_name": dataset_name,
                    "csv_path": csv_path,
                })
                continue
            print(
                "  CSV exists but misses breakdown columns; "
                f"re-collecting: {csv_path}"
            )

        print(f"Collecting: {seq_name}  ({rd_root})")
        rows = collect_rd_root(rd_root, seq_name, frame_ids=frame_ids)
        print(f"  {len(rows)} result(s)")

        # Use _build_csv_columns to include per-dim breakdown columns
        columns = _build_csv_columns(rows)
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in columns})

        print(f"  Wrote {len(rows)} rows to: {csv_path}\n")
        collected.append({
            "rd_root": rd_root,
            "sequence_name": seq_name,
            "dataset_name": dataset_name,
            "csv_path": csv_path,
        })

    if not collected:
        print("[WARN] No RD roots processed.")

    return collected


# ---------------------------------------------------------------------------
# Load + filter + parse rows from collected CSV
# ---------------------------------------------------------------------------

_NUMERIC_PARSE_KEYS = frozenset({
    "compressed_mb", "size_bytes", "decomp_psnr", "gt_psnr",
    "position_compressed_bytes",
    "attribute_compressed_bytes",
})


def _load_csv(csv_path: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Load collected CSV, return (parsed_rows, per_dim_cols)."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return [], []

    all_rows: list[dict[str, Any]] = []
    per_dim_cols: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            per_dim_cols = _detect_per_dim_columns(reader.fieldnames)
        numeric_keys = _NUMERIC_PARSE_KEYS | set(per_dim_cols)

        for row in reader:
            parsed: dict[str, Any] = dict(row)
            # Parse knob fields
            for key in KNOB_NAMES:
                raw = parsed.get(key)
                if raw in (None, ""):
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(raw)
                    except (TypeError, ValueError):
                        parsed[key] = None
            # Parse numeric fields
            for key in numeric_keys:
                raw = parsed.get(key)
                if raw in (None, ""):
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(raw)
                    except (TypeError, ValueError):
                        parsed[key] = None
            all_rows.append(parsed)

    return all_rows, per_dim_cols


def _filter_rows(
    all_rows: list[dict[str, Any]],
    sequence_name: str,
    frame_id: int,
    fixed: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter rows by sequence, frame, fixed knobs, and valid size+quality."""
    filtered = [
        r for r in all_rows
        if r.get("sequence_name") == sequence_name
        and r.get("frame_id") is not None
        and str(r.get("frame_id")) == str(frame_id)
    ]
    if fixed:
        filtered = [r for r in filtered if _matches_fixed(r, fixed)]
    filtered = [
        r for r in filtered
        if r.get("compressed_mb") is not None and r.get("decomp_psnr") is not None
    ]
    return filtered


# ---------------------------------------------------------------------------
# Plot 1: Stacked bar chart of hull points
# ---------------------------------------------------------------------------


def _plot_hull_stacked_bar(
    hull_rows: list[dict[str, Any]],
    per_dim_cols: list[str],
    sequence_name: str,
    dataset_name: str,
    frame_id: int,
    output_path: str,
) -> None:
    """Stacked bar chart: hull points sorted by total size, broken into components."""
    if not hull_rows:
        print(f"[WARN] No hull points for stacked bar: {sequence_name} frame {frame_id}")
        return

    hull_rows = sorted(hull_rows, key=lambda r: float(r.get("compressed_mb") or 0))
    n = len(hull_rows)

    component_data: dict[str, list[float]] = {comp: [] for comp in COMPONENT_ORDER}
    total_sizes: list[float] = []
    psnr_values: list[float] = []

    for row in hull_rows:
        comps = _aggregate_components(row, per_dim_cols)
        for comp in COMPONENT_ORDER:
            component_data[comp].append(comps.get(comp, 0))
        total_sizes.append(float(row.get("compressed_mb") or 0))
        psnr_values.append(float(row.get("decomp_psnr") or 0))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6 + 2), 6))
    x = np.arange(n)
    bar_width = 0.7

    bottom = np.zeros(n)
    for comp, color in zip(COMPONENT_ORDER, COMPONENT_COLORS):
        values = np.array(component_data[comp])
        if values.sum() < 1e-12:
            continue
        ax.bar(x, values, bar_width, bottom=bottom, label=comp, color=color, alpha=0.9)
        bottom += values

    # X-axis labels: total size + PSNR
    labels = [f"{sz:.2f} MB\n{psnr:.1f} dB" for sz, psnr in zip(total_sizes, psnr_values)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")

    ax.set_xlabel("Hull Operating Points (sorted by total compressed size)")
    ax.set_ylabel("Compressed Size (MB)")
    ax.set_title(
        f"Size Breakdown — Hull Points\n{dataset_name} / {sequence_name} frame {frame_id}"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved hull stacked bar: {output_path}")


def _plot_hull_stacked_bar_fallback(
    hull_rows: list[dict[str, Any]],
    sequence_name: str,
    dataset_name: str,
    frame_id: int,
    output_path: str,
) -> None:
    """Fallback stacked bar with coarse Position vs Attributes."""
    if not hull_rows:
        print(f"[WARN] No hull points for stacked bar: {sequence_name} frame {frame_id}")
        return

    hull_rows = sorted(hull_rows, key=lambda r: float(r.get("compressed_mb") or 0))
    n = len(hull_rows)

    component_data: dict[str, list[float]] = {comp: [] for comp in FALLBACK_COMPONENT_ORDER}
    total_sizes: list[float] = []
    psnr_values: list[float] = []

    for row in hull_rows:
        comps = _aggregate_components_fallback(row)
        for comp in FALLBACK_COMPONENT_ORDER:
            component_data[comp].append(comps.get(comp, 0))
        total_sizes.append(float(row.get("compressed_mb") or 0))
        psnr_values.append(float(row.get("decomp_psnr") or 0))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6 + 2), 6))
    x = np.arange(n)
    bar_width = 0.7

    bottom = np.zeros(n)
    for comp, color in zip(FALLBACK_COMPONENT_ORDER, FALLBACK_COMPONENT_COLORS):
        values = np.array(component_data[comp])
        if values.sum() < 1e-12:
            continue
        ax.bar(x, values, bar_width, bottom=bottom, label=comp, color=color, alpha=0.9)
        bottom += values

    labels = [f"{sz:.2f} MB\n{psnr:.1f} dB" for sz, psnr in zip(total_sizes, psnr_values)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")

    ax.set_xlabel("Hull Operating Points (sorted by total compressed size)")
    ax.set_ylabel("Compressed Size (MB)")
    ax.set_title(
        f"Size Breakdown (coarse) — Hull Points\n{dataset_name} / {sequence_name} frame {frame_id}"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved hull stacked bar (coarse): {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Normalized (100%) stacked bar of hull points
# ---------------------------------------------------------------------------


def _plot_hull_normalized_bar(
    hull_rows: list[dict[str, Any]],
    per_dim_cols: list[str],
    sequence_name: str,
    dataset_name: str,
    frame_id: int,
    output_path: str,
) -> None:
    """100% normalized stacked bar: proportions of each component at hull points."""
    if not hull_rows:
        print(f"[WARN] No hull points for normalized bar: {sequence_name} frame {frame_id}")
        return

    hull_rows = sorted(hull_rows, key=lambda r: float(r.get("compressed_mb") or 0))
    n = len(hull_rows)

    component_data: dict[str, list[float]] = {comp: [] for comp in COMPONENT_ORDER}
    total_sizes: list[float] = []
    psnr_values: list[float] = []

    for row in hull_rows:
        comps = _aggregate_components(row, per_dim_cols)
        total = sum(comps.values())
        if total < 1e-12:
            total = 1.0  # avoid division by zero
        for comp in COMPONENT_ORDER:
            component_data[comp].append(100.0 * comps.get(comp, 0) / total)
        total_sizes.append(float(row.get("compressed_mb") or 0))
        psnr_values.append(float(row.get("decomp_psnr") or 0))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6 + 2), 6))
    x = np.arange(n)
    bar_width = 0.7

    bottom = np.zeros(n)
    for comp, color in zip(COMPONENT_ORDER, COMPONENT_COLORS):
        values = np.array(component_data[comp])
        if values.sum() < 1e-12:
            continue
        ax.bar(x, values, bar_width, bottom=bottom, label=comp, color=color, alpha=0.9)
        # Add percentage label in the middle of each segment if > 8%
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 8:
                ax.text(
                    i, b + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=6, fontweight="bold",
                )
        bottom += values

    labels = [f"{sz:.2f} MB\n{psnr:.1f} dB" for sz, psnr in zip(total_sizes, psnr_values)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")

    ax.set_xlabel("Hull Operating Points (sorted by total compressed size)")
    ax.set_ylabel("Component Proportion (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Size Proportion — Hull Points\n{dataset_name} / {sequence_name} frame {frame_id}"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved hull normalized bar: {output_path}")


def _plot_hull_normalized_bar_fallback(
    hull_rows: list[dict[str, Any]],
    sequence_name: str,
    dataset_name: str,
    frame_id: int,
    output_path: str,
) -> None:
    """Fallback normalized bar with coarse Position vs Attributes."""
    if not hull_rows:
        print(f"[WARN] No hull points for normalized bar: {sequence_name} frame {frame_id}")
        return

    hull_rows = sorted(hull_rows, key=lambda r: float(r.get("compressed_mb") or 0))
    n = len(hull_rows)

    component_data: dict[str, list[float]] = {comp: [] for comp in FALLBACK_COMPONENT_ORDER}
    total_sizes: list[float] = []
    psnr_values: list[float] = []

    for row in hull_rows:
        comps = _aggregate_components_fallback(row)
        total = sum(comps.values())
        if total < 1e-12:
            total = 1.0
        for comp in FALLBACK_COMPONENT_ORDER:
            component_data[comp].append(100.0 * comps.get(comp, 0) / total)
        total_sizes.append(float(row.get("compressed_mb") or 0))
        psnr_values.append(float(row.get("decomp_psnr") or 0))

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6 + 2), 6))
    x = np.arange(n)
    bar_width = 0.7

    bottom = np.zeros(n)
    for comp, color in zip(FALLBACK_COMPONENT_ORDER, FALLBACK_COMPONENT_COLORS):
        values = np.array(component_data[comp])
        if values.sum() < 1e-12:
            continue
        ax.bar(x, values, bar_width, bottom=bottom, label=comp, color=color, alpha=0.9)
        bottom += values

    labels = [f"{sz:.2f} MB\n{psnr:.1f} dB" for sz, psnr in zip(total_sizes, psnr_values)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")

    ax.set_xlabel("Hull Operating Points (sorted by total compressed size)")
    ax.set_ylabel("Component Proportion (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        f"Size Proportion (coarse) — Hull Points\n{dataset_name} / {sequence_name} frame {frame_id}"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved hull normalized bar (coarse): {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Scatter colored by dominant component
# ---------------------------------------------------------------------------


def _plot_scatter_dominant(
    all_rows: list[dict[str, Any]],
    per_dim_cols: list[str],
    sequence_name: str,
    dataset_name: str,
    frame_id: int,
    output_path: str,
    hull_pts: list[tuple[float, float]],
) -> None:
    """Scatter plot of all operating points, colored by dominant size component."""
    if not all_rows:
        print(f"[WARN] No points for dominant scatter: {sequence_name} frame {frame_id}")
        return

    color_map = dict(zip(COMPONENT_ORDER, COMPONENT_COLORS))
    groups: dict[str, list[tuple[float, float]]] = {comp: [] for comp in COMPONENT_ORDER}

    for row in all_rows:
        comps = _aggregate_components(row, per_dim_cols)
        dominant = _dominant_component(comps)
        x = float(row.get("compressed_mb") or 0)
        y = float(row.get("decomp_psnr") or 0)
        groups[dominant].append((x, y))

    fig, ax = plt.subplots(figsize=(9, 6))

    for comp in COMPONENT_ORDER:
        pts = groups[comp]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(
            xs, ys, s=14, alpha=0.55, color=color_map[comp],
            label=f"{comp} ({len(pts)})", zorder=2,
        )

    # Overlay convex hull
    if len(hull_pts) >= 2:
        hx = [p[0] for p in hull_pts]
        hy = [p[1] for p in hull_pts]
        ax.plot(
            hx, hy, color="red", linewidth=2.0, linestyle="-",
            marker="D", markersize=4, label=f"Hull ({len(hull_pts)} pts)",
            zorder=3, alpha=0.8,
        )

    # Uncompressed PSNR reference line
    gt_psnr = next(
        (r.get("gt_psnr") for r in all_rows if r.get("gt_psnr") is not None), None,
    )
    if gt_psnr is not None:
        gt_val = float(gt_psnr)
        ax.axhline(
            gt_val, color="black", linestyle="--", linewidth=1.2,
            label=f"Uncompressed ({gt_val:.2f} dB)",
        )

    ax.set_xlabel("Compressed Size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(
        f"RD Scatter (dominant component)\n{dataset_name} / {sequence_name} frame {frame_id}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved dominant-component scatter: {output_path}")


def _plot_scatter_dominant_fallback(
    all_rows: list[dict[str, Any]],
    sequence_name: str,
    dataset_name: str,
    frame_id: int,
    output_path: str,
    hull_pts: list[tuple[float, float]],
) -> None:
    """Fallback scatter colored by dominant coarse component (Position/Attributes)."""
    if not all_rows:
        print(f"[WARN] No points for dominant scatter: {sequence_name} frame {frame_id}")
        return

    color_map = dict(zip(FALLBACK_COMPONENT_ORDER, FALLBACK_COMPONENT_COLORS))
    groups: dict[str, list[tuple[float, float]]] = {comp: [] for comp in FALLBACK_COMPONENT_ORDER}

    for row in all_rows:
        comps = _aggregate_components_fallback(row)
        dominant = max(comps, key=lambda k: comps[k])
        x = float(row.get("compressed_mb") or 0)
        y = float(row.get("decomp_psnr") or 0)
        groups[dominant].append((x, y))

    fig, ax = plt.subplots(figsize=(9, 6))

    for comp in FALLBACK_COMPONENT_ORDER:
        pts = groups[comp]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(
            xs, ys, s=14, alpha=0.55, color=color_map[comp],
            label=f"{comp} ({len(pts)})", zorder=2,
        )

    if len(hull_pts) >= 2:
        hx = [p[0] for p in hull_pts]
        hy = [p[1] for p in hull_pts]
        ax.plot(
            hx, hy, color="red", linewidth=2.0, linestyle="-",
            marker="D", markersize=4, label=f"Hull ({len(hull_pts)} pts)",
            zorder=3, alpha=0.8,
        )

    gt_psnr = next(
        (r.get("gt_psnr") for r in all_rows if r.get("gt_psnr") is not None), None,
    )
    if gt_psnr is not None:
        gt_val = float(gt_psnr)
        ax.axhline(
            gt_val, color="black", linestyle="--", linewidth=1.2,
            label=f"Uncompressed ({gt_val:.2f} dB)",
        )

    ax.set_xlabel("Compressed Size (MB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(
        f"RD Scatter (dominant coarse component)\n{dataset_name} / {sequence_name} frame {frame_id}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved dominant-component scatter (coarse): {output_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_breakdown_plots(
    csv_path: str,
    plot_dir: str,
    sequence_name: str,
    dataset_name: str,
    fixed: dict[str, Any],
) -> None:
    """Generate all three breakdown plots for one sequence across all its frames."""
    all_rows, per_dim_cols = _load_csv(csv_path)
    if not all_rows:
        return

    has_per_dim = bool(per_dim_cols)
    if not has_per_dim:
        print(
            f"[WARN] No per-dimension compressed-bytes columns found in {csv_path}.\n"
            "       Falling back to coarse breakdown: Position vs Attributes."
        )

    # Discover frame IDs for this sequence
    frame_ids: set[int] = set()
    for row in all_rows:
        if row.get("sequence_name") == sequence_name:
            try:
                frame_ids.add(int(row["frame_id"]))
            except (TypeError, ValueError, KeyError):
                continue

    if not frame_ids:
        print(f"[WARN] No frame IDs found for sequence: {sequence_name}")
        return

    for frame_id in sorted(frame_ids):
        print(f"\n  --- Frame {frame_id} ---")

        valid = _filter_rows(all_rows, sequence_name, frame_id, fixed)
        if not valid:
            print(f"  [WARN] No valid points for {sequence_name} frame {frame_id}")
            continue

        # Deduplicate operating points
        valid, removed = _deduplicate_operating_points(valid)
        if removed > 0:
            print(f"  Removed {removed} duplicate operating points")
        print(f"  {len(valid)} operating points after filtering")

        # Compute convex hull
        points = [(float(r["compressed_mb"]), float(r["decomp_psnr"])) for r in valid]
        point_to_row: dict[tuple[float, float], dict[str, Any]] = {}
        for r in valid:
            pt = (float(r["compressed_mb"]), float(r["decomp_psnr"]))
            point_to_row.setdefault(pt, r)

        hull_pts = _upper_convex_hull(points) if len(set(points)) >= 2 else []
        hull_rows = [point_to_row[pt] for pt in hull_pts if pt in point_to_row]
        print(f"  Convex hull: {len(hull_rows)} points")

        prefix = f"breakdown_{dataset_name}_{sequence_name}_frame{frame_id}"

        # Plot 1: Stacked bar of hull points
        if has_per_dim:
            _plot_hull_stacked_bar(
                hull_rows, per_dim_cols, sequence_name, dataset_name, frame_id,
                os.path.join(plot_dir, f"{prefix}_hull_stacked.png"),
            )
        else:
            _plot_hull_stacked_bar_fallback(
                hull_rows, sequence_name, dataset_name, frame_id,
                os.path.join(plot_dir, f"{prefix}_hull_stacked.png"),
            )

        # Plot 2: Normalized (100%) bar of hull points
        if has_per_dim:
            _plot_hull_normalized_bar(
                hull_rows, per_dim_cols, sequence_name, dataset_name, frame_id,
                os.path.join(plot_dir, f"{prefix}_hull_normalized.png"),
            )
        else:
            _plot_hull_normalized_bar_fallback(
                hull_rows, sequence_name, dataset_name, frame_id,
                os.path.join(plot_dir, f"{prefix}_hull_normalized.png"),
            )

        # Plot 3: Scatter colored by dominant component
        if has_per_dim:
            _plot_scatter_dominant(
                valid, per_dim_cols, sequence_name, dataset_name, frame_id,
                os.path.join(plot_dir, f"{prefix}_dominant_scatter.png"),
                hull_pts,
            )
        else:
            _plot_scatter_dominant_fallback(
                valid, sequence_name, dataset_name, frame_id,
                os.path.join(plot_dir, f"{prefix}_dominant_scatter.png"),
                hull_pts,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("LiVoGS RD Size Breakdown Plots (QUEEN)")
    print(f"  RD roots: {len(RD_OUTPUT_ROOTS)}")
    print(sep)

    # Step 1: Collect
    print(f"\n{sep}\nStep 1: Collect results\n{sep}")
    collected_infos = collect_all()

    # Step 2: Parse scatter spec
    print(f"{sep}\nStep 2: Resolve scatter spec\n{sep}")
    scatter_spec = _parse_scatter_spec(SCATTER_SPEC)
    spec_name = scatter_spec.get("name") or "unnamed"
    print(f"  Using spec: {spec_name}")

    # Step 3: Generate breakdown plots
    print(f"{sep}\nStep 3: Generate size breakdown plots\n{sep}")
    for info in collected_infos:
        rd_root = info["rd_root"]
        csv_path = info["csv_path"]
        seq_name = info["sequence_name"]
        dataset_name = info["dataset_name"]
        plot_dir = (
            PLOT_OUTPUT_DIR
            if PLOT_OUTPUT_DIR is not None
            else _default_plot_dir(rd_root)
        )

        print(f"  Sequence: {seq_name}  (dataset: {dataset_name})")
        print(f"    CSV:    {csv_path}")
        print(f"    Output: {plot_dir}")
        generate_breakdown_plots(
            csv_path=csv_path,
            plot_dir=plot_dir,
            sequence_name=seq_name,
            dataset_name=dataset_name,
            fixed=scatter_spec.get("fixed", {}),
        )

    print(f"\n{sep}")
    print("Done.")
    print(sep)


if __name__ == "__main__":
    main()
