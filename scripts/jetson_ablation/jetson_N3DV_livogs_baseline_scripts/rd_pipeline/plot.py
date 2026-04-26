#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Plot RD curves from aggregated LiVoGS CSV results."""

import argparse
import csv
import json
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt


ResultRow = dict[str, Any]

NUMERIC_COLUMNS = (
    "depth",
    "qp_sh",
    "beta",
    "qp_quats",
    "qp_scales",
    "qp_opacity",
    "compressed_bytes",
    "compressed_mb",
    "decomp_psnr",
    "gt_psnr",
)

HULL_CSV_COLUMNS = (
    "compressed_mb",
    "decomp_psnr",
    "qp_opacity",
    "qp_scales",
    "qp_quats",
    "qp_sh",
    "depth",
)

DEDUP_COMBO_KEYS = (
    "depth",
    "qp_sh",
    "qp_opacity",
    "qp_scales",
    "qp_quats",
)

def load_experiment_result(exp_dir: str, frame_id: int) -> Optional[ResultRow]:
    qp_config_path = os.path.join(exp_dir, "qp_config.json")
    benchmark_path = os.path.join(exp_dir, "benchmark_livogs.csv")
    eval_json_path = os.path.join(exp_dir, "evaluation", "evaluation_results.json")

    for p in (qp_config_path, benchmark_path, eval_json_path):
        if not os.path.exists(p):
            return None

    try:
        with open(qp_config_path, encoding="utf-8") as f:
            qp_config = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"  [SKIP] Invalid qp_config.json in {os.path.basename(exp_dir)}: {exc}")
        return None

    compressed_bytes = None
    with open(benchmark_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["frame_id"]) == frame_id:
                compressed_bytes = int(row["compressed_size_bytes"])
                break
    if compressed_bytes is None:
        return None

    try:
        with open(eval_json_path, encoding="utf-8") as f:
            eval_data = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"  [SKIP] Invalid evaluation_results.json in {os.path.basename(exp_dir)}: {exc}")
        return None

    decomp_psnr = None
    gt_psnr = None
    frame_str = str(frame_id).zfill(4)
    for fr in eval_data.get("per_frame", []):
        fr_val = fr["frame"]
        if fr_val == frame_id or fr_val == frame_str or str(fr_val) == str(frame_id):
            decomp_psnr = fr["decomp_psnr"]
            gt_psnr = fr["gt_psnr"]
            break
    if decomp_psnr is None:
        summary = eval_data.get("summary", {})
        decomp_psnr = summary.get("decomp_psnr")
        gt_psnr = summary.get("gt_psnr")
    if decomp_psnr is None:
        return None

    quantize_cfg = qp_config.get("quantize_config", {})
    try:
        qp_sh = float(qp_config.get("qp_sh", qp_config.get("sh_qp", qp_config.get("baseline_qp"))))
        beta = float(qp_config["beta"])
        qp_quats = float(qp_config.get("qp_quats", quantize_cfg.get("quats")))
        qp_scales = float(qp_config.get("qp_scales", quantize_cfg.get("scales")))
        qp_opacity = float(qp_config.get("qp_opacity", quantize_cfg.get("opacity")))
    except (KeyError, TypeError, ValueError):
        print(f"  [SKIP] Missing/invalid qp_sh, beta, or attr qps in {os.path.basename(exp_dir)}")
        return None

    return {
        "label": str(qp_config.get("label", os.path.basename(exp_dir))),
        "qp_sh": qp_sh,
        "beta": beta,
        "qp_quats": qp_quats,
        "qp_scales": qp_scales,
        "qp_opacity": qp_opacity,
        "compressed_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / (1024 * 1024),
        "decomp_psnr": float(decomp_psnr),
        "gt_psnr": float(gt_psnr) if gt_psnr is not None else None,
        "depth": -1,
    }


def collect_results(frame_dir: str, frame_id: int, depth: Optional[int] = None) -> list[ResultRow]:
    results: list[ResultRow] = []
    if not os.path.isdir(frame_dir):
        print(f"[WARN] Directory not found: {frame_dir}")
        return results

    for entry in sorted(os.listdir(frame_dir)):
        exp_dir = os.path.join(frame_dir, entry)
        if not os.path.isdir(exp_dir):
            continue
        result = load_experiment_result(exp_dir, frame_id)
        if result is None:
            print(f"  [SKIP] Incomplete results in: {entry}")
        else:
            if depth is not None:
                result["depth"] = depth
            results.append(result)

    print(f"Loaded {len(results)} experiments from {frame_dir}")
    return results


def _normalize_psnr_range(psnr_range: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
    if psnr_range is None:
        return None
    if len(psnr_range) != 2:
        print(f"[WARN] Ignoring invalid psnr_range={psnr_range!r}: expected [min, max].")
        return None

    try:
        psnr_min = float(psnr_range[0])
        psnr_max = float(psnr_range[1])
    except (TypeError, ValueError):
        print(f"[WARN] Ignoring invalid psnr_range={psnr_range!r}: values must be numeric.")
        return None

    if psnr_min >= psnr_max:
        print(f"[WARN] Ignoring invalid psnr_range={psnr_range!r}: min must be < max.")
        return None

    return psnr_min, psnr_max


def _parse_numeric_columns(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = dict(row)
    for key in NUMERIC_COLUMNS:
        raw_value = parsed.get(key)
        if raw_value in (None, ""):
            parsed[key] = None
            continue
        parsed[key] = float(raw_value)
    return parsed


def _parse_fixed_pairs(fixed_pairs: list[str]) -> dict[str, float]:
    fixed: dict[str, float] = {}
    for pair in fixed_pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --fixed item: {pair!r}. Expected key=value")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --fixed item: {pair!r}. Empty key")
        fixed[key] = float(value)
    return fixed


def _round6(value: float) -> float:
    return round(float(value), 6)


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


def _deduplicate_operating_points(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
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



def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Cross product of vectors OA and OB."""
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


def plot_rd(
    csv_path: str,
    curve_var: str,
    fixed: dict[str, float],
    output_path: str,
    sequence_name: str,
    frame_id: int,
    psnr_range: Optional[tuple[float, float]] = None,
    curve_values: Optional[list[float]] = None,
) -> None:
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return

    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                parsed = _parse_numeric_columns(row)
            except (TypeError, ValueError):
                continue
            rows.append(parsed)

    if not rows:
        print(f"[WARN] No rows loaded from: {csv_path}")
        return

    filtered: list[dict[str, Any]] = []
    for row in rows:
        keep = True
        for key, target_value in fixed.items():
            row_value = row.get(key)
            if row_value is None:
                keep = False
                break
            if isinstance(target_value, (list, tuple)):
                if not any(
                    round(float(row_value), 6) == round(float(tv), 6)
                    for tv in target_value
                ):
                    keep = False
                    break
            else:
                if round(float(row_value), 6) != round(float(target_value), 6):
                    keep = False
                    break
        if keep:
            filtered.append(row)

    if not filtered:
        print(f"[WARN] No rows match fixed filters: {fixed}")
        return

    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in filtered:
        curve_value = row.get(curve_var)
        if curve_value is None:
            continue
        grouped.setdefault(float(curve_value), []).append(row)

    if not grouped:
        print(f"[WARN] No rows contain curve variable '{curve_var}' after filtering.")
        return

    # Filter to selected curve_values if specified
    if curve_values is not None:
        allowed = {round(float(v), 6) for v in curve_values}
        grouped = {
            k: v for k, v in grouped.items()
            if round(k, 6) in allowed
        }
        if not grouped:
            print(f"[WARN] No rows match curve_values={curve_values} for '{curve_var}'.")
            return

    fig, ax = plt.subplots(figsize=(9, 6))
    all_points: list[tuple[float, float]] = []  # for convex hull
    for curve_value in sorted(grouped.keys()):
        curve_rows = grouped[curve_value]
        valid_rows = [
            r for r in curve_rows
            if r.get("compressed_mb") is not None and r.get("decomp_psnr") is not None
        ]
        if not valid_rows:
            continue
        valid_rows.sort(key=lambda r: float(r["compressed_mb"]))
        x = [float(r["compressed_mb"]) for r in valid_rows]
        y = [float(r["decomp_psnr"]) for r in valid_rows]
        all_points.extend(zip(x, y))
        ax.plot(x, y, marker="o", markersize=3, linewidth=0.8, linestyle="--", label=f"{curve_var}={curve_value:g}")

    # Convex hull across all operating points
    if len(all_points) >= 2:
        hull_pts = _upper_convex_hull(all_points)
        if len(hull_pts) >= 2:
            hx = [p[0] for p in hull_pts]
            hy = [p[1] for p in hull_pts]
            ax.plot(hx, hy, color="red", linewidth=2.5, linestyle="-",
                    label="Convex hull", zorder=0)

    gt_psnr = next((r.get("gt_psnr") for r in filtered if r.get("gt_psnr") is not None), None)
    if gt_psnr is not None:
        gt_psnr_float = float(gt_psnr)
        ax.axhline(
            gt_psnr_float,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=f"Uncompressed ({gt_psnr_float:.2f} dB)",
        )

    normalized_psnr_range = _normalize_psnr_range(psnr_range)
    ax.set_xlabel("Compressed size (MB)")
    ax.set_ylabel("PSNR (dB)")
    if normalized_psnr_range is not None:
        ax.set_ylim(normalized_psnr_range)
    ax.set_title(f"RD Curves - {sequence_name} frame {frame_id} ({curve_var} sweep)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def _write_hull_csv(
    hull_pts: list[tuple[float, float]],
    point_to_row: dict[tuple[float, float], dict[str, Any]],
    hull_csv_path: str,
) -> None:
    """Write convex-hull operating points to a CSV for later lookup."""
    os.makedirs(os.path.dirname(os.path.abspath(hull_csv_path)), exist_ok=True)
    with open(hull_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HULL_CSV_COLUMNS)
        writer.writeheader()
        for pt in hull_pts:
            row = point_to_row.get(pt, {})
            writer.writerow({col: row.get(col, "") for col in HULL_CSV_COLUMNS})
    print(f"Saved hull CSV ({len(hull_pts)} points): {hull_csv_path}")


def plot_rd_scatter(
    csv_path: str,
    output_path: str,
    sequence_name: str,
    frame_id: int,
    hull_csv_path: Optional[str] = None,
    psnr_range: Optional[tuple[float, float]] = None,
    fixed: Optional[dict[str, Any]] = None,
    deduplicate: bool = False,
) -> None:
    """Scatter-plot all RD operating points with upper convex hull.

    Unlike :func:`plot_rd` this does **not** group by a sweep variable —
    every matching row becomes a single dot.  The upper convex hull is
    overlaid and, when *hull_csv_path* is given, its vertices are exported
    to a CSV for downstream lookup.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return

    # ---- load rows ----
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                parsed = _parse_numeric_columns(row)
            except (TypeError, ValueError):
                continue
            rows.append(parsed)

    if not rows:
        print(f"[WARN] No rows loaded from: {csv_path}")
        return

    # ---- filter by sequence / frame ----
    filtered: list[dict[str, Any]] = [
        r for r in rows
        if r.get("sequence_name") == sequence_name
        and r.get("frame_id") is not None
        and int(r["frame_id"]) == frame_id
    ]

    # ---- optional fixed-knob filter ----
    if fixed:
        kept: list[dict[str, Any]] = []
        for row in filtered:
            keep = True
            for key, target in fixed.items():
                row_val = row.get(key)
                if row_val is None:
                    keep = False
                    break
                if isinstance(target, (list, tuple)):
                    if not any(
                        round(float(row_val), 6) == round(float(tv), 6)
                        for tv in target
                    ):
                        keep = False
                        break
                else:
                    if round(float(row_val), 6) != round(float(target), 6):
                        keep = False
                        break
            if keep:
                kept.append(row)
        filtered = kept

    # ---- extract valid (x, y) points ----
    valid = [
        r for r in filtered
        if r.get("compressed_mb") is not None and r.get("decomp_psnr") is not None
    ]
    if not valid:
        print(f"[WARN] No valid points for {sequence_name} frame {frame_id}")
        return

    if deduplicate:
        valid, removed = _deduplicate_operating_points(valid)
        if removed > 0:
            print(
                f"[INFO] Removed {removed} redundant rows for "
                f"{sequence_name} frame {frame_id} before plotting"
            )

    x = [float(r["compressed_mb"]) for r in valid]
    y = [float(r["decomp_psnr"]) for r in valid]

    # map (x, y) → row so hull vertices can be exported with full metadata
    point_to_row: dict[tuple[float, float], dict[str, Any]] = {}
    for r in valid:
        pt = (float(r["compressed_mb"]), float(r["decomp_psnr"]))
        point_to_row.setdefault(pt, r)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=12, alpha=0.6, zorder=2,
               label=f"All points ({len(valid)})")

    hull_pts: list[tuple[float, float]] = []
    if len(set(zip(x, y))) >= 2:
        hull_pts = _upper_convex_hull(list(zip(x, y)))
        if len(hull_pts) >= 2:
            hx = [p[0] for p in hull_pts]
            hy = [p[1] for p in hull_pts]
            ax.plot(hx, hy, color="red", linewidth=2.5, linestyle="-",
                    marker="D", markersize=5,
                    label=f"Convex hull ({len(hull_pts)} pts)", zorder=3)

    gt_psnr = next(
        (r.get("gt_psnr") for r in valid if r.get("gt_psnr") is not None), None
    )
    if gt_psnr is not None:
        gt_val = float(gt_psnr)
        ax.axhline(gt_val, color="black", linestyle="--", linewidth=1.4,
                   label=f"Uncompressed ({gt_val:.2f} dB)")

    normalized_psnr_range = _normalize_psnr_range(psnr_range)
    ax.set_xlabel("Compressed size (MB)")
    ax.set_ylabel("PSNR (dB)")
    if normalized_psnr_range is not None:
        ax.set_ylim(normalized_psnr_range)
    ax.set_title(f"RD Scatter — {sequence_name} frame {frame_id}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot: {output_path}")

    # ---- export hull CSV ----
    if hull_csv_path and hull_pts:
        _write_hull_csv(hull_pts, point_to_row, hull_csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot generic LiVoGS RD curves from all_results.csv")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--curve_var", required=True)
    parser.add_argument("--fixed", nargs="*", default=[])
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sequence_name", required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--psnr_range", nargs=2, type=float, default=None)
    args = parser.parse_args()

    try:
        fixed = _parse_fixed_pairs(args.fixed)
    except ValueError as exc:
        parser.error(str(exc))
        return

    plot_rd(
        csv_path=args.csv_path,
        curve_var=args.curve_var,
        fixed=fixed,
        output_path=args.output_path,
        sequence_name=args.sequence_name,
        frame_id=args.frame_id,
        psnr_range=tuple(args.psnr_range) if args.psnr_range is not None else None,
    )


if __name__ == "__main__":
    main()
