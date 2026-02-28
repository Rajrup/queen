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
    "baseline_qp",
    "beta",
    "qp_quats",
    "qp_scales",
    "qp_opacity",
    "compressed_bytes",
    "compressed_mb",
    "decomp_psnr",
    "gt_psnr",
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
        baseline_qp = float(qp_config["baseline_qp"])
        beta = float(qp_config["beta"])
        qp_quats = float(qp_config.get("qp_quats", quantize_cfg.get("quats")))
        qp_scales = float(qp_config.get("qp_scales", quantize_cfg.get("scales")))
        qp_opacity = float(qp_config.get("qp_opacity", quantize_cfg.get("opacity")))
    except (KeyError, TypeError, ValueError):
        print(f"  [SKIP] Missing/invalid baseline_qp, beta, or attr qps in {os.path.basename(exp_dir)}")
        return None

    return {
        "label": str(qp_config.get("label", os.path.basename(exp_dir))),
        "baseline_qp": baseline_qp,
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


def plot_rd(
    csv_path: str,
    curve_var: str,
    fixed: dict[str, float],
    output_path: str,
    sequence_name: str,
    frame_id: int,
    psnr_range: Optional[tuple[float, float]] = None,
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

    fig, ax = plt.subplots(figsize=(9, 6))
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
        ax.plot(x, y, marker="o", linewidth=1.6, label=f"{curve_var}={curve_value:g}")

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
