#!/usr/bin/env python3
"""Plot RD curves for LiVoGS single-frame compression experiments (QUEEN)."""

import csv
import json
import os
import sys
from typing import Any, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

import matplotlib.pyplot as plt


ResultRow = dict[str, Any]


def load_experiment_result(exp_dir: str, frame_id: int) -> Optional[ResultRow]:
    """Load one experiment's QP config and metrics. Returns None on failure."""
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

    try:
        baseline_qp = float(qp_config["baseline_qp"])
        beta = float(qp_config["beta"])
    except (KeyError, TypeError, ValueError):
        print(f"  [SKIP] Missing/invalid baseline_qp or beta in {os.path.basename(exp_dir)}")
        return None

    return {
        "label": str(qp_config.get("label", os.path.basename(exp_dir))),
        "baseline_qp": baseline_qp,
        "beta": beta,
        "compressed_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / (1024 * 1024),
        "decomp_psnr": float(decomp_psnr),
        "gt_psnr": float(gt_psnr) if gt_psnr is not None else None,
        "depth": -1,
    }


def collect_results(frame_dir: str, frame_id: int, depth: Optional[int] = None) -> List[ResultRow]:
    """Scan experiment subdirectories and return result dicts."""
    results: List[ResultRow] = []
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


def plot_rd_curves_by_beta(
    results: List[ResultRow],
    frame_id: int,
    output_path: str,
    sequence_name: str,
    octree_depth: int,
    beta_values: list[float],
    psnr_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot RD curves with one curve per beta value."""
    if not results:
        print("[WARN] No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    gt_psnr = next((r["gt_psnr"] for r in results if r["gt_psnr"] is not None), None)

    for beta in beta_values:
        beta_pts = [r for r in results if float(r["beta"]) == float(beta)]
        if not beta_pts:
            continue
        beta_pts.sort(key=lambda r: r["compressed_mb"])
        x = [r["compressed_mb"] for r in beta_pts]
        y = [r["decomp_psnr"] for r in beta_pts]
        ax.plot(x, y, marker="o", linewidth=1.6, label=f"beta={beta:.1f}")

    if gt_psnr is not None:
        ax.axhline(gt_psnr, color="black", linestyle="--", linewidth=1.4,
                   label=f"Uncompressed ({gt_psnr:.2f} dB)")

    ax.set_xlabel("Compressed size (MB)")
    ax.set_ylabel("PSNR (dB)")
    if psnr_range is not None:
        ax.set_ylim(psnr_range)
    ax.set_title(f"LiVoGS RD Curves - {sequence_name} frame {frame_id} J={octree_depth}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def plot_rd_curves_by_depth(
    results: List[ResultRow],
    frame_id: int,
    output_path: str,
    sequence_name: str,
    target_beta: float,
    plot_depths: List[int],
    psnr_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot RD curves with one curve per depth for a fixed beta."""
    if not results:
        print("[WARN] No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    gt_psnr = next((r["gt_psnr"] for r in results if r["gt_psnr"] is not None), None)

    beta_key = round(float(target_beta), 6)
    drawn = 0
    for depth in plot_depths:
        depth_pts = [
            r for r in results
            if int(r["depth"]) == int(depth) and round(float(r["beta"]), 6) == beta_key
        ]
        if not depth_pts:
            continue
        depth_pts.sort(key=lambda r: r["compressed_mb"])
        x = [r["compressed_mb"] for r in depth_pts]
        y = [r["decomp_psnr"] for r in depth_pts]
        ax.plot(x, y, marker="o", linewidth=1.6, label=f"J={depth}")
        drawn += 1

    if drawn == 0:
        print(f"[WARN] No depth curves found for beta={target_beta:.6g}.")
        plt.close(fig)
        return

    if gt_psnr is not None:
        ax.axhline(gt_psnr, color="black", linestyle="--", linewidth=1.4,
                   label=f"Uncompressed ({gt_psnr:.2f} dB)")

    ax.set_xlabel("Compressed size (MB)")
    ax.set_ylabel("PSNR (dB)")
    if psnr_range is not None:
        ax.set_ylim(psnr_range)
    ax.set_title(f"LiVoGS RD Curves - {sequence_name} frame {frame_id} beta={target_beta:.6g}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def _format_beta_tag(beta: float) -> str:
    return str(beta).replace("-", "m").replace(".", "p")


def _normalize_psnr_range(
    psnr_range: Optional[Union[List[float], Tuple[float, float]]],
) -> Optional[Tuple[float, float]]:
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


def main(
    frame_id: int,
    output_root: str,
    plot_output_dir: str,
    sequence_name: str,
    beta_values: List[float],
    baseline_qps: Optional[List[float]] = None,
    plot_mode: str = "beta_under_depth",
    octree_depth: int = config.J,
    target_beta: Optional[float] = None,
    plot_depths: Optional[List[int]] = None,
    psnr_range: Optional[Union[List[float], Tuple[float, float]]] = None,
) -> None:
    """Plot RD curves from experiment results."""
    if plot_mode == "beta_under_depth":
        frame_dir = os.path.join(output_root, f"frame_{frame_id}", f"J_{octree_depth}")
        results = collect_results(frame_dir, frame_id, depth=octree_depth)
    elif plot_mode == "depth_under_beta":
        depths = list(plot_depths) if plot_depths else [octree_depth]
        results = []
        for depth in depths:
            frame_dir = os.path.join(output_root, f"frame_{frame_id}", f"J_{depth}")
            results.extend(collect_results(frame_dir, frame_id, depth=depth))
        if target_beta is None:
            print("[WARN] --target_beta is required for plot_mode=depth_under_beta")
            return
    else:
        print(f"[WARN] Unsupported plot mode: {plot_mode}")
        return

    if baseline_qps is not None:
        qp_set = {round(q, 6) for q in baseline_qps}
        results = [r for r in results if round(r["baseline_qp"], 6) in qp_set]

    if not results:
        print("No valid results found - nothing to plot.")
        return

    normalized_psnr_range = _normalize_psnr_range(psnr_range)

    if plot_mode == "beta_under_depth":
        print(f"\n{'label':<30} {'beta':>6} {'qp':>8} {'MB':>8} {'PSNR':>8}")
        print("-" * 65)
        for r in sorted(results, key=lambda x: (x["beta"], x["baseline_qp"])):
            print(f"  {r['label']:<28} {r['beta']:>6.1f} {r['baseline_qp']:>8.4f} "
                  f"{r['compressed_mb']:>8.3f} {r['decomp_psnr']:>8.3f}")

        plot_path = os.path.join(
            plot_output_dir,
            f"rd_curves_{sequence_name}_frame{frame_id}_J{octree_depth}.png",
        )
        plot_rd_curves_by_beta(
            results,
            frame_id,
            plot_path,
            sequence_name,
            octree_depth,
            beta_values,
            psnr_range=normalized_psnr_range,
        )
    else:
        depths = list(plot_depths) if plot_depths else [octree_depth]
        beta_value = float(target_beta) if target_beta is not None else float("nan")
        print(f"\n{'label':<30} {'depth':>6} {'beta':>6} {'qp':>8} {'MB':>8} {'PSNR':>8}")
        print("-" * 73)
        for r in sorted(results, key=lambda x: (x["depth"], x["baseline_qp"])):
            print(f"  {r['label']:<28} {int(r['depth']):>6d} {r['beta']:>6.1f} {r['baseline_qp']:>8.4f} "
                  f"{r['compressed_mb']:>8.3f} {r['decomp_psnr']:>8.3f}")

        beta_tag = _format_beta_tag(beta_value)
        plot_path = os.path.join(
            plot_output_dir,
            f"rd_curves_{sequence_name}_frame{frame_id}_beta{beta_tag}_across_depths.png",
        )
        plot_rd_curves_by_depth(
            results,
            frame_id,
            plot_path,
            sequence_name,
            beta_value,
            depths,
            psnr_range=normalized_psnr_range,
        )


if __name__ == "__main__":
    import argparse as _ap

    parser = _ap.ArgumentParser(description="Plot LiVoGS RD curves for a single frame")
    parser.add_argument("--frame_id", type=int, default=1)
    parser.add_argument("--output_root", default=None,
                        help="Directory containing frame_N/ subdirs")
    parser.add_argument("--plot_output_dir", default=None)
    parser.add_argument("--sequence_name", default="cook_spinach")
    parser.add_argument("--beta_values", type=float, nargs="*",
                        default=[0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0])
    parser.add_argument("--baseline_qps", type=float, nargs="*", default=None)
    parser.add_argument("--plot_mode", default="beta_under_depth",
                        choices=["beta_under_depth", "depth_under_beta"])
    parser.add_argument("--octree_depth", type=int, default=config.J)
    parser.add_argument("--plot_depths", type=int, nargs="*", default=None)
    parser.add_argument("--target_beta", type=float, default=None)
    parser.add_argument("--psnr_range", type=float, nargs=2, default=None)
    args = parser.parse_args()

    default_root = config.rd_output_root(config.DATA_PATH, "Neural_3D_Video", args.sequence_name)
    output_root = args.output_root or default_root
    plot_dir = args.plot_output_dir or config.plot_output_dir(config.DATA_PATH, "Neural_3D_Video", args.sequence_name)

    main(
        frame_id=args.frame_id,
        output_root=output_root,
        plot_output_dir=plot_dir,
        sequence_name=args.sequence_name,
        beta_values=args.beta_values,
        baseline_qps=args.baseline_qps,
        plot_mode=args.plot_mode,
        octree_depth=args.octree_depth,
        target_beta=args.target_beta,
        plot_depths=args.plot_depths,
        psnr_range=args.psnr_range,
    )
