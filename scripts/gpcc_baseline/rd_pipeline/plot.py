#!/usr/bin/env python3
"""Plot RD curves from aggregated GPCC CSV results."""

import argparse
import csv
import os
import sys
from typing import Any, Optional

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def load_experiment_result(exp_dir: str, frame_id: int) -> Optional[dict[str, Any]]:
    """Load one experiment's QP config and metrics. Returns None on failure."""
    qp_config_path = os.path.join(exp_dir, "qp_config.json")
    benchmark_path = os.path.join(exp_dir, "benchmark_gpcc.csv")
    eval_json_path = os.path.join(exp_dir, "evaluation", "evaluation_results.json")

    # benchmark is required; qp_config and eval are optional
    if not os.path.exists(benchmark_path):
        return None

    # Load QP config if available
    qp_config = {}
    if os.path.exists(qp_config_path):
        import json
        try:
            with open(qp_config_path, encoding="utf-8") as f:
                qp_config = json.load(f)
        except Exception:
            pass

    # Load compressed bytes from benchmark CSV
    compressed_bytes = None
    with open(benchmark_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["frame_idx"]) == frame_id:
                compressed_bytes = int(row["total_compressed_bytes"])
                break
    if compressed_bytes is None:
        return None

    # Load quality metrics if available
    decomp_psnr = None
    gt_psnr = None
    if os.path.exists(eval_json_path):
        import json
        try:
            with open(eval_json_path, encoding="utf-8") as f:
                eval_data = json.load(f)
            for fr in eval_data.get("per_frame", []):
                if fr["frame"] == frame_id:
                    decomp_psnr = fr.get("decomp_psnr")
                    gt_psnr = fr.get("gt_psnr")
                    break
            if decomp_psnr is None:
                summary = eval_data.get("summary", {})
                decomp_psnr = summary.get("decomp_psnr")
                gt_psnr = summary.get("gt_psnr")
        except Exception:
            pass

    # Extract QP values from experiment name or qp_config
    exp_name = os.path.basename(exp_dir)
    f_rest_qp = qp_config.get("f_rest_qp")
    f_dc_qp = qp_config.get("f_dc_qp")
    opacity_qp = qp_config.get("opacity_qp")

    # Try to parse from experiment name (e.g. "rest40_dc4_opacity4")
    if f_rest_qp is None and "rest" in exp_name:
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

    return {
        "label": qp_config.get("label", exp_name),
        "f_rest_qp": f_rest_qp,
        "f_dc_qp": f_dc_qp,
        "opacity_qp": opacity_qp,
        "compressed_bytes": compressed_bytes,
        "compressed_mb": compressed_bytes / (1024 * 1024),
        "decomp_psnr": decomp_psnr,
        "gt_psnr": gt_psnr,
    }


def collect_results(frame_dir: str, frame_id: int) -> list[dict[str, Any]]:
    """Scan all experiment subdirectories and return a list of result dicts."""
    results: list[dict[str, Any]] = []
    if not os.path.isdir(frame_dir):
        print(f"[WARN] Directory not found: {frame_dir}")
        return results

    for entry in sorted(os.listdir(frame_dir)):
        exp_dir = os.path.join(frame_dir, entry)
        if not os.path.isdir(exp_dir):
            continue
        r = load_experiment_result(exp_dir, frame_id)
        if r is None:
            print(f"  [SKIP] Incomplete results in: {entry}")
            continue
        results.append(r)

    print(f"Loaded {len(results)} experiments from {frame_dir}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _upper_convex_hull(points):
    sorted_pts = sorted(set(points))
    if len(sorted_pts) <= 1:
        return list(sorted_pts)
    upper = []
    for p in sorted_pts:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(p)
    return upper


def plot_rd_scatter(
    results: list[dict[str, Any]],
    output_path: str,
    sequence_name: str,
    frame_id: int,
    psnr_range: Optional[tuple[float, float]] = None,
) -> None:
    """Scatter-plot all GPCC RD operating points with upper convex hull."""
    valid = [r for r in results if r.get("compressed_mb") is not None and r.get("decomp_psnr") is not None]
    if not valid:
        print(f"[WARN] No valid points for {sequence_name} frame {frame_id}")
        return

    x = [float(r["compressed_mb"]) for r in valid]
    y = [float(r["decomp_psnr"]) for r in valid]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=12, alpha=0.6, zorder=2, label=f"All points ({len(valid)})")

    if len(set(zip(x, y))) >= 2:
        hull_pts = _upper_convex_hull(list(zip(x, y)))
        if len(hull_pts) >= 2:
            hx = [p[0] for p in hull_pts]
            hy = [p[1] for p in hull_pts]
            ax.plot(hx, hy, color="red", linewidth=2.5, linestyle="-",
                    marker="D", markersize=5, label=f"Convex hull ({len(hull_pts)} pts)", zorder=3)

    gt_psnr = next((r.get("gt_psnr") for r in valid if r.get("gt_psnr") is not None), None)
    if gt_psnr is not None:
        ax.axhline(float(gt_psnr), color="black", linestyle="--", linewidth=1.4,
                   label=f"Uncompressed ({float(gt_psnr):.2f} dB)")

    ax.set_xlabel("Compressed size (MB)")
    ax.set_ylabel("PSNR (dB)")
    if psnr_range is not None:
        ax.set_ylim(psnr_range)
    ax.set_title(f"GPCC RD Scatter — {sequence_name} frame {frame_id}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for plotting RD curves."""
    parser = argparse.ArgumentParser(
        description="Plot RD curves from aggregated GPCC CSV results."
    )
    parser.add_argument(
        "frame_dir",
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--frame-id",
        type=int,
        default=0,
        help="Frame ID to plot (default: 0)",
    )
    parser.add_argument(
        "--sequence-name",
        default="sequence",
        help="Sequence name for plot title (default: 'sequence')",
    )
    parser.add_argument(
        "--output",
        default="rd_scatter.png",
        help="Output plot path (default: 'rd_scatter.png')",
    )
    parser.add_argument(
        "--psnr-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="PSNR range for y-axis (e.g., 20 40)",
    )

    args = parser.parse_args()

    results = collect_results(args.frame_dir, args.frame_id)
    if not results:
        print("[ERROR] No valid results found.")
        sys.exit(1)

    psnr_range = tuple(args.psnr_range) if args.psnr_range else None
    plot_rd_scatter(
        results,
        args.output,
        args.sequence_name,
        args.frame_id,
        psnr_range=psnr_range,
    )


if __name__ == "__main__":
    main()
