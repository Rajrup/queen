#!/usr/bin/env python3
"""Generate RD curve plots from collected GPCC experiment results.

Flow:
  1. Read collected_rd_results.csv (from collect_rd_results.py)
  2. Generate RD scatter plots with convex hull

Usage:
    python gpcc_baseline/plot_rd_results.py \\
        --experiment_dir /path/to/experiments \\
        --frame_idx 0
"""

import argparse
import csv
import os
import sys
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rd_pipeline.plot import plot_rd_scatter


def load_results_from_csv(csv_path: str, frame_idx: int) -> list[dict[str, Any]]:
    """Load results from collected_rd_results.csv for a specific frame."""
    results = []
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}")
        return results
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if int(row.get("frame_idx", 0)) != frame_idx:
                    continue
                total_bytes = row.get("total_compressed_bytes")
                decomp_psnr = row.get("decomp_psnr")
                if not total_bytes or not decomp_psnr:
                    continue
                results.append({
                    "experiment_name": row.get("experiment_name", ""),
                    "f_rest_qp": int(row["f_rest_qp"]) if row.get("f_rest_qp") else None,
                    "f_dc_qp": int(row["f_dc_qp"]) if row.get("f_dc_qp") else None,
                    "opacity_qp": int(row["opacity_qp"]) if row.get("opacity_qp") else None,
                    "compressed_mb": float(total_bytes) / (1024 * 1024),
                    "decomp_psnr": float(decomp_psnr),
                    "gt_psnr": float(row["gt_psnr"]) if row.get("gt_psnr") else None,
                })
            except (ValueError, KeyError):
                continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate RD curve plots from collected GPCC experiment results"
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Root directory containing experiment subdirs")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to collected_rd_results.csv (default: experiment_dir/collected_rd_results.csv)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: experiment_dir/plots)")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="Frame index to plot (default: 0)")
    parser.add_argument("--sequence_name", type=str, default="sequence",
                        help="Sequence name for plot title (default: sequence)")
    parser.add_argument("--psnr_range", nargs=2, type=float, default=None,
                        help="PSNR range for y-axis [min max]")
    args = parser.parse_args()
    
    csv_path = args.csv_path or os.path.join(args.experiment_dir, "collected_rd_results.csv")
    output_dir = args.output_dir or os.path.join(args.experiment_dir, "plots")
    
    print(f"Loading results from: {csv_path}")
    results = load_results_from_csv(csv_path, args.frame_idx)
    print(f"Loaded {len(results)} results for frame {args.frame_idx}")
    
    if not results:
        print("[WARN] No results to plot.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(
        output_dir,
        f"rd_scatter_{args.sequence_name}_frame{args.frame_idx}.png"
    )
    
    psnr_range = tuple(args.psnr_range) if args.psnr_range else None
    plot_rd_scatter(
        results=results,
        output_path=output_path,
        sequence_name=args.sequence_name,
        frame_id=args.frame_idx,
        psnr_range=psnr_range,
    )
    
    print(f"Done. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
