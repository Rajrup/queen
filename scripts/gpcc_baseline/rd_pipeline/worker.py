#!/usr/bin/env python3
"""Per-experiment execution worker for the GPCC RD pipeline.

Provides:
- ``run_experiment(experiment_config, global_config)`` — runs one QP experiment
  across all frames.
- ``worker_main(experiment_configs, global_config)`` — sequential batch driver.
- CLI entry point for standalone single-experiment execution.

Designed to be called by the orchestrator or run directly::

    python gpcc_baseline/rd_pipeline/worker.py \
        --dataset_path /data/train_output/Dataset/Seq/point_cloud \
        --experiment_dir /data/experiments/gpcc_rd \
        --tmc3_path /path/to/tmc3 \
        --f_rest_qp 40 --f_dc_qp 28 --opacity_qp 34 \
        --num_frames 10
"""

import argparse
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — ensure sibling modules are importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import GpccConfig
from .codec import run_gpcc_codec


# ---------------------------------------------------------------------------
# Core: run_experiment
# ---------------------------------------------------------------------------

def run_experiment(experiment_config: dict, global_config: GpccConfig) -> dict:
    """Run one GPCC experiment (one QP config, all frames).

    Args:
        experiment_config: Dict from generate_qp_configs() with keys:
            experiment_name, f_rest_qp, f_dc_qp, opacity_qp,
            voxel_depth, dataset_path, experiment_dir, tmc3_path
        global_config: GpccConfig instance

    Returns:
        Dict with summary stats including experiment_name, num_frames,
        csv_path, and whether the experiment was skipped.
    """
    exp_name = experiment_config["experiment_name"]
    exp_dir = os.path.join(global_config.experiment_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resumability check: skip if benchmark_gpcc.csv already exists
    # ------------------------------------------------------------------
    csv_path = os.path.join(exp_dir, "benchmark_gpcc.csv")
    if os.path.exists(csv_path):
        print(f"[SKIP] {exp_name}: benchmark_gpcc.csv already exists")
        return {"skipped": True, "experiment_name": exp_name}

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------
    benchmark_rows = []
    for frame_idx in range(global_config.num_frames):
        # Find PLY path: {dataset_path}/{frame_idx}/point_cloud/
        ckpt_path = os.path.join(
            global_config.dataset_path, str(frame_idx), "point_cloud",
        )
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Frame {frame_idx}: checkpoint not found, skipping")
            continue

        # Find max iteration directory
        saved_iters = [
            int(f.split("_")[-1])
            for f in os.listdir(ckpt_path)
            if "iteration_" in f
        ]
        if not saved_iters:
            print(f"[WARN] Frame {frame_idx}: no iteration dirs found, skipping")
            continue
        max_iter = max(saved_iters)

        ply_path = os.path.join(
            ckpt_path, f"iteration_{max_iter}", "point_cloud.ply",
        )
        if not os.path.exists(ply_path):
            print(f"[WARN] Frame {frame_idx}: {ply_path} not found, skipping")
            continue

        frame_output_dir = os.path.join(exp_dir, f"frame_{frame_idx}")

        result = run_gpcc_codec(
            frame_ply_path=ply_path,
            output_dir=frame_output_dir,
            qp_config=experiment_config,
            cfg=global_config,
        )

        benchmark_rows.append({
            "frame_idx": frame_idx,
            "encode_time_s": result["encode_time_s"],
            "decode_time_s": result["decode_time_s"],
            "total_compressed_bytes": result["total_compressed_bytes"],
            "opacity_bytes": result["opacity_bytes"],
            "dc_bytes": result["dc_bytes"],
            "rest_bytes": result["rest_bytes"],
            "scale_bytes": result["scale_bytes"],
            "rot_bytes": result["rot_bytes"],
            "num_points_input": result["num_points_input"],
            "num_points_output": result["num_points_output"],
        })

    # ------------------------------------------------------------------
    # Write benchmark CSV
    # ------------------------------------------------------------------
    if benchmark_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_idx", "encode_time_s", "decode_time_s",
                "total_compressed_bytes", "opacity_bytes", "dc_bytes",
                "rest_bytes", "scale_bytes", "rot_bytes",
                "num_points_input", "num_points_output",
            ])
            for row in benchmark_rows:
                writer.writerow([
                    row["frame_idx"],
                    f"{row['encode_time_s']:.6f}",
                    f"{row['decode_time_s']:.6f}",
                    row["total_compressed_bytes"],
                    row["opacity_bytes"],
                    row["dc_bytes"],
                    row["rest_bytes"],
                    row["scale_bytes"],
                    row["rot_bytes"],
                    row["num_points_input"],
                    row["num_points_output"],
                ])

    return {
        "skipped": False,
        "experiment_name": exp_name,
        "num_frames": len(benchmark_rows),
        "csv_path": csv_path,
    }


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def worker_main(experiment_configs: list, global_config: GpccConfig) -> None:
    """Process a batch of experiments sequentially on one GPU."""
    for exp_config in experiment_configs:
        try:
            result = run_experiment(exp_config, global_config)
            if result.get("skipped"):
                print(f"[SKIP] {result['experiment_name']}")
            else:
                print(f"[DONE] {result['experiment_name']}: {result['num_frames']} frames")
        except Exception as e:
            print(f"[ERROR] {exp_config.get('experiment_name', '?')}: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPCC RD worker — run a single QP experiment",
    )
    parser.add_argument("--dataset_path", required=True,
                        help="Path to dataset with frame subdirs")
    parser.add_argument("--experiment_dir", required=True,
                        help="Root directory for experiment outputs")
    parser.add_argument("--tmc3_path", default=GpccConfig.tmc3_path,
                        help="Path to tmc3 binary")
    parser.add_argument("--voxel_depth", type=int, default=GpccConfig.voxel_depth,
                        help="Voxel depth for geometry coding")
    parser.add_argument("--f_rest_qp", type=int, required=True,
                        help="QP for SH rest coefficients")
    parser.add_argument("--f_dc_qp", type=int, required=True,
                        help="QP for SH DC coefficients")
    parser.add_argument("--opacity_qp", type=int, required=True,
                        help="QP for opacity attribute")
    parser.add_argument("--num_frames", type=int, default=1,
                        help="Number of frames to process")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID (informational only)")
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    cfg = GpccConfig(
        tmc3_path=args.tmc3_path,
        voxel_depth=args.voxel_depth,
        experiment_dir=args.experiment_dir,
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
        gpu_id=args.gpu_id,
    )

    exp_name = f"rest{args.f_rest_qp}_dc{args.f_dc_qp}_op{args.opacity_qp}"
    exp_config = {
        "experiment_name": exp_name,
        "f_rest_qp": args.f_rest_qp,
        "f_dc_qp": args.f_dc_qp,
        "opacity_qp": args.opacity_qp,
        "voxel_depth": args.voxel_depth,
        "dataset_path": args.dataset_path,
        "experiment_dir": args.experiment_dir,
        "tmc3_path": args.tmc3_path,
    }

    sep = "=" * 70
    print(sep)
    print("GPCC Worker")
    print(f"  Dataset path:    {args.dataset_path}")
    print(f"  Experiment dir:  {args.experiment_dir}")
    print(f"  tmc3 path:       {args.tmc3_path}")
    print(f"  Voxel depth:     {args.voxel_depth}")
    print(f"  QP:              rest={args.f_rest_qp}  dc={args.f_dc_qp}  opacity={args.opacity_qp}")
    print(f"  Frames:          {args.num_frames}")
    print(f"  GPU:             {args.gpu_id}")
    print(sep)

    result = run_experiment(exp_config, cfg)
    if result.get("skipped"):
        print(f"\n[SKIP] {result['experiment_name']}: already completed")
    else:
        print(f"\n[DONE] {result['experiment_name']}: {result['num_frames']} frames → {result['csv_path']}")


if __name__ == "__main__":
    main()
