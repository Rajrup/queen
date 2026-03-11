#!/usr/bin/env python3
"""High-level orchestrator for the GPCC RD experiment pipeline.

Stages:
  1. GENERATE  — generate QP configs from Cartesian product (no GPU needed)
  2. EVALUATE  — run compress+decompress for each QP config across GPUs

Usage:
    python gpcc_baseline/run_rd_pipeline.py \\
        --dataset_path /path/to/frames \\
        --experiment_dir /path/to/output \\
        --tmc3_path /path/to/tmc3 \\
        --num_gpus 2 \\
        --num_frames 10 \\
        --stage both
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)




# ---------------------------------------------------------------------------
# Global configuration (edit here or pass via CLI)
# ---------------------------------------------------------------------------

DEFAULT_TMC3_PATH = "/ssd1/haodongw/workspace/3dstream/mpeg-pcc-tmc13/build/tmc3/tmc3"
DEFAULT_VOXEL_DEPTH = 15
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_FRAMES = 1


# ---------------------------------------------------------------------------
# Stage 1: Generate QP configs
# ---------------------------------------------------------------------------

def stage_generate(cfg) -> list[dict]:
    """Generate all QP experiment configs from Cartesian product."""
    from rd_pipeline.qp import generate_qp_configs

    sep = "=" * 70
    print(f"\n{sep}\nStage 1: Generate QP configs\n{sep}")
    configs = generate_qp_configs(cfg)
    print(f"  Generated {len(configs)} QP configs")

    configs_json_path = os.path.join(cfg.experiment_dir, "qp_configs.json")
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    with open(configs_json_path, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)
    print(f"  Saved configs to: {configs_json_path}")
    return configs


# ---------------------------------------------------------------------------
# Stage 2: Evaluate (compress + decompress)
# ---------------------------------------------------------------------------

def _worker_process(args_tuple):
    """Worker function for multiprocessing — runs a batch of experiments."""
    from rd_pipeline.config import GpccConfig
    from rd_pipeline.worker import worker_main

    experiment_configs, global_config_dict, gpu_id = args_tuple
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg = GpccConfig(**global_config_dict)
    cfg.gpu_id = gpu_id
    worker_main(experiment_configs, cfg)


def stage_evaluate(cfg, experiment_configs: list[dict], num_gpus: int) -> None:
    """Distribute experiments across GPUs and run them."""
    from rd_pipeline.worker import worker_main

    sep = "=" * 70
    print(f"\n{sep}\nStage 2: Evaluate ({len(experiment_configs)} experiments, {num_gpus} GPUs)\n{sep}")

    if not experiment_configs:
        print("  No experiments to run.")
        return

    gpu_batches = [[] for _ in range(num_gpus)]
    for i, exp_config in enumerate(experiment_configs):
        gpu_batches[i % num_gpus].append(exp_config)

    cfg_dict = {
        "tmc3_path": cfg.tmc3_path,
        "voxel_depth": cfg.voxel_depth,
        "experiment_dir": cfg.experiment_dir,
        "dataset_path": cfg.dataset_path,
        "num_frames": cfg.num_frames,
        "gpu_id": cfg.gpu_id,
    }

    if num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        worker_main(experiment_configs, cfg)
    else:
        args_list = [
            (batch, cfg_dict, gpu_id)
            for gpu_id, batch in enumerate(gpu_batches)
            if batch
        ]
        with ProcessPoolExecutor(max_workers=len(args_list)) as executor:
            futures = [executor.submit(_worker_process, args) for args in args_list]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Worker failed: {e}")

    print(f"\n{sep}")
    print("Stage 2 complete.")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPCC RD experiment pipeline orchestrator"
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to VideoGS checkpoint dir (contains frame subdirs)")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Output directory for all experiments")
    parser.add_argument("--tmc3_path", type=str, default=DEFAULT_TMC3_PATH,
                        help="Path to tmc3 binary")
    parser.add_argument("--voxel_depth", type=int, default=DEFAULT_VOXEL_DEPTH,
                        help="Voxelization depth J")
    parser.add_argument("--num_gpus", type=int, default=DEFAULT_NUM_GPUS,
                        help="Number of GPUs to use")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Number of frames to process per experiment")
    parser.add_argument("--stage", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Which stage to run: 1=generate, 2=evaluate, both=all")
    return parser.parse_args()


def main():
    from rd_pipeline.config import GpccConfig

    args = parse_args()

    cfg = GpccConfig(
        tmc3_path=args.tmc3_path,
        voxel_depth=args.voxel_depth,
        experiment_dir=args.experiment_dir,
        dataset_path=args.dataset_path,
        num_frames=args.num_frames,
    )

    sep = "=" * 70
    print(sep)
    print("GPCC RD Experiment Pipeline")
    print(f"  Dataset:        {args.dataset_path}")
    print(f"  Experiment dir: {args.experiment_dir}")
    print(f"  tmc3 path:      {args.tmc3_path}")
    print(f"  Voxel depth:    {args.voxel_depth}")
    print(f"  Num GPUs:       {args.num_gpus}")
    print(f"  Num frames:     {args.num_frames}")
    print(f"  Stage:          {args.stage}")
    print(sep)

    experiment_configs = None

    if args.stage in ("1", "both"):
        experiment_configs = stage_generate(cfg)

    if args.stage in ("2", "both"):
        if experiment_configs is None:
            configs_json_path = os.path.join(cfg.experiment_dir, "qp_configs.json")
            if not os.path.exists(configs_json_path):
                print(f"[ERROR] No qp_configs.json found at {configs_json_path}. Run stage 1 first.")
                return
            with open(configs_json_path, encoding="utf-8") as f:
                experiment_configs = json.load(f)
            print(f"  Loaded {len(experiment_configs)} configs from {configs_json_path}")

        stage_evaluate(cfg, experiment_configs, args.num_gpus)

    print("\nDone.")


if __name__ == "__main__":
    main()
