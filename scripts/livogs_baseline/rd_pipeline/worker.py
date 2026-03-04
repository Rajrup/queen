#!/usr/bin/env python3
"""GPU worker for LiVoGS compression experiments in QUEEN."""

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.setup_livogs_imports()

import codec


def main() -> None:
    """Run compression and optional evaluation for one QP configuration."""
    parser = argparse.ArgumentParser(
        description="LiVoGS compression worker (compress + decompress + eval)",
    )
    parser.add_argument("--data_path", default=config.DATA_PATH)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--sequence_name", required=True)
    parser.add_argument("--rd_output_subdir", default=config.RD_OUTPUT_SUBDIR,
                        help="RD output subdirectory under compression/")
    parser.add_argument("--resolution", type=int, default=config.RESOLUTION)
    parser.add_argument("--frame_id", type=int, default=None,
                        help="Single frame to process (overrides --frame_start/--frame_end)")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=20)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=config.SH_DEGREE)
    parser.add_argument("--j", type=int, default=config.J,
                        help="Octree depth for voxelization")
    parser.add_argument("--quantize_step", type=float, default=0.0001,
                        help="Uniform quantization step (ignored when --qp_config_json is set)")
    parser.add_argument("--sh_color_space", default=config.SH_COLOR_SPACE,
                        choices=["rgb", "yuv", "klt"])
    parser.add_argument("--rlgr_block_size", type=int, default=config.RLGR_BLOCK_SIZE)
    parser.add_argument("--qp_config_json", default=None,
                        help="Path to QP config JSON (overrides --quantize_step)")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device for codec (use cuda:0 with CUDA_VISIBLE_DEVICES pinning)")
    parser.add_argument("--disable_ply_saving", action="store_true",
                        help="Skip saving decompressed PLY files")
    parser.add_argument("--disable_image_saving", action="store_true",
                        help="Skip quality evaluation (image rendering)")
    parser.add_argument("--nvcomp_algorithm", type=str, default=config.NVCOMP_ALGORITHM,
                        help="nvCOMP lossless compression algorithm for positions "
                             "(e.g. ANS, LZ4, Snappy, None to disable)")
    args = parser.parse_args()

    if args.frame_id is not None:
        args.frame_start = args.frame_id
        args.frame_end = args.frame_id

    nvcomp_algorithm = None if args.nvcomp_algorithm == "None" else args.nvcomp_algorithm

    qp_label = None
    qp_data = None
    frame_id = args.frame_id
    if args.qp_config_json is not None:
        with open(args.qp_config_json, encoding="utf-8") as f:
            qp_data = json.load(f)
        qp_label = qp_data["label"]
        if frame_id is None:
            frame_id = int(qp_data.get("frame_id"))
            args.frame_start = frame_id
            args.frame_end = frame_id

    if qp_label is not None and frame_id is not None:
        output_folder = config.experiment_dir(
            args.data_path,
            args.dataset_name,
            args.sequence_name,
            frame_id,
            args.j,
            qp_label,
            rd_subdir_name=args.rd_output_subdir,
        )
    else:
        output_folder = config.standard_output_dir(
            args.data_path,
            args.dataset_name,
            args.sequence_name,
            args.j,
            args.quantize_step,
            args.sh_color_space,
        )

    gt_model_path = config.checkpoint_dir(args.data_path, args.dataset_name, args.sequence_name)
    dataset_path = config.dataset_dir(args.data_path, args.dataset_name, args.sequence_name)

    print("=" * 70)
    print("LiVoGS Worker")
    print(f"  Dataset:       {args.dataset_name}")
    print(f"  Sequence:      {args.sequence_name}")
    print(f"  GT model path: {gt_model_path}")
    print(f"  Output folder: {output_folder}")
    print(f"  Frames:        {args.frame_start}..{args.frame_end} (interval={args.interval})")
    print(f"  Device:        {args.device}")
    if qp_label and qp_data is not None:
        print(f"  QP label:      {qp_label} (beta={qp_data['beta']}, baseline_qp={qp_data['baseline_qp']})")
    else:
        print(f"  J:             {args.j} | quantize_step: {args.quantize_step} | color space: {args.sh_color_space}")
    print("=" * 70)

    os.makedirs(output_folder, exist_ok=True)
    if args.qp_config_json is not None:
        shutil.copy(args.qp_config_json, os.path.join(output_folder, "qp_config.json"))

    if args.qp_config_json is not None and qp_data is not None:
        quantize_step = qp_data["quantize_config"]
    else:
        qs = args.quantize_step
        quantize_step = {
            "quats": qs,
            "scales": qs,
            "opacity": qs,
            "sh_dc": qs,
            "sh_rest": [qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
        }

    print(f"\n{'=' * 70}\nStep 1: LiVoGS Compress + Decompress\n{'=' * 70}")
    codec.compress_decompress(
        ply_path=gt_model_path,
        output_folder=output_folder,
        output_ply_folder=os.path.join(output_folder, "decompressed_ply"),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        interval=args.interval,
        sh_degree=args.sh_degree,
        J=args.j,
        quantize_step=quantize_step,
        sh_color_space=args.sh_color_space,
        rlgr_block_size=args.rlgr_block_size,
        device=args.device,
        skip_save_ply=args.disable_ply_saving,
        nvcomp_algorithm=nvcomp_algorithm,
    )

    if args.disable_image_saving:
        print("[INFO] --disable_image_saving: skipping quality evaluation.")
    else:
        print(f"\n{'=' * 70}\nStep 2: Evaluate Decompression Quality\n{'=' * 70}")
        eval_cmd = [
            sys.executable,
            os.path.join("scripts", "evaluate_decompress.py"),
            "--config",
            "configs/dynerf.yaml",
            "-s",
            dataset_path,
            "-m",
            gt_model_path,
            "--decompressed_ply_path",
            os.path.join(output_folder, "decompressed_ply"),
            "--output_render_path",
            os.path.join(output_folder, "evaluation"),
            "--save_renders",
            "--frame_start",
            str(args.frame_start),
            "--frame_end",
            str(args.frame_end),
            "--interval",
            str(args.interval),
        ]
        result = subprocess.run(eval_cmd, cwd=config.QUEEN_ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print(f"\n{'=' * 70}")
    print(f"Done! Results in: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
