#!/usr/bin/env python3
"""LiVoGS compression + decompression for QUEEN-trained Gaussian splat models."""

import csv
import json
import os
import sys
import time
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.setup_livogs_imports()

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from compress_decompress import decode_livogs, encode_livogs
from utils.system_utils import searchForMaxIteration


def find_queen_ply_path(frame_dir: str) -> str:
    """Find PLY file in a QUEEN frame directory."""
    canonical = os.path.join(frame_dir, "point_cloud.ply")
    if os.path.exists(canonical):
        return canonical

    iter_dir = os.path.join(frame_dir, "point_cloud")
    max_iter = searchForMaxIteration(iter_dir)
    per_iter = os.path.join(iter_dir, f"iteration_{max_iter}", "point_cloud.ply")
    if os.path.exists(per_iter):
        return per_iter

    raise FileNotFoundError(
        f"Missing checkpoint PLY for max iteration {max_iter}: {per_iter}"
    )


def load_queen_ply(ply_path: str, device: str = "cuda") -> tuple[dict[str, torch.Tensor], int]:
    """Load QUEEN PLY and return LiVoGS-compatible params on target device."""
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rest_names:
        sh_rest = np.stack([vertex[name] for name in rest_names], axis=1)
    else:
        sh_rest = np.zeros((len(vertex), 0), dtype=np.float32)
    colors = np.concatenate([sh_dc, sh_rest], axis=1)

    opacities = np.asarray(vertex["opacity"])
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)
    quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1)

    params = {
        "means": torch.from_numpy(means.copy()).float().to(device),
        "quats": torch.from_numpy(quats.copy()).float().to(device),
        "scales": torch.from_numpy(scales.copy()).float().to(device),
        "opacities": torch.from_numpy(opacities.copy()).float().to(device),
        "colors": torch.from_numpy(colors.copy()).float().to(device),
    }

    uncompressed_size_bytes = sum(v.numel() * v.element_size() for v in params.values())

    params["quats"] = F.normalize(params["quats"], p=2, dim=1)
    params["opacities"] = torch.sigmoid(params["opacities"])
    params["scales"] = torch.exp(params["scales"])
    return params, uncompressed_size_bytes


def save_queen_ply(
    params: dict[str, torch.Tensor],
    output_path: str,
    sh_degree: int = config.SH_DEGREE,
    eps: float = 1e-6,
) -> None:
    """Save reconstructed params to QUEEN-compatible PLY."""
    del sh_degree
    means = params["means"].detach().cpu().float().numpy()
    quats = params["quats"].detach().cpu().float().numpy()
    scales = params["scales"].detach().cpu().float().numpy()
    opacities = params["opacities"].detach().cpu().float().numpy()
    colors = params["colors"].detach().cpu().float().numpy()

    num_vertices = means.shape[0]

    opacities_c = np.clip(opacities, eps, 1.0 - eps)
    opacities_logit = np.log(opacities_c / (1.0 - opacities_c))
    scales_log = np.log(np.clip(scales, eps, None))

    normals = np.zeros((num_vertices, 3), dtype=np.float32)
    n_dc = 3
    n_rest = colors.shape[1] - n_dc
    sh_dc = colors[:, :n_dc]
    sh_rest = colors[:, n_dc:]
    vertex_ids = np.arange(num_vertices, dtype=np.int32)

    dtype_full = [(attr, "f4") for attr in ["x", "y", "z", "nx", "ny", "nz"]]
    dtype_full.extend([(f"f_dc_{i}", "f4") for i in range(n_dc)])
    dtype_full.extend([(f"f_rest_{i}", "f4") for i in range(n_rest)])
    dtype_full.append(("opacity", "f4"))
    dtype_full.extend([(f"scale_{i}", "f4") for i in range(3)])
    dtype_full.extend([(f"rot_{i}", "f4") for i in range(4)])
    dtype_full.append(("vertex_id", "i4"))

    elements = np.empty(num_vertices, dtype=dtype_full)
    elements["x"] = means[:, 0]
    elements["y"] = means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]
    for i in range(n_dc):
        elements[f"f_dc_{i}"] = sh_dc[:, i]
    for i in range(n_rest):
        elements[f"f_rest_{i}"] = sh_rest[:, i]
    elements["opacity"] = opacities_logit.reshape(-1)
    for i in range(3):
        elements[f"scale_{i}"] = scales_log[:, i]
    for i in range(4):
        elements[f"rot_{i}"] = quats[:, i]
    elements["vertex_id"] = vertex_ids

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vertex_el = PlyElement.describe(elements, "vertex")
    PlyData([vertex_el]).write(output_path)


def compress_decompress(
    ply_path: str,
    output_folder: str,
    output_ply_folder: str,
    frame_start: int,
    frame_end: int,
    interval: int = 1,
    sh_degree: int = config.SH_DEGREE,
    J: int = config.J,
    quantize_step: Optional[dict[str, Any]] = None,
    sh_color_space: str = config.SH_COLOR_SPACE,
    rlgr_block_size: int = config.RLGR_BLOCK_SIZE,
    device: str = config.DEVICE,
    skip_save_ply: bool = False,
) -> list[dict[str, Any]]:
    """Run LiVoGS compress + decompress for an inclusive QUEEN frame range."""
    if quantize_step is None:
        qs = 0.0001
        quantize_step = {
            "quats": qs,
            "scales": qs,
            "opacity": qs,
            "sh_dc": qs,
            "sh_rest": [qs] * (3 * ((sh_degree + 1) ** 2 - 1)),
        }

    if device.startswith("cuda:"):
        device_id = int(device.split(":")[1])
    else:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_ply_folder, exist_ok=True)

    print("=" * 70)
    print("LiVoGS Compress + Decompress Pipeline")
    print("=" * 70)
    print(f"  PLY path:           {ply_path}")
    print(f"  Output folder:      {output_folder}")
    print(f"  Output PLY folder:  {output_ply_folder}")
    print(f"  Frames:             {frame_start} to {frame_end} (interval={interval})")
    print(f"  SH degree:          {sh_degree}")
    print(f"  Device:             {device}")
    print(f"  J (octree depth):   {J}")
    print(
        "  Quantize steps:     "
        f"quats={quantize_step['quats']}, scales={quantize_step['scales']}, "
        f"opacity={quantize_step['opacity']}, sh_dc={quantize_step['sh_dc']}, "
        f"sh_rest={quantize_step['sh_rest']}"
    )
    print(f"  SH color space:     {sh_color_space}")
    print(f"  RLGR block size:    {rlgr_block_size}")
    print("=" * 70)

    print("Warmup GPU...")
    frame = frame_start
    frame_str = str(frame).zfill(4)
    frame_dir = os.path.join(ply_path, "frames", frame_str)
    ply_file_path = find_queen_ply_path(frame_dir)

    params, _ = load_queen_ply(ply_file_path, device=device)
    torch.cuda.synchronize(device_id)
    compressed_state = encode_livogs(
        params,
        J=J,
        device=device,
        device_id=device_id,
        sh_color_space=sh_color_space,
        quantize_step=quantize_step,
        rlgr_block_size=rlgr_block_size,
    )
    torch.cuda.synchronize(device_id)
    decode_livogs(compressed_state, device=device, device_id=device_id)
    torch.cuda.synchronize(device_id)
    print("Warmup GPU done.")

    benchmark_rows: list[dict[str, Any]] = []
    for frame in tqdm(range(frame_start, frame_end + 1, interval), desc="Frames"):
        frame_str = str(frame).zfill(4)
        frame_dir = os.path.join(ply_path, "frames", frame_str)
        ply_file_path = find_queen_ply_path(frame_dir)

        params, uncompressed_size_bytes = load_queen_ply(ply_file_path, device=device)
        num_original = params["means"].shape[0]

        torch.cuda.synchronize(device_id)
        t_enc_start = time.perf_counter()
        compressed_state = encode_livogs(
            params,
            J=J,
            device=device,
            device_id=device_id,
            sh_color_space=sh_color_space,
            quantize_step=quantize_step,
            rlgr_block_size=rlgr_block_size,
        )
        torch.cuda.synchronize(device_id)
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000

        nvox = compressed_state["Nvox"]
        compressed_size_bytes = compressed_state["total_compressed_bytes"]
        position_compressed_bytes = compressed_state["position_compressed_bytes"]
        attribute_compressed_bytes = compressed_state["attribute_compressed_bytes"]

        t_dec_start = time.perf_counter()
        decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)
        torch.cuda.synchronize(device_id)
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000

        if not skip_save_ply:
            ply_out_path = os.path.join(output_ply_folder, "frames", frame_str, "point_cloud.ply")
            save_queen_ply(decoded_params, ply_out_path, sh_degree)

        benchmark_rows.append(
            {
                "frame": frame,
                "encode_time_ms": encode_time_ms,
                "decode_time_ms": decode_time_ms,
                "original_points": num_original,
                "voxelized_points": nvox,
                "uncompressed_size_bytes": uncompressed_size_bytes,
                "compressed_size_bytes": compressed_size_bytes,
                "position_compressed_bytes": position_compressed_bytes,
                "attribute_compressed_bytes": attribute_compressed_bytes,
            }
        )

        tqdm.write(
            f"  Frame {frame}: N={num_original}→{nvox} voxels, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        del params, compressed_state, decoded_params
        torch.cuda.empty_cache()

    if benchmark_rows:
        csv_path = os.path.join(output_folder, "benchmark_livogs.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_id",
                    "encode_time_ms",
                    "decode_time_ms",
                    "original_points",
                    "voxelized_points",
                    "uncompressed_size_bytes",
                    "compressed_size_bytes",
                    "position_compressed_bytes",
                    "attribute_compressed_bytes",
                ]
            )
            for row in benchmark_rows:
                writer.writerow(
                    [
                        row["frame"],
                        f"{row['encode_time_ms']:.2f}",
                        f"{row['decode_time_ms']:.2f}",
                        row["original_points"],
                        row["voxelized_points"],
                        row["uncompressed_size_bytes"],
                        row["compressed_size_bytes"],
                        row["position_compressed_bytes"],
                        row["attribute_compressed_bytes"],
                    ]
                )

        livogs_cfg = {
            "J": J,
            "quantize_step": quantize_step,
            "sh_color_space": sh_color_space,
            "rlgr_block_size": rlgr_block_size,
            "sh_degree": sh_degree,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "interval": interval,
        }
        with open(os.path.join(output_folder, "livogs_config.json"), "w", encoding="utf-8") as f:
            json.dump(livogs_cfg, f, indent=4)

        n = len(benchmark_rows)
        total_enc_ms = sum(row["encode_time_ms"] for row in benchmark_rows)
        total_dec_ms = sum(row["decode_time_ms"] for row in benchmark_rows)
        total_uncomp = sum(row["uncompressed_size_bytes"] for row in benchmark_rows)
        total_comp = sum(row["compressed_size_bytes"] for row in benchmark_rows)
        total_pos = sum(row["position_compressed_bytes"] for row in benchmark_rows)
        total_attr = sum(row["attribute_compressed_bytes"] for row in benchmark_rows)
        total_orig = sum(row["original_points"] for row in benchmark_rows)
        total_vox = sum(row["voxelized_points"] for row in benchmark_rows)

        print("\n" + "=" * 70)
        print("Benchmark Summary (LiVoGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(
            f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  "
            f"(avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)"
        )
        print(
            f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  "
            f"(avg {total_comp / n / 1024 / 1024:.2f} MB/frame)"
        )
        print(
            f"  Total position compressed: {total_pos / 1024 / 1024:.2f} MB  "
            f"(avg {total_pos / n / 1024 / 1024:.2f} MB/frame)"
        )
        print(
            f"  Total attribute compressed: {total_attr / 1024 / 1024:.2f} MB  "
            f"(avg {total_attr / n / 1024 / 1024:.2f} MB/frame)"
        )
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  Avg point reduction:       {total_orig / n:.0f} → {total_vox / n:.0f} ({total_orig / total_vox:.2f}x)")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
    return benchmark_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LiVoGS compress + decompress for QUEEN-trained models",
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to QUEEN checkpoint dir containing frames/NNNN")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder", type=str, required=True,
                        help="Folder for decompressed PLY output")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=300)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=config.SH_DEGREE)
    parser.add_argument("--J", type=int, default=config.J, help="Octree depth for voxelization")
    parser.add_argument("--quantize_step", type=float, default=0.0001,
                        help="Uniform quantization step for all attributes")
    parser.add_argument("--quantize_step_quats", type=float, default=None)
    parser.add_argument("--quantize_step_scales", type=float, default=None)
    parser.add_argument("--quantize_step_opacity", type=float, default=None)
    parser.add_argument("--quantize_step_sh_dc", type=float, default=None)
    parser.add_argument("--quantize_step_sh_rest", type=float, default=None)
    parser.add_argument("--sh_color_space", type=str, default=config.SH_COLOR_SPACE,
                        choices=["rgb", "yuv", "klt"])
    parser.add_argument("--rlgr_block_size", type=int, default=config.RLGR_BLOCK_SIZE)
    parser.add_argument("--quantize_config_json", type=str, default=None,
                        help="Path to JSON with full quantize_config (overrides --quantize_step_*)")
    parser.add_argument("--device", type=str, default=config.DEVICE)
    parser.add_argument("--skip_save_ply", action="store_true",
                        help="Skip saving decompressed PLY files (fast mode)")
    cli_args = parser.parse_args()

    if cli_args.quantize_config_json is not None:
        with open(cli_args.quantize_config_json, encoding="utf-8") as f:
            qp_data = json.load(f)
        qs_dict = qp_data["quantize_config"]
        print(
            f"  Loaded quantize config: {cli_args.quantize_config_json} "
            f"(label: {qp_data.get('label', '?')})"
        )
    else:
        qs = cli_args.quantize_step
        qs_dict = {
            "quats": cli_args.quantize_step_quats or qs,
            "scales": cli_args.quantize_step_scales or qs,
            "opacity": cli_args.quantize_step_opacity or qs,
            "sh_dc": cli_args.quantize_step_sh_dc or qs,
            "sh_rest": [cli_args.quantize_step_sh_rest or qs] * (3 * ((cli_args.sh_degree + 1) ** 2 - 1)),
        }

    compress_decompress(
        ply_path=cli_args.ply_path,
        output_folder=cli_args.output_folder,
        output_ply_folder=cli_args.output_ply_folder,
        frame_start=cli_args.frame_start,
        frame_end=cli_args.frame_end,
        interval=cli_args.interval,
        sh_degree=cli_args.sh_degree,
        J=cli_args.J,
        quantize_step=qs_dict,
        sh_color_space=cli_args.sh_color_space,
        rlgr_block_size=cli_args.rlgr_block_size,
        device=cli_args.device,
        skip_save_ply=cli_args.skip_save_ply,
    )
