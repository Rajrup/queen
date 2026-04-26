#!/usr/bin/env python3
"""
RAHT Ablation: PyTorch vs CUDA implementation.

For each frame:
  1. Run Morton → Voxelization → Merge to get voxelized attributes and Morton codes.
  2. Benchmark both RAHT implementations on the SAME prepared attributes:
       pytorch  – RAHT_param_reorder_fast (prelude) + RAHT2_optimized (fwd) + inverse_RAHT_optimized (inv)
       cuda     – raht_cuda.raht_prelude  (prelude) + raht_cuda.raht_transform x2 (fwd + inv)
  3. Each step (prelude / forward / inverse) is timed separately and also summed.

CSV columns per row:
  frame_id, variant, N_orig, N_vox, n_channels,
  prelude_ms, forward_ms, inverse_ms, total_raht_ms

Usage:
  python scripts/livogs_baseline/ablation/ablation_raht.py \\
      --ply_path /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1 \\
      --output_folder /home/rajrup/Queen/.../ablation/livogs_raht \\
      --J 15 --sh_color_space klt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# sys.path setup (mirror ablation_rlgr.py)
# ---------------------------------------------------------------------------
_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
_ABLATION_DIR   = _THIS_DIR
_LIVOGS_BL_DIR  = os.path.dirname(_ABLATION_DIR)       # livogs_baseline/
_SCRIPTS_DIR    = os.path.dirname(_LIVOGS_BL_DIR)      # scripts/
_QUEEN_ROOT     = os.path.dirname(_SCRIPTS_DIR)        # Queen/
_LIVOGS_COMP    = os.path.join(_QUEEN_ROOT, "LiVoGS", "compression")
_RAHT_PY_ROOT   = os.path.join(_LIVOGS_COMP, "RAHT-3DGS-codec", "python")

for _p in (_QUEEN_ROOT, _LIVOGS_COMP, _RAHT_PY_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse PLY loader from compress_decompress_pipeline.py
sys.path.insert(0, _LIVOGS_BL_DIR)
try:
    from compress_decompress_pipeline import find_queen_ply_path, load_queen_ply
except ImportError:
    find_queen_ply_path = None
    load_queen_ply = None

from compress_decompress_pipeline_optimized import load_videogs_ply, searchForMaxIteration

from gpu_octree_codec import calc_morton
from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from color_space_transforms import (
    normalize_attributes,
    rgb_to_yuv,
    rgb_to_klt15,
)
from RAHT_param import RAHT_param_reorder_fast
from RAHT import RAHT2_optimized
from iRAHT import inverse_RAHT_optimized
import raht_cuda


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _detect_dataset_type(ply_path: str) -> str:
    """Return 'videogs' if checkpoint/<frame>/point_cloud/ exists, else 'queen'."""
    if os.path.isdir(os.path.join(ply_path, "0", "point_cloud")):
        return "videogs"
    if os.path.isdir(os.path.join(ply_path, "frames")):
        return "queen"
    raise ValueError(f"Cannot detect dataset type from {ply_path}")


def _load_frame(ply_path: str, frame: int, dataset_type: str, device: str):
    """Load a single frame's PLY, returning (params, uncompressed_size_bytes)."""
    if dataset_type == "videogs":
        ckpt_dir = os.path.join(ply_path, str(frame), "point_cloud")
        if not os.path.isdir(ckpt_dir):
            return None, 0
        max_iter = searchForMaxIteration(ckpt_dir)
        ply_file = os.path.join(ckpt_dir, f"iteration_{max_iter}", "point_cloud.ply")
        return load_videogs_ply(ply_file, device=device)
    else:
        frame_dir = os.path.join(ply_path, "frames", str(frame).zfill(4))
        if find_queen_ply_path is None:
            raise RuntimeError("Queen PLY loader not available")
        ply_file = find_queen_ply_path(frame_dir)
        if ply_file is None:
            return None, 0
        return load_queen_ply(ply_file, device=device)


def prepare_raht_inputs(
    params: dict,
    J: int,
    sh_color_space: str,
    device: str,
    device_id: int,
) -> tuple:
    """Morton → voxelize → merge → colour-space transform.

    Returns
    -------
    attrs : (Nvox, C) float32 tensor on GPU
    morton_codes : (Nvox,) int64 tensor on GPU
    Nvox : int
    N_orig : int
    dummy_V : (Nvox, 3) float32 tensor  (device placeholder for PyTorch prelude)
    """
    N = params["means"].shape[0]

    # --- (a) Morton ---
    V_means = params["means"]
    vmin    = V_means.min(dim=0)[0]
    V0      = V_means - vmin.unsqueeze(0)
    width   = V0.max()
    voxel_size = width / (2.0 ** J)
    V0_int  = torch.clamp(torch.floor(V0 / voxel_size).long(), 0, 2 ** J - 1).int()

    morton_result = calc_morton(
        V0_int, voxel_grid_depth=J,
        force_64bit_codes=True, device=device_id, return_torch=True,
    )
    mc_points = morton_result["morton_codes"]
    if mc_points.dtype == torch.uint64:
        mc_points = mc_points.to(torch.int64)

    # --- (b) Voxelize ---
    sorted_mc, sort_idx = torch.sort(mc_points)
    boundary     = sorted_mc[1:] - sorted_mc[:-1]
    voxel_start  = torch.cat([torch.tensor([0], device=device),
                               torch.where(boundary != 0)[0] + 1])
    Nvox = len(voxel_start)
    morton_codes = sorted_mc[voxel_start]  # one Morton code per voxel

    # --- (c) Merge ---
    cluster_indices = sort_idx.int()
    cluster_offsets = torch.cat([
        voxel_start,
        torch.tensor([N], dtype=torch.int32, device=device),
    ]).int()

    _, merged_quats, merged_scales, merged_opacities, merged_colors = \
        merge_gaussian_clusters_with_indices(
            params["means"], params["quats"], params["scales"],
            params["opacities"], params["colors"],
            cluster_indices, cluster_offsets, weight_by_opacity=True,
        )

    # --- (d) Colour-space transform ---
    colors_norm, _ = normalize_attributes(merged_colors, target_range=1.0)
    if sh_color_space == "yuv":
        colors_t = rgb_to_yuv(colors_norm)
    elif sh_color_space == "klt":
        colors_t, _ = rgb_to_klt15(colors_norm)
    else:
        colors_t = colors_norm  # rgb

    attrs = torch.cat([
        merged_quats,
        merged_scales,
        merged_opacities.unsqueeze(1),
        colors_t,
    ], dim=1).float()

    # dummy_V: only its `.device` is used by RAHT_param_reorder_fast
    dummy_V = torch.zeros((Nvox, 3), device=device, dtype=torch.float32)

    return attrs, morton_codes, Nvox, N, dummy_V


def sync(device_id: int) -> None:
    torch.cuda.synchronize(device_id)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def benchmark_pytorch(
    attrs: torch.Tensor,
    morton_codes: torch.Tensor,
    Nvox: int,
    J: int,
    dummy_V: torch.Tensor,
    device_id: int,
) -> dict[str, float]:
    """RAHT_param_reorder_fast + RAHT2_optimized + inverse_RAHT_optimized."""

    # --- Prelude ---
    sync(device_id)
    t0 = time.perf_counter()
    List_py, Flags_py, weights_py, order_py = RAHT_param_reorder_fast(
        dummy_V, None, None, J, morton_codes
    )
    sync(device_id)
    prelude_ms = (time.perf_counter() - t0) * 1000

    # --- Forward ---
    attrs_fp64 = attrs.to(torch.float64)
    sync(device_id)
    t0 = time.perf_counter()
    Coeff, _ = RAHT2_optimized(attrs_fp64, List_py, Flags_py, weights_py, return_weights=False)
    sync(device_id)
    forward_ms = (time.perf_counter() - t0) * 1000

    # --- Inverse ---
    sync(device_id)
    t0 = time.perf_counter()
    _ = inverse_RAHT_optimized(Coeff, List_py, Flags_py, weights_py)
    sync(device_id)
    inverse_ms = (time.perf_counter() - t0) * 1000

    return {
        "prelude_ms": prelude_ms,
        "forward_ms": forward_ms,
        "inverse_ms": inverse_ms,
        "total_raht_ms": prelude_ms + forward_ms + inverse_ms,
    }


def benchmark_cuda(
    attrs: torch.Tensor,
    morton_codes: torch.Tensor,
    Nvox: int,
    J: int,
    device_id: int,
) -> dict[str, float]:
    """raht_cuda.raht_prelude + raht_cuda.raht_transform (fwd + inv)."""

    # --- Prelude ---
    sync(device_id)
    t0 = time.perf_counter()
    List_cu, Flags_cu, weights_cu, _ = raht_cuda.raht_prelude(morton_codes, J, Nvox)
    sync(device_id)
    prelude_ms = (time.perf_counter() - t0) * 1000

    # --- Forward ---
    sync(device_id)
    t0 = time.perf_counter()
    Coeff_cu = raht_cuda.raht_transform(attrs, List_cu, Flags_cu, weights_cu, False)
    sync(device_id)
    forward_ms = (time.perf_counter() - t0) * 1000

    # --- Inverse ---
    sync(device_id)
    t0 = time.perf_counter()
    _ = raht_cuda.raht_transform(Coeff_cu, List_cu, Flags_cu, weights_cu, True)
    sync(device_id)
    inverse_ms = (time.perf_counter() - t0) * 1000

    return {
        "prelude_ms": prelude_ms,
        "forward_ms": forward_ms,
        "inverse_ms": inverse_ms,
        "total_raht_ms": prelude_ms + forward_ms + inverse_ms,
    }


VARIANTS = ["pytorch", "cuda"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="RAHT ablation: PyTorch vs CUDA")
    p.add_argument("--ply_path",        type=str, required=True,
                   help="Path to QUEEN model output dir (queen_compressed_<seq>)")
    p.add_argument("--output_folder",   type=str, required=True)
    p.add_argument("--frame_start",     type=int, default=1)
    p.add_argument("--frame_end",       type=int, default=300)
    p.add_argument("--interval",        type=int, default=10)
    p.add_argument("--sh_degree",       type=int, default=2)
    p.add_argument("--J",               type=int, default=15)
    p.add_argument("--sh_color_space",  type=str, default="klt",
                   choices=["rgb", "yuv", "klt"])
    # QP params (saved to config for reproducibility; do not affect RAHT timing)
    p.add_argument("--qps",             type=float, default=None,
                   help="Quantization step for scales")
    p.add_argument("--qpq",             type=float, default=None,
                   help="Quantization step for rotations (quaternions)")
    p.add_argument("--qpo",             type=float, default=None,
                   help="Quantization step for opacity")
    p.add_argument("--qpdc",            type=float, default=None,
                   help="Quantization step for SH DC")
    p.add_argument("--qpac",            type=float, default=None,
                   help="Quantization step for SH AC (rest)")
    p.add_argument("--rlgr_block_size", type=int,   default=512,
                   help="RLGR block size used in the full pipeline (config only)")
    p.add_argument("--device",          type=str, default="cuda:0")
    args = p.parse_args()

    device    = args.device
    device_id = int(device.split(":")[1]) if device.startswith("cuda:") else 0

    os.makedirs(args.output_folder, exist_ok=True)

    print("=" * 70)
    print("RAHT Ablation Study: PyTorch vs CUDA")
    print("=" * 70)
    print(f"  PLY path:         {args.ply_path}")
    print(f"  Output folder:    {args.output_folder}")
    print(f"  Frames:           {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  J={args.J}, sh_degree={args.sh_degree}, sh_color_space={args.sh_color_space}")
    print(f"  QP (config only):  scales={args.qps}, quats={args.qpq}, opacity={args.qpo}, "
          f"sh_dc={args.qpdc}, sh_ac={args.qpac}")
    print(f"  RLGR block size (config only): {args.rlgr_block_size}")
    print(f"  Variants:         {VARIANTS}")
    print("=" * 70)

    # --- GPU warmup ---
    print("Warming up GPU...")
    dataset_type = _detect_dataset_type(args.ply_path)
    print(f"  Dataset type: {dataset_type}")

    params_wu, _ = _load_frame(args.ply_path, args.frame_start, dataset_type, device)
    if params_wu is None:
        raise ValueError(f"PLY not found for warmup frame {args.frame_start}")
    # trim SH to requested degree
    target_sh = 3 * ((args.sh_degree + 1) ** 2)
    if params_wu["colors"].shape[1] > target_sh:
        params_wu["colors"] = params_wu["colors"][:, :target_sh]

    attrs_wu, mc_wu, nvox_wu, _, dummy_wu = prepare_raht_inputs(
        params_wu, args.J, args.sh_color_space, device, device_id
    )
    benchmark_pytorch(attrs_wu, mc_wu, nvox_wu, args.J, dummy_wu, device_id)
    benchmark_cuda(attrs_wu, mc_wu, nvox_wu, args.J, device_id)
    del params_wu, attrs_wu, mc_wu, dummy_wu
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    # --- Main benchmark loop ---
    csv_path = os.path.join(args.output_folder, "ablation_raht.csv")
    csv_columns = [
        "frame_id", "variant", "N_orig", "N_vox", "n_channels",
        "prelude_ms", "forward_ms", "inverse_ms", "total_raht_ms",
    ]

    all_rows: list[dict] = []
    frames   = list(range(args.frame_start, args.frame_end + 1, args.interval))

    for frame in tqdm(frames, desc="Frames"):
        params, _ = _load_frame(args.ply_path, frame, dataset_type, device)
        if params is None:
            tqdm.write(f"  WARNING: PLY not found for frame {frame}, skipping")
            continue
        if params["colors"].shape[1] > target_sh:
            params["colors"] = params["colors"][:, :target_sh]

        attrs, morton_codes, Nvox, N_orig, dummy_V = prepare_raht_inputs(
            params, args.J, args.sh_color_space, device, device_id
        )
        n_channels = attrs.shape[1]

        for variant in VARIANTS:
            if variant == "pytorch":
                result = benchmark_pytorch(attrs, morton_codes, Nvox, args.J, dummy_V, device_id)
            else:
                result = benchmark_cuda(attrs, morton_codes, Nvox, args.J, device_id)

            row = {
                "frame_id":    frame,
                "variant":     variant,
                "N_orig":      N_orig,
                "N_vox":       Nvox,
                "n_channels":  n_channels,
                **result,
            }
            all_rows.append(row)

            tqdm.write(
                f"  frame={frame:4d} {variant:>8s}  "
                f"prelude={result['prelude_ms']:7.2f}ms  "
                f"fwd={result['forward_ms']:7.2f}ms  "
                f"inv={result['inverse_ms']:7.2f}ms  "
                f"total={result['total_raht_ms']:7.2f}ms"
            )

        del params, attrs, morton_codes, dummy_V
        torch.cuda.empty_cache()

    # --- Write CSV ---
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_columns)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nCSV saved: {csv_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"{'Variant':<10s}  {'Prelude':>10s}  {'Forward':>10s}  {'Inverse':>10s}  {'Total':>10s}  {'Speedup':>8s}")
    print("-" * 70)
    means: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        rows_v = [r for r in all_rows if r["variant"] == variant]
        if not rows_v:
            continue
        means[variant] = {k: float(np.mean([r[k] for r in rows_v]))
                          for k in ("prelude_ms", "forward_ms", "inverse_ms", "total_raht_ms")}
    for variant in VARIANTS:
        if variant not in means:
            continue
        m = means[variant]
        ref = means.get("pytorch", {}).get("total_raht_ms", 1.0)
        speedup = ref / m["total_raht_ms"] if m["total_raht_ms"] > 0 else 0
        print(f"{variant:<10s}  {m['prelude_ms']:>10.2f}  {m['forward_ms']:>10.2f}  "
              f"{m['inverse_ms']:>10.2f}  {m['total_raht_ms']:>10.2f}  {speedup:>7.2f}x")
    print("=" * 70)

    # --- Save config ---
    config = {
        "ply_path":        args.ply_path,
        "frame_start":     args.frame_start,
        "frame_end":       args.frame_end,
        "interval":        args.interval,
        "sh_degree":       args.sh_degree,
        "J":               args.J,
        "sh_color_space":  args.sh_color_space,
        "variants":        VARIANTS,
        "pipeline_params": {
            "qp_scales":        args.qps,
            "qp_quats":         args.qpq,
            "qp_opacity":       args.qpo,
            "qp_sh_dc":         args.qpdc,
            "qp_sh_ac":         args.qpac,
            "rlgr_block_size":  args.rlgr_block_size,
        },
    }
    config_path = os.path.join(args.output_folder, "ablation_raht_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
