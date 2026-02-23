#!/usr/bin/env python3
"""
LiVoGS Compression + Decompression for QUEEN-trained Gaussian Splat Models.

For each frame:
  1. Load PLY from QUEEN output (not timed)
  2. encode_livogs(): Morton → Voxelize → Merge → Position encode → RAHT → Quantize → RLGR
  3. decode_livogs(): RLGR → Dequant → Position decode → RAHT prelude → iRAHT
  4. save_to_ply(): Save reconstructed model to disk (not timed)

The compressed bytestream stays on GPU (no GPU→CPU transfer).
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from plyfile import PlyData, PlyElement

# --- Setup sys.path for LiVoGS imports ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QUEEN_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_LIVOGS_COMPRESSION = os.path.join(_QUEEN_ROOT, "LiVoGS", "compression")

if _LIVOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _LIVOGS_COMPRESSION)

_RAHT_PY_ROOT = os.path.join(_LIVOGS_COMPRESSION, "RAHT-3DGS-codec", "python")
if _RAHT_PY_ROOT not in sys.path:
    sys.path.append(_RAHT_PY_ROOT)

_OCTREE_ROOT = os.path.join(_LIVOGS_COMPRESSION, "Octree_Compression_GPU")
sys.path.insert(0, os.path.join(_OCTREE_ROOT, 'python'))
sys.path.insert(0, os.path.join(_OCTREE_ROOT, 'build'))

from gpu_octree_codec import calc_morton, compress_positions_gpu_resident, decompress_positions_gpu_resident
from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from voxelize_pc import voxelize_pc
from color_space_transforms import (
    normalize_attributes,
    denormalize_attributes,
    rgb_to_yuv,
    yuv_to_rgb,
    compute_klt_matrices_per_degree,
    rgb_to_klt,
    klt_to_rgb,
)

# Import GPU-based RAHT implementations
try:
    import raht_cuda
except ImportError as e:
    print(f"Failed to import raht_cuda: {e}")
    print("Ensure you have built and installed the extension.")
    sys.exit(1)

import rlgr_gpu

DEFAULT_SH_DEGREE = 2
DEFAULT_QUANTIZE_STEP = {
    'quats': 0.0001,
    'scales': 0.0001,
    'opacity': 0.0001,
    'sh_dc': 0.0001,
    'sh_rest': [0.0001] * (3 * ((DEFAULT_SH_DEGREE + 1) ** 2 - 1)),
}

# ---------------------------------------------------------------------------
# PLY I/O (QUEEN-compatible)
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def find_queen_ply_path(frame_dir):
    """Find PLY file in a QUEEN frame directory.

    Checks canonical point_cloud.ply first, then falls back to per-iteration.
    """
    canonical = os.path.join(frame_dir, "point_cloud.ply")
    if os.path.exists(canonical):
        return canonical
    iter_dir = os.path.join(frame_dir, "point_cloud")
    if os.path.exists(iter_dir):
        max_iter = searchForMaxIteration(iter_dir)
        per_iter = os.path.join(iter_dir, f"iteration_{max_iter}", "point_cloud.ply")
        if os.path.exists(per_iter):
            return per_iter
    return None


def load_queen_ply(ply_path, device='cuda'):
    """Load a QUEEN-trained PLY and return LiVoGS-compatible param dict on GPU.

    QUEEN PLY attribute order (SH degree 2):
        x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..23, opacity,
        scale_0..2, rot_0..3, vertex_id (int)

    Normals and vertex_id are ignored.
    Opacities are stored in logit space → converted to [0,1] via sigmoid.
    Scales are stored in log space → converted to positive via exp.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith('f_rest_')],
        key=lambda x: int(x.split('_')[-1])
    )
    if rest_names:
        sh_rest = np.stack([vertex[name] for name in rest_names], axis=1)
    else:
        sh_rest = np.zeros((len(vertex), 0), dtype=np.float32)
    colors = np.concatenate([sh_dc, sh_rest], axis=1)

    opacities = np.asarray(vertex['opacity'])
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)

    params = {
        'means': torch.from_numpy(means.copy()).float().to(device),
        'quats': torch.from_numpy(quats.copy()).float().to(device),
        'scales': torch.from_numpy(scales.copy()).float().to(device),
        'opacities': torch.from_numpy(opacities.copy()).float().to(device),
        'colors': torch.from_numpy(colors.copy()).float().to(device),
    }

    uncompressed_size_bytes = sum(
        v.numel() * v.element_size() for v in params.values()
    )

    params['quats'] = F.normalize(params['quats'], p=2, dim=1)
    params['opacities'] = torch.sigmoid(params['opacities'])
    params['scales'] = torch.exp(params['scales'])

    return params, uncompressed_size_bytes


def save_queen_ply(params, output_path, sh_degree=2, eps=1e-6):
    """Save reconstructed params to QUEEN-compatible PLY.

    Converts opacities back to logit space and scales back to log space so that
    QUEEN's GaussianModel.load_ply() can consume them directly.
    Includes vertex_id (int) field to match QUEEN's save_ply format.
    """
    means = params['means'].detach().cpu().float().numpy()
    quats = params['quats'].detach().cpu().float().numpy()
    scales = params['scales'].detach().cpu().float().numpy()
    opacities = params['opacities'].detach().cpu().float().numpy()
    colors = params['colors'].detach().cpu().float().numpy()

    N = means.shape[0]

    # Convert back to raw (logit / log) space for QUEEN compatibility
    opacities_c = np.clip(opacities, eps, 1.0 - eps)
    opacities_logit = np.log(opacities_c / (1.0 - opacities_c))
    scales_log = np.log(np.clip(scales, eps, None))

    normals = np.zeros((N, 3), dtype=np.float32)
    n_dc = 3
    n_rest = colors.shape[1] - n_dc
    sh_dc = colors[:, :n_dc]
    sh_rest = colors[:, n_dc:]
    vertex_ids = np.arange(N, dtype=np.int32)

    dtype_full = [(attr, 'f4') for attr in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
    dtype_full.extend([(f'f_dc_{i}', 'f4') for i in range(n_dc)])
    dtype_full.extend([(f'f_rest_{i}', 'f4') for i in range(n_rest)])
    dtype_full.append(('opacity', 'f4'))
    dtype_full.extend([(f'scale_{i}', 'f4') for i in range(3)])
    dtype_full.extend([(f'rot_{i}', 'f4') for i in range(4)])
    dtype_full.append(('vertex_id', 'i4'))

    elements = np.empty(N, dtype=dtype_full)
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    for i in range(n_dc):
        elements[f'f_dc_{i}'] = sh_dc[:, i]
    for i in range(n_rest):
        elements[f'f_rest_{i}'] = sh_rest[:, i]
    elements['opacity'] = opacities_logit.reshape(-1)
    for i in range(3):
        elements[f'scale_{i}'] = scales_log[:, i]
    for i in range(4):
        elements[f'rot_{i}'] = quats[:, i]
    elements['vertex_id'] = vertex_ids

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------

def encode_livogs(
    params, 
    J=15, 
    device="cuda:0",
    device_id=0,
    precision="fp32",
    sh_color_space='rgb', 
    color_rescale=True,
    quantize_step=DEFAULT_QUANTIZE_STEP,
    rlgr_block_size=4096,
):
    """Encode Gaussian parameters using LiVoGS (all on GPU).

    Steps:
        a. Morton code calculation
        b. Voxelization
        c. Merging Gaussians per voxel
        d. Position encoding (GPU Octree codec)
        e. RAHT (prelude + forward transform)
        f. Quantization of RAHT coefficients
        g. RLGR GPU encoding

    Returns a compressed_state dict with everything the decoder needs.
    """

    float_precision_type = torch.float32
    if precision == "fp64":
        float_precision_type = torch.float64
    elif precision == "fp32":
        float_precision_type = torch.float32
    elif precision == "fp16":
        float_precision_type = torch.float16
    elif precision == "bf16":
        float_precision_type = torch.bfloat16

    N = params['means'].shape[0]
    apply_color_rescale = color_rescale

    # --- (a) Morton code calculation ---
    try:
        V_means = params['means']
        vmin = V_means.min(dim=0)[0]
        V0 = V_means - vmin.unsqueeze(0)
        width = V0.max()
        voxel_size = width / (2.0 ** J)
        V0_integer = torch.clamp(torch.floor(V0 / voxel_size).long(), 0, 2**J - 1).int()

        morton_result = calc_morton(
            V0_integer,
            voxel_grid_depth=J,
            force_64bit_codes=True, 
            device=device_id,
            return_torch=True
        )
        morton_codes_points = morton_result['morton_codes']
        if morton_codes_points.dtype == torch.uint64:
            morton_codes_points = morton_codes_points.to(torch.int64)
    except Exception as e:
        raise RuntimeError(f"Morton code calculation failed: {e}")

    # --- (b) Voxelization ---
    PCvox, PCsorted, voxel_indices, DeltaPC, voxel_info = voxelize_pc(
        params['means'],
        vmin=vmin,
        width=width,
        J=J,
        device=device,
        morton_codes=morton_codes_points
    )
    Nvox = voxel_info['Nvox']
    sort_idx = voxel_info['sort_idx']

    # --- (c) Merging ---
    cluster_indices = sort_idx.int()
    cluster_offsets = torch.cat([
        voxel_indices,
        torch.tensor([N], dtype=torch.int32, device=device)
    ]).int()

    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = \
        merge_gaussian_clusters_with_indices(
            params['means'],
            params['quats'],
            params['scales'],
            params['opacities'],
            params['colors'],
            cluster_indices,
            cluster_offsets,
            weight_by_opacity=True
        )

    V = PCvox[:, :3]
    morton_codes_voxels = voxel_info.get('voxel_morton_codes')
    morton_codes_for_raht = morton_codes_voxels

    # --- (d) Position encoding ---
    compressed_positions = None
    position_compressed_bytes = 0

    morton_codes_sorted_uint64 = morton_codes_voxels.to(torch.uint64)
    compress_result = compress_positions_gpu_resident(
        morton_codes=morton_codes_sorted_uint64,
        octree_depth=J,
        voxel_grid_depth=J,
        force_64bit_codes=True,
        device=device_id
    )
    compressed_positions = compress_result['compressed_data']
    position_compressed_bytes = compress_result['compressed_size_bytes']
    assert position_compressed_bytes == compressed_positions.shape[0], "Position compressed bytes mismatch"

    # --- (e) RAHT ---
     # Apply color space transform + normalization to SH coefficients before RAHT
    # All paths normalize first, then optionally apply color transform
    klt_info = None
    norm_range_str = "[0, 255]" if apply_color_rescale else "[0, 1]"

    target_range = 255.0 if apply_color_rescale else 1.0
    merged_colors_normalized, sh_norm_info = normalize_attributes(merged_colors, target_range=target_range)

    if sh_color_space == "rgb":
        merged_colors_transformed = merged_colors_normalized
        print(f"  SH color space: RGB (min/max normalize to {norm_range_str})")
    elif sh_color_space == "yuv":
        merged_colors_transformed = rgb_to_yuv(merged_colors_normalized)
        print(f"  SH color space: YUV (min/max normalize to {norm_range_str} + RGB->YUV)")
    elif sh_color_space == "klt":
        klt_info = compute_klt_matrices_per_degree(merged_colors_normalized)
        merged_colors_transformed = rgb_to_klt(merged_colors_normalized, klt_info)
        print(f"  SH color space: KLT (min/max normalize to {norm_range_str} + per-degree PCA)")
    else:
        raise ValueError(f"Invalid sh_color_space: {sh_color_space}")

    attributes_to_compress = torch.cat([
        merged_quats,
        merged_scales,
        merged_opacities.unsqueeze(1),
        merged_colors_transformed
    ], dim=1)

    # Convert to specified precision for CUDA RAHT
    print(f"  Using {precision.upper()} precision for RAHT")
    attributes_to_compress = attributes_to_compress.to(float_precision_type)

    ListC, FlagsC, weightsC, order_RAGFT = raht_cuda.raht_prelude(
        morton_codes_for_raht, J, Nvox
    )
    Coeff = raht_cuda.raht_transform(attributes_to_compress, ListC, FlagsC, weightsC, inverse=False)

    # --- (f) Quantization ---
    n_channels = Coeff.shape[1]
    num_sh_coeffs = n_channels - 8
    num_sh_dc = min(3, num_sh_coeffs)
    num_sh_rest = max(0, num_sh_coeffs - 3)

    quantize_step_tensor = torch.zeros(1, n_channels, dtype=float_precision_type, device=device)
    quantize_step_tensor[0, 0:4] = quantize_step['quats']
    quantize_step_tensor[0, 4:7] = quantize_step['scales']
    quantize_step_tensor[0, 7] = quantize_step['opacity']

    sh_dc_qstep = quantize_step['sh_dc']
    if isinstance(sh_dc_qstep, (list, tuple)):
        for i, qstep in enumerate(sh_dc_qstep):
            quantize_step_tensor[0, 8 + i] = qstep
    else:
        quantize_step_tensor[0, 8:8 + num_sh_dc] = sh_dc_qstep

    sh_rest_qstep = quantize_step['sh_rest']
    if num_sh_rest > 0:
        if isinstance(sh_rest_qstep, (list, tuple)):
            for i, qstep in enumerate(sh_rest_qstep):
                quantize_step_tensor[0, 8 + num_sh_dc + i] = qstep
        else:
            quantize_step_tensor[0, 8 + num_sh_dc:] = sh_rest_qstep
    


    # Dead-zone quantizer
    quantize_offset = 0.25 * quantize_step_tensor # dead-zone control
    abs_Coeff = torch.abs(Coeff)
    Coeff_enc = torch.sign(Coeff) * torch.floor((abs_Coeff + quantize_offset) / quantize_step_tensor)

    # Coefficient reordering for encoding
    coeff_reordered = Coeff_enc.index_select(0, order_RAGFT)

    # --- (g) RLGR GPU encoding ---
    coeff_gpu_int32 = coeff_reordered.to(dtype=torch.int32)
    encoder_gpu = rlgr_gpu.EncoderGPU(block_size=rlgr_block_size, flagSigned=1) # 1 => signed integers (RAHT coefficients can be negative)
    compressed_attributes = encoder_gpu.rlgrEncode(coeff_gpu_int32)

    attribute_compressed_bytes = compressed_attributes['compressed_data'].shape[0]
    total_compressed_bytes = position_compressed_bytes + attribute_compressed_bytes

    return {
        'compressed_attributes': compressed_attributes,
        'compressed_positions': compressed_positions,
        'quantize_step_tensor': quantize_step_tensor,
        'quantize_offset': quantize_offset,
        'J': J,
        'N_original': N,
        'voxel_info': voxel_info,
        'Nvox': Nvox,
        'sh_norm_info': sh_norm_info,
        'klt_info': klt_info,
        'sh_color_space': sh_color_space,
        'total_compressed_bytes': total_compressed_bytes,
        'attribute_compressed_bytes': attribute_compressed_bytes,
        'position_compressed_bytes': position_compressed_bytes,
        'float_precision_type': float_precision_type,
    }


def decode_livogs(compressed_state, device='cuda:0', device_id=0):
    """Decode compressed state using LiVoGS (all on GPU).

    Steps:
        a. RLGR GPU decoding
        b. Dequantization
        c. Position decoding (GPU Octree codec)
        d. RAHT prelude (decoder-side, from decoded positions)
        e. iRAHT (inverse RAHT)
        f. Reconstruct gaussian splat model

    Returns reconstructed params dict (all tensors on GPU).
    """
    compressed_attributes = compressed_state['compressed_attributes']
    compressed_positions = compressed_state['compressed_positions']
    J = compressed_state['J']
    voxel_info = compressed_state['voxel_info']
    sh_norm_info = compressed_state['sh_norm_info']
    klt_info = compressed_state['klt_info']
    sh_color_space = compressed_state['sh_color_space']
    quantize_step_tensor = compressed_state['quantize_step_tensor']
    quantize_offset = compressed_state['quantize_offset']
    float_precision_type = compressed_state['float_precision_type']

    # --- (a) RLGR GPU decoding ---
    decoder_gpu = rlgr_gpu.DecoderGPU()
    decoded_gpu_tensor, _ = decoder_gpu.rlgrDecode(compressed_attributes)

    # --- (b) Dequantization ---
    Coeff_dec = decoded_gpu_tensor.to(dtype=float_precision_type)
    adjustment = quantize_step_tensor / 2.0 - quantize_offset
    Coeff_dec = Coeff_dec * quantize_step_tensor + torch.sign(Coeff_dec) * adjustment

    # --- (c) Position decoding ---
    decompress_result = decompress_positions_gpu_resident(
        compressed_data=compressed_positions,
        octree_depth=J,
        voxel_grid_depth=J,
        force_64bit_codes=True,
        device=device_id
    )
    V_decompressed = decompress_result['positions']
    if V_decompressed.dtype == torch.uint32:
        V_decompressed = V_decompressed.to(torch.int32)
    V_for_decode = V_decompressed.to(dtype=torch.float32)

    # --- (d) RAHT prelude (decoder-side) ---
    V_dec_int = V_for_decode.int()
    morton_result_dec = calc_morton(
        V_dec_int,
        voxel_grid_depth=J,
        force_64bit_codes=True,
        device=device_id,
        return_torch=True
    )
    morton_codes_for_decode = morton_result_dec['morton_codes']
    if morton_codes_for_decode.dtype == torch.uint64:
        morton_codes_for_decode = morton_codes_for_decode.to(torch.int64)

    Nvox_dec = morton_codes_for_decode.shape[0]
    ListC_dec, FlagsC_dec, weightsC_dec, order_RAGFT_dec = raht_cuda.raht_prelude(
        morton_codes_for_decode, J, Nvox_dec
    )

    # --- (e) iRAHT (inverse RAHT) ---
    order_RAGFT_inv = torch.argsort(order_RAGFT_dec)
    Coeff_to_decompress = Coeff_dec[order_RAGFT_inv, :]

    attributes_reconstructed = raht_cuda.raht_transform(Coeff_to_decompress, ListC_dec, FlagsC_dec, weightsC_dec, inverse=True)

    # --- (f) Reconstruct gaussian splat model ---
    recon_quats_raw = attributes_reconstructed[:, 0:4]
    recon_scales_raw = attributes_reconstructed[:, 4:7]
    recon_opacities_raw = attributes_reconstructed[:, 7]
    recon_colors_sh_raw = attributes_reconstructed[:, 8:]

    V_final = V_for_decode

    # Inverse color space transform
    if sh_color_space == "yuv":
        recon_colors_sh_raw = yuv_to_rgb(recon_colors_sh_raw)
    elif sh_color_space == "klt":
        recon_colors_sh_raw = klt_to_rgb(recon_colors_sh_raw, klt_info)
    recon_colors_sh_raw = denormalize_attributes(recon_colors_sh_raw, sh_norm_info)

    # World-space positions from voxel coordinates
    voxel_positions_world = (V_final + 0.5) * voxel_info['voxel_size'] + voxel_info['vmin']
    recon_means = voxel_positions_world
    # Use F.normalize to safely handle edge cases (e.g., zero-norm quaternions)
    recon_quats = F.normalize(recon_quats_raw, p=2, dim=1)
    recon_scales = torch.abs(recon_scales_raw)
    recon_opacities = torch.clamp(recon_opacities_raw, 0, 1)
    recon_colors = recon_colors_sh_raw

    return {
        'means': recon_means.float(),
        'quats': recon_quats.float(),
        'scales': recon_scales.float(),
        'opacities': recon_opacities.float(),
        'colors': recon_colors.float(),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiVoGS compress + decompress for QUEEN-trained models"
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to QUEEN model output dir (e.g. output/cook_spinach_trained_compressed)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder", type=str, required=True,
                        help="Folder for decompressed PLY output")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=300)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=2)
    # LiVoGS-specific parameters
    parser.add_argument("--J", type=int, default=15,
                        help="Octree depth for voxelization (default: 15)")
    parser.add_argument("--quantize_step", type=float, default=0.0001,
                        help="Uniform quantization step for all attributes (default: 0.0001)")
    parser.add_argument("--quantize_step_quats", type=float, default=None,
                        help="Override quantization step for quaternions")
    parser.add_argument("--quantize_step_scales", type=float, default=None,
                        help="Override quantization step for scales")
    parser.add_argument("--quantize_step_opacity", type=float, default=None,
                        help="Override quantization step for opacity")
    parser.add_argument("--quantize_step_sh_dc", type=float, default=None,
                        help="Override quantization step for SH DC")
    parser.add_argument("--quantize_step_sh_rest", type=float, default=None,
                        help="Override quantization step for SH rest")
    parser.add_argument("--sh_color_space", type=str, default="rgb",
                        choices=["rgb", "yuv", "klt"],
                        help="Color space for SH coefficients (default: rgb)")
    parser.add_argument("--color_rescale", action="store_true", default=True,
                        help="Rescale SH to [0, 255] before color transform (default: True)")
    parser.add_argument("--no_color_rescale", action="store_true",
                        help="Disable SH color rescaling")
    parser.add_argument("--rlgr_block_size", type=int, default=4096,
                        help="RLGR parallel block size (default: 4096)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.no_color_rescale:
        args.color_rescale = False

    # Build quantize_step dict
    qs = args.quantize_step
    quantize_step = {
        'quats': args.quantize_step_quats if args.quantize_step_quats is not None else qs,
        'scales': args.quantize_step_scales if args.quantize_step_scales is not None else qs,
        'opacity': args.quantize_step_opacity if args.quantize_step_opacity is not None else qs,
        'sh_dc': args.quantize_step_sh_dc if args.quantize_step_sh_dc is not None else qs,
        'sh_rest': [args.quantize_step_sh_rest if args.quantize_step_sh_rest is not None else qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
    }

    device = args.device
    if device.startswith('cuda:'):
        device_id = int(device.split(':')[1])
    else:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_ply_folder, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("LiVoGS Compress + Decompress Pipeline")
    print("=" * 70)
    print(f"  PLY path:           {args.ply_path}")
    print(f"  Output folder:      {args.output_folder}")
    print(f"  Output PLY folder:  {args.output_ply_folder}")
    print(f"  Frames:             {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Device:             {device}")
    print(f"  J (octree depth):   {args.J}")
    print(f"  Quantize steps:     quats={quantize_step['quats']}, scales={quantize_step['scales']}, "
          f"opacity={quantize_step['opacity']}, sh_dc={quantize_step['sh_dc']}, sh_rest={quantize_step['sh_rest']}")
    print(f"  SH color space:     {args.sh_color_space}")
    print(f"  Color rescale:      {args.color_rescale}")
    print(f"  RLGR block size:    {args.rlgr_block_size}")
    print("=" * 70)

    # Warmup GPU
    print("Warmup GPU...")
    frame = args.frame_start
    frame_str = str(frame).zfill(4)
    frame_dir = os.path.join(args.ply_path, "frames", frame_str)
    ply_file_path = find_queen_ply_path(frame_dir)
    if ply_file_path is None:
        raise ValueError(f"PLY not found for warmup frame {frame_str} at {frame_dir}")

    params, _ = load_queen_ply(ply_file_path, device=device)

    torch.cuda.synchronize(device_id)
    compressed_state = encode_livogs(
        params, J=args.J, device=device, device_id=device_id,
        sh_color_space=args.sh_color_space,
        color_rescale=args.color_rescale,
        quantize_step=quantize_step,
        rlgr_block_size=args.rlgr_block_size,
    )
    torch.cuda.synchronize(device_id)

    decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)

    torch.cuda.synchronize(device_id)
    print("Warmup GPU done.")

    benchmark_rows = []

    for frame in tqdm(range(args.frame_start, args.frame_end + 1, args.interval), desc="Frames"):

        # --- 1. Load PLY (not timed) ---
        frame_str = str(frame).zfill(4)
        frame_dir = os.path.join(args.ply_path, "frames", frame_str)
        ply_file_path = find_queen_ply_path(frame_dir)
        if ply_file_path is None:
            print(f"Warning: PLY not found for frame {frame_str} at {frame_dir}, skipping")
            continue

        params, uncompressed_size_bytes = load_queen_ply(ply_file_path, device=device)
        N_original = params['means'].shape[0]

        # --- 2. Encode (timed) ---
        torch.cuda.synchronize(device_id)
        t_enc_start = time.perf_counter()

        compressed_state = encode_livogs(
            params, J=args.J, device=device, device_id=device_id,
            sh_color_space=args.sh_color_space,
            color_rescale=args.color_rescale,
            quantize_step=quantize_step,
            rlgr_block_size=args.rlgr_block_size,
        )

        torch.cuda.synchronize(device_id)
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000

        Nvox = compressed_state['Nvox']
        compressed_size_bytes = compressed_state['total_compressed_bytes']
        position_compressed_bytes = compressed_state['position_compressed_bytes']
        attribute_compressed_bytes = compressed_state['attribute_compressed_bytes']

        # --- 3. Decode (timed) ---
        t_dec_start = time.perf_counter()

        decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)

        torch.cuda.synchronize(device_id)
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000

        # --- 4. Save PLY (not timed) ---
        frame_ply_dir = os.path.join(args.output_ply_folder, "frames", frame_str)
        os.makedirs(frame_ply_dir, exist_ok=True)
        ply_out_path = os.path.join(frame_ply_dir, "point_cloud.ply")
        save_queen_ply(decoded_params, ply_out_path, args.sh_degree)

        benchmark_rows.append({
            "frame": frame,
            "encode_time_ms": encode_time_ms,
            "decode_time_ms": decode_time_ms,
            "original_points": N_original,
            "voxelized_points": Nvox,
            "uncompressed_size_bytes": uncompressed_size_bytes,
            "position_compressed_bytes": position_compressed_bytes,
            "attribute_compressed_bytes": attribute_compressed_bytes,
            "compressed_size_bytes": compressed_size_bytes,
        })

        tqdm.write(
            f"  Frame {frame}: N={N_original}→{Nvox} voxels, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        # Clean up
        del params, compressed_state, decoded_params
        torch.cuda.empty_cache()

    # --- Benchmark CSV and summary ---
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_livogs.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "encode_time_ms", "decode_time_ms",
                         "original_points", "voxelized_points", 
                         "uncompressed_size_bytes", "compressed_size_bytes",
                         "position_compressed_bytes", "attribute_compressed_bytes"])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"],
                    f"{r['encode_time_ms']:.2f}",
                    f"{r['decode_time_ms']:.2f}",
                    r["original_points"],
                    r["voxelized_points"],
                    r["uncompressed_size_bytes"],
                    r["compressed_size_bytes"],
                    r["position_compressed_bytes"],
                    r["attribute_compressed_bytes"],
                ])

        n = len(benchmark_rows)
        total_enc_ms = sum(r["encode_time_ms"] for r in benchmark_rows)
        total_dec_ms = sum(r["decode_time_ms"] for r in benchmark_rows)
        total_uncomp = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_pos_comp = sum(r["position_compressed_bytes"] for r in benchmark_rows)
        total_attr_comp = sum(r["attribute_compressed_bytes"] for r in benchmark_rows)
        total_comp = sum(r["compressed_size_bytes"] for r in benchmark_rows)
        total_orig_points = sum(r["original_points"] for r in benchmark_rows)
        total_vox_points = sum(r["voxelized_points"] for r in benchmark_rows)

        # Save config JSON for reproducibility
        import json
        config = {
            "J": args.J,
            "quantize_step": quantize_step,
            "sh_color_space": args.sh_color_space,
            "color_rescale": args.color_rescale,
            "rlgr_block_size": args.rlgr_block_size,
            "sh_degree": args.sh_degree,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "interval": args.interval,
        }
        with open(os.path.join(args.output_folder, "livogs_config.json"), "w") as f:
            json.dump(config, f, indent=4)

        print("\n" + "=" * 70)
        print("Benchmark Summary (LiVoGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  (avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total position compressed size:   {total_pos_comp / 1024 / 1024:.2f} MB  (avg {total_pos_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total attribute compressed size:   {total_attr_comp / 1024 / 1024:.2f} MB  (avg {total_attr_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  (avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  Avg point reduction:       {total_orig_points / n:.0f} → {total_vox_points / n:.0f} "
              f"({total_orig_points / total_vox_points:.2f}x)")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
