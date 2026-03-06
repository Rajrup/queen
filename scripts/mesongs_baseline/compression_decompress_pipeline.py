#!/usr/bin/env python3
"""
MesonGS Compression + Decompression for QUEEN-trained Gaussian Splat Models.

For each frame:
  1. Load PLY from QUEEN output into MesonGS GaussianModel
  2. Compute importance via cal_imp() (renders from train cameras)
  3. Optional: prune low-importance Gaussians
  4. encode_mesongs(): Octree -> VQ -> RAHT -> Block Quantize -> LZ77
  5. decode_mesongs(): LZ77 -> Dequant -> iRAHT -> VQ lookup -> Octree decode
  6. Convert Euler angles -> quaternions, save as QUEEN-compatible PLY

Must be run from the MesonGS directory in the queen conda environment.
"""

import os
import sys
import csv
import json
import time
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from plyfile import PlyData, PlyElement

# --- sys.path setup: MesonGS root must be on path (for raht_torch etc.) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QUEEN_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_MESONGS_ROOT = os.path.join(_QUEEN_ROOT, "MesonGS")
if _MESONGS_ROOT not in sys.path:
    sys.path.insert(0, _MESONGS_ROOT)

from scene import GaussianModel
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo, getNerfppNorm
from arguments import OptimizationParams
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov
from mesongs import cal_imp, prune_mask
from compression.compress_decompress import encode_mesongs, decode_mesongs
from compression.utils import euler_to_quaternion

# ---------------------------------------------------------------------------
# Neural_3D_Video Config (same structure as universal_config in mesongs.py)
# ---------------------------------------------------------------------------
DEFAULT_DEPTH = 17
DEFAULT_NUM_BITS = 8
DEFAULT_N_BLOCK = 57
DEFAULT_CODEBOOK_SIZE = 2048
DEFAULT_PRUNE_PERCENT = 0.0

neural3d_config = {
    'prune': {
        'cook_spinach': DEFAULT_PRUNE_PERCENT,
        'cut_roasted_beef': DEFAULT_PRUNE_PERCENT,
        'flame_salmon_1': DEFAULT_PRUNE_PERCENT,
        'flame_steak': DEFAULT_PRUNE_PERCENT,
        'sear_steak': DEFAULT_PRUNE_PERCENT,
        'coffee_martini': DEFAULT_PRUNE_PERCENT,
    },
    'depth': {
        'cook_spinach': DEFAULT_DEPTH,
        'cut_roasted_beef': DEFAULT_DEPTH,
        'flame_salmon_1': DEFAULT_DEPTH,
        'flame_steak': DEFAULT_DEPTH,
        'sear_steak': DEFAULT_DEPTH,
        'coffee_martini': DEFAULT_DEPTH,
    },
    'n_block': {
        'cook_spinach': DEFAULT_N_BLOCK,
        'cut_roasted_beef': DEFAULT_N_BLOCK,
        'flame_salmon_1': DEFAULT_N_BLOCK,
        'flame_steak': DEFAULT_N_BLOCK,
        'sear_steak': DEFAULT_N_BLOCK,
        'coffee_martini': DEFAULT_N_BLOCK,
    },
    'cb': {
        'cook_spinach': DEFAULT_CODEBOOK_SIZE,
        'cut_roasted_beef': DEFAULT_CODEBOOK_SIZE,
        'flame_salmon_1': DEFAULT_CODEBOOK_SIZE,
        'flame_steak': DEFAULT_CODEBOOK_SIZE,
        'sear_steak': DEFAULT_CODEBOOK_SIZE,
        'coffee_martini': DEFAULT_CODEBOOK_SIZE,
    },
    'num_bits': {
        'cook_spinach': DEFAULT_NUM_BITS,
        'cut_roasted_beef': DEFAULT_NUM_BITS,
        'flame_salmon_1': DEFAULT_NUM_BITS,
        'flame_steak': DEFAULT_NUM_BITS,
        'sear_steak': DEFAULT_NUM_BITS,
        'coffee_martini': DEFAULT_NUM_BITS,
    },
}

# ---------------------------------------------------------------------------
# Camera Loading (from Neural_3D_Video poses_bounds.npy)
#
# Replicates the coordinate transform from QUEEN's readCamerasFromPoseBounds()
# in scene/dataset_readers.py, with test_indices=[0] (cam00 = test).
# ---------------------------------------------------------------------------

def parse_dynerf_cameras(dataset_path, test_indices=None):
    """Parse camera extrinsics from poses_bounds.npy (LLFF format).

    Returns (train_cam_params, cameras_extent) where train_cam_params is a list
    of dicts {uid, R, T, FovX, FovY, cam_dir} for each training camera.
    """
    if test_indices is None:
        test_indices = [0]

    image = Image.open(os.path.join(dataset_path, 'cam00', 'images', '0000.png'))
    image_wh = (image.width, image.height)

    poses_arr = np.load(os.path.join(dataset_path, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)

    H, W, focal = poses[0, :, -1]
    scale = W / image_wh[0]
    focal = focal / scale
    focal = [focal, focal]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    videos = sorted(glob.glob(os.path.join(dataset_path, "cam*")))
    cam_dirs = [os.path.basename(v) for v in videos]
    n_cams = poses.shape[0]

    FovX = focal2fov(focal[0], image_wh[0])
    FovY = focal2fov(focal[1], image_wh[1])

    train_cam_params = []
    cam_infos_for_norm = []
    train_uid = 0

    for idx in range(n_cams):
        if idx in test_indices:
            continue

        pose = np.array(poses[idx])
        R = pose[:3, :3]
        R = -R
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)

        cam_dir = cam_dirs[idx] if idx < len(cam_dirs) else f"cam{idx:02d}"

        train_cam_params.append({
            'uid': train_uid,
            'colmap_id': idx,
            'R': R,
            'T': T,
            'FovX': FovX,
            'FovY': FovY,
            'cam_dir': cam_dir,
        })

        cam_infos_for_norm.append(CameraInfo(
            uid=train_uid, R=R, T=T, FovY=FovY, FovX=FovX,
            image=None, image_path="", image_name=cam_dir,
            width=image_wh[0], height=image_wh[1]
        ))
        train_uid += 1

    nerf_norm = getNerfppNorm(cam_infos_for_norm)
    cameras_extent = nerf_norm["radius"]

    total_cams = n_cams
    print(f"Total cameras: {total_cams}, Train cameras (test_indices={test_indices}): {len(train_cam_params)}")

    return train_cam_params, cameras_extent, image_wh


def build_train_cameras(train_cam_params, dataset_path, frame_str, image_wh):
    """Build MesonGS Camera objects for a specific frame by loading images.

    frame_str: 4-digit zero-padded frame ID (e.g. '0001').
    """
    cameras = []
    for p in train_cam_params:
        image_path = os.path.join(dataset_path, p['cam_dir'], 'images', f'{frame_str}.png')
        if not os.path.exists(image_path):
            print(f"Warning: image not found: {image_path}")
            continue

        image_pil = Image.open(image_path).convert("RGB")
        resized_image = PILtoTorch(image_pil, image_wh)
        gt_image = resized_image[:3, ...]

        cam = Camera(
            colmap_id=p['colmap_id'], R=p['R'], T=p['T'],
            FoVx=p['FovX'], FoVy=p['FovY'],
            image=gt_image, gt_alpha_mask=None,
            image_name=f"{p['cam_dir']}_{frame_str}", uid=p['uid'],
        )
        cameras.append(cam)

    return cameras

# ---------------------------------------------------------------------------
# PLY utilities
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def find_queen_ply_path(frame_dir):
    """Find PLY file in a QUEEN frame directory."""
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


def compute_uncompressed_size(gaussians, sh_degree=2):
    """Compute uncompressed size in bytes from GaussianModel attributes (float32)."""
    N = gaussians.get_xyz.shape[0]
    n_sh_rest = 3 * ((sh_degree + 1) ** 2 - 1)
    n_floats_per_point = 3 + 3 + n_sh_rest + 1 + 3 + 4  # xyz + f_dc + f_rest + opacity + scale + rot
    return N * n_floats_per_point * 4  # float32 = 4 bytes


def save_decoded_ply(decoded_gaussians, output_path):
    """Convert decoded model (Euler angles) to quaternions and save as QUEEN-compatible PLY.

    QUEEN PLYs include a vertex_id (int) field. Opacities and scales from
    MesonGS decode are already in logit/log space respectively.
    """
    with torch.no_grad():
        quats = euler_to_quaternion(decoded_gaussians._euler.detach())

    xyz = decoded_gaussians._xyz.detach().cpu().numpy()
    N = xyz.shape[0]
    normals = np.zeros_like(xyz)
    f_dc = decoded_gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = decoded_gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = decoded_gaussians._opacity.detach().cpu().numpy()
    scales = decoded_gaussians._scaling.detach().cpu().numpy()
    rotation = quats.detach().cpu().numpy()
    vertex_ids = np.arange(N, dtype=np.int32)

    n_dc = f_dc.shape[1]
    n_rest = f_rest.shape[1]

    dtype_full = [(attr, 'f4') for attr in ['x', 'y', 'z', 'nx', 'ny', 'nz']]
    dtype_full.extend([(f'f_dc_{i}', 'f4') for i in range(n_dc)])
    dtype_full.extend([(f'f_rest_{i}', 'f4') for i in range(n_rest)])
    dtype_full.append(('opacity', 'f4'))
    dtype_full.extend([(f'scale_{i}', 'f4') for i in range(3)])
    dtype_full.extend([(f'rot_{i}', 'f4') for i in range(4)])
    dtype_full.append(('vertex_id', 'i4'))

    elements = np.empty(N, dtype=dtype_full)
    attributes_f = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotation), axis=1)
    for i, (name, _) in enumerate(dtype_full[:-1]):
        elements[name] = attributes_f[:, i]
    elements['vertex_id'] = vertex_ids

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MesonGS compress + decompress for QUEEN-trained models"
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to QUEEN model output dir (e.g. pretrained_output/.../queen_compressed_cook_spinach)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to Neural_3D_Video dataset (containing cam*/images/*.png and poses_bounds.npy)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder", type=str, required=True,
                        help="Folder for decompressed PLY output")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=300)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--scene_name", type=str, required=True,
                        help="Neural_3D_Video sequence name (e.g. cook_spinach)")
    parser.add_argument("--save_bitstreams", action="store_true",
                        help="Write .npz bitstreams and .zip to disk per frame")
    parser.add_argument("--white_background", action="store_true")

    # MesonGS hyperparameters (override config defaults)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Octree depth (default: from config)")
    parser.add_argument("--n_block", type=int, default=DEFAULT_N_BLOCK, help="Block quantization count (default: from config)")
    parser.add_argument("--codebook_size", type=int, default=DEFAULT_CODEBOOK_SIZE, help="VQ codebook size (default: from config)")
    parser.add_argument("--prune", action="store_true", help="Enable pruning before compression")
    parser.add_argument("--prune_percent", type=float, default=DEFAULT_PRUNE_PERCENT,
                        help="Prune fraction (default: from config)")

    # MesonGS defaults that rarely change
    parser.add_argument("--oct_merge", type=str, default="mean", choices=["mean", "imp", "rand"])
    parser.add_argument("--batch_size", type=int, default=262144)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num_bits", type=int, default=DEFAULT_NUM_BITS)

    args = parser.parse_args()

    # --- Load config (from neural3d_config, overridable by CLI) ---
    scene = args.scene_name
    if scene not in neural3d_config['depth']:
        print(f"Warning: scene '{scene}' not in neural3d_config, using defaults")

    depth = args.depth if args.depth is not None else neural3d_config['depth'].get(scene, DEFAULT_DEPTH)
    n_block = args.n_block if args.n_block is not None else neural3d_config['n_block'].get(scene, DEFAULT_N_BLOCK)
    codebook_size = args.codebook_size if args.codebook_size is not None else neural3d_config['cb'].get(scene, DEFAULT_CODEBOOK_SIZE)
    prune_percent = args.prune_percent if args.prune_percent is not None else neural3d_config['prune'].get(scene, DEFAULT_PRUNE_PERCENT)
    num_bits = args.num_bits if args.num_bits is not None else neural3d_config['num_bits'].get(scene, DEFAULT_NUM_BITS)

    # --- Build dataset_args (SimpleNamespace matching what encode/decode_mesongs expects) ---
    from types import SimpleNamespace
    dataset_args = SimpleNamespace(
        sh_degree=args.sh_degree,
        depth=depth,
        num_bits=num_bits,
        oct_merge=args.oct_merge,
        raht=True,
        per_block_quant=True,
        per_channel_quant=False,
        n_block=n_block,
        codebook_size=codebook_size,
        batch_size=args.batch_size,
        steps=args.steps,
        percent=prune_percent,
        white_background=args.white_background,
    )

    pipe_args = SimpleNamespace(
        convert_SHs_python=True,
        compute_cov3D_python=False,
        scene_imp=scene,
        debug=False,
    )

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_ply_folder, exist_ok=True)

    # --- Print configuration ---
    print("=" * 70)
    print("MesonGS Compress + Decompress Pipeline (QUEEN)")
    print("=" * 70)
    print(f"  PLY path:           {args.ply_path}")
    print(f"  Dataset path:       {args.dataset_path}")
    print(f"  Output folder:      {args.output_folder}")
    print(f"  Output PLY folder:  {args.output_ply_folder}")
    print(f"  Frames:             {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  Scene:              {scene}")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Octree depth:       {depth}")
    print(f"  N block:            {n_block}")
    print(f"  Codebook size:      {codebook_size}")
    print(f"  Pruning:            {args.prune} (percent={prune_percent})")
    print(f"  Oct merge:          {args.oct_merge}")
    print(f"  VQ steps:           {args.steps}")
    print(f"  Save bitstreams:    {args.save_bitstreams}")
    print("=" * 70)

    # --- Parse camera params once (extrinsics are fixed across frames) ---
    print("\nParsing camera parameters...")
    train_cam_params, cameras_extent, image_wh = parse_dynerf_cameras(args.dataset_path)
    print(f"Loaded {len(train_cam_params)} train camera params, cameras_extent={cameras_extent:.4f}")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --- Setup OptimizationParams for training_setup (needed if --prune) ---
    opt_parser = argparse.ArgumentParser()
    op = OptimizationParams(opt_parser)
    opt_args = op.extract(opt_parser.parse_args([]))

    # --- Per-frame loop ---
    benchmark_rows = []

    for frame in tqdm(range(args.frame_start, args.frame_end + 1, args.interval), desc="Frames"):
        frame_str = str(frame).zfill(4)

        # --- 1. Load PLY ---
        frame_dir = os.path.join(args.ply_path, "frames", frame_str)
        ply_file_path = find_queen_ply_path(frame_dir)
        if ply_file_path is None:
            print(f"Warning: PLY not found for frame {frame_str} at {frame_dir}, skipping")
            continue

        gaussians = GaussianModel(args.sh_degree, depth=depth, num_bits=num_bits)
        gaussians.load_ply(ply_file_path, og_number_points=-1, spatial_lr_scale=cameras_extent)
        N_original = gaussians.get_xyz.shape[0]
        uncompressed_size_bytes = compute_uncompressed_size(gaussians, args.sh_degree)

        # --- 2. Setup optimizer if pruning ---
        if args.prune:
            gaussians.training_setup(opt_args)

        # --- 3. Build train cameras with images for this frame ---
        train_cameras = build_train_cameras(train_cam_params, args.dataset_path, frame_str, image_wh)

        # --- 4. Importance calculation ---
        torch.cuda.synchronize()
        t_enc_start = time.perf_counter()
        with torch.no_grad():
            imp = cal_imp(gaussians, train_cameras, pipe_args, background)

        # --- 5. Optional pruning ---
        N_after_prune = N_original
        if args.prune:
            pmask = prune_mask(dataset_args.percent, imp)
            imp = imp[torch.logical_not(pmask)]
            gaussians.prune_points(pmask)
            N_after_prune = gaussians.get_xyz.shape[0]

        # --- 6. Encode (timed) ---
        bitstreams = encode_mesongs(
            gaussians, dataset_args, imp,
            output_dir=os.path.join(args.output_folder, f"frame_{frame_str}") if args.save_bitstreams else "",
            save_to_disk=args.save_bitstreams,
        )

        torch.cuda.synchronize()
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000

        compressed_size_bytes = sum(len(v) for v in bitstreams.values())
        N_after_octree = gaussians.get_xyz.shape[0]

        # --- 7. Decode (timed) ---
        torch.cuda.synchronize()
        t_dec_start = time.perf_counter()

        decoded_gaussians = decode_mesongs(bitstreams, dataset_args)

        torch.cuda.synchronize()
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000

        N_decoded = decoded_gaussians.get_xyz.shape[0]

        # --- 8. Save PLY (Euler -> quaternion -> QUEEN-compatible PLY) ---
        frame_ply_dir = os.path.join(args.output_ply_folder, "frames", frame_str)
        os.makedirs(frame_ply_dir, exist_ok=True)
        ply_out_path = os.path.join(frame_ply_dir, "point_cloud.ply")
        save_decoded_ply(decoded_gaussians, ply_out_path)

        benchmark_rows.append({
            "frame": frame_str,
            "encode_time_ms": encode_time_ms,
            "decode_time_ms": decode_time_ms,
            "original_points": N_original,
            "after_prune_points": N_after_prune,
            "after_octree_points": N_after_octree,
            "decoded_points": N_decoded,
            "uncompressed_size_bytes": uncompressed_size_bytes,
            "compressed_size_bytes": compressed_size_bytes,
        })

        tqdm.write(
            f"  Frame {frame_str}: N={N_original}"
            f"{'→' + str(N_after_prune) + ' pruned' if args.prune else ''}"
            f"→{N_after_octree} octree→{N_decoded} decoded, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        del gaussians, decoded_gaussians, bitstreams, imp, train_cameras
        torch.cuda.empty_cache()

    # --- Benchmark CSV and summary ---
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_mesongs.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "encode_time_ms", "decode_time_ms",
                         "original_points", "after_prune_points",
                         "after_octree_points", "decoded_points",
                         "uncompressed_size_bytes", "compressed_size_bytes"])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"],
                    f"{r['encode_time_ms']:.2f}",
                    f"{r['decode_time_ms']:.2f}",
                    r["original_points"],
                    r["after_prune_points"],
                    r["after_octree_points"],
                    r["decoded_points"],
                    r["uncompressed_size_bytes"],
                    r["compressed_size_bytes"],
                ])

        n = len(benchmark_rows)
        total_enc_ms = sum(r["encode_time_ms"] for r in benchmark_rows)
        total_dec_ms = sum(r["decode_time_ms"] for r in benchmark_rows)
        total_uncomp = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_comp = sum(r["compressed_size_bytes"] for r in benchmark_rows)
        total_orig_points = sum(r["original_points"] for r in benchmark_rows)
        total_octree_points = sum(r["after_octree_points"] for r in benchmark_rows)
        total_decoded_points = sum(r["decoded_points"] for r in benchmark_rows)

        config_out = {
            "scene_name": scene,
            "depth": depth,
            "n_block": n_block,
            "codebook_size": codebook_size,
            "prune": args.prune,
            "prune_percent": prune_percent,
            "oct_merge": args.oct_merge,
            "vq_steps": args.steps,
            "sh_degree": args.sh_degree,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "interval": args.interval,
        }
        with open(os.path.join(args.output_folder, "mesongs_config.json"), "w") as f:
            json.dump(config_out, f, indent=4)

        print("\n" + "=" * 70)
        print("Benchmark Summary (MesonGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  (avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  (avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  Avg point flow:            {total_orig_points / n:.0f} → {total_octree_points / n:.0f} octree → {total_decoded_points / n:.0f} decoded")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
