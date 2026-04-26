#!/usr/bin/env python3
"""
Evaluate decompression quality against GT VideoGS-trained models.

For each frame, loads the GT PLY and the VideoGS-decompressed PLY, renders
test cameras using the gaussian renderer, and computes PSNR / SSIM.

Output: per-frame CSV, summary JSON, and optionally saved rendered images.
"""

import os
import csv
import numpy as np
import cv2
import json
import argparse
import torch
import sys
import time
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import PipelineParams
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov
from utils.general_utils import PILtoTorch
from scene.colmap_loader import rotmat2qvec, qvec2rotmat


def load_test_cameras(dataset_path, first_frame, resolution, llffhold=8):
    """Load only test cameras (every llffhold-th) from the first frame's transforms.json.
    Parses transforms.json directly -- no DynamicScene overhead, no loading of train cameras."""

    frame_path = os.path.join(dataset_path, str(first_frame))
    with open(os.path.join(frame_path, "transforms.json")) as f:
        contents = json.load(f)
    frames = contents["frames"]

    all_indices = list(range(len(frames)))
    test_indices = [idx for idx in all_indices if idx % llffhold == 0]

    print(f"Total cameras: {len(frames)}, Test cameras (llffhold={llffhold}): {len(test_indices)}")
    print(f"Test camera indices: {test_indices}")

    cameras = []
    cam_file_paths = []
    cam_resolutions = []

    flip_mat = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    for uid, idx in enumerate(test_indices):
        entry = frames[idx]
        cam_name = entry["file_path"]
        cam_file_paths.append(cam_name)

        # Extrinsics (same logic as readCamerasFromTransforms in dataset_readers.py)
        matrix = np.linalg.inv(np.matmul(np.array(entry["transform_matrix"]), flip_mat))
        R = np.transpose(qvec2rotmat(-rotmat2qvec(matrix[:3, :3])))
        T = matrix[:3, 3]

        # Load image for this frame
        image_path = os.path.join(frame_path, cam_name)
        image_pil = Image.open(image_path).convert("RGB")

        orig_w, orig_h = image_pil.size

        # Intrinsics
        fx = entry["fl_x"]
        fy = entry["fl_y"]
        FovY = focal2fov(fy, orig_h)
        FovX = focal2fov(fx, orig_w)

        # Resolution (same logic as loadCam in camera_utils.py, resolution_scale=1.0)
        if resolution in [1, 2, 4, 8]:
            target_res = (round(orig_w / resolution), round(orig_h / resolution))
        else:
            target_res = (orig_w, orig_h)

        cam_resolutions.append(target_res)

        resized_image = PILtoTorch(image_pil, target_res)
        gt_image = resized_image[:3, ...]

        cam = Camera(
            colmap_id=idx, R=R, T=T,
            FoVx=FovX, FoVy=FovY,
            image=gt_image, gt_alpha_mask=None,
            image_name=Path(cam_name).stem, uid=uid,
            data_device="cpu"
        )
        cameras.append(cam)

    return cameras, cam_file_paths, cam_resolutions


def _load_single_image(args):
    """Load and resize a single image. Runs in a thread."""
    image_path, target_res = args
    image_pil = Image.open(image_path).convert("RGB")
    resized_image = PILtoTorch(image_pil, target_res)
    return resized_image[:3, ...]


def update_camera_images(cameras, dataset_path, frame, cam_file_paths, cam_resolutions):
    """Swap GT images in existing Camera objects for a new frame. Loads test images in parallel."""
    frame_path = os.path.join(dataset_path, str(frame))
    load_args = [
        (os.path.join(frame_path, cam_file_paths[idx]), cam_resolutions[idx])
        for idx in range(len(cameras))
    ]
    with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        images = list(executor.map(_load_single_image, load_args))
    for idx, cam in enumerate(cameras):
        cam.original_image = images[idx]


def render_and_evaluate(gaussians, cameras, background, pipeline, psnr_metric, ssim_metric, save=False):
    """Render all cameras and compute metrics against GT images."""
    metrics_psnr = []
    metrics_ssim = []
    rendered_images = []

    with torch.no_grad():
        gt_images = [
            torch.clamp(cam.original_image.cuda(non_blocking=True), 0.0, 1.0)
            for cam in cameras
        ]

        for cam, gt_image in zip(cameras, gt_images):
            render_pkg = render(cam, gaussians, pipeline, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)

            psnr_val = psnr_metric(image.unsqueeze(0), gt_image.unsqueeze(0))
            ssim_val = ssim_metric(image.unsqueeze(0), gt_image.unsqueeze(0))

            metrics_psnr.append(psnr_val)
            metrics_ssim.append(ssim_val)

            if save:
                rendered_images.append(image.detach().cpu())

            del render_pkg, image

    avg_metrics = {
        "psnr": torch.stack(metrics_psnr).mean().item(),
        "ssim": torch.stack(metrics_ssim).mean().item(),
    }
    return avg_metrics, rendered_images


def save_images(images, save_dir, frame, prefix):
    """Save a list of [C,H,W] tensors (CPU or CUDA) to disk as PNGs."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, img in enumerate(images):
        img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, f"frame{frame}_view{idx}_{prefix}.png"), img_bgr)


def evaluate_decompression_quality(args):
    """Evaluate decompression quality frame-by-frame."""

    if args.save_renders and not args.output_render_path:
        print("Error: --output_render_path is required when --save_renders is set")
        sys.exit(1)

    # --- Setup ---
    pipeline = PipelineParams(argparse.ArgumentParser())

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    gt_metrics_all = {'psnr': [], 'ssim': []}
    decomp_metrics_all = {'psnr': [], 'ssim': []}
    results = []

    # --- Resolve frame list ---
    if getattr(args, 'frame_ids', None) is not None:
        frame_list = sorted(int(x.strip()) for x in args.frame_ids.split(","))
    else:
        frame_list = list(range(args.frame_start, args.frame_end, args.interval))

    # --- Load test cameras ONCE from the first frame ---
    first_frame = frame_list[0]
    print("Loading test cameras from first frame...")
    cameras, cam_file_paths, cam_resolutions = load_test_cameras(
        args.dataset_path, first_frame, args.resolution, args.llffhold
    )

    n_test_cams = len(cameras)

    print(f"\nEvaluating decompression quality")
    print(f"  GT PLY path:           {args.gt_ply_path}")
    print(f"  Decompressed PLY path: {args.decompressed_ply_path}")
    if getattr(args, 'frame_ids', None) is not None:
        print(f"  Frames: {frame_list}")
    else:
        print(f"  Frames: [{args.frame_start}, {args.frame_end}) interval {args.interval}")
    print(f"  Test cameras: {n_test_cams}\n")

    # --- Per-frame evaluation loop ---
    for frame in tqdm(frame_list, desc="Evaluating Frames"):

        # GT PLY
        ckpt_path = os.path.join(args.gt_ply_path, str(frame), "point_cloud")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"GT checkpoint folder not found for frame {frame}: {ckpt_path}"
            )
        max_iter = searchForMaxIteration(ckpt_path)
        gt_ply_file = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")

        # Decompressed PLY
        decomp_ply_file = os.path.join(args.decompressed_ply_path, str(frame), "point_cloud", "point_cloud.ply")
        if not os.path.exists(decomp_ply_file):
            print(f"Warning: Decompressed PLY not found: {decomp_ply_file}, skipping frame {frame}")
            continue

        # Update GT images for this frame
        t0 = time.time()
        update_camera_images(cameras, args.dataset_path, frame, cam_file_paths, cam_resolutions)
        t1 = time.time()

        # Evaluate GT model (always load with degree 3, since GT PLY has full SH)
        gt_gaussians = GaussianModel(3)
        gt_gaussians.load_ply(gt_ply_file)
        gt_metrics, gt_renders = render_and_evaluate(
            gt_gaussians, cameras, background, pipeline, psnr_metric, ssim_metric,
            save=args.save_renders
        )
        del gt_gaussians
        t2 = time.time()

        # Evaluate Decompressed model (use target sh_degree matching compressed PLY)
        decomp_gaussians = GaussianModel(args.sh_degree)
        decomp_gaussians.load_ply(decomp_ply_file)
        decomp_metrics, decomp_renders = render_and_evaluate(
            decomp_gaussians, cameras, background, pipeline, psnr_metric, ssim_metric,
            save=args.save_renders
        )
        del decomp_gaussians
        t3 = time.time()

        print(f"  [Timing] update_images: {(t1-t0)*1000:.0f}ms, "
              f"eval_gt: {(t2-t1)*1000:.0f}ms, eval_decomp: {(t3-t2)*1000:.0f}ms")

        for k in gt_metrics:
            gt_metrics_all[k].append(gt_metrics[k])
            decomp_metrics_all[k].append(decomp_metrics[k])

        frame_result = {
            "frame": frame,
            "gt_psnr": gt_metrics['psnr'],
            "gt_ssim": gt_metrics['ssim'],
            "decomp_psnr": decomp_metrics['psnr'],
            "decomp_ssim": decomp_metrics['ssim'],
            "psnr_drop": gt_metrics['psnr'] - decomp_metrics['psnr'],
            "ssim_drop": gt_metrics['ssim'] - decomp_metrics['ssim'],
        }
        results.append(frame_result)

        print(f"  Frame {frame}: GT PSNR={gt_metrics['psnr']:.2f}, "
              f"Decomp PSNR={decomp_metrics['psnr']:.2f}, "
              f"Drop={frame_result['psnr_drop']:.2f} | "
              f"GT SSIM={gt_metrics['ssim']:.4f}, "
              f"Decomp SSIM={decomp_metrics['ssim']:.4f}")

        if args.save_renders:
            gt_images = [cam.original_image for cam in cameras]
            save_images(gt_images, os.path.join(args.output_render_path, "gt_images"),
                        frame, "gt_image")
            save_images(gt_renders, os.path.join(args.output_render_path, "gt_model_renders"),
                        frame, "gt_render")
            save_images(decomp_renders, os.path.join(args.output_render_path, "decomp_model_renders"),
                        frame, "decomp_render")

        del gt_renders, decomp_renders

    # --- Summary ---
    if results:
        print("\n" + "=" * 70)
        print(f"Evaluation Summary ({len(results)} frames, {n_test_cams} test cameras)")
        print("=" * 70)
        print(f"  GT Model       -> PSNR: {np.mean(gt_metrics_all['psnr']):.4f}, "
              f"SSIM: {np.mean(gt_metrics_all['ssim']):.4f}")
        print(f"  Decomp Model   -> PSNR: {np.mean(decomp_metrics_all['psnr']):.4f}, "
              f"SSIM: {np.mean(decomp_metrics_all['ssim']):.4f}")
        print(f"  Quality Drop   -> PSNR: "
              f"{np.mean(gt_metrics_all['psnr']) - np.mean(decomp_metrics_all['psnr']):.4f}, "
              f"SSIM: {np.mean(gt_metrics_all['ssim']) - np.mean(decomp_metrics_all['ssim']):.6f}")
        print("=" * 70)

        if args.output_render_path:
            os.makedirs(args.output_render_path, exist_ok=True)

            with open(os.path.join(args.output_render_path, "evaluation_results.json"), "w") as f:
                json.dump({
                    "config": {
                        "gt_ply_path": args.gt_ply_path,
                        "decompressed_ply_path": args.decompressed_ply_path,
                        "llffhold": args.llffhold,
                        "num_test_cameras": n_test_cams,
                        "resolution": args.resolution,
                        "frame_list": frame_list,
                    },
                    "summary": {
                        "gt_psnr": np.mean(gt_metrics_all['psnr']),
                        "gt_ssim": np.mean(gt_metrics_all['ssim']),
                        "decomp_psnr": np.mean(decomp_metrics_all['psnr']),
                        "decomp_ssim": np.mean(decomp_metrics_all['ssim']),
                        "psnr_drop": np.mean(gt_metrics_all['psnr']) - np.mean(decomp_metrics_all['psnr']),
                        "ssim_drop": np.mean(gt_metrics_all['ssim']) - np.mean(decomp_metrics_all['ssim']),
                    },
                    "per_frame": results,
                }, f, indent=4)

            csv_path = os.path.join(args.output_render_path, "evaluation_results.csv")
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["frame_id", "gt_psnr", "gt_ssim",
                                 "decomp_psnr", "decomp_ssim", "psnr_drop", "ssim_drop"])
                for r in results:
                    writer.writerow([
                        r["frame"],
                        f"{r['gt_psnr']:.4f}",
                        f"{r['gt_ssim']:.6f}",
                        f"{r['decomp_psnr']:.4f}",
                        f"{r['decomp_ssim']:.6f}",
                        f"{r['psnr_drop']:.4f}",
                        f"{r['ssim_drop']:.6f}",
                    ])
                writer.writerow([
                    "avg",
                    f"{np.mean(gt_metrics_all['psnr']):.4f}",
                    f"{np.mean(gt_metrics_all['ssim']):.6f}",
                    f"{np.mean(decomp_metrics_all['psnr']):.4f}",
                    f"{np.mean(decomp_metrics_all['ssim']):.6f}",
                    f"{np.mean(gt_metrics_all['psnr']) - np.mean(decomp_metrics_all['psnr']):.4f}",
                    f"{np.mean(gt_metrics_all['ssim']) - np.mean(decomp_metrics_all['ssim']):.6f}",
                ])
            print(f"  CSV saved to: {csv_path}")
    else:
        print("No frames were evaluated.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate decompression quality against GT models"
    )
    parser.add_argument("--gt_ply_path", type=str, required=True,
                        help="Path to training checkpoint dir (e.g. .../checkpoint)")
    parser.add_argument("--decompressed_ply_path", type=str, required=True,
                        help="Folder containing decompressed PLY files (<frame_id>/point_cloud/point_cloud.ply)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to processed dataset (containing frame folders with transforms.json)")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=2,
                        help="Resolution scale used during training (1, 2, 4, 8)")
    parser.add_argument("--llffhold", type=int, default=8,
                        help="Every llffhold-th camera is used for test evaluation (default: 8)")
    parser.add_argument("--white_background", action="store_true", default=True,
                        help="Use white background (default True for HiFi4G)")
    parser.add_argument("--no_white_background", action="store_true",
                        help="Use black background instead")
    parser.add_argument("--save_renders", action="store_true",
                        help="Save GT images, GT model renders, and decompressed model renders")
    parser.add_argument("--output_render_path", type=str, default=None,
                        help="Folder to save rendered images (required if --save_renders)")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=200)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--frame_ids", type=str, default=None,
                        help="Comma-separated frame IDs (overrides --frame_start/--frame_end/--interval)")
    args = parser.parse_args()

    if args.no_white_background:
        args.white_background = False

    evaluate_decompression_quality(args)
