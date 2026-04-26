#!/usr/bin/env python3
"""
Evaluate LiVoGS decompression quality against GT QUEEN-trained models.

For each frame, loads the GT PLY and the LiVoGS-decompressed PLY, renders
test cameras using QUEEN's gaussian renderer, and computes PSNR / SSIM.

Output: per-frame CSV, summary JSON, and optionally saved rendered images.
"""

import os
import csv
import numpy as np
import json
import torch
import sys
import time
import yaml
import socket
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from torchvision.utils import save_image

# Add project root to path
_QUEEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _QUEEN_ROOT not in sys.path:
    sys.path.insert(0, _QUEEN_ROOT)

from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
# from utils.image_utils import psnr as psnr_fn
# from utils.loss_utils import ssim as ssim_fn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.loader_utils import MultiViewVideoDataset, SequentialMultiviewSampler
from utils.camera_utils import updateCam
from utils.compress_utils import search_for_max_iteration
from arguments import (ModelParams, PipelineParams, OptimizationParams, QuantizeParams,
                       OptimizationParamsInitial, OptimizationParamsRest)


def find_ply_path(frame_dir):
    """Find PLY file: prefer canonical point_cloud.ply, fallback to per-iteration."""
    canonical = os.path.join(frame_dir, "point_cloud.ply")
    if os.path.exists(canonical):
        return canonical

    iter_dir = os.path.join(frame_dir, "point_cloud")
    if not os.path.isdir(iter_dir):
        raise FileNotFoundError(f"Checkpoint folder not found: {iter_dir}")

    max_iter = search_for_max_iteration(iter_dir)
    if max_iter <= 0:
        raise FileNotFoundError(
            f"No valid iteration_<N> checkpoints found in: {iter_dir}"
        )

    per_iter = os.path.join(iter_dir, f"iteration_{max_iter}", "point_cloud.ply")
    if os.path.exists(per_iter):
        return per_iter

    raise FileNotFoundError(
        f"Missing checkpoint PLY for max iteration {max_iter}: {per_iter}"
    )


def render_and_evaluate(gaussians, cameras, background, pipeline, psnr_metric, ssim_metric,
                        save=False):
    """Render all cameras and compute metrics against GT images."""
    metrics = {'psnr': [], 'ssim': []}
    rendered_images = []

    with torch.no_grad():
        for cam in cameras:
            render_pkg = render(cam, gaussians, pipeline, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(cam.original_image[:3], 0.0, 1.0)

            psnr_val = psnr_metric(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
            ssim_val = ssim_metric(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

            metrics['psnr'].append(psnr_val)
            metrics['ssim'].append(ssim_val)

            if save:
                rendered_images.append(image.cpu())

            del render_pkg, image, gt_image
            torch.cuda.empty_cache()

    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if v}
    return avg_metrics, rendered_images


def save_images(images, save_dir, frame, prefix):
    """Save a list of [C,H,W] tensors to disk as PNGs."""
    os.makedirs(save_dir, exist_ok=True)
    for idx, img in enumerate(images):
        save_image(img, os.path.join(save_dir, f"frame{frame}_view{idx}_{prefix}.png"))


# ---------------------------------------------------------------------------
# QUEEN-specific: camera / model setup
# (VideoGS equivalent: load_test_cameras + GaussianModel(sh_degree))
# ---------------------------------------------------------------------------

def setup_queen_evaluation(dataset, opt, qp, args):
    """Set up QUEEN datasets, cameras, Scene, and GaussianModel instances."""

    train_image_dataset = MultiViewVideoDataset(
        dataset.source_path, split='train',
        test_indices=dataset.test_indices,
        max_frames=dataset.max_frames,
        start_idx=dataset.start_idx,
        img_format=dataset.img_fmt,
    )
    test_image_dataset = MultiViewVideoDataset(
        dataset.source_path, split='test',
        test_indices=dataset.test_indices,
        max_frames=dataset.max_frames,
        start_idx=dataset.start_idx,
        img_format=dataset.img_fmt,
    )

    if test_image_dataset.n_cams == 0:
        raise RuntimeError("No test cameras found. Cannot evaluate.")

    train_sampler = SequentialMultiviewSampler(train_image_dataset)
    train_loader = iter(torch.utils.data.DataLoader(
        train_image_dataset, batch_size=train_image_dataset.n_cams,
        sampler=train_sampler, num_workers=4,
    ))
    test_sampler = SequentialMultiviewSampler(test_image_dataset)
    test_loader = iter(torch.utils.data.DataLoader(
        test_image_dataset, batch_size=test_image_dataset.n_cams,
        sampler=test_sampler, num_workers=4,
    ))

    train_data = next(train_loader)
    train_images, train_paths = train_data
    train_image_data = {'image': train_images.cuda(), 'path': train_paths, 'frame_idx': 0}

    test_data = next(test_loader)
    test_images, test_paths = test_data
    test_image_data = {'image': test_images.cuda(), 'path': test_paths, 'frame_idx': 0}

    qp.seed = dataset.seed

    gt_gaussians = GaussianModel(dataset.sh_degree, qp, dataset)
    scene = Scene(dataset, gt_gaussians,
                  train_image_data=train_image_data,
                  test_image_data=test_image_data)
    gt_gaussians.training_setup(opt)

    decomp_gaussians = GaussianModel(dataset.sh_degree, qp, dataset)
    Scene(dataset, decomp_gaussians,
          train_image_data=train_image_data,
          test_image_data=test_image_data)
    decomp_gaussians.training_setup(opt)

    n_test_cams = test_image_dataset.n_cams
    n_total_cams = train_image_dataset.n_cams + n_test_cams
    n_frames = dataset.max_frames

    print(f"Total cameras: {n_total_cams}, Test cameras: {n_test_cams}")

    return (gt_gaussians, decomp_gaussians, scene, test_loader,
            n_test_cams, n_frames, dataset.start_idx)


def update_queen_cameras(scene, test_loader, current_frame_idx, target_frame_idx, args):
    if target_frame_idx < current_frame_idx:
        raise ValueError(
            f"target_frame_idx ({target_frame_idx}) must be >= current_frame_idx ({current_frame_idx})"
        )
    if target_frame_idx == current_frame_idx:
        return list(scene.getTestCameras()), current_frame_idx

    for frame_idx in range(current_frame_idx + 1, target_frame_idx + 1):
        try:
            test_data = next(test_loader)
        except StopIteration as exc:
            raise RuntimeError(
                f"Ran out of test frames while advancing to frame {target_frame_idx}"
            ) from exc

        if frame_idx != target_frame_idx:
            continue

        test_images, test_paths = test_data[0].cuda(), test_data[1]
        for idx, cam in enumerate(scene.getTestCameras()):
            updateCam(args, test_images[idx], test_paths[idx], frame_idx, cam, 1.0)

    return list(scene.getTestCameras()), target_frame_idx


# ---------------------------------------------------------------------------
# Main evaluation (same loop structure as evaluate_decompress_videogs.py)
# ---------------------------------------------------------------------------

def evaluate_livogs_quality(dataset, opt, pipeline, qp, args):
    """Evaluate LiVoGS decompression quality frame-by-frame."""

    if args.save_renders and not args.output_render_path:
        print("Error: --output_render_path is required when --save_renders is set")
        sys.exit(1)

    # --- Setup (QUEEN-specific) ---
    (gt_gaussians, decomp_gaussians, scene, test_loader,
     n_test_cams, n_frames, start_idx) = setup_queen_evaluation(dataset, opt, qp, args)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # psnr_metric = psnr_fn
    # ssim_metric = ssim_fn

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()

    gt_metrics_all = {'psnr': [], 'ssim': []}
    decomp_metrics_all = {'psnr': [], 'ssim': []}
    results = []

    frame_start = max(1, args.frame_start)
    frame_end = args.frame_end if args.frame_end > 0 else n_frames
    frame_end = min(frame_end, n_frames)
    interval = max(1, args.interval)
    frames_to_eval = sorted(set(range(frame_start, frame_end + 1, interval)))

    print(f"\nEvaluating LiVoGS decompression quality")
    print(f"  Model path:            {args.model_path}")
    print(f"  Decompressed PLY path: {args.decompressed_ply_path}")
    print(f"  Frames: {n_frames} total, evaluating {len(frames_to_eval)} "
          f"(range [{frame_start}, {frame_end}], interval {interval})")
    print(f"  Test cameras: {n_test_cams}\n")

    if not frames_to_eval:
        print("No frames were selected for evaluation.")
        return results

    # --- Per-frame evaluation loop ---
    current_frame_idx = 1
    for frame_idx in tqdm(frames_to_eval, desc="Evaluating Frames", total=len(frames_to_eval)):
        t0 = time.time()
        cameras, current_frame_idx = update_queen_cameras(
            scene,
            test_loader,
            current_frame_idx,
            frame_idx,
            args,
        )
        t1 = time.time()

        frame = str(start_idx + frame_idx).zfill(4)

        # GT PLY
        gt_frame_dir = os.path.join(args.model_path, 'frames', frame)
        gt_ply_file = find_ply_path(gt_frame_dir)

        # Decompressed PLY
        decomp_frame_dir = os.path.join(args.decompressed_ply_path, 'frames', frame)
        decomp_ply_file = find_ply_path(decomp_frame_dir)

        # Evaluate GT model
        gt_gaussians.frame_idx = frame_idx
        gt_gaussians.load_ply(gt_ply_file)
        gt_metrics, gt_renders = render_and_evaluate(
            gt_gaussians, cameras, background, pipeline, psnr_metric, ssim_metric,
            save=args.save_renders
        )
        torch.cuda.empty_cache()
        t2 = time.time()

        # Evaluate Decompressed model
        decomp_gaussians.frame_idx = frame_idx
        decomp_gaussians.load_ply(decomp_ply_file)
        decomp_metrics, decomp_renders = render_and_evaluate(
            decomp_gaussians, cameras, background, pipeline, psnr_metric, ssim_metric,
            save=args.save_renders
        )
        torch.cuda.empty_cache()
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
            "gt_size_bytes": os.path.getsize(gt_ply_file),
            "decomp_size_bytes": os.path.getsize(decomp_ply_file),
        }
        results.append(frame_result)

        print(f"  Frame {frame}: GT PSNR={gt_metrics['psnr']:.2f}, "
              f"Decomp PSNR={decomp_metrics['psnr']:.2f}, "
              f"Drop={frame_result['psnr_drop']:.2f} | "
              f"GT SSIM={gt_metrics['ssim']:.4f}, "
              f"Decomp SSIM={decomp_metrics['ssim']:.4f}")

        if args.save_renders:
            gt_images = []
            for cam in cameras:
                if cam.original_image is None:
                    raise ValueError("Camera original_image is None during evaluation.")
                original_image = cam.original_image
                gt_images.append(torch.clamp(original_image[:3], 0.0, 1.0))
            save_images(gt_images, os.path.join(args.output_render_path, "gt_images"),
                        frame, "gt_image")
            save_images(gt_renders, os.path.join(args.output_render_path, "gt_model_renders"),
                        frame, "gt_render")
            save_images(decomp_renders, os.path.join(args.output_render_path, "decomp_model_renders"),
                        frame, "decomp_render")

        del gt_renders, decomp_renders
        torch.cuda.empty_cache()

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
                        "model_path": args.model_path,
                        "decompressed_ply_path": args.decompressed_ply_path,
                        "num_test_cameras": n_test_cams,
                        "n_frames": len(results),
                        "frame_start": frame_start,
                        "frame_end": frame_end,
                        "interval": interval,
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
                                 "decomp_psnr", "decomp_ssim",
                                 "psnr_drop", "ssim_drop"])
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
    print('Running on', socket.gethostname())

    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    parser = ArgumentParser(
        description="Evaluate LiVoGS decompression quality against GT QUEEN models"
    )

    lp = ModelParams(parser, config['model_params'])
    op_i = OptimizationParamsInitial(parser, config['opt_params_initial'])
    op_r = OptimizationParamsRest(parser, config['opt_params_rest'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp_cls = QuantizeParams(parser, config['quantize_params'])

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--decompressed_ply_path", type=str, required=True,
                        help="Folder containing decompressed PLY files (frames/<frame_str>/point_cloud.ply)")
    parser.add_argument("--save_renders", action="store_true",
                        help="Save GT images, GT model renders, and decompressed model renders")
    parser.add_argument("--output_render_path", type=str, default=None,
                        help="Folder to save rendered images and evaluation results")
    parser.add_argument("--frame_start", type=int, default=1,
                        help="First frame index to evaluate (1-based, default: 1)")
    parser.add_argument("--frame_end", type=int, default=0,
                        help="Last frame index to evaluate (0 = all frames, default: 0)")
    parser.add_argument("--interval", type=int, default=1,
                        help="Evaluate every N-th frame (default: 1)")
    args = parser.parse_args(sys.argv[1:])

    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print(f"Model path: {args.model_path}")
    print(f"Decompressed PLY path: {args.decompressed_ply_path}")
    safe_state(args.quiet)

    evaluate_livogs_quality(
        lp.extract(args), op, pp.extract(args), qp_cls.extract(args), args
    )
