# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Evaluate rendering quality of PLY (uncompressed) and PKL (compressed) models
against ground truth test camera views.

Loads saved PLY/PKL models from disk, renders test camera views using QUEEN's
rendering pipeline, and computes PSNR/SSIM/LPIPS against ground truth images.

Sanity check: PLY mode results should match metrics_video.py output (the
--quantize_renders flag applies uint8 rounding to match the PNG save/load
quantization used in metrics_video.py).

Usage:
    # Evaluate PLY model quality vs GT (should match metrics_video.py output)
    python eval_ply_vs_pkl.py --config configs/dynerf.yaml \
        -s data/multipleview/cook_spinach \
        -m ./output/cook_spinach_trained_compressed \
        --mode ply

    # Evaluate PKL (compressed) model quality vs GT
    python eval_ply_vs_pkl.py --config configs/dynerf.yaml \
        -s data/multipleview/cook_spinach \
        -m ./output/cook_spinach_trained_compressed \
        --mode pkl

    # Evaluate both and report comparison
    python eval_ply_vs_pkl.py --config configs/dynerf.yaml \
        -s data/multipleview/cook_spinach \
        -m ./output/cook_spinach_trained_compressed \
        --mode both
"""

import torch
import os
import sys
import yaml
import json
import socket
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict

from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from lpipsPyTorch import LPIPS
from utils.loader_utils import MultiViewVideoDataset, SequentialMultiviewSampler
from utils.camera_utils import updateCam
from utils.compress_utils import search_for_max_iteration
from arguments import (ModelParams, PipelineParams, OptimizationParams, QuantizeParams,
                       OptimizationParamsInitial, OptimizationParamsRest)


def quantize_to_uint8(image):
    """Simulate PNG save/load quantization (float32 -> uint8 -> float32)."""
    return torch.clamp(torch.round(image * 255.0) / 255.0, 0.0, 1.0)


def find_ply_path(frame_dir):
    """Find PLY file: prefer canonical point_cloud.ply, fallback to per-iteration."""
    canonical = os.path.join(frame_dir, "point_cloud.ply")
    if os.path.exists(canonical):
        return canonical
    iter_dir = os.path.join(frame_dir, "point_cloud")
    if os.path.exists(iter_dir):
        max_iter = search_for_max_iteration(iter_dir)
        per_iter = os.path.join(iter_dir, f"iteration_{max_iter}", "point_cloud.ply")
        if os.path.exists(per_iter):
            return per_iter
    return None


def evaluate_model(dataset, opt, pipeline, qp, args, mode, quantize_renders):
    """
    Load saved models (PLY or PKL) from disk, render test camera views,
    and compute quality metrics vs GT.

    Args:
        mode: 'ply' | 'pkl' | 'both'
        quantize_renders: if True, apply uint8 quantization to rendered images
                          to match metrics_video.py behavior
    """
    assert mode in ('ply', 'pkl', 'both'), f"Invalid mode: {mode}"

    with torch.no_grad():
        n_frames = dataset.max_frames

        # --- Data loaders (same as render.py / train.py) ---
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

        train_sampler = SequentialMultiviewSampler(train_image_dataset)
        train_loader = iter(torch.utils.data.DataLoader(
            train_image_dataset, batch_size=train_image_dataset.n_cams,
            sampler=train_sampler, num_workers=4,
        ))

        has_test = test_image_dataset.n_cams > 0
        if not has_test:
            print("ERROR: No test cameras found. Cannot evaluate.")
            return None

        test_sampler = SequentialMultiviewSampler(test_image_dataset)
        test_loader = iter(torch.utils.data.DataLoader(
            test_image_dataset, batch_size=test_image_dataset.n_cams,
            sampler=test_sampler, num_workers=4,
        ))

        # Load first frame data (needed for Scene initialization)
        train_data = next(train_loader)
        train_images, train_paths = train_data
        train_image_data = {'image': train_images.cuda(), 'path': train_paths, 'frame_idx': 0}

        test_data = next(test_loader)
        test_images, test_paths = test_data
        test_image_data = {'image': test_images.cuda(), 'path': test_paths, 'frame_idx': 0}

        # --- Create Gaussian model and Scene ---
        qp.seed = dataset.seed
        gaussians = GaussianModel(dataset.sh_degree, qp, dataset)
        scene = Scene(dataset, gaussians,
                      train_image_data=train_image_data,
                      test_image_data=test_image_data)
        gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        lpips_fn = LPIPS(net_type='vgg', version='0.1').to("cuda")

        # For 'both' mode, create a second GaussianModel for PKL
        gaussians_pkl = None
        if mode == 'both':
            gaussians_pkl = GaussianModel(dataset.sh_degree, qp, dataset)
            Scene(dataset, gaussians_pkl,
                  train_image_data=train_image_data,
                  test_image_data=test_image_data)
            gaussians_pkl.training_setup(opt)

        run_modes = [mode] if mode != 'both' else ['ply', 'pkl']
        results = {m: {'psnr': [], 'ssim': [], 'lpips': [],
                        'size_bytes': []} for m in run_modes}

        print(f"\nEvaluating mode(s): {run_modes}")
        print(f"Model path: {args.model_path}")
        print(f"Frames: {n_frames}, Test cameras: {test_image_dataset.n_cams}")
        print(f"Quantize renders (match metrics_video.py): {quantize_renders}\n")

        for frame_idx in tqdm(range(1, n_frames + 1), desc="Evaluating frames"):
            frame_str = str(dataset.start_idx + frame_idx).zfill(4)
            frame_dir = os.path.join(args.model_path, 'frames', frame_str)

            # Load next frame GT test images and update test cameras only (frame 2+)
            if frame_idx > 1:
                test_data = next(test_loader)
                test_images, test_paths = test_data[0].cuda(), test_data[1]
                for idx, cam in enumerate(scene.getTestCameras()):
                    updateCam(args, test_images[idx], test_paths[idx],
                              frame_idx, cam, 1.0)

            test_cameras = scene.getTestCameras()

            # --- PLY evaluation ---
            if 'ply' in run_modes:
                gaussians.frame_idx = frame_idx
                ply_path = find_ply_path(frame_dir)
                if ply_path is None:
                    print(f"WARNING: PLY not found for frame {frame_idx} at {frame_dir}")
                    continue
                gaussians.load_ply(ply_path)
                results['ply']['size_bytes'].append(os.path.getsize(ply_path))

                for cam in test_cameras:
                    rendered = render(cam, gaussians, pipeline, background)
                    image = torch.clamp(rendered["render"], 0.0, 1.0)
                    if quantize_renders:
                        image = quantize_to_uint8(image)
                    gt_image = torch.clamp(cam.original_image[:3], 0.0, 1.0)

                    results['ply']['psnr'].append(psnr(image, gt_image).item())
                    results['ply']['ssim'].append(ssim(image, gt_image).item())
                    results['ply']['lpips'].append(
                        lpips_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).item())

            # --- PKL evaluation ---
            if 'pkl' in run_modes:
                gs = gaussians_pkl if mode == 'both' else gaussians
                gs.frame_idx = frame_idx

                if frame_idx == 1:
                    ply_path = find_ply_path(frame_dir)
                    if ply_path is None:
                        print(f"WARNING: PLY not found for frame 1 at {frame_dir}")
                        continue
                    gs.load_ply(ply_path)
                    results['pkl']['size_bytes'].append(os.path.getsize(ply_path))
                else:
                    pkl_path = os.path.join(frame_dir, "compressed",
                                            "point_cloud.pkl")
                    if not os.path.exists(pkl_path):
                        print(f"WARNING: PKL not found at {pkl_path}, skipping")
                        continue
                    gs.load_compressed_pkl(pkl_path)
                    results['pkl']['size_bytes'].append(os.path.getsize(pkl_path))

                for cam in test_cameras:
                    rendered = render(cam, gs, pipeline, background)
                    image = torch.clamp(rendered["render"], 0.0, 1.0)
                    if quantize_renders:
                        image = quantize_to_uint8(image)
                    gt_image = torch.clamp(cam.original_image[:3], 0.0, 1.0)

                    results['pkl']['psnr'].append(psnr(image, gt_image).item())
                    results['pkl']['ssim'].append(ssim(image, gt_image).item())
                    results['pkl']['lpips'].append(
                        lpips_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).item())

                # Sequential state: update prev_atts for next frame's PKL decoding
                if frame_idx != n_frames:
                    for att_name in gs.get_atts:
                        gs.prev_atts[att_name] = gs.get_decoded_atts[att_name].clone()
                        gs.prev_latents[att_name] = gs.get_atts[att_name].clone()
                        gs.prev_atts[att_name].requires_grad_(False)
                        gs.prev_latents[att_name].requires_grad_(False)

        # --- Aggregate and report ---
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        output = {}
        for m in run_modes:
            if not results[m]['psnr']:
                print(f"\n  {m.upper()}: No frames evaluated")
                continue

            avg_psnr = np.mean(results[m]['psnr'])
            avg_ssim = np.mean(results[m]['ssim'])
            avg_lpips = np.mean(results[m]['lpips'])
            total_size_bytes = sum(results[m]['size_bytes'])
            total_size_mb = total_size_bytes / (1024 * 1024)
            avg_size_mb = total_size_mb / len(results[m]['size_bytes'])

            label = "PLY (uncompressed)" if m == 'ply' else "PKL (compressed)"
            print(f"\n  {label}:")
            print(f"    PSNR:  {avg_psnr:.4f} dB")
            print(f"    SSIM:  {avg_ssim:.6f}")
            print(f"    LPIPS: {avg_lpips:.6f}")

            output[m] = {
                'PSNR': round(float(avg_psnr), 4),
                'SSIM': round(float(avg_ssim), 6),
                'LPIPS': round(float(avg_lpips), 6),
                'total_size_MB': round(total_size_mb, 3),
                'avg_frame_size_MB': round(avg_size_mb, 3),
                'per_frame_psnr': results[m]['psnr'],
                'per_frame_ssim': results[m]['ssim'],
                'per_frame_lpips': results[m]['lpips'],
                'per_frame_size_bytes': results[m]['size_bytes'],
            }

        # Print size summary
        if 'ply' in output:
            print(f"\n  Avg uncompressed size per frame (MB):   {output['ply']['avg_frame_size_MB']:.3f}")
            print(f"  Total uncompressed size (MB):           {output['ply']['total_size_MB']:.3f}")
        if 'pkl' in output:
            print(f"  Avg compressed size per frame (MB):     {output['pkl']['avg_frame_size_MB']:.3f}")
            print(f"  Total compressed size (MB):             {output['pkl']['total_size_MB']:.3f}")
        if 'ply' in output and 'pkl' in output:
            compression_ratio = (output['ply']['total_size_MB']
                                 / output['pkl']['total_size_MB']
                                 if output['pkl']['total_size_MB'] > 0 else float('inf'))
            print(f"  Compression ratio:                      {compression_ratio:.3f}x")

        if mode == 'both' and 'ply' in output and 'pkl' in output:
            diff_psnr = output['pkl']['PSNR'] - output['ply']['PSNR']
            diff_ssim = output['pkl']['SSIM'] - output['ply']['SSIM']
            diff_lpips = output['pkl']['LPIPS'] - output['ply']['LPIPS']
            print(f"\n  Quality difference (PKL - PLY):")
            print(f"    dPSNR:  {diff_psnr:+.4f} dB")
            print(f"    dSSIM:  {diff_ssim:+.6f}")
            print(f"    dLPIPS: {diff_lpips:+.6f}")

        # Sanity check hint
        existing_results = os.path.join(args.model_path, "results.json")
        if 'ply' in output and os.path.exists(existing_results):
            with open(existing_results, 'r') as f:
                mv_results = json.load(f)
            print(f"\n  Sanity check vs metrics_video.py ({existing_results}):")
            print(f"    metrics_video.py PSNR:  {mv_results.get('PSNR', 'N/A')}")
            print(f"    This script PLY PSNR:   {output['ply']['PSNR']}")
            if 'PSNR' in mv_results:
                psnr_diff = abs(output['ply']['PSNR'] - mv_results['PSNR'])
                print(f"    Absolute difference:    {psnr_diff:.4f} dB")
                if psnr_diff < 0.05:
                    print(f"    Status: PASS (within 0.05 dB)")
                else:
                    print(f"    Status: CHECK (difference > 0.05 dB)")

        print("=" * 70)

        # Save results
        out_path = os.path.join(args.model_path, f"eval_{mode}_results.json")
        serializable = {}
        for m in output:
            serializable[m] = {
                'PSNR': output[m]['PSNR'],
                'SSIM': output[m]['SSIM'],
                'LPIPS': output[m]['LPIPS'],
                'total_size_MB': output[m]['total_size_MB'],
                'avg_frame_size_MB': output[m]['avg_frame_size_MB'],
            }
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {out_path}")

        # Save per-frame results
        per_frame_path = os.path.join(args.model_path, f"eval_{mode}_per_frame.json")
        per_frame_out = {}
        for m in output:
            per_frame_out[m] = {
                'per_frame_psnr': output[m]['per_frame_psnr'],
                'per_frame_ssim': output[m]['per_frame_ssim'],
                'per_frame_lpips': output[m]['per_frame_lpips'],
                'per_frame_size_bytes': output[m]['per_frame_size_bytes'],
            }
        with open(per_frame_path, 'w') as f:
            json.dump(per_frame_out, f, indent=2)
        print(f"Per-frame results saved to: {per_frame_path}")

        return output


if __name__ == "__main__":
    print('Running on', socket.gethostname())

    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else None
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    parser = ArgumentParser(description="Evaluate PLY vs PKL rendering quality")

    lp = ModelParams(parser, config['model_params'])
    op_i = OptimizationParamsInitial(parser, config['opt_params_initial'])
    op_r = OptimizationParamsRest(parser, config['opt_params_rest'])
    pp = PipelineParams(parser, config['pipe_params'])
    qp_cls = QuantizeParams(parser, config['quantize_params'])

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["ply", "pkl", "both"],
                        help="Which model format to evaluate: ply, pkl, or both")
    parser.add_argument("--quantize_renders", action="store_true", default=True,
                        help="Apply uint8 quantization to match metrics_video.py "
                             "(default: True)")
    parser.add_argument("--no_quantize_renders", action="store_false",
                        dest="quantize_renders",
                        help="Skip uint8 quantization for raw float32 comparison")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_format", type=str, default='ply')

    args = parser.parse_args(sys.argv[1:])
    op = OptimizationParams(op_i.extract(args), op_r.extract(args))

    print(f"Evaluating: {args.model_path}")
    print(f"Mode: {args.mode}")
    safe_state(args.quiet)

    lp_args = lp.extract(args)
    pp_args = pp.extract(args)
    qp_args = qp_cls.extract(args)

    evaluate_model(lp_args, op, pp_args, qp_args, args, args.mode,
                   args.quantize_renders)

'''
# PLY only (sanity check against metrics_video.py)
python eval_ply_vs_pkl.py --config configs/dynerf.yaml \
    -s data/multipleview/cook_spinach \
    -m ./output/cook_spinach_trained_compressed \
    --mode ply

# PKL only (compressed model quality)
python eval_ply_vs_pkl.py --config configs/dynerf.yaml \
    -s data/multipleview/cook_spinach \
    -m ./output/cook_spinach_trained_compressed \
    --mode pkl

# Both with comparison
python eval_ply_vs_pkl.py --config configs/dynerf.yaml \
    -s data/multipleview/cook_spinach \
    -m ./output/cook_spinach_trained_compressed \
    --mode both
'''