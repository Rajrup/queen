# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Compute model sizes for PLY (uncompressed) and PKL (compressed) frames.

For PLY: uses os.path.getsize() (raw on-disk size).

For PKL: loads the pickle and computes the actual compressed data payload
(bitstream + decoder weights + metadata), excluding reconstruction-only
overhead like decoded_att. This gives the true compressed frame size, not
the inflated PKL file size.

PKL files on disk are much larger than the actual compressed payload because
they also store decoded_att (previous frame decoded attributes needed for
residual decoding). In a streaming scenario, decoded_att is derived from the
previous frame output on the receiver side and need not be transmitted.
See: https://github.com/NVlabs/queen/issues/3

Usage:
    python compute_model_sizes.py -m ./output/cook_spinach_trained_compressed
    python compute_model_sizes.py -m ./output/cook_spinach_trained_compressed --compare_training_metrics
"""

import os
import sys
import pickle
import json
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm


def _search_for_max_iteration(folder):
    saved_iters = [
        int(fname.split("_")[-1])
        for fname in os.listdir(folder)
        if fname.startswith("iteration_")
    ]
    return max(saved_iters) if saved_iters else 0


class _CompressedLatentsStub:
    """Minimal stand-in for unpickling CompressedLatents without triggering
    the circular import in utils.compress_utils -> scene -> gaussian_model."""
    pass


class _PKLUnpickler(pickle.Unpickler):
    """Custom unpickler that redirects CompressedLatents to our stub."""
    def find_class(self, module, name):
        if name == "CompressedLatents":
            return _CompressedLatentsStub
        return super().find_class(module, name)


def _is_compressed_latents(obj):
    return isinstance(obj, _CompressedLatentsStub)


def find_ply_path(frame_dir):
    """Find PLY file: prefer canonical point_cloud.ply, fallback to per-iteration."""
    canonical = os.path.join(frame_dir, "point_cloud.ply")
    if os.path.exists(canonical):
        return canonical
    iter_dir = os.path.join(frame_dir, "point_cloud")
    if os.path.exists(iter_dir):
        max_iter = _search_for_max_iteration(iter_dir)
        per_iter = os.path.join(
            iter_dir, "iteration_" + str(max_iter), "point_cloud.ply"
        )
        if os.path.exists(per_iter):
            return per_iter
    return None


def compressed_latents_size(cl):
    """
    Compute compressed payload size of a CompressedLatents object in bytes.

    Includes: torchac bitstream, CDF probability table, value mapping,
    and tail location indices (rare values encoded separately).
    """
    size = len(cl.byte_stream)
    size += cl.cdf.nbytes
    size += len(cl.mapping) * 8
    for _val, locs in cl.tail_locs.items():
        size += 4 + locs.numel() * locs.element_size()
    return size


def compute_pkl_payload_size(pkl_path):
    """
    Compute the compressed data payload from a PKL file.

    Returns (paper_bytes, full_payload_bytes, pkl_disk_bytes) where:

    paper_bytes: matches the paper's gaussians.size() methodology —
        compressed latent bitstreams + decoder weights, excluding flow
        and the xyz reorder mapping. This is what the paper reports.

    full_payload_bytes: everything needed for reconstruction including
        flow and xyz mapping, still excluding decoded_att.

    pkl_disk_bytes: raw PKL file size on disk.
    """
    pkl_disk_bytes = os.path.getsize(pkl_path)

    with open(pkl_path, "rb") as f:
        state = _PKLUnpickler(f).load()

    paper_bytes = 0
    full_bytes = 0

    decoder_types = state.get("latent_decoders_dict", {})

    for attr, latent_data in state["latents"].items():
        attr_bytes = 0
        skip_for_paper = (attr == "flow")

        if _is_compressed_latents(latent_data):
            attr_bytes = compressed_latents_size(latent_data)
        elif isinstance(latent_data, dict):
            mapping_bytes = 0
            if "mapping" in latent_data:
                m = latent_data["mapping"]
                mapping_bytes = m.numel() * m.element_size()
            rest_bytes = 0
            if "ungated_indices" in latent_data:
                idx = latent_data["ungated_indices"]
                rest_bytes += idx.numel() * idx.element_size()
            if "ungated_residuals_compressed" in latent_data:
                rest_bytes += compressed_latents_size(
                    latent_data["ungated_residuals_compressed"]
                )
            attr_bytes = mapping_bytes + rest_bytes
            full_bytes += attr_bytes
            if not skip_for_paper:
                paper_bytes += rest_bytes
            continue
        elif isinstance(latent_data, torch.Tensor):
            attr_bytes = latent_data.numel() * latent_data.element_size()

        full_bytes += attr_bytes
        if not skip_for_paper:
            paper_bytes += attr_bytes

    for _attr, sd in state.get("decoder_state_dict", {}).items():
        for key, param in sd.items():
            if key == "decoded_att":
                continue
            if isinstance(param, torch.Tensor):
                sz = param.numel() * param.element_size()
                paper_bytes += sz
                full_bytes += sz

    return paper_bytes, full_bytes, pkl_disk_bytes


def main():
    parser = ArgumentParser(description="Compute PLY vs PKL model sizes")
    parser.add_argument(
        "-m", "--model_path", required=True, help="Model output directory"
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Limit number of frames (default: all)"
    )
    parser.add_argument(
        "--compare_training_metrics", action="store_true",
        help="Compare with training_metrics.json sizes"
    )
    args = parser.parse_args()

    frames_dir = os.path.join(args.model_path, "frames")
    if not os.path.isdir(frames_dir):
        print("ERROR: Frames directory not found: " + frames_dir)
        sys.exit(1)

    frame_dirs = sorted(d for d in os.listdir(frames_dir) if d.isdigit())
    if not frame_dirs:
        print("ERROR: No frame directories found")
        sys.exit(1)

    n_frames = len(frame_dirs) if args.max_frames is None else args.max_frames
    frame_dirs = frame_dirs[:n_frames]

    print("Model path:  " + args.model_path)
    print("Frames:      {} ({} .. {})".format(
        n_frames, frame_dirs[0], frame_dirs[-1]))

    ply_sizes = []
    pkl_paper_sizes = []
    pkl_full_sizes = []
    pkl_disk_sizes = []

    for i, frame_name in enumerate(tqdm(frame_dirs, desc="Computing sizes")):
        frame_dir = os.path.join(frames_dir, frame_name)
        frame_idx = i + 1

        ply_path = find_ply_path(frame_dir)
        if ply_path:
            ply_sizes.append(os.path.getsize(ply_path))

        if frame_idx == 1:
            if ply_path:
                ply_sz = os.path.getsize(ply_path)
                pkl_paper_sizes.append(ply_sz)
                pkl_full_sizes.append(ply_sz)
        else:
            pkl_path = os.path.join(
                frame_dir, "compressed", "point_cloud.pkl"
            )
            if os.path.exists(pkl_path):
                paper, full, disk = compute_pkl_payload_size(pkl_path)
                pkl_paper_sizes.append(paper)
                pkl_full_sizes.append(full)
                pkl_disk_sizes.append(disk)

    MB = 1024 * 1024

    print("")
    print("=" * 60)
    print("MODEL SIZE REPORT")
    print("=" * 60)

    has_ply = len(ply_sizes) > 0
    has_pkl = len(pkl_paper_sizes) > 0

    total_ply_mb = avg_ply_mb = 0.0
    total_pkl_mb = avg_pkl_mb = 0.0
    ratio = 0.0

    if has_ply:
        total_ply_mb = sum(ply_sizes) / MB
        avg_ply_mb = total_ply_mb / len(ply_sizes)

    if has_pkl:
        total_pkl_mb = sum(pkl_paper_sizes) / MB
        avg_pkl_mb = total_pkl_mb / len(pkl_paper_sizes)

    if has_ply:
        print("")
        print("  Avg uncompressed size per frame (MB):   {:.3f}".format(
            avg_ply_mb))
        print("  Total uncompressed size (MB):           {:.3f}".format(
            total_ply_mb))
    if has_pkl:
        print("  Avg compressed size per frame (MB):     {:.3f}".format(
            avg_pkl_mb))
        print("  Total compressed size (MB):             {:.3f}".format(
            total_pkl_mb))
    if has_ply and has_pkl:
        ratio = (total_ply_mb / total_pkl_mb
                 if total_pkl_mb > 0 else float("inf"))
        print("  Compression ratio:                      {:.3f}x".format(
            ratio))

    if pkl_full_sizes:
        total_full_mb = sum(pkl_full_sizes) / MB
        avg_full_mb = total_full_mb / len(pkl_full_sizes)
        print("")
        print("  [Full payload incl. flow+mapping]")
        print("    Avg per frame (MB):                   {:.3f}".format(
            avg_full_mb))
        print("    Total (MB):                           {:.3f}".format(
            total_full_mb))

    if pkl_disk_sizes:
        total_disk_mb = sum(pkl_disk_sizes) / MB
        avg_disk_mb = total_disk_mb / len(pkl_disk_sizes)
        print("")
        print("  [Raw PKL file on disk (incl. decoded_att overhead)]")
        print("    Avg per frame (MB):                   {:.3f}".format(
            avg_disk_mb))
        print("    Total (MB):                           {:.3f}".format(
            total_disk_mb))

    if args.compare_training_metrics:
        tm_path = os.path.join(args.model_path, "training_metrics.json")
        if os.path.exists(tm_path):
            with open(tm_path) as f:
                tm = json.load(f)
            tm_sizes = [fm["Size (MB)"] for fm in tm[:n_frames]]
            print("")
            print("  [training_metrics.json (entropy estimate)]")
            print("    Avg per frame (MB):                   {:.3f}".format(
                np.mean(tm_sizes)))
            print("    Total (MB):                           {:.3f}".format(
                sum(tm_sizes)))
        else:
            print("")
            print("  training_metrics.json not found at " + tm_path)

    print("=" * 60)

    output = {}
    if has_ply:
        output["uncompressed"] = {
            "avg_frame_size_MB": round(avg_ply_mb, 3),
            "total_size_MB": round(total_ply_mb, 3),
            "per_frame_bytes": ply_sizes,
        }
    if has_pkl:
        output["compressed"] = {
            "avg_frame_size_MB": round(avg_pkl_mb, 3),
            "total_size_MB": round(total_pkl_mb, 3),
            "per_frame_bytes": pkl_paper_sizes,
        }
        output["compressed_full_payload"] = {
            "avg_frame_size_MB": round(
                sum(pkl_full_sizes) / MB / len(pkl_full_sizes), 3),
            "total_size_MB": round(sum(pkl_full_sizes) / MB, 3),
            "per_frame_bytes": pkl_full_sizes,
        }
    if has_ply and has_pkl:
        output["compression_ratio"] = round(ratio, 3)

    out_path = os.path.join(args.model_path, "model_sizes.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("")
    print("Results saved to: " + out_path)


if __name__ == "__main__":
    main()
