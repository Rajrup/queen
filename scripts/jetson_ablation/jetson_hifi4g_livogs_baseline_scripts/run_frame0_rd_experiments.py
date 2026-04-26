#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false
import csv
import inspect
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_WORKSPACE = os.path.dirname(_VIDEOGS_ROOT)
_LIVOGS_ROOT = os.path.join(_WORKSPACE, "LiVoGS")
_COMPRESSION_DIR = os.path.join(_LIVOGS_ROOT, "compression")

for path in [os.path.join(_COMPRESSION_DIR, "RAHT-3DGS-codec", "python"), _COMPRESSION_DIR]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from color_space_transforms import normalize_attributes, rgb_to_klt3
from data_util import load_3dgs
from deploy_3dgs_codec import deploy_compress_decompress
from gpu_octree_codec import calc_morton
from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from voxelize_pc import voxelize_pc


_DEPLOY_SUPPORTS_COLOR_RESCALE = "color_rescale" in inspect.signature(deploy_compress_decompress).parameters

QP_SH_VALUES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]
BETA_VALUES = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
J = 15
SH_COLOR_SPACE = "klt"
COLOR_RESCALE = False
DEVICE = "cuda:0"
BASELINE_QUANTIZE_STEP: Dict[str, float] = {
    "quats": 0.00005,
    "scales": 0.0001,
    "opacity": 0.0001,
}
OUTPUT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "../../results/frame0_rd_experiments"))


def _parse_float_list_env(var_name: str) -> Optional[List[float]]:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return None
    return [float(item) for item in values]


_env_qp_sh_values = _parse_float_list_env("FRAME0_QP_SH_VALUES")
RUNTIME_QP_SH_VALUES = _env_qp_sh_values if _env_qp_sh_values is not None else QP_SH_VALUES

_env_beta_values = _parse_float_list_env("FRAME0_BETA_VALUES")
RUNTIME_BETA_VALUES = _env_beta_values if _env_beta_values is not None else BETA_VALUES

_env_output_dir = os.getenv("FRAME0_OUTPUT_DIR", "").strip()
RUNTIME_OUTPUT_DIR = os.path.abspath(_env_output_dir) if _env_output_dir else OUTPUT_DIR


class SequenceConfig(TypedDict):
    name: str
    checkpoint_path: str
    dataset_config: Dict[str, Any]


SEQUENCES: List[SequenceConfig] = [
    {
        "name": "HiFi4G_4K_Actor1_Greeting",
        "checkpoint_path": "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/checkpoint/0/point_cloud/iteration_16000/point_cloud.ply",
        "dataset_config": {
            "format": "hifi4g",
            "data_dir": "/synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor1_Greeting/0",
            "split": "val",
            "llffhold": 8,
            "resolution": 2,
        },
    },
    {
        "name": "HiFi4G_4K_Actor2_Dancing",
        "checkpoint_path": "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor2_Dancing/checkpoint/0/point_cloud/iteration_12000/point_cloud.ply",
        "dataset_config": {
            "format": "hifi4g",
            "data_dir": "/synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor2_Dancing/0",
            "split": "val",
            "llffhold": 8,
            "resolution": 2,
        },
    },
    {
        "name": "HiFi4G_4K_Actor3_Violin",
        "checkpoint_path": "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor3_Violin/checkpoint/0/point_cloud/iteration_16000/point_cloud.ply",
        "dataset_config": {
            "format": "hifi4g",
            "data_dir": "/synology/rajrup/VideoGS/HiFi4G_Dataset_processed/4K_Actor3_Violin/0",
            "split": "val",
            "llffhold": 8,
            "resolution": 2,
        },
    },
]


def compute_energy_rms(checkpoint_path: str, J: int, device: str) -> Tuple[NDArray[np.float64], float]:
    params = load_3dgs(checkpoint_path, device=device)
    n_points = params["means"].shape[0]

    v_means = params["means"]
    vmin = v_means.min(dim=0)[0]
    v0 = v_means - vmin.unsqueeze(0)
    width = v0.max()
    voxel_size = width / (2.0**J)
    v0_integer = torch.clamp(torch.floor(v0 / voxel_size).long(), 0, 2**J - 1).int()

    device_id = int(device.split(":")[1]) if ":" in device else 0
    morton_result = calc_morton(
        v0_integer,
        voxel_grid_depth=J,
        force_64bit_codes=True,
        device=device_id,
        return_torch=True,
    )
    morton_codes = morton_result["morton_codes"]
    if morton_codes.dtype == torch.uint64:
        morton_codes = morton_codes.to(torch.int64)

    _, _, voxel_indices, _, voxel_info = voxelize_pc(
        params["means"],
        vmin=vmin,
        width=width,
        J=J,
        device=device,
        morton_codes=morton_codes,
    )

    sort_idx = voxel_info["sort_idx"]
    cluster_indices = sort_idx.int()
    cluster_offsets = torch.cat(
        [
            voxel_indices,
            torch.tensor([n_points], dtype=torch.int32, device=device),
        ]
    ).int()

    _, _, _, _, merged_colors = merge_gaussian_clusters_with_indices(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        params["colors"],
        cluster_indices,
        cluster_offsets,
        weight_by_opacity=True,
    )

    merged_colors_normalized, _ = normalize_attributes(merged_colors)
    # Keep KLT3 to match analyze_sh_energy_and_qp.py energy estimation flow.
    klt_colors, _, _ = rgb_to_klt3(merged_colors_normalized)

    energy = (klt_colors.double() ** 2).mean(dim=0).cpu().numpy()
    rms = np.sqrt(energy)
    rms = np.where(rms > 0, rms, rms.max() * 1e-6)
    return rms, float(rms.max())


def generate_qp_sets(
    rms: NDArray[np.float64],
    rms_max: float,
    qp_sh_values: List[float],
    beta_values: List[float],
) -> List[Dict[str, Any]]:
    qp_sets: List[Dict[str, Any]] = []
    next_id = 0
    for qp_sh in qp_sh_values:
        for beta in beta_values:
            qps = qp_sh * (rms_max / rms) ** beta
            qp_sets.append(
                {
                    "id": next_id,
                    "label": f"qp{qp_sh}_beta_{beta:.1f}",
                    "values": qps.tolist(),
                    "qp_sh": qp_sh,
                    "beta": beta,
                }
            )
            next_id += 1
    return qp_sets


def create_quantize_config(sh_values: Any) -> Dict[str, Any]:
    config: Dict[str, Any] = dict(BASELINE_QUANTIZE_STEP)
    if isinstance(sh_values, (list, tuple)) and len(sh_values) > 3:
        config["sh_dc"] = list(sh_values[:3])
        config["sh_rest"] = list(sh_values[3:])
    else:
        config["sh_dc"] = sh_values
        config["sh_rest"] = sh_values
    return config


def run_single_experiment(
    qp_set: Dict[str, Any],
    checkpoint_path: str,
    J: int,
    dataset_config: Dict[str, Any],
    output_dir: str,
    device: str,
) -> Dict[str, Any]:
    exp_output_dir = os.path.join(output_dir, f"exp_{qp_set['label']}")
    os.makedirs(exp_output_dir, exist_ok=True)
    quantize_config = create_quantize_config(qp_set["values"])

    deploy_kwargs: Dict[str, Any] = {
        "ckpt_path": checkpoint_path,
        "J": J,
        "output_dir": exp_output_dir,
        "device": device,
        "sh_color_space": SH_COLOR_SPACE,
        "use_entropy_encoding": True,
        "quantize_step": quantize_config,
        "dataset_config": dataset_config,
        "verify_lossless_checks": False,
        "save_images": False,
    }
    if _DEPLOY_SUPPORTS_COLOR_RESCALE:
        deploy_kwargs["color_rescale"] = COLOR_RESCALE

    try:
        results = deploy_compress_decompress(**deploy_kwargs)

        psnr_avg = results["rendering_metrics"]["decoded"]["psnr_avg"]
        attr_bytes = results["attribute_compression"]["compressed_bytes"]
        pos_bytes = results["position_compression"]["compressed_bytes"]
        total_compressed_bytes = attr_bytes + pos_bytes
        total_compressed_mb = total_compressed_bytes / (1024 * 1024)

        return {
            "qp_sh": qp_set["qp_sh"],
            "beta": qp_set["beta"],
            "psnr_avg": psnr_avg,
            "total_compressed_bytes": total_compressed_bytes,
            "total_compressed_mb": total_compressed_mb,
            "attr_compressed_bytes": attr_bytes,
            "pos_compressed_bytes": pos_bytes,
        }
    except Exception as exc:
        print(f"[WARN] Experiment failed for {qp_set['label']}: {exc}")
        return {
            "qp_sh": qp_set["qp_sh"],
            "beta": qp_set["beta"],
            "error": str(exc),
        }


def save_results_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    valid_rows = [row for row in results if "error" not in row]
    failed_rows = [row for row in results if "error" in row]
    if failed_rows:
        print(f"[WARN] Skipping {len(failed_rows)} failed rows when writing CSV: {csv_path}")

    fieldnames = [
        "qp_sh",
        "beta",
        "psnr_avg",
        "total_compressed_bytes",
        "total_compressed_mb",
        "attr_compressed_bytes",
        "pos_compressed_bytes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)


def plot_rd_curves(results: List[Dict[str, Any]], output_path: str, seq_name: str) -> None:
    valid_rows = [row for row in results if "error" not in row]
    if not valid_rows:
        print(f"[WARN] No valid rows to plot for {seq_name}")
        return

    grouped = {}
    for row in valid_rows:
        grouped.setdefault(row["beta"], []).append(row)

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(8, 6))

    for idx, beta in enumerate(sorted(grouped.keys())):
        points = sorted(grouped[beta], key=lambda item: item["total_compressed_mb"])
        x = [p["total_compressed_mb"] for p in points]
        y = [p["psnr_avg"] for p in points]
        plt.plot(x, y, marker="o", label=f"β={beta}", color=cmap(idx % 10))

    plt.xlabel("Compressed Size (MB)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Rate-Distortion Curves — {seq_name} (J={J}, KLT)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(RUNTIME_OUTPUT_DIR, exist_ok=True)
    print(f"Saving outputs to: {RUNTIME_OUTPUT_DIR}")

    for seq in SEQUENCES:
        seq_name = seq["name"]
        checkpoint_path = seq["checkpoint_path"]
        dataset_config = seq["dataset_config"]
        seq_output_dir = os.path.join(RUNTIME_OUTPUT_DIR, seq_name)
        os.makedirs(seq_output_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"Sequence: {seq_name}")
        print(f"Checkpoint: {checkpoint_path}")

        rms, rms_max = compute_energy_rms(checkpoint_path, J, DEVICE)
        qp_sets = generate_qp_sets(rms, rms_max, RUNTIME_QP_SH_VALUES, RUNTIME_BETA_VALUES)

        sequence_results = []
        for qp_set in qp_sets:
            result = run_single_experiment(
                qp_set=qp_set,
                checkpoint_path=checkpoint_path,
                J=J,
                dataset_config=dataset_config,
                output_dir=seq_output_dir,
                device=DEVICE,
            )
            sequence_results.append(result)
            torch.cuda.empty_cache()

        csv_path = os.path.join(seq_output_dir, "rd_results.csv")
        plot_path = os.path.join(seq_output_dir, "rd_curves.png")
        save_results_csv(sequence_results, csv_path)
        plot_rd_curves(sequence_results, plot_path, seq_name)

        print(f"Saved CSV: {csv_path}")
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
