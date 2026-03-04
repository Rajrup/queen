#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Generate per-channel QP configs for LiVoGS RD experiments (QUEEN)."""

import json
import os
import sys
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUEEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if QUEEN_ROOT not in sys.path:
    sys.path.insert(0, QUEEN_ROOT)
from scripts.livogs_baseline.rd_pipeline import config

config.setup_livogs_imports()

import numpy as np
import torch

from scripts.livogs_baseline.rd_pipeline import codec
from color_space_transforms import normalize_attributes, rgb_to_klt3
from gpu_octree_codec import calc_morton
from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from voxelize_pc import voxelize_pc


def compute_energy_rms(
    ply_path: str,
    j: int,
    device: str,
) -> tuple[np.ndarray[Any, Any], float]:
    """Compute per-channel KLT3 energy RMS from one frame PLY."""
    params, _ = codec.load_queen_ply(ply_path, device=device)
    num_gaussians = params["means"].shape[0]

    v_means = params["means"]
    vmin = v_means.min(dim=0)[0]
    v0 = v_means - vmin.unsqueeze(0)
    width = v0.max()
    voxel_size = width / (2.0**j)
    v0_integer = torch.clamp(torch.floor(v0 / voxel_size).long(), 0, 2**j - 1).int()
    device_id = int(device.split(":")[1]) if ":" in device else 0

    morton_result = calc_morton(
        v0_integer,
        voxel_grid_depth=j,
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
        J=j,
        device=device,
        morton_codes=morton_codes,
    )

    sort_idx = voxel_info["sort_idx"]
    cluster_indices = sort_idx.int()
    cluster_offsets = torch.cat(
        [
            voxel_indices,
            torch.tensor([num_gaussians], dtype=torch.int32, device=device),
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
    klt_colors, _, _ = rgb_to_klt3(merged_colors_normalized)
    energy = (klt_colors.double() ** 2).mean(dim=0).cpu().numpy()
    rms = np.sqrt(energy)
    return rms, float(rms.max())


def generate_qp_sets(
    rms: np.ndarray[Any, Any],
    rms_max: float,
    qp_sh_values: list[float],
    beta_values: list[float],
) -> list[dict[str, Any]]:
    """Generate QP sets for all (qp_sh, beta) combinations."""
    safe_rms = np.where(rms > 0, rms, rms.max() * 1e-6)
    qp_sets: list[dict[str, Any]] = []
    set_id = 0
    for qp_sh in qp_sh_values:
        for beta in beta_values:
            qps = qp_sh * (rms_max / safe_rms) ** beta
            qp_sets.append(
                {
                    "id": set_id,
                    "label": f"qp{qp_sh}_beta_{beta:.1f}",
                    "values": qps.tolist(),
                    "qp_sh": qp_sh,
                    "beta": beta,
                }
            )
            set_id += 1
    return qp_sets


def create_quantize_config(
    sh_values: list[float],
    qp_quats: Optional[float] = None,
    qp_scales: Optional[float] = None,
    qp_opacity: Optional[float] = None,
) -> dict[str, Any]:
    """Build quantize_config from SH QPs (27 entries for SH degree 2: 3 DC + 24 rest)."""
    quantize_cfg: dict[str, Any] = dict(config.BASELINE_QUANTIZE_STEP)
    if qp_quats is not None:
        quantize_cfg["quats"] = qp_quats
    if qp_scales is not None:
        quantize_cfg["scales"] = qp_scales
    if qp_opacity is not None:
        quantize_cfg["opacity"] = qp_opacity
    if isinstance(sh_values, (list, tuple)) and len(sh_values) > 3:
        quantize_cfg["sh_dc"] = list(sh_values[:3])
        quantize_cfg["sh_rest"] = list(sh_values[3:])
    else:
        quantize_cfg["sh_dc"] = sh_values
        quantize_cfg["sh_rest"] = sh_values
    return quantize_cfg


def generate(
    sequences: list[config.SequenceCfg],
    frame_ids: list[int],
    qp_sh_values: list[float],
    beta_values: list[float],
    output_root: str = config.QP_CONFIGS_ROOT,
    data_path: str = config.DATA_PATH,
    j: int = config.J,
    device: str = config.DEVICE,
    selected_qp_dir_names: Optional[list[str]] = None,
    qp_quats_list: Optional[list[float]] = None,
    qp_scales_list: Optional[list[float]] = None,
    qp_opacity_list: Optional[list[float]] = None,
) -> None:
    """Generate QP config JSONs for selected sequences and frames."""
    output_root = os.path.abspath(output_root)

    if selected_qp_dir_names is not None:
        requested = set(selected_qp_dir_names)
        active = [s for s in sequences if s["qp_dir_name"] in requested]
        missing = sorted(requested - {s["qp_dir_name"] for s in active})
        if missing:
            print(f"[WARN] Unknown qp_dir_names ignored: {missing}")
    else:
        active = list(sequences)

    if not active:
        raise SystemExit("No matching sequences to generate. Check --qp_dir_names.")
    if not frame_ids:
        raise SystemExit("No frame IDs requested. Check --frame_ids.")

    for seq in active:
        qp_dir_name = seq["qp_dir_name"]
        checkpoint_root = config.checkpoint_dir(data_path, seq["dataset_name"], seq["sequence_name"])

        print(f"\n{'=' * 70}")
        print(f"Sequence: {qp_dir_name}")

        for frame_id in frame_ids:
            frame_str = str(frame_id).zfill(4)
            frame_dir = os.path.join(checkpoint_root, "frames", frame_str)
            try:
                ply_path = codec.find_queen_ply_path(frame_dir)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Failed to resolve checkpoint PLY for sequence={qp_dir_name}, frame={frame_id}. {exc}"
                ) from exc

            print(f"\n  Frame {frame_id}: computing energy RMS from {ply_path}")
            rms, rms_max = compute_energy_rms(ply_path, j, device)
            print(f"  RMS: shape={rms.shape}  min={rms.min():.4f}  max={rms_max:.4f}")

            qp_sets = generate_qp_sets(rms, rms_max, qp_sh_values, beta_values)

            _qp_quats = qp_quats_list or [config.BASELINE_QUANTIZE_STEP["quats"]]
            _qp_scales = qp_scales_list or [config.BASELINE_QUANTIZE_STEP["scales"]]
            _qp_opacity = qp_opacity_list or [config.BASELINE_QUANTIZE_STEP["opacity"]]
            out_dir = config.qp_json_output_dir(output_root, qp_dir_name, frame_id)
            os.makedirs(out_dir, exist_ok=True)

            total_configs = len(qp_sets) * len(_qp_quats) * len(_qp_scales) * len(_qp_opacity)
            for qp_set in qp_sets:
                for qp_quats in _qp_quats:
                    for qp_scales in _qp_scales:
                        for qp_opacity in _qp_opacity:
                            quantize_cfg = create_quantize_config(
                                qp_set["values"],
                                qp_quats=qp_quats,
                                qp_scales=qp_scales,
                                qp_opacity=qp_opacity,
                            )
                            label = f"qp{qp_set['qp_sh']}_b{qp_set['beta']:.1f}_q{qp_quats}_s{qp_scales}_o{qp_opacity}"
                            payload = {
                                "label": label,
                                "qp_sh": qp_set["qp_sh"],
                                "beta": qp_set["beta"],
                                "qp_quats": qp_quats,
                                "qp_scales": qp_scales,
                                "qp_opacity": qp_opacity,
                                "frame_id": frame_id,
                                "sequence_name": qp_dir_name,
                                "octree_depth": j,
                                "quantize_config": quantize_cfg,
                            }
                            out_path = os.path.join(out_dir, f"qp_{label}.json")
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2)

            print(f"  Saved {total_configs} QP configs -> {out_dir}")
            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("Done.")


_STANDALONE_SEQUENCES: list[config.SequenceCfg] = [
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "coffee_martini",
        "qp_dir_name": "DyNeRF_coffee_martini",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "cook_spinach",
        "qp_dir_name": "DyNeRF_cook_spinach",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "cut_roasted_beef",
        "qp_dir_name": "DyNeRF_cut_roasted_beef",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "flame_salmon_1",
        "qp_dir_name": "DyNeRF_flame_salmon_1",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "flame_steak",
        "qp_dir_name": "DyNeRF_flame_steak",
    },
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "sear_steak",
        "qp_dir_name": "DyNeRF_sear_steak",
    },
]

_STANDALONE_QP_SH_VALUES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]
_STANDALONE_BETA_VALUES = [0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0]
_STANDALONE_FRAME_IDS = [1]

if __name__ == "__main__":
    import argparse as _ap

    parser = _ap.ArgumentParser(
        description="Generate per-channel QP configs for LiVoGS RD experiments",
    )
    parser.add_argument("--output_dir", default=None,
                        help="Override output root (absolute path)")
    parser.add_argument("--qp_sh_values", nargs="+", type=float, default=None,
                        help="Override baseline QPs (space-separated floats)")
    parser.add_argument("--beta_values", nargs="+", type=float, default=None,
                        help="Override beta values (space-separated floats)")
    parser.add_argument("--frame_ids", nargs="+", type=int, default=None,
                        help="Override frame IDs (space-separated ints)")
    parser.add_argument("--qp_dir_names", nargs="+", default=None,
                        help="Subset of qp_dir_name values to generate")
    parser.add_argument("--qp_quats", nargs="+", type=float, default=None,
                        help="Override quat quantization step sweep")
    parser.add_argument("--qp_scales", nargs="+", type=float, default=None,
                        help="Override scale quantization step sweep")
    parser.add_argument("--qp_opacity", nargs="+", type=float, default=None,
                        help="Override opacity quantization step sweep")
    args = parser.parse_args()

    generate(
        sequences=_STANDALONE_SEQUENCES,
        frame_ids=args.frame_ids or _STANDALONE_FRAME_IDS,
        qp_sh_values=args.qp_sh_values or _STANDALONE_QP_SH_VALUES,
        beta_values=args.beta_values or _STANDALONE_BETA_VALUES,
        output_root=args.output_dir or config.QP_CONFIGS_ROOT,
        selected_qp_dir_names=args.qp_dir_names,
        qp_quats_list=args.qp_quats,
        qp_scales_list=args.qp_scales,
        qp_opacity_list=args.qp_opacity,
    )
