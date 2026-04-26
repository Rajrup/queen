#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Generate per-channel QP configs for LiVoGS RD experiments.

For each (sequence, frame), computes SH energy RMS in KLT3 space and generates
JSON QP config files for all (qp_sh, beta) combinations.

Output: {output_root}/{qp_dir_name}/frame_{frame_id}/qp_{label}.json

Usable as a library (``qp.generate(...)``) or standalone CLI::

    python scripts/livogs_baseline/rd_pipeline/qp.py --qp_sh_values 1 5 10

JSON schema::

    {
        "label":           "shqp0.005_beta_0.8",
        "qp_sh":           float,
        "beta":            float,
        "frame_id":        int,
        "sequence_name":   str,
        "octree_depth":    int,
        "quantize_config": { "quats", "scales", "opacity", "sh_dc", "sh_rest" }
    }
"""

import json
import os
import sys
from typing import Any, Optional

# -- config import (must come before LiVoGS imports) --------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.setup_livogs_imports()

import numpy as np
import torch

from color_space_transforms import normalize_attributes, rgb_to_klt3
from data_util import load_3dgs
from gpu_octree_codec import calc_morton
from merge_cluster_cuda import merge_gaussian_clusters_with_indices
from utils.system_utils import searchForMaxIteration
from voxelize_pc import voxelize_pc


# ---------------------------------------------------------------------------
# Library functions
# ---------------------------------------------------------------------------

def compute_energy_rms(
    checkpoint_path: str, j: int, device: str,
) -> tuple[np.ndarray[Any, Any], float]:
    """Compute per-channel KLT3 energy RMS for merged SH coefficients.

    NOTE: uses KLT3 (per-degree PCA) for energy analysis.  The coding path
    uses KLT15, so the QP adaptation is approximate but correlates with
    coding importance.
    """
    params = load_3dgs(checkpoint_path, device=device)
    num_gaussians = params["means"].shape[0]

    v_means = params["means"]
    vmin = v_means.min(dim=0)[0]
    v0 = v_means - vmin.unsqueeze(0)
    width = v0.max()
    voxel_size = width / (2.0 ** j)
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
    cluster_offsets = torch.cat([
        voxel_indices,
        torch.tensor([num_gaussians], dtype=torch.int32, device=device),
    ]).int()

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
    """Generate QP value sets for all (qp_sh, beta) combinations.

    For beta=0: uniform QPs (all channels get qp_sh).
    For beta>0: channels with lower RMS (less energy) get higher QP (coarser).
    """
    safe_rms = np.where(rms > 0, rms, rms.max() * 1e-6)
    qp_sets: list[dict[str, Any]] = []
    set_id = 0
    for qp_sh in qp_sh_values:
        for beta in beta_values:
            qps = qp_sh * (rms_max / safe_rms) ** beta
            qp_sets.append({
                "id": set_id,
                "label": f"shqp{qp_sh}_beta_{beta:.1f}",
                "values": qps.tolist(),
                "qp_sh": qp_sh,
                "beta": beta,
            })
            set_id += 1
    return qp_sets


def create_quantize_config(
    sh_values: list[float],
    qp_quats: Optional[float] = None,
    qp_scales: Optional[float] = None,
    qp_opacity: Optional[float] = None,
) -> dict[str, Any]:
    """Build full quantize_config from per-channel SH QP values.

    *sh_values* has 48 entries (3 DC + 45 higher-order for SH degree 3).
    Optional *qp_quats*, *qp_scales*, *qp_opacity* override the defaults from
    ``config.BASELINE_QUANTIZE_STEP``; if ``None`` the baseline value is kept.
    """
    cfg: dict[str, Any] = dict(config.BASELINE_QUANTIZE_STEP)
    if qp_quats is not None:
        cfg["quats"] = qp_quats
    if qp_scales is not None:
        cfg["scales"] = qp_scales
    if qp_opacity is not None:
        cfg["opacity"] = qp_opacity
    if isinstance(sh_values, (list, tuple)) and len(sh_values) > 3:
        cfg["sh_dc"]   = list(sh_values[:3])
        cfg["sh_rest"] = list(sh_values[3:])
    else:
        cfg["sh_dc"]   = sh_values
        cfg["sh_rest"] = sh_values
    return cfg

# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

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
    """Generate QP config JSONs for given sequences, frames, and QP sweep.

    Parameters
    ----------
    sequences : list[SequenceCfg]
        Sequences to process.  ``qp_dir_name`` selects the output subdirectory;
        ``dataset_name`` + ``sequence_name`` locate the checkpoint.
    selected_qp_dir_names : list[str] | None
        If given, only generate for sequences whose ``qp_dir_name`` is in this set.
    """
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
        raise SystemExit("No matching sequences to generate.  Check --qp_dir_names.")
    if not frame_ids:
        raise SystemExit("No frame IDs requested.  Check --frame_ids.")

    for seq in active:
        qp_dir_name     = seq["qp_dir_name"]
        checkpoint_root  = config.checkpoint_dir(data_path, seq["dataset_name"], seq["sequence_name"])

        print(f"\n{'=' * 70}")
        print(f"Sequence: {qp_dir_name}")

        for frame_id in frame_ids:
            point_cloud_root = os.path.join(checkpoint_root, str(frame_id), "point_cloud")
            try:
                max_iter = searchForMaxIteration(point_cloud_root)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Failed to resolve max iteration checkpoint for sequence={qp_dir_name}, "
                    f"frame={frame_id}. {e}"
                ) from e

            ckpt_path = os.path.join(
                point_cloud_root, f"iteration_{max_iter}", "point_cloud.ply",
            )

            print(f"\n  Frame {frame_id}: computing energy RMS from {ckpt_path}")
            rms, rms_max = compute_energy_rms(ckpt_path, j, device)
            print(f"  RMS: shape={rms.shape}  min={rms.min():.4f}  max={rms_max:.4f}")

            qp_sets = generate_qp_sets(rms, rms_max, qp_sh_values, beta_values)

            _qp_quats = qp_quats_list or [config.BASELINE_QUANTIZE_STEP["quats"]]
            _qp_scales = qp_scales_list or [config.BASELINE_QUANTIZE_STEP["scales"]]
            _qp_opacity = qp_opacity_list or [config.BASELINE_QUANTIZE_STEP["opacity"]]
            multi_attr = (len(_qp_quats) > 1 or len(_qp_scales) > 1 or len(_qp_opacity) > 1)

            out_dir = config.qp_json_output_dir(output_root, qp_dir_name, frame_id)
            os.makedirs(out_dir, exist_ok=True)

            total_configs = len(qp_sets) * len(_qp_quats) * len(_qp_scales) * len(_qp_opacity)
            for qp_set in qp_sets:
                for qp_quats in _qp_quats:
                    for qp_scales in _qp_scales:
                        for qp_opacity in _qp_opacity:
                            quantize_cfg = create_quantize_config(
                                qp_set["values"], qp_quats, qp_scales, qp_opacity,
                            )
                            label = f"shqp{qp_set['qp_sh']}_b{qp_set['beta']:.1f}_q{qp_quats}_s{qp_scales}_o{qp_opacity}"
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
                            with open(out_path, "w") as f:
                                json.dump(payload, f, indent=2)

            print(f"  Saved {total_configs} QP configs → {out_dir}")
            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("Done.")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

_STANDALONE_SEQUENCES: list[config.SequenceCfg] = [
    {
        "dataset_name":  "HiFi4G_Dataset",
        "sequence_name": "4K_Actor1_Greeting",
        "qp_dir_name":   "HiFi4G_4K_Actor1_Greeting",
    },
]

_STANDALONE_SH_QPS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]
_STANDALONE_BETA_VALUES  = [0.0, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0]
_STANDALONE_FRAME_IDS    = [0]

if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser(
        description="Generate per-channel QP configs for LiVoGS RD experiments",
    )
    _parser.add_argument("--output_dir",   default=None,
                         help="Override output root (absolute path)")
    _parser.add_argument("--qp_sh_values", nargs="+", type=float, default=None,
                         help="Override SH_QPS (space-separated floats)")
    _parser.add_argument("--beta_values",  nargs="+", type=float, default=None,
                         help="Override BETA_VALUES (space-separated floats)")
    _parser.add_argument("--qp_quats", nargs="+", type=float, default=None,
                         help="Override attr QP list for quats")
    _parser.add_argument("--qp_scales", nargs="+", type=float, default=None,
                         help="Override attr QP list for scales")
    _parser.add_argument("--qp_opacity", nargs="+", type=float, default=None,
                         help="Override attr QP list for opacity")
    _parser.add_argument("--frame_ids",    nargs="+", type=int,   default=None,
                         help="Override FRAME_IDS (space-separated ints)")
    _parser.add_argument("--qp_dir_names", nargs="+", default=None,
                         help="Subset of qp_dir_name values to generate")
    _args = _parser.parse_args()

    generate(
        sequences=_STANDALONE_SEQUENCES,
        frame_ids=_args.frame_ids or _STANDALONE_FRAME_IDS,
        qp_sh_values=_args.qp_sh_values or _STANDALONE_SH_QPS,
        beta_values=_args.beta_values or _STANDALONE_BETA_VALUES,
        output_root=_args.output_dir or config.QP_CONFIGS_ROOT,
        selected_qp_dir_names=_args.qp_dir_names,
        qp_quats_list=_args.qp_quats,
        qp_scales_list=_args.qp_scales,
        qp_opacity_list=_args.qp_opacity,
    )
