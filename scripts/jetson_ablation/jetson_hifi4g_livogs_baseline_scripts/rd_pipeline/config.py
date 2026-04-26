#!/usr/bin/env python3
"""Shared configuration, constants, and path builders for the LiVoGS RD pipeline.

Every script in scripts/livogs_baseline/rd_pipeline/ imports from here instead of
re-declaring DATA_PATH, SH_DEGREE, output-directory conventions, etc.
"""

import os
import sys
from typing import TypedDict

# ---------------------------------------------------------------------------
# Directory roots (derived from this file's location)
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOGS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
LIVOGS_COMPRESSION = os.path.join(VIDEOGS_ROOT, "LiVoGS", "compression")
SCRIPTS_DIR = os.path.dirname(os.path.dirname(THIS_DIR))  # scripts/


def setup_livogs_imports() -> None:
    """Ensure VideoGS root and LiVoGS compression dirs are on sys.path.

    Call this at module level in any script that needs LiVoGS or VideoGS
    project imports (e.g. ``from compress_decompress import encode_livogs``).
    """
    if VIDEOGS_ROOT in sys.path:
        sys.path.remove(VIDEOGS_ROOT)
    sys.path.insert(0, VIDEOGS_ROOT)
    if LIVOGS_COMPRESSION in sys.path:
        sys.path.remove(LIVOGS_COMPRESSION)
    sys.path.append(LIVOGS_COMPRESSION)


# ---------------------------------------------------------------------------
# Shared parameter defaults
# ---------------------------------------------------------------------------

DATA_PATH       = "/synology/rajrup/VideoGS"
RESOLUTION      = 2
SH_DEGREE       = 3
SH_COLOR_SPACE  = "klt"
RLGR_BLOCK_SIZE = 4096
NVCOMP_ALGORITHM: str | None = None
J               = 15          # octree depth for voxelization
DEVICE          = "cuda:0"

# Fixed quantize steps for non-colour attributes (used by QP generation)
BASELINE_QUANTIZE_STEP: dict[str, float] = {
    "quats":   0.00005,
    "scales":  0.0001,
    "opacity": 0.0001,
}

# Default QP configs output root
QP_CONFIGS_ROOT = os.path.join(VIDEOGS_ROOT, "results", "rd_qp_configs")
RD_OUTPUT_SUBDIR = "livogs_rd"


# ---------------------------------------------------------------------------
# Sequence configuration type
# ---------------------------------------------------------------------------

class SequenceCfg(TypedDict):
    dataset_name: str
    sequence_name: str
    qp_dir_name: str


# ---------------------------------------------------------------------------
# Path builders — single source of truth for directory conventions
# ---------------------------------------------------------------------------

def checkpoint_dir(data_path: str, dataset_name: str, sequence_name: str) -> str:
    """``{data}/train_output/{dataset}/{sequence}/checkpoint``"""
    return os.path.join(
        data_path, "train_output", dataset_name, sequence_name, "checkpoint",
    )


def processed_dataset_dir(data_path: str, dataset_name: str, sequence_name: str) -> str:
    """``{data}/{dataset}_processed/{sequence}``"""
    return os.path.join(data_path, f"{dataset_name}_processed", sequence_name)


def rd_output_root(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """``{data}/train_output/{dataset}/{sequence}/compression/{rd_subdir_name}/``"""
    return os.path.join(
        data_path, "train_output", dataset_name, sequence_name,
        "compression", rd_subdir_name,
    )


def experiment_dir(
    data_path: str, dataset_name: str, sequence_name: str,
    frame_id: int, depth: int, label: str,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """``…/{rd_subdir_name}/frame_{id}/J_{depth}/{label}/``"""
    return os.path.join(
        rd_output_root(data_path, dataset_name, sequence_name, rd_subdir_name=rd_subdir_name),
        f"frame_{frame_id}", f"J_{depth}", label,
    )


def standard_output_dir(
    data_path: str, dataset_name: str, sequence_name: str,
    j: int, quantize_step: float, sh_color_space: str,
) -> str:
    """Standard (non-RD) output: ``…/compression/livogs/J_{j}_qstep_{qs}_{cs}``"""
    return os.path.join(
        data_path, "train_output", dataset_name, sequence_name,
        "compression", "livogs",
        f"J_{j}_qstep_{quantize_step}_{sh_color_space}",
    )


def plot_output_dir(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """``…/{rd_subdir_name}/plots/``"""
    return os.path.join(
        rd_output_root(data_path, dataset_name, sequence_name, rd_subdir_name=rd_subdir_name),
        "plots",
    )


def qp_json_pattern(qp_configs_root: str, qp_dir_name: str, frame_id: int) -> str:
    """Glob pattern: ``{root}/{qp_dir}/frame_{id}/qp_*.json``"""
    return os.path.join(qp_configs_root, qp_dir_name, f"frame_{frame_id}", "qp_*.json")


def qp_json_output_dir(qp_configs_root: str, qp_dir_name: str, frame_id: int) -> str:
    """Directory for QP config JSONs: ``{root}/{qp_dir}/frame_{id}/``"""
    return os.path.join(qp_configs_root, qp_dir_name, f"frame_{frame_id}")


def all_results_csv(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    frame_id: int,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """``…/{rd_subdir_name}/frame_{id}/all_results.csv``"""
    return os.path.join(
        rd_output_root(data_path, dataset_name, sequence_name, rd_subdir_name=rd_subdir_name),
        f"frame_{frame_id}", "all_results.csv",
    )


# Valid knob names for plot specs
KNOB_NAMES = frozenset({"depth", "qp_sh", "beta", "qp_quats", "qp_scales", "qp_opacity"})
