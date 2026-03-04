#!/usr/bin/env python3
"""Shared configuration, constants, and path builders for the LiVoGS RD pipeline (QUEEN)."""

import os
import sys
from typing import TypedDict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QUEEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
LIVOGS_COMPRESSION = os.path.join(QUEEN_ROOT, "LiVoGS", "compression")
SCRIPTS_DIR = os.path.dirname(os.path.dirname(THIS_DIR))  # scripts/


def setup_livogs_imports() -> None:
    """Ensure QUEEN root and LiVoGS compression dirs are on sys.path."""
    if QUEEN_ROOT in sys.path:
        sys.path.remove(QUEEN_ROOT)
    sys.path.insert(0, QUEEN_ROOT)
    if LIVOGS_COMPRESSION in sys.path:
        sys.path.remove(LIVOGS_COMPRESSION)
    sys.path.insert(0, LIVOGS_COMPRESSION)


DATA_PATH = "/synology/rajrup/Queen"
RESOLUTION = 2
SH_DEGREE = 2  # QUEEN uses SH degree 2 (27 coefficients total)
SH_COLOR_SPACE = "klt"
RLGR_BLOCK_SIZE = 4096
NVCOMP_ALGORITHM: str | None = None
J = 15
DEVICE = "cuda:0"

BASELINE_QUANTIZE_STEP: dict[str, float] = {
    "quats": 0.00005,
    "scales": 0.0001,
    "opacity": 0.0001,
}

QP_CONFIGS_ROOT = os.path.join(QUEEN_ROOT, "results", "rd_qp_configs")
RD_OUTPUT_SUBDIR = "livogs_rd"


class SequenceCfg(TypedDict):
    dataset_name: str
    sequence_name: str
    qp_dir_name: str


def checkpoint_dir(data_path: str, dataset_name: str, sequence_name: str) -> str:
    """{data}/pretrained_output/{dataset}/queen_compressed_{seq}/"""
    return os.path.join(
        data_path,
        "pretrained_output",
        dataset_name,
        f"queen_compressed_{sequence_name}",
    )


def dataset_dir(data_path: str, dataset_name: str, sequence_name: str) -> str:
    """{data}/{dataset_name}/{sequence_name}"""
    return os.path.join(data_path, dataset_name, sequence_name)


def rd_output_root(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """{checkpoint_dir}/compression/livogs_rd/"""
    return os.path.join(
        checkpoint_dir(data_path, dataset_name, sequence_name),
        "compression",
        rd_subdir_name,
    )


def experiment_dir(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    frame_id: int,
    depth: int,
    label: str,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """.../livogs_rd/frame_{id}/J_{depth}/{label}/"""
    return os.path.join(
        rd_output_root(data_path, dataset_name, sequence_name, rd_subdir_name=rd_subdir_name),
        f"frame_{frame_id}",
        f"J_{depth}",
        label,
    )


def standard_output_dir(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    j: int,
    quantize_step: float,
    sh_color_space: str,
) -> str:
    """Standard (non-RD) output: .../compression/livogs/J_{j}_qstep_{qs}_{cs}"""
    return os.path.join(
        checkpoint_dir(data_path, dataset_name, sequence_name),
        "compression",
        "livogs",
        f"J_{j}_qstep_{quantize_step}_{sh_color_space}",
    )


def plot_output_dir(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """.../livogs_rd/plots/"""
    return os.path.join(
        rd_output_root(data_path, dataset_name, sequence_name, rd_subdir_name=rd_subdir_name),
        "plots",
    )


def qp_json_pattern(qp_configs_root: str, qp_dir_name: str, frame_id: int) -> str:
    """Glob pattern: {root}/{qp_dir}/frame_{id}/qp_*.json"""
    return os.path.join(qp_configs_root, qp_dir_name, f"frame_{frame_id}", "qp_*.json")


def qp_json_output_dir(qp_configs_root: str, qp_dir_name: str, frame_id: int) -> str:
    """Directory for QP config JSONs: {root}/{qp_dir}/frame_{id}/"""
    return os.path.join(qp_configs_root, qp_dir_name, f"frame_{frame_id}")


def all_results_csv(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    frame_id: int,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    return os.path.join(
        rd_output_root(data_path, dataset_name, sequence_name, rd_subdir_name=rd_subdir_name),
        f"frame_{frame_id}", "all_results.csv",
    )


KNOB_NAMES = frozenset({"depth", "baseline_qp", "beta", "qp_quats", "qp_scales", "qp_opacity"})
