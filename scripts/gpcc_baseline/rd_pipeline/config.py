#!/usr/bin/env python3
"""Shared configuration, constants, and path builders for the GPCC RD pipeline.

Every script in scripts/gpcc_baseline/rd_pipeline/ imports from here instead of
re-declaring tmc3_path, voxel_depth, output-directory conventions, etc.
"""

import os
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Directory roots (derived from this file's location)
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOGS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR)))
LIVOGS_COMPRESSION = os.path.join(VIDEOGS_ROOT, "LiVoGS", "compression")
SCRIPTS_DIR = os.path.dirname(os.path.dirname(THIS_DIR))  # scripts/


def setup_gpcc_imports() -> None:
    """Ensure VideoGS root and LiVoGS compression dirs are on sys.path.

    Call this at module level in any script that needs VideoGS or LiVoGS
    project imports.
    """
    if VIDEOGS_ROOT not in sys.path:
        sys.path.insert(0, VIDEOGS_ROOT)
    if LIVOGS_COMPRESSION not in sys.path:
        sys.path.append(LIVOGS_COMPRESSION)


# ---------------------------------------------------------------------------
# Default QP values (from Adaptive encoder.py:287-289)
# ---------------------------------------------------------------------------

PQ_OPACITY = [4, 16, 24, 28, 34, 40]
PQ_DC = [4, 16, 24, 28]
PQ_REST = [40, 34, 28, 24, 16, 4]


# ---------------------------------------------------------------------------
# GpccConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class GpccConfig:
    """Configuration for GPCC compression pipeline."""
    tmc3_path: str = "/ssd1/haodongw/workspace/3dstream/mpeg-pcc-tmc13/build/tmc3/tmc3"
    voxel_depth: int = 15
    experiment_dir: str = ""
    dataset_path: str = ""
    num_frames: int = 1
    gpu_id: int = 0
    pq_opacity: list = field(default_factory=lambda: list(PQ_OPACITY))
    pq_dc: list = field(default_factory=lambda: list(PQ_DC))
    pq_rest: list = field(default_factory=lambda: list(PQ_REST))


# ---------------------------------------------------------------------------
# TMC3 parameters (from Adaptive encoder.py:294-363)
# ---------------------------------------------------------------------------

TMC3_PARAMS = {
    'opacity': {
        '--mode': '0', '--geomTreeType': '0', '--partitionMethod': '3',
        '--convertPlyColourspace': '1',
        '--transformType': '0', '--rahtExtension': '0', '--rahtPredictionEnabled': '0',
        '--bitdepth': '8', '--colourMatrix': '2', '--attrOffset': '0', '--attrScale': '257',
        '--attrInterPredSearchRange': '-1', '--attribute': 'reflectance'
    },
    'dc': {
        '--mode': '0', '--geomTreeType': '0', '--partitionMethod': '3',
        '--convertPlyColourspace': '0',
        '--transformType': '0', '--rahtExtension': '0', '--rahtPredictionEnabled': '0',
        '--bitdepth': '8', '--colourMatrix': '0', '--attrOffset': '0', '--attrScale': '1',
        '--attrInterPredSearchRange': '-1', '--attribute': 'color'
    },
    'rest': {
        '--mode': '0', '--geomTreeType': '0', '--partitionMethod': '3',
        '--convertPlyColourspace': '0',
        '--transformType': '0', '--rahtExtension': '0', '--rahtPredictionEnabled': '0',
        '--bitdepth': '8', '--colourMatrix': '0', '--attrOffset': '0', '--attrScale': '1',
        '--attrInterPredSearchRange': '-1', '--attribute': 'color'
    },
    'scale': {
        '--mode': '0', '--mergeDuplicatedPoints': '0', '--positionQuantizationScale': '1',
        '--trisoupNodeSizeLog2': '0', '--neighbourAvailBoundaryLog2': '8',
        '--intra_pred_max_node_size_log2': '6', '--inferredDirectCodingMode': '1',
        '--maxNumQtBtBeforeOt': '4', '--minQtbtSizeLog2': '0', '--planarEnabled': '1',
        '--planarModeIdcmUse': '0', '--convertPlyColourspace': '0', '--transformType': '0',
        '--qp': '4', '--bitdepth': '16', '--attrOffset': '0', '--attrScale': '1',
        '--attribute': 'reflectance'
    },
    'rot': {
        '--mode': '0', '--mergeDuplicatedPoints': '0', '--positionQuantizationScale': '1',
        '--trisoupNodeSizeLog2': '0', '--neighbourAvailBoundaryLog2': '8',
        '--intra_pred_max_node_size_log2': '6', '--inferredDirectCodingMode': '1',
        '--maxNumQtBtBeforeOt': '4', '--minQtbtSizeLog2': '0', '--planarEnabled': '1',
        '--planarModeIdcmUse': '0', '--convertPlyColourspace': '0', '--transformType': '0',
        '--qp': '4', '--bitdepth': '16', '--attrOffset': '0', '--attrScale': '1',
        '--attribute': 'reflectance'
    },
}


# ---------------------------------------------------------------------------
# RD output path helpers
# ---------------------------------------------------------------------------

RD_OUTPUT_SUBDIR = "gpcc_rd"


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


def experiment_dir_path(
    data_path: str,
    dataset_name: str,
    sequence_name: str,
    frame_id: int,
    depth: int,
    label: str,
    rd_subdir_name: str = RD_OUTPUT_SUBDIR,
) -> str:
    """``…/{rd_subdir_name}/frame_{id}/J_{depth}/{label}/``"""
    return os.path.join(
        rd_output_root(data_path, dataset_name, sequence_name, rd_subdir_name=rd_subdir_name),
        f"frame_{frame_id}", f"J_{depth}", label,
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
