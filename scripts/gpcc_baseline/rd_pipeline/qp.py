#!/usr/bin/env python3
"""Generate QP config combinations for GPCC RD experiments."""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GpccConfig, PQ_OPACITY, PQ_DC, PQ_REST


def generate_qp_configs(cfg: GpccConfig) -> list[dict]:
    """Generate all (f_rest_qp, f_dc_qp, opacity_qp) combinations.
    
    Returns list of experiment config dicts, each with:
    - experiment_name: str (e.g. "rest40_dc4_opacity4")
    - f_rest_qp: int
    - f_dc_qp: int
    - opacity_qp: int
    - voxel_depth: int (from cfg)
    - dataset_path: str (from cfg)
    - experiment_dir: str (from cfg)
    - tmc3_path: str (from cfg)
    """
    configs = []
    for f_rest_qp, f_dc_qp, opacity_qp in itertools.product(
        cfg.pq_rest, cfg.pq_dc, cfg.pq_opacity
    ):
        experiment_name = f"rest{f_rest_qp}_dc{f_dc_qp}_opacity{opacity_qp}"
        configs.append({
            "experiment_name": experiment_name,
            "f_rest_qp": f_rest_qp,
            "f_dc_qp": f_dc_qp,
            "opacity_qp": opacity_qp,
            "voxel_depth": cfg.voxel_depth,
            "dataset_path": cfg.dataset_path,
            "experiment_dir": cfg.experiment_dir,
            "tmc3_path": cfg.tmc3_path,
        })
    return configs


if __name__ == "__main__":
    cfg = GpccConfig()
    configs = generate_qp_configs(cfg)
    print(f"Generated {len(configs)} QP configurations")
