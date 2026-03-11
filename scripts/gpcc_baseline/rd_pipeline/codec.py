#!/usr/bin/env python3
"""GPCC codec wrapper for the RD pipeline.

Thin wrapper around encode_gpcc / decode_gpcc from compress_decompress_pipeline.
Provides the interface expected by worker.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compress_decompress_pipeline import encode_gpcc, decode_gpcc


def run_gpcc_codec(
    frame_ply_path: str,
    output_dir: str,
    qp_config: dict,
    cfg,  # GpccConfig
) -> dict:
    """Compress and decompress a single frame PLY using GPCC/tmc3.
    
    Args:
        frame_ply_path: Path to input VideoGS PLY file
        output_dir: Directory for compressed files and temp data
        qp_config: Dict with keys: f_rest_qp, f_dc_qp, opacity_qp
        cfg: GpccConfig instance
    
    Returns:
        Dict with keys: encode_time_s, decode_time_s, total_compressed_bytes,
        opacity_bytes, dc_bytes, rest_bytes, scale_bytes, rot_bytes,
        num_points_input, num_points_output, output_ply_path
    """
    output_ply_path = os.path.join(output_dir, "decompressed", "point_cloud.ply")
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    
    enc = encode_gpcc(
        ply_path=frame_ply_path,
        output_dir=output_dir,
        qp_config={
            "qp_rest": qp_config["f_rest_qp"],
            "qp_dc": qp_config["f_dc_qp"],
            "qp_opacity": qp_config["opacity_qp"],
        },
        tmc3_path=cfg.tmc3_path,
        voxel_depth=cfg.voxel_depth,
    )
    
    dec = decode_gpcc(
        compressed_dir=enc["compressed_dir"],
        output_ply_path=output_ply_path,
        metadata_json_path=enc["metadata_json_path"],
        tmc3_path=cfg.tmc3_path,
    )
    
    return {
        "encode_time_s": enc["encode_time_s"],
        "decode_time_s": dec["decode_time_s"],
        "total_compressed_bytes": enc["total_compressed_bytes"],
        "opacity_bytes": enc["opacity_bytes"],
        "dc_bytes": enc["dc_bytes"],
        "rest_bytes": enc["rest_bytes"],
        "scale_bytes": enc["scale_bytes"],
        "rot_bytes": enc["rot_bytes"],
        "num_points_input": enc["num_points_input"],
        "num_points_output": dec["num_points_output"],
        "output_ply_path": output_ply_path,
    }
