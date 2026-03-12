#!/usr/bin/env python3
"""GPCC compression/decompression pipeline for VideoGS checkpoints."""

import argparse
import concurrent.futures
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, cast

np = importlib.import_module("numpy")
_plyfile = importlib.import_module("plyfile")
PlyData = _plyfile.PlyData
PlyElement = _plyfile.PlyElement

try:
    torch = importlib.import_module("torch")
    torch_f = importlib.import_module("torch.nn.functional")
except Exception:
    torch = None
    torch_f = None


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_LIVOGS_COMPRESSION = os.path.join(_VIDEOGS_ROOT, "LiVoGS", "compression")
_OCTREE_GPU_PYTHON = os.path.join(_LIVOGS_COMPRESSION, "Octree_Compression_GPU", "python")
if _VIDEOGS_ROOT not in sys.path:
    sys.path.insert(0, _VIDEOGS_ROOT)
if _LIVOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _LIVOGS_COMPRESSION)
if _OCTREE_GPU_PYTHON not in sys.path:
    sys.path.insert(0, _OCTREE_GPU_PYTHON)

try:
    merge_gaussian_clusters_with_indices = (
        importlib.import_module("merge_cluster_cuda").merge_gaussian_clusters_with_indices
    )
    calc_morton = importlib.import_module("gpu_octree_codec").calc_morton
    _livogs_import_error = None
except Exception as exc:
    merge_gaussian_clusters_with_indices = None
    calc_morton = None
    _livogs_import_error = exc


def adaptive_normalize(values, dtype):
    if np.any(np.isnan(values)) or np.all(values == values[0]):
        normalized_values = np.zeros_like(values, dtype=dtype)
        return normalized_values, 0, 0

    min_value = np.min(values)
    max_value = np.max(values)
    if max_value - min_value == 0:
        normalized_values = np.zeros_like(values, dtype=dtype)
        return normalized_values, min_value, max_value

    normalized_values = (
        (values - min_value) / (max_value - min_value) * np.iinfo(dtype).max
    ).astype(dtype)
    return normalized_values, min_value, max_value


def numpy_to_native(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(data)
    if isinstance(data, (np.float64, np.float32, np.float16)):
        return float(data)
    return data


def convert_rgb2yuv(rgb):
    rgb2yuv = np.array(
        [
            [0.212600, -0.114572, 0.50000],
            [0.715200, -0.385428, -0.454153],
            [0.072200, 0.50000, -0.045847],
        ]
    )
    return np.dot(rgb, rgb2yuv)


def convert_yuv2rgb(yuv):
    yuv2rgb = np.array(
        [
            [1.0, 1.0, 1.0],
            [-2.94998246485263e-07, -0.18732418151803, 1.85559996313466],
            [1.57479993240292, -0.468124212249768, -4.02047471671933e-07],
        ]
    )
    return np.dot(yuv, yuv2rgb)


def uchar_to_float(values, min_val, max_val):
    return (values / 255.0) * (max_val - min_val) + min_val


def uint16_to_float(values, min_val, max_val):
    return (values / 65535.0) * (max_val - min_val) + min_val


def _get_morton_code(coords, bits):
    coords = coords.astype(np.uint64)
    morton = np.zeros(coords.shape[0], dtype=np.uint64)
    for i in range(bits):
        b = (coords >> i) & 1
        morton |= (b[:, 2] << (3 * i + 0)) | (b[:, 1] << (3 * i + 1)) | (b[:, 0] << (3 * i + 2))
    return morton


def morton_order_sort(points):
    points = np.asarray(points)
    if points.size == 0:
        return points, np.array([], dtype=np.int64)

    p_int = np.rint(points).astype(np.int64)
    p_int = p_int - p_int.min(axis=0, keepdims=True)
    max_coord = int(np.max(p_int))
    bits = int(np.ceil(np.log2(max_coord + 1))) if max_coord > 0 else 1
    morton = _get_morton_code(p_int, bits)
    sort_idx = np.argsort(morton, kind="stable")
    return points[sort_idx], sort_idx


def search_for_max_iteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def load_videogs_ply(ply_path, device="cuda"):
    if torch is None or torch_f is None:
        raise RuntimeError("PyTorch is required for GPCC voxelization pipeline")

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    sh_rest = np.stack([vertex[name] for name in rest_names], axis=1) if rest_names else np.zeros((len(vertex), 0), dtype=np.float32)
    colors = np.concatenate([sh_dc, sh_rest], axis=1)

    opacities = np.asarray(vertex["opacity"])
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)
    quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1)

    params = {
        "means": torch.from_numpy(means.copy()).float().to(device),
        "quats": torch.from_numpy(quats.copy()).float().to(device),
        "scales": torch.from_numpy(scales.copy()).float().to(device),
        "opacities": torch.from_numpy(opacities.copy()).float().to(device),
        "colors": torch.from_numpy(colors.copy()).float().to(device),
    }
    params["quats"] = torch_f.normalize(params["quats"], p=2, dim=1)
    if params["opacities"].min() < 0 or params["opacities"].max() > 1:
        params["opacities"] = torch.sigmoid(params["opacities"])
    if params["scales"].min() < 0:
        params["scales"] = torch.exp(params["scales"])

    return params


def save_videogs_ply_from_arrays(means, colors, opacity, scales, quats, output_path, eps=1e-6):
    means = means.astype(np.float32)
    colors = colors.astype(np.float32)
    opacity = opacity.astype(np.float32)
    scales = scales.astype(np.float32)
    quats = quats.astype(np.float32)
    n_points = means.shape[0]

    opacity_c = np.clip(opacity, eps, 1.0 - eps)
    opacity_logit = np.log(opacity_c / (1.0 - opacity_c))
    scales_log = np.log(np.clip(scales, eps, None))
    normals = np.zeros((n_points, 3), dtype=np.float32)

    attr_names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attr_names.append(f"f_dc_{i}")
    n_rest = colors.shape[1] - 3
    for i in range(n_rest):
        attr_names.append(f"f_rest_{i}")
    attr_names.append("opacity")
    for i in range(3):
        attr_names.append(f"scale_{i}")
    for i in range(4):
        attr_names.append(f"rot_{i}")

    data = np.concatenate(
        [means, normals, colors, opacity_logit.reshape(-1, 1), scales_log, quats],
        axis=1,
    ).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {n_points}\n".encode())
        for name in attr_names:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())


def _write_reflectance_ply(output_path, xyz, reflectance_u16):
    vertices = np.array(
        list(zip(xyz[:, 0], xyz[:, 1], xyz[:, 2], reflectance_u16)),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("reflectance", "u2")],
    )
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(output_path)


def _write_color_ply(output_path, xyz, r_u8, g_u8, b_u8):
    vertices = np.array(
        list(zip(xyz[:, 0], xyz[:, 1], xyz[:, 2], r_u8, g_u8, b_u8)),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(output_path)


def _read_attribute_ply(file_path):
    plydata = PlyData.read(file_path)
    v = plydata["vertex"]
    points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(v.data.dtype.names):
        colors = np.stack([v["red"], v["green"], v["blue"]], axis=1)
        return points, colors, None
    if "reflectance" in v.data.dtype.names:
        return points, None, np.asarray(v["reflectance"])
    if "refc" in v.data.dtype.names:
        return points, None, np.asarray(v["refc"])
    raise ValueError(f"Unsupported PLY attributes in {file_path}")


def _require_livogs_imports():
    if _livogs_import_error is not None:
        raise RuntimeError(
            "LiVoGS voxelization imports failed. Build/install LiVoGS CUDA modules first."
        ) from _livogs_import_error


def _run_tmc3_command(command):
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"TMC3 command failed ({proc.returncode}): {' '.join(command)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _build_encode_command(tmc3_path, input_ply, output_bin, attr_type, qp, voxel_depth):
    # _DYNAMIC is replaced at build time with the actual qp / voxel_depth values.
    # Argument order matters: tmc3 treats --attribute as the boundary that
    # finalises the current attribute context, so --qp must precede it.
    _DYNAMIC = None
    params = {
        "opacity": {
            "--mode": "0",
            "--geomTreeType": "0",
            "--partitionMethod": "3",
            "--partitionOctreeDepth": _DYNAMIC,
            "--convertPlyColourspace": "1",
            "--transformType": "0",
            "--rahtExtension": "0",
            "--rahtPredictionEnabled": "0",
            "--qp": _DYNAMIC,
            "--bitdepth": "8",
            "--colourMatrix": "2",
            "--attrOffset": "0",
            "--attrScale": "257",
            "--attrInterPredSearchRange": "-1",
            "--attribute": "reflectance",
        },
        "dc": {
            "--mode": "0",
            "--geomTreeType": "0",
            "--partitionMethod": "3",
            "--partitionOctreeDepth": _DYNAMIC,
            "--convertPlyColourspace": "0",
            "--transformType": "0",
            "--rahtExtension": "0",
            "--rahtPredictionEnabled": "0",
            "--qp": _DYNAMIC,
            "--bitdepth": "8",
            "--colourMatrix": "0",
            "--attrOffset": "0",
            "--attrScale": "1",
            "--attrInterPredSearchRange": "-1",
            "--attribute": "color",
        },
        "rest": {
            "--mode": "0",
            "--geomTreeType": "0",
            "--partitionMethod": "3",
            "--partitionOctreeDepth": _DYNAMIC,
            "--convertPlyColourspace": "0",
            "--transformType": "0",
            "--rahtExtension": "0",
            "--rahtPredictionEnabled": "0",
            "--qp": _DYNAMIC,
            "--bitdepth": "8",
            "--colourMatrix": "0",
            "--attrOffset": "0",
            "--attrScale": "1",
            "--attrInterPredSearchRange": "-1",
            "--attribute": "color",
        },
        "scale": {
            "--mode": "0",
            "--mergeDuplicatedPoints": "0",
            "--positionQuantizationScale": "1",
            "--trisoupNodeSizeLog2": "0",
            "--neighbourAvailBoundaryLog2": "8",
            "--intra_pred_max_node_size_log2": "6",
            "--inferredDirectCodingMode": "1",
            "--maxNumQtBtBeforeOt": "4",
            "--minQtbtSizeLog2": "0",
            "--planarEnabled": "1",
            "--planarModeIdcmUse": "0",
            "--convertPlyColourspace": "0",
            "--transformType": "1",
            "--numberOfNearestNeighborsInPrediction": "3",
            "--intraLodPredictionSkipLayers": "0",
            "--interComponentPredictionEnabled": "0",
            "--adaptivePredictionThreshold": "64",
            "--qp": _DYNAMIC,
            "--bitdepth": "16",
            "--attrOffset": "0",
            "--attrScale": "1",
            "--attribute": "reflectance",
        },
        "rot": {
            "--mode": "0",
            "--mergeDuplicatedPoints": "0",
            "--positionQuantizationScale": "1",
            "--trisoupNodeSizeLog2": "0",
            "--neighbourAvailBoundaryLog2": "8",
            "--intra_pred_max_node_size_log2": "6",
            "--inferredDirectCodingMode": "1",
            "--maxNumQtBtBeforeOt": "4",
            "--minQtbtSizeLog2": "0",
            "--planarEnabled": "1",
            "--planarModeIdcmUse": "0",
            "--convertPlyColourspace": "0",
            "--transformType": "1",
            "--numberOfNearestNeighborsInPrediction": "3",
            "--intraLodPredictionSkipLayers": "0",
            "--interComponentPredictionEnabled": "0",
            "--adaptivePredictionThreshold": "64",
            "--qp": _DYNAMIC,
            "--bitdepth": "16",
            "--attrOffset": "0",
            "--attrScale": "1",
            "--attribute": "reflectance",
        },
    }

    dynamic_values = {
        "--partitionOctreeDepth": str(voxel_depth),
        "--qp": str(qp),
    }

    command = [
        tmc3_path,
        f"--uncompressedDataPath={input_ply}",
        f"--compressedStreamPath={output_bin}",
    ]
    for key, value in params[attr_type].items():
        if value is _DYNAMIC:
            value = dynamic_values[key]
        command.append(f"{key}={value}")

    return command


def _build_decode_command(tmc3_path, input_bin, output_ply):
    return [
        tmc3_path,
        "--mode=1",
        "--outputBinaryPly=1",
        f"--compressedStreamPath={input_bin}",
        f"--reconstructedDataPath={output_ply}",
    ]


def _voxelize_and_merge(params, voxel_depth, device):
    _require_livogs_imports()
    if torch is None:
        raise RuntimeError("PyTorch is required for voxelization")
    assert calc_morton is not None
    assert merge_gaussian_clusters_with_indices is not None

    means = params["means"]
    n_points = means.shape[0]
    vmin = means.min(dim=0)[0]
    v0 = means - vmin.unsqueeze(0)
    width = v0.max()
    if width <= 0:
        width = torch.tensor(1.0, device=means.device)
    voxel_size = width / (2.0 ** voxel_depth)
    v0_integer = torch.clamp(torch.floor(v0 / voxel_size).long(), 0, 2**voxel_depth - 1).int()

    morton_result = calc_morton(
        v0_integer,
        voxel_grid_depth=voxel_depth,
        force_64bit_codes=True,
        device=device,
        return_torch=True,
    )
    morton_codes_points = morton_result["morton_codes"]
    if morton_codes_points.dtype == torch.uint64:
        morton_codes_points = morton_codes_points.to(torch.int64)

    sorted_morton, sort_idx = torch.sort(morton_codes_points)
    voxel_boundary = sorted_morton[1:] - sorted_morton[:-1]
    voxel_indices = torch.cat([
        torch.tensor([0], device=means.device),
        torch.where(voxel_boundary != 0)[0] + 1,
    ])

    cluster_indices = sort_idx.int()
    cluster_offsets = torch.cat(
        [voxel_indices.int(), torch.tensor([n_points], dtype=torch.int32, device=means.device)],
        dim=0,
    ).int()

    merged_means, merged_quats, merged_scales, merged_opacities, merged_colors = merge_gaussian_clusters_with_indices(
        params["means"],
        params["quats"],
        params["scales"],
        params["opacities"],
        params["colors"],
        cluster_indices,
        cluster_offsets,
        weight_by_opacity=True,
    )

    voxel_xyz = v0_integer[sort_idx[voxel_indices]].float()
    return {
        "voxel_xyz": voxel_xyz.detach().cpu().numpy().astype(np.float32),
        "colors": merged_colors.detach().cpu().numpy().astype(np.float32),
        "opacity": merged_opacities.detach().cpu().numpy().astype(np.float32),
        "scales": merged_scales.detach().cpu().numpy().astype(np.float32),
        "quats": merged_quats.detach().cpu().numpy().astype(np.float32),
        "vmin": vmin.detach().cpu().numpy().astype(np.float32),
        "voxel_size": float(voxel_size.item()),
    }


def encode_gpcc(ply_path, output_dir, qp_config, tmc3_path, voxel_depth):
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    temp_encode_dir = os.path.join(output_dir, "temp_encode")
    attr_ply_dir = os.path.join(temp_encode_dir, "attribute_ply")
    os.makedirs(attr_ply_dir, exist_ok=True)
    compressed_dirs = {
        "opacity": os.path.join(temp_encode_dir, "opacity_compressed"),
        "dc": os.path.join(temp_encode_dir, "dc_compressed"),
        "rest": os.path.join(temp_encode_dir, "rest_compressed"),
        "scale": os.path.join(temp_encode_dir, "scale_compressed"),
        "rot": os.path.join(temp_encode_dir, "rot_compressed"),
    }
    for d in compressed_dirs.values():
        os.makedirs(d, exist_ok=True)

    device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    device_id = int(device.split(":")[1]) if device.startswith("cuda:") else 0

    params = load_videogs_ply(ply_path, device=device)
    merged = cast(Dict[str, Any], _voxelize_and_merge(params, voxel_depth=voxel_depth, device=device_id))

    xyz = cast(Any, merged["voxel_xyz"])
    colors = cast(Any, merged["colors"])
    if colors.shape[1] < 3:
        raise ValueError(f"Expected at least 3 color channels after merge, got {colors.shape[1]}")
    n_rest_channels = int(colors.shape[1] - 3)
    if n_rest_channels % 3 != 0:
        raise ValueError(f"Expected rest channel count divisible by 3, got {n_rest_channels}")
    opacity = cast(Any, merged["opacity"])
    scales = cast(Any, merged["scales"])
    quats = cast(Any, merged["quats"])

    metadata: Dict[str, Any] = {
        "Geometry": {
            "vmin": merged["vmin"],
            "voxel_size": merged["voxel_size"],
            "voxel_depth": voxel_depth,
        },
        "Attribute": {},
        "files": {"opacity": [], "dc": [], "rest": [], "scale": [], "rot": []},
    }

    compression_jobs: List[Any] = []

    opacity_u16, mn, mx = adaptive_normalize(opacity, np.uint16)
    metadata["Attribute"]["opacity"] = {"min": mn, "max": mx}
    opacity_ply = os.path.join(attr_ply_dir, "opacity.ply")
    _write_reflectance_ply(opacity_ply, xyz, opacity_u16)
    opacity_bin = os.path.join(compressed_dirs["opacity"], "opacity.bin")
    metadata["files"]["opacity"].append("opacity.bin")
    compression_jobs.append(("opacity", opacity_ply, opacity_bin, qp_config["qp_opacity"]))

    dc_rgb = colors[:, :3]
    dc_yuv = convert_rgb2yuv(dc_rgb)
    dc_u8 = []
    for i in range(3):
        arr_u8, mn, mx = adaptive_normalize(dc_yuv[:, i], np.uint8)
        dc_u8.append(arr_u8)
        metadata["Attribute"][f"f_dc_{i}"] = {"min": mn, "max": mx}
    dc_ply = os.path.join(attr_ply_dir, "dc.ply")
    _write_color_ply(dc_ply, xyz, dc_u8[0], dc_u8[1], dc_u8[2])
    dc_bin = os.path.join(compressed_dirs["dc"], "dc.bin")
    metadata["files"]["dc"].append("dc.bin")
    compression_jobs.append(("dc", dc_ply, dc_bin, qp_config["qp_dc"]))

    rest_rgb = colors[:, 3 : 3 + n_rest_channels]
    for i in range(0, n_rest_channels, 3):
        rest_triplet_rgb = rest_rgb[:, i : i + 3]
        rest_triplet_yuv = convert_rgb2yuv(rest_triplet_rgb)
        c_u8 = []
        for c in range(3):
            arr_u8, mn, mx = adaptive_normalize(rest_triplet_yuv[:, c], np.uint8)
            c_u8.append(arr_u8)
            metadata["Attribute"][f"f_rest_{i + c}"] = {"min": mn, "max": mx}
        name = f"rest_{i:02d}_{i+1:02d}_{i+2:02d}"
        rest_ply = os.path.join(attr_ply_dir, f"{name}.ply")
        _write_color_ply(rest_ply, xyz, c_u8[0], c_u8[1], c_u8[2])
        rest_bin = os.path.join(compressed_dirs["rest"], f"{name}.bin")
        metadata["files"]["rest"].append(f"{name}.bin")
        compression_jobs.append(("rest", rest_ply, rest_bin, qp_config["qp_rest"]))

    for i in range(3):
        scale_u16, mn, mx = adaptive_normalize(scales[:, i], np.uint16)
        metadata["Attribute"][f"scale_{i}"] = {"min": mn, "max": mx}
        name = f"scale_{i}"
        scale_ply = os.path.join(attr_ply_dir, f"{name}.ply")
        _write_reflectance_ply(scale_ply, xyz, scale_u16)
        scale_bin = os.path.join(compressed_dirs["scale"], f"{name}.bin")
        metadata["files"]["scale"].append(f"{name}.bin")
        compression_jobs.append(("scale", scale_ply, scale_bin, 4))

    for i in range(4):
        rot_u16, mn, mx = adaptive_normalize(quats[:, i], np.uint16)
        metadata["Attribute"][f"rot_{i}"] = {"min": mn, "max": mx}
        name = f"rot_{i}"
        rot_ply = os.path.join(attr_ply_dir, f"{name}.ply")
        _write_reflectance_ply(rot_ply, xyz, rot_u16)
        rot_bin = os.path.join(compressed_dirs["rot"], f"{name}.bin")
        metadata["files"]["rot"].append(f"{name}.bin")
        compression_jobs.append(("rot", rot_ply, rot_bin, 4))

    metadata_json_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=numpy_to_native)

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for attr_type, input_ply, output_bin, qp in compression_jobs:
            command = _build_encode_command(
                tmc3_path=tmc3_path,
                input_ply=input_ply,
                output_bin=output_bin,
                attr_type=attr_type,
                qp=qp,
                voxel_depth=voxel_depth,
            )
            futures.append(executor.submit(_run_tmc3_command, command))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    shutil.rmtree(attr_ply_dir)

    size_stats = {
        "opacity_bytes": sum(os.path.getsize(os.path.join(compressed_dirs["opacity"], n)) for n in metadata["files"]["opacity"]),
        "dc_bytes": sum(os.path.getsize(os.path.join(compressed_dirs["dc"], n)) for n in metadata["files"]["dc"]),
        "rest_bytes": sum(os.path.getsize(os.path.join(compressed_dirs["rest"], n)) for n in metadata["files"]["rest"]),
        "scale_bytes": sum(os.path.getsize(os.path.join(compressed_dirs["scale"], n)) for n in metadata["files"]["scale"]),
        "rot_bytes": sum(os.path.getsize(os.path.join(compressed_dirs["rot"], n)) for n in metadata["files"]["rot"]),
    }
    return {
        **size_stats,
        "total_compressed_bytes": sum(size_stats.values()),
        "num_points_input": int(params["means"].shape[0]),
        "num_points_voxelized": int(xyz.shape[0]),
        "encode_time_s": time.perf_counter() - t0,
        "compressed_dir": temp_encode_dir,
        "metadata_json_path": metadata_json_path,
    }


def decode_gpcc(compressed_dir, output_ply_path, metadata_json_path, tmc3_path):
    t0 = time.perf_counter()
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        metadata = cast(Dict[str, Any], json.load(f))

    temp_decode_dir = os.path.join(compressed_dir, "temp_decode")
    if os.path.exists(temp_decode_dir):
        shutil.rmtree(temp_decode_dir, ignore_errors=True)
    os.makedirs(temp_decode_dir, exist_ok=True)

    category_dirs = {
        "opacity": os.path.join(compressed_dir, "opacity_compressed"),
        "dc": os.path.join(compressed_dir, "dc_compressed"),
        "rest": os.path.join(compressed_dir, "rest_compressed"),
        "scale": os.path.join(compressed_dir, "scale_compressed"),
        "rot": os.path.join(compressed_dir, "rot_compressed"),
    }

    decomp_ply_paths: Dict[str, List[str]] = {"opacity": [], "dc": [], "rest": [], "scale": [], "rot": []}
    decode_jobs: List[Any] = []
    for category in ["opacity", "dc", "rest", "scale", "rot"]:
        out_cat_dir = os.path.join(temp_decode_dir, f"{category}_decompressed")
        os.makedirs(out_cat_dir, exist_ok=True)
        for bin_name in metadata["files"][category]:
            input_bin = os.path.join(category_dirs[category], bin_name)
            output_ply = os.path.join(out_cat_dir, os.path.splitext(bin_name)[0] + ".ply")
            decode_jobs.append(_build_decode_command(tmc3_path, input_bin, output_ply))
            decomp_ply_paths[category].append(output_ply)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_tmc3_command, cmd) for cmd in decode_jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # --- Compute morton sort index ONCE from the first decompressed PLY ---
    # All 24 attribute streams share identical voxel geometry, so the sort
    # order is the same for every file.  Computing it once avoids 23
    # redundant morton-code calculations + argsorts.
    first_ply = decomp_ply_paths["opacity"][0]
    ref_points, _, _ = _read_attribute_ply(first_ply)
    points_ref, _ = morton_order_sort(ref_points)
    n_points = points_ref.shape[0]

    geom_meta = metadata.get("Geometry", {})
    vmin = np.asarray(geom_meta.get("vmin", [0.0, 0.0, 0.0]), dtype=np.float32)
    voxel_size = float(geom_meta.get("voxel_size", 1.0))
    points_world = (points_ref.astype(np.float32) + 0.5) * voxel_size + vmin.reshape(1, 3)

    def _read_sorted_reflectance(ply_path):
        points, _, refl = _read_attribute_ply(ply_path)
        if refl is None:
            raise ValueError(f"Decompressed PLY missing reflectance: {ply_path}")
        points_sorted, local_sort_idx = morton_order_sort(points)
        if points_sorted.shape != points_ref.shape or not np.array_equal(points_sorted, points_ref):
            raise ValueError(f"Geometry/order mismatch after sorting for reflectance stream: {ply_path}")
        return refl[local_sort_idx].astype(np.float32)

    def _read_sorted_colors(ply_path):
        points, colors, _ = _read_attribute_ply(ply_path)
        if colors is None:
            raise ValueError(f"Decompressed PLY missing color attributes: {ply_path}")
        points_sorted, local_sort_idx = morton_order_sort(points)
        if points_sorted.shape != points_ref.shape or not np.array_equal(points_sorted, points_ref):
            raise ValueError(f"Geometry/order mismatch after sorting for color stream: {ply_path}")
        return colors[local_sort_idx].astype(np.float32)

    opacity_denorm = uint16_to_float(
        _read_sorted_reflectance(decomp_ply_paths["opacity"][0]),
        metadata["Attribute"]["opacity"]["min"],
        metadata["Attribute"]["opacity"]["max"],
    )

    dc_sorted = _read_sorted_colors(decomp_ply_paths["dc"][0])
    y0 = uchar_to_float(dc_sorted[:, 0], metadata["Attribute"]["f_dc_0"]["min"], metadata["Attribute"]["f_dc_0"]["max"])
    y1 = uchar_to_float(dc_sorted[:, 1], metadata["Attribute"]["f_dc_1"]["min"], metadata["Attribute"]["f_dc_1"]["max"])
    y2 = uchar_to_float(dc_sorted[:, 2], metadata["Attribute"]["f_dc_2"]["min"], metadata["Attribute"]["f_dc_2"]["max"])
    dc_rgb = convert_yuv2rgb(np.stack([y0, y1, y2], axis=1)).astype(np.float32)

    rest_attr_indices = sorted(
        int(k.split("_")[-1])
        for k in metadata["Attribute"].keys()
        if k.startswith("f_rest_")
    )
    if len(rest_attr_indices) % 3 != 0:
        raise ValueError(f"Invalid number of rest attributes in metadata: {len(rest_attr_indices)}")

    rest_rgb_channels = []
    for rest_idx, rest_ply in enumerate(decomp_ply_paths["rest"]):
        rest_sorted = _read_sorted_colors(rest_ply)
        idx_base = rest_idx * 3
        if idx_base + 2 >= len(rest_attr_indices):
            raise ValueError("Rest stream count does not match metadata attributes")
        k0 = f"f_rest_{rest_attr_indices[idx_base]}"
        k1 = f"f_rest_{rest_attr_indices[idx_base + 1]}"
        k2 = f"f_rest_{rest_attr_indices[idx_base + 2]}"
        r0 = uchar_to_float(rest_sorted[:, 0], metadata["Attribute"][k0]["min"], metadata["Attribute"][k0]["max"])
        r1 = uchar_to_float(rest_sorted[:, 1], metadata["Attribute"][k1]["min"], metadata["Attribute"][k1]["max"])
        r2 = uchar_to_float(rest_sorted[:, 2], metadata["Attribute"][k2]["min"], metadata["Attribute"][k2]["max"])
        rest_rgb_triplet = convert_yuv2rgb(np.stack([r0, r1, r2], axis=1)).astype(np.float32)
        rest_rgb_channels.extend([rest_rgb_triplet[:, 0], rest_rgb_triplet[:, 1], rest_rgb_triplet[:, 2]])

    scales = np.zeros((n_points, 3), dtype=np.float32)
    for i, scale_ply in enumerate(decomp_ply_paths["scale"]):
        scales[:, i] = uint16_to_float(
            _read_sorted_reflectance(scale_ply),
            metadata["Attribute"][f"scale_{i}"]["min"],
            metadata["Attribute"][f"scale_{i}"]["max"],
        )

    quats = np.zeros((n_points, 4), dtype=np.float32)
    for rot_ply in decomp_ply_paths["rot"]:
        stem = os.path.splitext(os.path.basename(rot_ply))[0]
        comp_name = stem.replace("_dec", "")
        if not comp_name.startswith("rot_"):
            raise ValueError(f"Unexpected rotation stream name: {rot_ply}")
        comp_idx = int(comp_name.split("_")[-1])
        quats[:, comp_idx] = uint16_to_float(
            _read_sorted_reflectance(rot_ply),
            metadata["Attribute"][f"rot_{comp_idx}"]["min"],
            metadata["Attribute"][f"rot_{comp_idx}"]["max"],
        )

    if not decomp_ply_paths["rot"]:
        raise ValueError("No rotation streams were decoded")

    if rest_rgb_channels:
        rest_rgb = np.stack(rest_rgb_channels, axis=1).astype(np.float32)
        colors = np.concatenate([dc_rgb, rest_rgb], axis=1).astype(np.float32)
    else:
        colors = dc_rgb.astype(np.float32)

    save_videogs_ply_from_arrays(
        means=points_world,
        colors=colors,
        opacity=opacity_denorm.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=quats.astype(np.float32),
        output_path=output_ply_path,
    )

    shutil.rmtree(temp_decode_dir, ignore_errors=True)
    return {
        "decode_time_s": time.perf_counter() - t0,
        "num_points_output": int(n_points),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="GPCC compress + decompress pipeline for VideoGS checkpoints")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to VideoGS checkpoint dir (contains frame subdirs)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for compressed files and benchmark CSV")
    parser.add_argument("--output_ply_dir", type=str, required=True, help="Output directory for decompressed PLY files")
    parser.add_argument(
        "--tmc3_path",
        type=str,
        default="/ssd1/haodongw/workspace/3dstream/mpeg-pcc-tmc13/build/tmc3/tmc3",
        help="Path to tmc3 binary",
    )
    parser.add_argument("--voxel_depth", type=int, default=15, help="Voxelization depth J")
    parser.add_argument("--qp_rest", type=int, default=40, help="QP for rest attributes")
    parser.add_argument("--qp_dc", type=int, default=4, help="QP for dc attributes")
    parser.add_argument("--qp_opacity", type=int, default=4, help="QP for opacity")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames to process")
    parser.add_argument("--frame_start", type=int, default=0, help="Starting frame index")
    return parser.parse_args()


def _resolve_input_checkpoint_dir(input_dir: str, frame_idx: int) -> str | None:
    base = Path(input_dir)
    candidates = [
        base / str(frame_idx) / "point_cloud",
        base / f"{frame_idx:04d}" / "point_cloud",
        base / "frames" / f"{frame_idx:04d}" / "point_cloud",
        base / "frames" / str(frame_idx) / "point_cloud",
    ]
    for cand in candidates:
        if cand.is_dir():
            return str(cand)
    return None


def _build_output_ply_path(output_ply_dir: str, frame_idx: int, queen_layout: bool) -> str:
    if queen_layout:
        return str(Path(output_ply_dir) / "frames" / f"{frame_idx:04d}" / "point_cloud.ply")
    return str(Path(output_ply_dir) / str(frame_idx) / "point_cloud" / "point_cloud.ply")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_ply_dir, exist_ok=True)
    queen_layout = Path(args.input_dir, "frames").is_dir()

    qp_config = {
        "qp_rest": args.qp_rest,
        "qp_dc": args.qp_dc,
        "qp_opacity": args.qp_opacity,
    }

    benchmark_rows = []
    for frame_idx in range(args.frame_start, args.frame_start + args.num_frames):
        ckpt_path = _resolve_input_checkpoint_dir(args.input_dir, frame_idx)
        if ckpt_path is None:
            continue

        max_iter = search_for_max_iteration(ckpt_path)
        ply_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")
        if not os.path.exists(ply_path):
            continue

        frame_output_dir = os.path.join(args.output_dir, f"frame_{frame_idx}")
        frame_output_ply = _build_output_ply_path(args.output_ply_dir, frame_idx, queen_layout)

        enc = encode_gpcc(
            ply_path=ply_path,
            output_dir=frame_output_dir,
            qp_config=qp_config,
            tmc3_path=args.tmc3_path,
            voxel_depth=args.voxel_depth,
        )
        dec = decode_gpcc(
            compressed_dir=enc["compressed_dir"],
            output_ply_path=frame_output_ply,
            metadata_json_path=enc["metadata_json_path"],
            tmc3_path=args.tmc3_path,
        )

        benchmark_rows.append(
            {
                "frame_idx": frame_idx,
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
            }
        )

    csv_path = os.path.join(args.output_dir, "benchmark_gpcc.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "encode_time_s",
                "decode_time_s",
                "total_compressed_bytes",
                "opacity_bytes",
                "dc_bytes",
                "rest_bytes",
                "scale_bytes",
                "rot_bytes",
                "num_points_input",
                "num_points_output",
            ]
        )
        for row in benchmark_rows:
            writer.writerow(
                [
                    row["frame_idx"],
                    f"{row['encode_time_s']:.6f}",
                    f"{row['decode_time_s']:.6f}",
                    row["total_compressed_bytes"],
                    row["opacity_bytes"],
                    row["dc_bytes"],
                    row["rest_bytes"],
                    row["scale_bytes"],
                    row["rot_bytes"],
                    row["num_points_input"],
                    row["num_points_output"],
                ]
            )


if __name__ == "__main__":
    main()
