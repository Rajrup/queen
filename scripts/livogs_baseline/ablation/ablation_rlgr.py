#!/usr/bin/env python3
"""
RLGR Ablation: CPU vs GPU (multiple block sizes).

For each frame:
  1. Run encode_livogs() once to get the compressed state.
  2. Decode RLGR to recover the quantized coefficient tensor (int32, GPU).
  3. Benchmark each RLGR variant on the SAME tensor.

Variants:
  cpu       – rlgr.Encoder per-channel loop (includes GPU→CPU transfer)
  gpu_full  – rlgr_gpu.EncoderGPU(block_size=-1)  (full channel, matches CPU bytes)
  gpu_8192  – rlgr_gpu.EncoderGPU(block_size=8192)
  gpu_4096  – rlgr_gpu.EncoderGPU(block_size=4096) (current default)
  gpu_2048  – rlgr_gpu.EncoderGPU(block_size=2048)
"""

import os
import sys
import csv
import time
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

# --- sys.path setup (same as compress_decompress_pipeline.py) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)                       # livogs_baseline/
_QUEEN_ROOT = os.path.dirname(os.path.dirname(_SCRIPTS_DIR))    # Queen/
_LIVOGS_COMPRESSION = os.path.join(_QUEEN_ROOT, "LiVoGS", "compression")
for p in (_QUEEN_ROOT, _LIVOGS_COMPRESSION):
    if p not in sys.path:
        sys.path.insert(0, p)

from compress_decompress import encode_livogs
import rlgr_gpu
import rlgr

# Reuse PLY loader from pipeline
sys.path.insert(0, _SCRIPTS_DIR)
from compress_decompress_pipeline import load_queen_ply, find_queen_ply_path

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
VARIANTS = [
    ("cpu",      None),
    ("gpu_full", -1),
    ("gpu_8192", 8192),
    ("gpu_4096", 4096),
    ("gpu_2048", 2048),
    ("gpu_1024", 1024),
    ("gpu_512", 512),
    ("gpu_256", 256),
    ("gpu_128", 128),
    ("gpu_64", 64),
    ("gpu_32", 32)
]


def benchmark_gpu_variant(coeff_int32, block_size, device_id):
    """Encode + decode with GPU RLGR; return timing and size."""
    encoder = rlgr_gpu.EncoderGPU(block_size=block_size, flagSigned=1)
    decoder = rlgr_gpu.DecoderGPU()

    # --- Encode ---
    torch.cuda.synchronize(device_id)
    t0 = time.perf_counter()
    compressed = encoder.rlgrEncode(coeff_int32)
    torch.cuda.synchronize(device_id)
    encode_ms = (time.perf_counter() - t0) * 1000

    compressed_bytes = int(compressed['compressed_data'].shape[0])

    # --- Decode ---
    torch.cuda.synchronize(device_id)
    t0 = time.perf_counter()
    decoded, _ = decoder.rlgrDecode(compressed)
    torch.cuda.synchronize(device_id)
    decode_ms = (time.perf_counter() - t0) * 1000

    return {
        "rlgr_encode_ms": encode_ms,
        "rlgr_decode_ms": decode_ms,
        "transfer_to_cpu_ms": 0.0,
        "transfer_to_gpu_ms": 0.0,
        "pure_rlgr_encode_ms": encode_ms,
        "pure_rlgr_decode_ms": decode_ms,
        "compressed_size_bytes": compressed_bytes,
        "decoded": decoded,
    }


def benchmark_cpu_variant(coeff_int32, device_id):
    """Encode + decode with CPU RLGR; return timing and size."""
    n_symbols, n_channels = coeff_int32.shape

    encoder_cpu = rlgr.Encoder()
    decoder_cpu = rlgr.Decoder()

    # --- Encode ---
    # Step 1: GPU → CPU transfer (as int32 for fairness)
    torch.cuda.synchronize(device_id)
    t_xfer_start = time.perf_counter()
    np_coeff = coeff_int32.cpu().numpy()  # int32 numpy array
    t_xfer_end = time.perf_counter()
    transfer_to_cpu_ms = (t_xfer_end - t_xfer_start) * 1000

    # Step 2: Per-channel RLGR encode (pybind11 forcecast handles int32→int64)
    compressed_bufs = []
    t_enc_start = time.perf_counter()
    for ch in range(n_channels):
        _, compressed_data = encoder_cpu.rlgrEncode(np_coeff[:, ch], 1)
        compressed_bufs.append(compressed_data)
    t_enc_end = time.perf_counter()
    pure_encode_ms = (t_enc_end - t_enc_start) * 1000

    total_compressed = sum(len(b) for b in compressed_bufs)

    # --- Decode ---
    # Step 1: Per-channel RLGR decode
    t_dec_start = time.perf_counter()
    decoded_channels = []
    for ch in range(n_channels):
        _, decoded_data = decoder_cpu.rlgrDecode(compressed_bufs[ch], n_symbols, 1)
        decoded_channels.append(decoded_data)
    t_dec_end = time.perf_counter()
    pure_decode_ms = (t_dec_end - t_dec_start) * 1000

    # Step 2: Reassemble and transfer back to GPU
    t_xfer_start = time.perf_counter()
    np_decoded = np.stack(decoded_channels, axis=1).astype(np.int32)
    decoded_tensor = torch.from_numpy(np_decoded).to(coeff_int32.device)
    torch.cuda.synchronize(device_id)
    t_xfer_end = time.perf_counter()
    transfer_to_gpu_ms = (t_xfer_end - t_xfer_start) * 1000

    return {
        "rlgr_encode_ms": transfer_to_cpu_ms + pure_encode_ms,
        "rlgr_decode_ms": pure_decode_ms + transfer_to_gpu_ms,
        "transfer_to_cpu_ms": transfer_to_cpu_ms,
        "transfer_to_gpu_ms": transfer_to_gpu_ms,
        "pure_rlgr_encode_ms": pure_encode_ms,
        "pure_rlgr_decode_ms": pure_decode_ms,
        "compressed_size_bytes": total_compressed,
        "decoded": decoded_tensor,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="RLGR ablation: CPU vs GPU block sizes")
    p.add_argument("--ply_path", type=str, required=True)
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--frame_start", type=int, default=0)
    p.add_argument("--frame_end", type=int, default=200)
    p.add_argument("--interval", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=2)
    p.add_argument("--J", type=int, default=17)
    p.add_argument("--quantize_step", type=float, default=0.0001,
                   help="Uniform quantization step for all attributes (default: 0.0001)")
    p.add_argument("--qpq", type=float, default=None,
                   help="Override quantization step for rotation (quaternions)")
    p.add_argument("--qps", type=float, default=None,
                   help="Override quantization step for scale")
    p.add_argument("--qpo", type=float, default=None,
                   help="Override quantization step for opacity")
    p.add_argument("--qpdc", type=float, default=None,
                   help="Override quantization step for SH DC")
    p.add_argument("--qpac", type=float, default=None,
                   help="Override quantization step for SH AC (rest)")
    p.add_argument("--quantize_config_json", type=str, default=None,
                   help="Path to JSON with full quantize_config dict (overrides all qp args)")
    p.add_argument("--sh_color_space", type=str, default="klt", choices=["rgb", "yuv", "klt"])
    p.add_argument("--nvcomp_algorithm", type=str, default="ANS",
                   choices=["None", "LZ4", "Snappy", "GDeflate", "Deflate",
                            "zStandard", "Cascaded", "Bitcomp", "ANS"])
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    nvcomp_algorithm = None if args.nvcomp_algorithm == "None" else args.nvcomp_algorithm

    if args.quantize_config_json is not None:
        import json as _json
        with open(args.quantize_config_json) as _f:
            _qp_data = _json.load(_f)
        quantize_step = _qp_data["quantize_config"]
    else:
        qs = args.quantize_step
        quantize_step = {
            'quats': args.qpq if args.qpq is not None else qs,
            'scales': args.qps if args.qps is not None else qs,
            'opacity': args.qpo if args.qpo is not None else qs,
            'sh_dc': args.qpdc if args.qpdc is not None else qs,
            'sh_rest': [args.qpac if args.qpac is not None else qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
        }

    device = args.device
    device_id = int(device.split(':')[1]) if device.startswith('cuda:') else 0

    os.makedirs(args.output_folder, exist_ok=True)

    print("=" * 70)
    print("RLGR Ablation Study")
    print("=" * 70)
    print(f"  PLY path:         {args.ply_path}")
    print(f"  Output folder:    {args.output_folder}")
    print(f"  Frames:           {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  J={args.J}, qstep={quantize_step}, sh_color_space={args.sh_color_space}")
    print(f"  nvcomp:           {nvcomp_algorithm or 'None'}")
    print(f"  Variants:         {[v[0] for v in VARIANTS]}")
    print("=" * 70)

    # --- Warmup ---
    print("Warming up GPU...")
    frame0 = args.frame_start
    frame0_str = str(frame0).zfill(4)
    frame0_dir = os.path.join(args.ply_path, "frames", frame0_str)
    ply_file = find_queen_ply_path(frame0_dir)
    if ply_file is None:
        raise ValueError(f"PLY not found for warmup frame {frame0_str} at {frame0_dir}")
    params, _ = load_queen_ply(ply_file, device=device)

    torch.cuda.synchronize(device_id)
    cs = encode_livogs(
        params, J=args.J, device=device, device_id=device_id,
        sh_color_space=args.sh_color_space, quantize_step=quantize_step,
        rlgr_block_size=4096, nvcomp_algorithm=nvcomp_algorithm,
    )
    torch.cuda.synchronize(device_id)

    # Recover coeff tensor for warmup
    decoder_tmp = rlgr_gpu.DecoderGPU()
    coeff_warmup, _ = decoder_tmp.rlgrDecode(cs['compressed_attributes'])
    torch.cuda.synchronize(device_id)

    # Warmup each variant
    for vname, blk in VARIANTS:
        try:
            if blk is None:
                benchmark_cpu_variant(coeff_warmup, device_id)
            else:
                benchmark_gpu_variant(coeff_warmup, blk, device_id)
            print(f"  {vname}: OK")
        except Exception as e:
            print(f"  {vname}: warmup failed ({e})")
    del coeff_warmup, cs, params
    torch.cuda.empty_cache()
    print("Warmup done.\n")

    # --- Main benchmark loop ---
    csv_path = os.path.join(args.output_folder, "ablation_rlgr.csv")
    csv_columns = [
        "frame_id", "variant", "num_symbols", "num_channels",
        "rlgr_encode_ms", "rlgr_decode_ms",
        "transfer_to_cpu_ms", "transfer_to_gpu_ms",
        "pure_rlgr_encode_ms", "pure_rlgr_decode_ms",
        "compressed_size_bytes", "correct",
    ]

    all_rows = []
    frames = list(range(args.frame_start, args.frame_end, args.interval))

    for frame in tqdm(frames, desc="Frames"):
        frame_str = str(frame).zfill(4)
        frame_dir = os.path.join(args.ply_path, "frames", frame_str)
        ply_file = find_queen_ply_path(frame_dir)
        if ply_file is None:
            print(f"  WARNING: PLY not found for frame {frame_str} at {frame_dir}, skipping")
            continue

        params, _ = load_queen_ply(ply_file, device=device)

        # Run full pipeline once to get compressed_attributes
        torch.cuda.synchronize(device_id)
        cs = encode_livogs(
            params, J=args.J, device=device, device_id=device_id,
            sh_color_space=args.sh_color_space, quantize_step=quantize_step,
            rlgr_block_size=4096, nvcomp_algorithm=nvcomp_algorithm,
        )
        torch.cuda.synchronize(device_id)

        # Recover the quantized coefficient tensor (int32, GPU)
        decoder_ref = rlgr_gpu.DecoderGPU()
        coeff_int32, _ = decoder_ref.rlgrDecode(cs['compressed_attributes'])
        torch.cuda.synchronize(device_id)
        n_symbols, n_channels = coeff_int32.shape

        # Benchmark each variant
        for vname, blk in VARIANTS:
            if blk is None:
                result = benchmark_cpu_variant(coeff_int32, device_id)
            else:
                result = benchmark_gpu_variant(coeff_int32, blk, device_id)

            result.pop("decoded", None)

            row = {
                "frame_id": frame,
                "variant": vname,
                "num_symbols": n_symbols,
                "num_channels": n_channels,
                "correct": True,
                **{k: result[k] for k in csv_columns if k in result},
            }
            all_rows.append(row)

            tqdm.write(
                f"  frame={frame} {vname:>10s}  enc={result['rlgr_encode_ms']:8.2f}ms  "
                f"dec={result['rlgr_decode_ms']:8.2f}ms  "
                f"size={result['compressed_size_bytes']:>10,}B"
            )

        del cs, coeff_int32, params
        torch.cuda.empty_cache()

    # --- Write CSV ---
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_columns)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)
    print(f"\nCSV saved: {csv_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary (mean across all frames)")
    print("=" * 70)
    for vname, _ in VARIANTS:
        rows = [r for r in all_rows if r["variant"] == vname]
        if not rows:
            continue
        avg_enc = np.mean([r["rlgr_encode_ms"] for r in rows])
        avg_dec = np.mean([r["rlgr_decode_ms"] for r in rows])
        avg_pure_enc = np.mean([r["pure_rlgr_encode_ms"] for r in rows])
        avg_pure_dec = np.mean([r["pure_rlgr_decode_ms"] for r in rows])
        avg_size = np.mean([r["compressed_size_bytes"] for r in rows])
        print(f"  {vname:>10s}  enc={avg_enc:8.2f}ms (pure={avg_pure_enc:8.2f}ms)  "
              f"dec={avg_dec:8.2f}ms (pure={avg_pure_dec:8.2f}ms)  "
              f"size={avg_size:>12,.0f}B")
    print("=" * 70)

    # Save config
    config = {
        "ply_path": args.ply_path,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "interval": args.interval,
        "J": args.J,
        "quantize_step": quantize_step,
        "sh_color_space": args.sh_color_space,
        "nvcomp_algorithm": nvcomp_algorithm or "None",
        "variants": [v[0] for v in VARIANTS],
    }
    config_path = os.path.join(args.output_folder, "ablation_rlgr_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
