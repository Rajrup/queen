"""
Stacked area plot of VideoGS encode and decode time per frame.
Requires: benchmark_videogs_pipeline.csv under input_folder/<output_tag>/

Layers (bottom to top): Quantize, H.264 encode, H.264 decode, Dequantize.
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt


def add_qp_args(parser):
    parser.add_argument("--qp", type=int, default=22)
    parser.add_argument("--qfd", type=int, default=22)
    parser.add_argument("--qfr1", type=int, default=22)
    parser.add_argument("--qfr2", type=int, default=22)
    parser.add_argument("--qo", type=int, default=22)
    parser.add_argument("--qs", type=int, default=22)
    parser.add_argument("--qr", type=int, default=22)


def build_output_tag(args):
    return (f"qp_{args.qp}_qfd_{args.qfd}_qfr1_{args.qfr1}_qfr2_{args.qfr2}"
            f"_qo_{args.qo}_qs_{args.qs}_qr_{args.qr}")


def load_pipeline_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "frame": int(row["frame_id"]),
                "quantize_ms": float(row["quantize_ms"]),
                "encode_ms": float(row["encode_ms"]),
                "decode_ms": float(row["decode_ms"]),
                "dequantize_ms": float(row["dequantize_ms"]),
            })
    return sorted(rows, key=lambda x: x["frame"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Base folder for videogs_compression")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sequence_name", type=str, required=True)
    add_qp_args(parser)
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder for plot PNG")
    args = parser.parse_args()

    output_tag = build_output_tag(args)
    config_dir = os.path.join(args.input_folder, output_tag)
    csv_path = os.path.join(config_dir, "benchmark_videogs_pipeline.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, output_tag)
    out_path = os.path.join(out_dir, "compression_time.png")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"Required file not found: {csv_path}")

    rows = load_pipeline_csv(csv_path)
    n = len(rows)
    frame_ids = [r["frame"] for r in rows]

    quantize = [r["quantize_ms"] for r in rows]
    encode = [r["encode_ms"] for r in rows]
    decode = [r["decode_ms"] for r in rows]
    dequantize = [r["dequantize_ms"] for r in rows]

    x = list(range(n))
    tick_every = 10
    labels = ["Quantize", "H.264 encode", "H.264 decode", "Dequantize"]
    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(x, quantize, encode, decode, dequantize,
                 labels=labels, colors=colors, alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, 4000)
    ax.set_xticks(x[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"VideoGS encode + decode time per frame [{output_tag}]\n{args.dataset_name}/{args.sequence_name}")

    avg_q = sum(quantize) / n
    avg_e = sum(encode) / n
    avg_d = sum(decode) / n
    avg_dq = sum(dequantize) / n
    avg_total = avg_q + avg_e + avg_d + avg_dq
    ax.annotate(
        f"avg/frame: quant={avg_q:.1f}, enc={avg_e:.1f}, "
        f"dec={avg_d:.1f}, dequant={avg_dq:.1f}, total={avg_total:.1f}ms",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
