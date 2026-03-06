"""
Plot VideoGS compressed size breakdown and point count per frame.
Requires: benchmark_videogs_pipeline.csv under input_folder/<output_tag>/

Generates two plots:
  1. Size: uncompressed vs compressed (MP4)
  2. Point count: original_points per frame
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt

def build_output_tag(args):
    return f"qp_{args.qp}"


def load_pipeline_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "frame": int(row["frame_id"]),
                "uncompressed_size_bytes": int(row["uncompressed_size_bytes"]),
                "compressed_size_bytes": int(row["compressed_size_bytes"]),
                "original_points": int(row["original_points"]),
            })
    return sorted(rows, key=lambda x: x["frame"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Base folder for videogs_compression")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sequence_name", type=str, required=True)
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder for plot PNG")
    args = parser.parse_args()

    output_tag = build_output_tag(args)
    config_dir = os.path.join(args.input_folder, output_tag)
    csv_path = os.path.join(config_dir, "benchmark_videogs_pipeline.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, output_tag)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise SystemExit(f"Required file not found: {csv_path}")

    rows = load_pipeline_csv(csv_path)
    n = len(rows)
    frame_ids = [r["frame"] for r in rows]
    x = range(n)
    tick_every = 10

    uncompressed_mb = [r["uncompressed_size_bytes"] / 1024 / 1024 for r in rows]
    compressed_mb = [r["compressed_size_bytes"] / 1024 / 1024 for r in rows]

    # ---- Plot 1: Size breakdown ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, uncompressed_mb, "o-", label="Uncompressed", color="gray", markersize=2, alpha=0.7)
    ax.plot(x, compressed_mb, "s-", label="Compressed (MP4)", color="coral", markersize=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"VideoGS size per frame [{output_tag}]\n{args.dataset_name}/{args.sequence_name}")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    avg_unc = sum(uncompressed_mb) / n
    avg_comp = sum(compressed_mb) / n
    ratio = avg_unc / avg_comp if avg_comp > 0 else 0
    ax.annotate(
        f"avg uncompressed={avg_unc:.2f} MB, compressed={avg_comp:.2f} MB ({ratio:.1f}x)",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    size_path = os.path.join(out_dir, "compressed_size.png")
    fig.savefig(size_path, dpi=150)
    print(f"Saved: {size_path}")
    plt.close(fig)

    # ---- Plot 2: Point counts ----
    orig_pts = [r["original_points"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, orig_pts, "o-", label="Original points", color="steelblue", markersize=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Number of Gaussians")
    ax.set_title(f"VideoGS point count per frame\n{args.dataset_name}/{args.sequence_name}")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    avg_orig = sum(orig_pts) / n
    ax.annotate(
        f"avg original points = {avg_orig:.0f}",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    pts_path = os.path.join(out_dir, "point_counts.png")
    fig.savefig(pts_path, dpi=150)
    print(f"Saved: {pts_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
