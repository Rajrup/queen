"""
Plot VideoGS compressed size breakdown and point count per frame.
Requires: benchmark_compress_to_png.csv, benchmark_compress_png_2_video.csv under input_folder/qp_<qp>/

Generates two plots:
  1. Size: uncompressed, PNG compressed, MP4 (avg/frame)
  2. Point count: original_points per frame
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt


def load_png_csv(path):
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


def load_video_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "group_id": int(row["group_id"]),
                "group_compressed_size_bytes": int(row["group_compressed_size_bytes"]),
                "num_frames": int(row["num_frames"]),
            })
    return sorted(rows, key=lambda x: x["group_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Base folder for videogs_compression")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g. Neural_3D_Video)")
    parser.add_argument("--sequence_name", type=str, required=True, help="Sequence name (e.g. cook_spinach)")
    parser.add_argument("--qp", type=int, required=True, help="QP value (e.g. 22)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder for plot PNG")
    args = parser.parse_args()

    qp_dir = os.path.join(args.input_folder, f"qp_{args.qp}")
    png_csv = os.path.join(qp_dir, "compressed_png", "benchmark_compress_to_png.csv")
    video_csv = os.path.join(qp_dir, "compressed_video", "benchmark_compress_png_2_video.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, f"qp_{args.qp}")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(png_csv):
        raise SystemExit(f"Required file not found: {png_csv}")
    if not os.path.isfile(video_csv):
        raise SystemExit(f"Required file not found: {video_csv}")

    png_rows = load_png_csv(png_csv)
    video_rows = load_video_csv(video_csv)
    if not video_rows:
        raise SystemExit("No video benchmark rows")

    frame_to_group = []
    for row in video_rows:
        for _ in range(row["num_frames"]):
            frame_to_group.append(row["group_id"])

    frame_ids = [r["frame"] for r in png_rows]
    n = len(frame_ids)
    x = range(n)
    tick_every = 10

    uncompressed_mb = [r["uncompressed_size_bytes"] / 1024 / 1024 for r in png_rows]
    png_sizes_mb = [r["compressed_size_bytes"] / 1024 / 1024 for r in png_rows]
    video_sizes_mb = []
    for i, r in enumerate(png_rows):
        if i < len(frame_to_group):
            gid = frame_to_group[i]
            vr = next(x for x in video_rows if x["group_id"] == gid)
            video_sizes_mb.append(vr["group_compressed_size_bytes"] / vr["num_frames"] / 1024 / 1024)
        else:
            video_sizes_mb.append(0)

    # ---- Plot 1: Size breakdown ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, uncompressed_mb, "o-", label="Uncompressed", color="gray", markersize=2, alpha=0.7)
    ax.plot(x, png_sizes_mb, "s-", label="PNG compressed", color="steelblue", markersize=2)
    ax.plot(x, video_sizes_mb, "^-", label="MP4 (avg/frame)", color="coral", markersize=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"VideoGS size per frame [QP={args.qp}, {args.dataset_name}, {args.sequence_name}]")
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(x)[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    avg_unc = sum(uncompressed_mb) / n
    avg_png = sum(png_sizes_mb) / n
    avg_mp4 = sum(video_sizes_mb) / n
    ax.annotate(
        f"avg uncompressed={avg_unc:.2f} MB, PNG={avg_png:.2f} MB, MP4={avg_mp4:.2f} MB/frame",
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
    orig_pts = [r["original_points"] for r in png_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, orig_pts, "o-", label="Original points", color="steelblue", markersize=2)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Number of Gaussians")
    ax.set_title(f"VideoGS point count per frame [QP={args.qp}, {args.dataset_name}, {args.sequence_name}]")
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
