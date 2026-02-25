"""
Stacked area plot of compression and decompression time per frame.
Order from bottom to top: Compress PNG encode, Compress H.264 encode, Decompress H.264 decode, Decompress PNG decode.
Requires same CSVs as plot_compression_time.py under input_folder/qp_<qp>/.
"""
import os
import argparse
import csv
import matplotlib.pyplot as plt


def load_png_compress_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"frame": int(row["frame_id"]), "time_ms": float(row["time_ms"])})
    return sorted(rows, key=lambda x: x["frame"])

def load_video_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "group_id": int(row["group_id"]),
                "time_ms": float(row["time_ms"]),
                "num_frames": int(row["num_frames"]),
            })
    return sorted(rows, key=lambda x: x["group_id"])

def load_png_decompress_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"frame": int(row["frame_id"]), "time_ms": float(row["time_ms"])})
    return sorted(rows, key=lambda x: x["frame"])

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
    png_compress_csv = os.path.join(qp_dir, "compressed_png", "benchmark_compress_to_png.csv")
    video_compress_csv = os.path.join(qp_dir, "compressed_video", "benchmark_compress_png_2_video.csv")
    video_decompress_csv = os.path.join(qp_dir, "decompressed_png", "benchmark_decompress_video_2_png.csv")
    png_decompress_csv = os.path.join(qp_dir, "decompressed_ply", "benchmark_decompress_from_png.csv")
    out_dir = os.path.join(args.output_folder, "plots", args.dataset_name, args.sequence_name, f"qp_{args.qp}")
    out_path = os.path.join(out_dir, "compression_time.png")
    os.makedirs(out_dir, exist_ok=True)

    for p in [png_compress_csv, video_compress_csv, video_decompress_csv, png_decompress_csv]:
        if not os.path.isfile(p):
            raise SystemExit(f"Required file not found: {p}")

    png_rows = load_png_compress_csv(png_compress_csv)
    video_compress_rows = load_video_csv(video_compress_csv)
    video_decompress_rows = load_video_csv(video_decompress_csv)
    png_decompress_rows = load_png_decompress_csv(png_decompress_csv)

    frame_to_group = []
    for row in video_compress_rows:
        for _ in range(row["num_frames"]):
            frame_to_group.append(row["group_id"])

    frame_ids = [r["frame"] for r in png_rows]
    n = len(frame_ids)
    compress_png = [r["time_ms"] for r in png_rows]
    compress_video = []
    for r in png_rows:
        fid = r["frame"]
        if fid < len(frame_to_group):
            gid = frame_to_group[fid]
            vr = next(x for x in video_compress_rows if x["group_id"] == gid)
            compress_video.append(vr["time_ms"] / vr["num_frames"])
        else:
            compress_video.append(0)

    frame_to_group_dec = []
    for row in video_decompress_rows:
        for _ in range(row["num_frames"]):
            frame_to_group_dec.append(row["group_id"])
    decompress_video = []
    for r in png_rows:
        fid = r["frame"]
        if fid < len(frame_to_group_dec):
            gid = frame_to_group_dec[fid]
            vr = next(x for x in video_decompress_rows if x["group_id"] == gid)
            decompress_video.append(vr["time_ms"] / vr["num_frames"])
        else:
            decompress_video.append(0)

    png_decompress_by_frame = {r["frame"]: r["time_ms"] for r in png_decompress_rows}
    decompress_png_to_ply = [png_decompress_by_frame.get(fid, 0) for fid in frame_ids]

    # Stacked area: bottom to top = Compress PNG encode, Compress H.264 encode, Decompress H.264 decode, Decompress PNG decode
    x = list(range(n))
    labels = [
        "Compress: PNG encode",
        "Compress: H.264 encode",
        "Decompress: H.264 decode",
        "Decompress: PNG decode",
    ]
    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        x,
        compress_png,
        compress_video,
        decompress_video,
        decompress_png_to_ply,
        labels=labels,
        colors=colors,
        alpha=0.8,
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time (ms)")
    tick_every = 10
    ax.set_xticks(x[::tick_every])
    ax.set_xticklabels(frame_ids[::tick_every], rotation=90)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"VideoGS compression and decompression time per frame [QP={args.qp}, {args.dataset_name}, {args.sequence_name}]")

    avg_cpng = sum(compress_png) / n
    avg_cvid = sum(compress_video) / n
    avg_dvid = sum(decompress_video) / n
    avg_dply = sum(decompress_png_to_ply) / n
    avg_total = avg_cpng + avg_cvid + avg_dvid + avg_dply
    ax.annotate(
        f"avg/frame: PNG enc={avg_cpng:.1f}, H.264 enc={avg_cvid:.1f}, "
        f"H.264 dec={avg_dvid:.1f}, PNG dec={avg_dply:.1f}, total={avg_total:.1f}ms",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
