#!/usr/bin/env python3
"""Publication-quality plots for the latency & resource benchmark (QUEEN / Neural_3D_Video).

Generates:
  1. Per-sequence stacked latency comparison (1x4 subplots, log-scale Y)
  2. Summary 2x2 grouped bar chart (encode, decode log-scale; CPU/GPU util 0-100%)
  3. Per-sequence time-varying CPU & GPU utilization (1x4 subplots per metric)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "Neural_3D_Video"

PIPELINES = ["livogs", "videogs", "dracogs", "mesongs"]
PIPELINE_LABELS = {
    "livogs": "LiVoGS",
    "videogs": "VideoGS",
    "dracogs": "DracoGS",
    "mesongs": "MesonGS",
}
PIPELINE_COLORS = {
    "livogs": "#1f77b4",
    "videogs": "#ff7f0e",
    "dracogs": "#2ca02c",
    "mesongs": "#d62728",
}
PIPELINE_HATCHES = {
    "livogs": "/",
    "videogs": "\\",
    "dracogs": "x",
    "mesongs": ".",
}

SEQUENCES = [
    "coffee_martini",
    "cook_spinach",
    "cut_roasted_beef",
    "flame_salmon_1",
    "flame_steak",
    "sear_steak",
]

SEQ_ABBREV = {
    "coffee_martini": "Coffee",
    "cook_spinach": "Spinach",
    "cut_roasted_beef": "Beef",
    "flame_salmon_1": "Salmon",
    "flame_steak": "F-Steak",
    "sear_steak": "S-Steak",
}

BENCHMARK_CSV_NAMES = {
    "livogs": "benchmark_livogs.csv",
    "videogs": "benchmark_videogs_pipeline.csv",
    "dracogs": "benchmark_dracogs.csv",
    "mesongs": "benchmark_mesongs.csv",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_benchmark_csv(path: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame": int(row["frame_id"]),
                "encode_ms": float(row["total_encode_ms"]),
                "decode_ms": float(row["total_decode_ms"]),
            })
    return sorted(rows, key=lambda r: r["frame"])


def load_summary_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_resource_timeseries(path: str) -> dict[str, list[float]]:
    """Load elapsed_sec, cpu_pct, gpu_util_pct from resource_timeseries.csv."""
    data: dict[str, list[float]] = {"elapsed_sec": [], "cpu_pct": [], "gpu_util_pct": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["elapsed_sec"].append(float(row["elapsed_sec"]))
            data["cpu_pct"].append(float(row["cpu_pct"]))
            data["gpu_util_pct"].append(float(row["gpu_util_pct"]))
    return data


def get_benchmark_dir(data_path: str, seq: str, pipeline: str) -> str:
    return str(
        Path(data_path) / "pretrained_output" / DATASET_NAME
        / f"queen_compressed_{seq}" / "latency_benchmark" / pipeline
    )


# ---------------------------------------------------------------------------
# Figure 1: Per-sequence stacked latency comparison (log-scale Y)
# ---------------------------------------------------------------------------
def plot_per_sequence(
    data_path: str,
    sequences: list[str],
    pipelines: list[str],
    output_folder: str,
    fmt: str,
) -> None:
    for seq in sequences:
        fig, axes = plt.subplots(1, len(pipelines), figsize=(5 * len(pipelines), 5),
                                 sharey=True)
        if len(pipelines) == 1:
            axes = [axes]

        plot_data: list[tuple[str, list[dict[str, float]]]] = []

        for pipeline in pipelines:
            bench_dir = get_benchmark_dir(data_path, seq, pipeline)
            csv_name = BENCHMARK_CSV_NAMES[pipeline]
            csv_path = os.path.join(bench_dir, csv_name)
            if not os.path.isfile(csv_path):
                print(f"  WARNING: Missing {csv_path}, skipping {pipeline}/{seq}")
                plot_data.append((pipeline, []))
                continue
            rows = load_benchmark_csv(csv_path)
            plot_data.append((pipeline, rows))

        for ax, (pipeline, rows) in zip(axes, plot_data):
            label = PIPELINE_LABELS[pipeline]
            if not rows:
                ax.set_title(f"{label}\n(no data)", fontsize=12)
                ax.set_xlabel("Frame", fontsize=11)
                continue

            n = len(rows)
            frame_ids = [int(r["frame"]) for r in rows]
            encode_ms = np.array([r["encode_ms"] for r in rows])
            decode_ms = np.array([r["decode_ms"] for r in rows])

            ax.plot(frame_ids, encode_ms, marker="o", markersize=3,
                    linewidth=1.2, label="Encode", color="#4C8BF5")
            ax.plot(frame_ids, decode_ms, marker="s", markersize=3,
                    linewidth=1.2, label="Decode", color="#F5794C")

            ax.set_yscale("log")

            tick_every = max(1, n // 20)
            ax.set_xticks(frame_ids[::tick_every])
            ax.set_xticklabels([str(f) for f in frame_ids[::tick_every]],
                               rotation=90, fontsize=8)
            ax.set_xlabel("Frame", fontsize=11)
            ax.grid(True, alpha=0.3, which="both")

            avg_enc = float(np.mean(encode_ms))
            avg_dec = float(np.mean(decode_ms))
            ax.set_title(f"{label}", fontsize=12)
            ax.annotate(
                f"avg enc={avg_enc:.1f}ms\navg dec={avg_dec:.1f}ms\ntotal={avg_enc + avg_dec:.1f}ms",
                xy=(0.03, 0.97), xycoords="axes fraction",
                fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            )

        axes[0].set_ylabel("Time (ms)", fontsize=11)
        axes[0].legend(loc="upper right", fontsize=9)

        fig.suptitle(f"Codec Latency Per Frame — {seq}", fontsize=14, y=1.02)
        fig.tight_layout()

        out_path = os.path.join(output_folder, f"codec_latency_{seq}.{fmt}")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Summary 2x2 grouped bar chart
#   - Latency subplots use log scale
#   - Utilization subplots subtract baseline and clamp to 0-100%
# ---------------------------------------------------------------------------
def plot_summary(
    data_path: str,
    sequences: list[str],
    pipelines: list[str],
    output_folder: str,
    fmt: str,
) -> None:
    # Per-sequence mean for bar chart
    encode_data: dict[str, list[float]] = {p: [] for p in pipelines}
    encode_std: dict[str, list[float]] = {p: [] for p in pipelines}
    decode_data: dict[str, list[float]] = {p: [] for p in pipelines}
    decode_std: dict[str, list[float]] = {p: [] for p in pipelines}
    cpu_data: dict[str, list[float]] = {p: [] for p in pipelines}
    gpu_data: dict[str, list[float]] = {p: [] for p in pipelines}
    ram_data: dict[str, list[float]] = {p: [] for p in pipelines}
    gpu_mem_data: dict[str, list[float]] = {p: [] for p in pipelines}

    # All per-frame values pooled across sequences (for stats box)
    all_enc_frames: dict[str, list[float]] = {p: [] for p in pipelines}
    all_dec_frames: dict[str, list[float]] = {p: [] for p in pipelines}

    valid_sequences: list[str] = []

    for seq in sequences:
        has_any = False
        for pipeline in pipelines:
            bench_dir = get_benchmark_dir(data_path, seq, pipeline)
            csv_name = BENCHMARK_CSV_NAMES[pipeline]
            csv_path = os.path.join(bench_dir, csv_name)
            summary_path = os.path.join(bench_dir, "summary.json")

            if os.path.isfile(csv_path):
                rows = load_benchmark_csv(csv_path)
                enc_vals = [r["encode_ms"] for r in rows]
                dec_vals = [r["decode_ms"] for r in rows]
                encode_data[pipeline].append(np.mean(enc_vals) if enc_vals else 0)
                encode_std[pipeline].append(np.std(enc_vals) if enc_vals else 0)
                decode_data[pipeline].append(np.mean(dec_vals) if dec_vals else 0)
                decode_std[pipeline].append(np.std(dec_vals) if dec_vals else 0)
                all_enc_frames[pipeline].extend(enc_vals)
                all_dec_frames[pipeline].extend(dec_vals)
                has_any = True
            else:
                encode_data[pipeline].append(0)
                encode_std[pipeline].append(0)
                decode_data[pipeline].append(0)
                decode_std[pipeline].append(0)

            if os.path.isfile(summary_path):
                summary = load_summary_json(summary_path)
                metrics = summary.get("metrics", {})
                baseline = summary.get("baseline", {})
                cpu_val = max(0.0, metrics.get("avg_cpu_pct", 0) - baseline.get("avg_cpu_pct", 0))
                gpu_val = max(0.0, metrics.get("peak_gpu_util_pct", 0) - baseline.get("peak_gpu_util_pct", 0))
                ram_val = max(0.0, metrics.get("avg_ram_used_mb", 0) - baseline.get("avg_ram_used_mb", 0))
                gpu_mem_val = max(0.0, metrics.get("avg_gpu_mem_used_mb", 0) - baseline.get("avg_gpu_mem_used_mb", 0))
                cpu_data[pipeline].append(cpu_val)
                gpu_data[pipeline].append(gpu_val)
                ram_data[pipeline].append(ram_val)
                gpu_mem_data[pipeline].append(gpu_mem_val)
            else:
                cpu_data[pipeline].append(0)
                gpu_data[pipeline].append(0)
                ram_data[pipeline].append(0)
                gpu_mem_data[pipeline].append(0)

        if has_any:
            valid_sequences.append(seq)

    if not valid_sequences:
        print("  WARNING: No data found for summary plot")
        return

    for pipeline in pipelines:
        enc_vals = [v for v in encode_data[pipeline] if v > 0]
        dec_vals = [v for v in decode_data[pipeline] if v > 0]
        cpu_vals = cpu_data[pipeline][:]
        gpu_vals = gpu_data[pipeline][:]
        ram_vals = ram_data[pipeline][:]
        gpu_mem_vals = gpu_mem_data[pipeline][:]
        encode_data[pipeline].append(np.mean(enc_vals) if enc_vals else 0)
        encode_std[pipeline].append(np.mean([v for v in encode_std[pipeline] if v > 0]) if enc_vals else 0)
        decode_data[pipeline].append(np.mean(dec_vals) if dec_vals else 0)
        decode_std[pipeline].append(np.mean([v for v in decode_std[pipeline] if v > 0]) if dec_vals else 0)
        cpu_data[pipeline].append(np.mean(cpu_vals) if cpu_vals else 0)
        gpu_data[pipeline].append(np.mean(gpu_vals) if gpu_vals else 0)
        ram_data[pipeline].append(np.mean(ram_vals) if ram_vals else 0)
        gpu_mem_data[pipeline].append(np.mean(gpu_mem_vals) if gpu_mem_vals else 0)

    seq_labels = [SEQ_ABBREV.get(s, s) for s in sequences] + ["Average"]
    n_groups = len(seq_labels)
    n_bars = len(pipelines)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    def _draw_grouped_bars(
        ax: plt.Axes,
        title: str,
        data: dict[str, list[float]],
        std_data: dict[str, list[float]] | None,
        scale_mode: str,
        stats_pool: dict[str, list[float]] | None = None,
    ) -> None:
        for i, pipeline in enumerate(pipelines):
            offset = (i - n_bars / 2 + 0.5) * bar_width
            vals = data[pipeline]
            kw: dict[str, Any] = dict(
                width=bar_width,
                label=PIPELINE_LABELS[pipeline],
                color=PIPELINE_COLORS[pipeline],
                hatch=PIPELINE_HATCHES[pipeline],
                edgecolor="white",
                alpha=0.85,
            )
            if std_data is not None:
                kw["yerr"] = std_data[pipeline]
                kw["capsize"] = 2
            ax.bar(x + offset, vals, **kw)

        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_labels, rotation=45, ha="right", fontsize=10)
        if scale_mode == "log":
            ax.set_yscale("log")
            ax.grid(True, axis="y", alpha=0.3, which="both")
        elif scale_mode == "pct":
            ax.set_ylim(0, 100)
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.set_ylim(bottom=0)
            ax.grid(True, axis="y", alpha=0.3)

        if stats_pool is not None:
            lines = []
            for p in pipelines:
                arr = np.array(stats_pool[p]) if stats_pool[p] else np.array([0.0])
                lbl = PIPELINE_LABELS[p]
                lines.append(f"{lbl}: avg={np.mean(arr):.1f}  med={np.median(arr):.1f}  std={np.std(arr):.1f}")
            ax.annotate(
                "\n".join(lines),
                xy=(0.98, 0.97), xycoords="axes fraction",
                fontsize=7.5, va="top", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
            )

    def _save_fig(fig: plt.Figure, filename: str) -> None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=n_bars,
                   fontsize=11, bbox_to_anchor=(0.5, 1.03))
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(output_folder, f"{filename}.{fmt}")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # --- Plot 1: Latency (1x2) ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 5))
    _draw_grouped_bars(axes1[0], "Mean Encode Latency (ms)",
                       encode_data, encode_std, "log", all_enc_frames)
    _draw_grouped_bars(axes1[1], "Mean Decode Latency (ms)",
                       decode_data, decode_std, "log", all_dec_frames)
    _save_fig(fig1, "summary_latency")

    # --- Plot 2: CPU & GPU Utilization (1x2) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    _draw_grouped_bars(axes2[0], "Avg CPU Utilization — baseline (%)",
                       cpu_data, None, "pct")
    _draw_grouped_bars(axes2[1], "Peak GPU Utilization — baseline (%)",
                       gpu_data, None, "pct")
    _save_fig(fig2, "summary_utilization")

    # --- Plot 3: RAM & GPU Memory (1x2) ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5))
    _draw_grouped_bars(axes3[0], "Avg RAM Usage — baseline (MB)",
                       ram_data, None, "auto")
    _draw_grouped_bars(axes3[1], "Avg GPU Memory — baseline (MB)",
                       gpu_mem_data, None, "auto")
    _save_fig(fig3, "summary_memory")

    # --- Plot 4: P95 CPU & GPU Utilization (computed from timeseries) ---
    cpu_p95_data: dict[str, list[float]] = {p: [] for p in pipelines}
    gpu_p95_data: dict[str, list[float]] = {p: [] for p in pipelines}
    for seq in sequences:
        for pipeline in pipelines:
            bench_dir = get_benchmark_dir(data_path, seq, pipeline)
            ts_path = os.path.join(bench_dir, "resource_timeseries.csv")
            summary_path = os.path.join(bench_dir, "summary.json")
            if os.path.isfile(ts_path) and os.path.isfile(summary_path):
                ts = load_resource_timeseries(ts_path)
                bl = load_summary_json(summary_path).get("baseline", {})
                cpu_arr = np.array(ts["cpu_pct"])
                gpu_arr = np.array(ts["gpu_util_pct"])
                cpu_p95_data[pipeline].append(max(0.0, float(np.percentile(cpu_arr, 95)) - bl.get("avg_cpu_pct", 0)))
                gpu_p95_data[pipeline].append(max(0.0, float(np.percentile(gpu_arr, 95)) - bl.get("avg_gpu_util_pct", 0)))
            else:
                cpu_p95_data[pipeline].append(0)
                gpu_p95_data[pipeline].append(0)
    for pipeline in pipelines:
        cpu_p95_data[pipeline].append(np.mean(cpu_p95_data[pipeline]) if cpu_p95_data[pipeline] else 0)
        gpu_p95_data[pipeline].append(np.mean(gpu_p95_data[pipeline]) if gpu_p95_data[pipeline] else 0)

    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 5))
    _draw_grouped_bars(axes4[0], "P95 CPU Utilization — baseline (%)",
                       cpu_p95_data, None, "pct")
    _draw_grouped_bars(axes4[1], "P95 GPU Utilization — baseline (%)",
                       gpu_p95_data, None, "pct")
    _save_fig(fig4, "summary_utilization_p95")


# ---------------------------------------------------------------------------
# Figure 3: Per-sequence time-varying CPU & GPU utilization
# ---------------------------------------------------------------------------
def plot_utilization_timeseries(
    data_path: str,
    sequences: list[str],
    pipelines: list[str],
    output_folder: str,
    fmt: str,
) -> None:
    metrics = [
        ("cpu_pct", "CPU Utilization (%)"),
        ("gpu_util_pct", "GPU Utilization (%)"),
    ]

    for seq in sequences:
        n_rows = len(metrics)
        n_cols = len(pipelines)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows),
                                 sharey=True, squeeze=False)

        for row, (metric_key, metric_label) in enumerate(metrics):
            for col, pipeline in enumerate(pipelines):
                ax = axes[row][col]
                label = PIPELINE_LABELS[pipeline]
                bench_dir = get_benchmark_dir(data_path, seq, pipeline)
                ts_path = os.path.join(bench_dir, "resource_timeseries.csv")

                if not os.path.isfile(ts_path):
                    ax.set_title(f"{label}\n(no data)", fontsize=12)
                    ax.set_xlabel("Time (s)", fontsize=11)
                    continue

                ts = load_resource_timeseries(ts_path)
                t = np.array(ts["elapsed_sec"])
                vals = np.array(ts[metric_key])

                ax.fill_between(t, 0, vals, alpha=0.4,
                                color=PIPELINE_COLORS[pipeline])
                ax.plot(t, vals, linewidth=0.5,
                        color=PIPELINE_COLORS[pipeline], alpha=0.8)

                if row == 0:
                    ax.set_title(label, fontsize=12)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                if row == n_rows - 1:
                    ax.set_xlabel("Time (s)", fontsize=11)

            axes[row][0].set_ylabel(metric_label, fontsize=11)

        fig.suptitle(f"Resource Utilization Over Time — {seq}", fontsize=14, y=1.02)
        fig.tight_layout()

        out_path = os.path.join(output_folder, f"util_timeseries_{seq}.{fmt}")
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Plot latency benchmark results (QUEEN / Neural_3D_Video)")
    p.add_argument("--data_path", type=str, default="/synology/rajrup/Queen")
    p.add_argument("--sequences", nargs="+", default=SEQUENCES)
    p.add_argument("--output_folder", type=str,
                   default=str(Path(__file__).resolve().parent / "plots" / "latency_benchmark"))
    p.add_argument("--format", type=str, choices=["pdf", "png"], default="pdf")
    args = p.parse_args()

    print(f"Plotting latency benchmark results from: {args.data_path}")
    print(f"Output format: {args.format}")
    print(f"Output folder: {args.output_folder}")
    print()

    print("Generating per-sequence stacked latency figures (log-scale)...")
    plot_per_sequence(args.data_path, args.sequences, PIPELINES, args.output_folder, args.format)

    print("\nGenerating summary comparison figure...")
    plot_summary(args.data_path, args.sequences, PIPELINES, args.output_folder, args.format)

    print("\nGenerating per-sequence utilization time-series figures...")
    plot_utilization_timeseries(args.data_path, args.sequences, PIPELINES, args.output_folder, args.format)

    print("\nDone!")


if __name__ == "__main__":
    main()

'''
conda activate queen
python scripts/plot_latency_benchmark.py
python scripts/plot_latency_benchmark.py --format png
'''
