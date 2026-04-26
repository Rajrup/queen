#!/usr/bin/env python3
"""Bar plot comparing LiVoGS decode latency for SH0 vs SH3 across all sequences.

Reads benchmark_livogs.csv from each sequence's latency_benchmark output directory
and produces a grouped bar chart of average decode time (ms) per sequence.

Usage:
    python scripts/plot_livogs_decode_latency.py
    python scripts/plot_livogs_decode_latency.py --data_path /path/to/VideoGS
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASET_NAME = "HiFi4G_Dataset"

SEQUENCES = [
    "4K_Actor1_Greeting",
    "4K_Actor2_Dancing",
    "4K_Actor3_Violin",
    "4K_Actor4_Dancing",
    "4K_Actor5_Oil-paper_Umbrella",
    "4K_Actor6_Changing_Clothes",
    "4K_Actor7_Nunchaku",
]

SEQ_LABELS = {
    "4K_Actor1_Greeting": "A1\nGreeting",
    "4K_Actor2_Dancing": "A2\nDancing",
    "4K_Actor3_Violin": "A3\nViolin",
    "4K_Actor4_Dancing": "A4\nDancing",
    "4K_Actor5_Oil-paper_Umbrella": "A5\nUmbrella",
    "4K_Actor6_Changing_Clothes": "A6\nClothes",
    "4K_Actor7_Nunchaku": "A7\nNunchaku",
}

SH_DEGREES = [0, 3]
SH_COLORS = {0: "#4C72B0", 3: "#DD8452"}
SH_HATCHES = {0: "//", 3: "\\\\"}


def read_decode_times(csv_path: Path) -> list[float]:
    times: list[float] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["total_decode_ms"]))
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LiVoGS decode latency SH0 vs SH3")
    parser.add_argument("--data_path", type=str, default="/home/rajrup/VideoGS",
                        help="Root data path containing train_output/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: scripts/)")
    parser.add_argument("--skip_warmup", type=int, default=1,
                        help="Number of initial frames to skip (warmup)")
    args = parser.parse_args()

    data_root = Path(args.data_path) / "train_output" / DATASET_NAME
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent

    # Collect avg and std decode times per (sequence, sh_degree)
    avg_times: dict[int, list[float]] = {sh: [] for sh in SH_DEGREES}
    std_times: dict[int, list[float]] = {sh: [] for sh in SH_DEGREES}
    valid_sequences: list[str] = []

    for seq in SEQUENCES:
        all_found = True
        for sh in SH_DEGREES:
            csv_path = data_root / seq / "latency_benchmark" / f"livogs_sh_degree_{sh}" / "benchmark_livogs.csv"
            if not csv_path.exists():
                print(f"  WARNING: missing {csv_path}")
                all_found = False
                break
        if not all_found:
            continue

        valid_sequences.append(seq)
        for sh in SH_DEGREES:
            csv_path = data_root / seq / "latency_benchmark" / f"livogs_sh_degree_{sh}" / "benchmark_livogs.csv"
            times = read_decode_times(csv_path)
            times = times[args.skip_warmup:]
            avg_times[sh].append(np.mean(times))
            std_times[sh].append(np.std(times))

    if not valid_sequences:
        print("ERROR: No valid sequence data found.")
        return

    n_seq = len(valid_sequences)
    n_bars = n_seq + 1  # +1 for the "Average" group
    x = np.arange(n_bars)
    bar_width = 0.35

    # Append overall averages and stds
    plot_avg: dict[int, list[float]] = {}
    plot_std: dict[int, list[float]] = {}
    for sh in SH_DEGREES:
        plot_avg[sh] = avg_times[sh] + [float(np.mean(avg_times[sh]))]
        plot_std[sh] = std_times[sh] + [float(np.mean(std_times[sh]))]

    labels = [SEQ_LABELS[s] for s in valid_sequences] + ["Average"]

    fig, ax = plt.subplots(figsize=(max(8, n_bars * 1.4), 5))

    for i, sh in enumerate(SH_DEGREES):
        offset = (i - 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            plot_avg[sh],
            bar_width,
            yerr=plot_std[sh],
            label=f"SH{sh}",
            color=SH_COLORS[sh],
            hatch=SH_HATCHES[sh],
            edgecolor="white",
            linewidth=0.6,
            capsize=3,
            error_kw={"linewidth": 0.8},
            zorder=3,
        )
        # Darker shade for the "Average" bar
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.2)

        for bar, val in zip(bars, plot_avg[sh]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="medium",
            )

    # Vertical separator line before the "Average" group
    ax.axvline(n_seq - 0.5, color="grey", linestyle=":", linewidth=0.8, zorder=1)

    ax.set_ylabel("Decode Latency (ms)", fontsize=11)
    ax.set_xlabel("Sequence", fontsize=11)
    ax.set_title("LiVoGS Decode Latency: SH0 vs SH3 (Jetson Orin)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = output_dir / "livogs_decode_latency_sh0_vs_sh3.pdf"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    out_png = out_path.with_suffix(".png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")

    # Print summary table
    print(f"\n{'Sequence':<35s}  {'SH0 (ms)':>10s}  {'SH3 (ms)':>10s}  {'Δ (ms)':>10s}  {'Ratio':>8s}")
    print("-" * 80)
    for i, seq in enumerate(valid_sequences):
        sh0 = avg_times[0][i]
        sh3 = avg_times[3][i]
        delta = sh3 - sh0
        ratio = sh3 / sh0 if sh0 > 0 else float("inf")
        print(f"{seq:<35s}  {sh0:>10.2f}  {sh3:>10.2f}  {delta:>+10.2f}  {ratio:>7.2f}x")
    print("-" * 80)
    print(f"{'Average':<35s}  {np.mean(avg_times[0]):>10.2f}  {np.mean(avg_times[3]):>10.2f}  "
          f"{np.mean(avg_times[3]) - np.mean(avg_times[0]):>+10.2f}  "
          f"{np.mean(avg_times[3]) / np.mean(avg_times[0]):>7.2f}x")


if __name__ == "__main__":
    main()
