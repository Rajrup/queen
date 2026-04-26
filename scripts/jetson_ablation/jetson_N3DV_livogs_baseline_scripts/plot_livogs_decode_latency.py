#!/usr/bin/env python3
"""Bar plot comparing LiVoGS decode latency (SH0 vs SH*) across sequences.

QUEEN sequences (flame_salmon_1, sear_steak) use SH2 as SH*.
HiFi4G sequences (4K_Actor1_Greeting) use SH3 as SH*.
Both are labelled "SH*" and share the same bar colour.

Usage:
    python scripts/plot_livogs_decode_latency.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Per-sequence config
#   label      : x-axis tick label
#   sh0_csv    : path to benchmark_livogs.csv for SH0 run
#   shstar_csv : path to benchmark_livogs.csv for SH* run (SH2 or SH3)
# ---------------------------------------------------------------------------
_QUEEN_ROOT   = Path("/home/rajrup/Queen/pretrained_output/Neural_3D_Video")
_VIDEOGS_ROOT = Path("/home/rajrup/VideoGS/train_output/HiFi4G_Dataset")


def _queen_csv(seq: str, sh: int) -> Path:
    return _QUEEN_ROOT / f"queen_compressed_{seq}" / "latency_benchmark" / f"livogs_sh_degree_{sh}" / "benchmark_livogs.csv"


def _videogs_csv(seq: str, sh: int) -> Path:
    return _VIDEOGS_ROOT / seq / "latency_benchmark" / f"livogs_sh_degree_{sh}" / "benchmark_livogs.csv"


SEQUENCE_CONFIG = [
    {
        "name":       "4K_Actor1_Greeting",
        "label":      "Actor1\nGreeting",
        "sh0_csv":    _videogs_csv("4K_Actor1_Greeting", 0),
        "shstar_csv": _videogs_csv("4K_Actor1_Greeting", 3),
    },
    {
        "name":       "flame_salmon_1",
        "label":      "Flame\nSalmon",
        "sh0_csv":    _queen_csv("flame_salmon_1", 0),
        "shstar_csv": _queen_csv("flame_salmon_1", 2),
    },
    {
        "name":       "sear_steak",
        "label":      "Sear\nSteak",
        "sh0_csv":    _queen_csv("sear_steak", 0),
        "shstar_csv": _queen_csv("sear_steak", 2),
    },
]

COLOR_SH0    = "#4C72B0"
COLOR_SHSTAR = "#DD8452"
HATCH_SH0    = "//"
HATCH_SHSTAR = "\\\\"

SKIP_WARMUP = 1
OUTPUT_DIR  = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
def read_decode_times(csv_path: Path) -> list[float]:
    times: list[float] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["total_decode_ms"]))
    return times


def main() -> None:
    avg_sh0:    list[float] = []
    std_sh0:    list[float] = []
    avg_shstar: list[float] = []
    std_shstar: list[float] = []
    valid: list[dict] = []

    for cfg in SEQUENCE_CONFIG:
        missing = [p for p in (cfg["sh0_csv"], cfg["shstar_csv"]) if not p.exists()]
        if missing:
            for p in missing:
                print(f"  WARNING: missing {p}")
            continue

        t0 = read_decode_times(cfg["sh0_csv"])[SKIP_WARMUP:]
        ts = read_decode_times(cfg["shstar_csv"])[SKIP_WARMUP:]
        avg_sh0.append(float(np.mean(t0)))
        std_sh0.append(float(np.std(t0)))
        avg_shstar.append(float(np.mean(ts)))
        std_shstar.append(float(np.std(ts)))
        valid.append(cfg)

    if not valid:
        print("ERROR: No valid sequence data found.")
        return

    n_seq  = len(valid)
    n_bars = n_seq + 1          # +1 for "Average"
    x      = np.arange(n_bars)
    bar_width = 0.35

    # Append overall averages
    plot_avg0 = avg_sh0    + [float(np.mean(avg_sh0))]
    plot_std0 = std_sh0    + [float(np.mean(std_sh0))]
    plot_avgs = avg_shstar + [float(np.mean(avg_shstar))]
    plot_stds = std_shstar + [float(np.mean(std_shstar))]

    labels = [cfg["label"] for cfg in valid] + ["Average"]

    fig, ax = plt.subplots(figsize=(max(8, n_bars * 1.6), 5))

    def draw_bars(pos_offset, avgs, stds, color, hatch, legend_label):
        bars = ax.bar(
            x + pos_offset, avgs, bar_width,
            yerr=stds,
            label=legend_label,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.6,
            capsize=3,
            error_kw={"linewidth": 0.8},
            zorder=3,
        )
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.2)
        for bar, val in zip(bars, avgs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="medium",
            )

    draw_bars(-0.5 * bar_width, plot_avg0, plot_std0, COLOR_SH0,    HATCH_SH0,    "SH0")
    draw_bars(+0.5 * bar_width, plot_avgs, plot_stds, COLOR_SHSTAR, HATCH_SHSTAR, "SH*")

    ax.axvline(n_seq - 0.5, color="grey", linestyle=":", linewidth=0.8, zorder=1)

    ax.set_ylabel("Decode Latency (ms)", fontsize=11)
    ax.set_xlabel("Sequence", fontsize=11)
    ax.set_title("LiVoGS Decode Latency: SH0 vs SH* (Jetson Orin)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        out = OUTPUT_DIR / f"livogs_decode_latency_sh0_vs_shstar{suffix}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    # Summary table
    print(f"\n{'Sequence':<25s}  {'SH0 (ms)':>10s}  {'SH* (ms)':>10s}  {'Δ (ms)':>10s}  {'Ratio':>8s}")
    print("-" * 72)
    for i, cfg in enumerate(valid):
        v0, vs = avg_sh0[i], avg_shstar[i]
        print(f"{cfg['name']:<25s}  {v0:>10.2f}  {vs:>10.2f}  {vs - v0:>+10.2f}  {vs / v0:>7.2f}x")
    print("-" * 72)
    m0, ms = np.mean(avg_sh0), np.mean(avg_shstar)
    print(f"{'Average':<25s}  {m0:>10.2f}  {ms:>10.2f}  {ms - m0:>+10.2f}  {ms / m0:>7.2f}x")


if __name__ == "__main__":
    main()
