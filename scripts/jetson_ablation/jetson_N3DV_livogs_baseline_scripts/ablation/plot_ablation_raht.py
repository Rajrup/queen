#!/usr/bin/env python3
'''
Plot RAHT ablation results from ablation_raht.csv files.

Two modes:
  1. Single-SH mode (--input_csvs): one CSV per sequence, one SH degree.
     Generates: raht_latency, raht_speedup.

  2. Combined SH mode (--sh0_csvs + --shstar_csvs): two sets of CSVs (SH0 and SH*),
     one per sequence each. Generates: raht_latency_sh_combined, raht_speedup_sh_combined.
     Within each sequence group, bars are: SH0-PyTorch | SH0-CUDA | SH*-PyTorch | SH*-CUDA.

Usage (single-SH):
  python scripts/livogs_baseline/ablation/plot_ablation_raht.py \\
      --input_csvs /path/salmon/ablation_raht.csv /path/steak/ablation_raht.csv \\
      --seq_labels "Flame Salmon" "Sear Steak" \\
      --output_folder scripts/livogs_baseline/ablation/plots

Usage (combined SH0 + SH*):
  python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
    --sh0_csvs \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_raht_sh0/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_raht_sh0/ablation_raht.csv \
    --shstar_csvs \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_raht_sh2/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_raht_sh2/ablation_raht.csv \
    --seq_labels "Flame Salmon" "Sear Steak" \
    --output_folder scripts/livogs_baseline/ablation/plots

python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
    --sh0_csvs \
        /home/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/ablation/livogs_raht_sh0/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_raht_sh0/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_raht_sh0/ablation_raht.csv \
    --shstar_csvs \
        /home/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/ablation/livogs_raht_sh3/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_raht_sh2/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_raht_sh2/ablation_raht.csv \
    --seq_labels "Actor1" "Flame Salmon" "Sear Steak" \
    --output_folder scripts/livogs_baseline/ablation/plots
'''

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARIANTS       = ["pytorch", "cuda"]
VARIANT_LABELS = {"pytorch": "PyTorch", "cuda": "CUDA"}
PHASES         = ["prelude_ms", "inverse_ms"]
PHASE_LABELS   = {"prelude_ms": "Prelude", "inverse_ms": "Inverse"}

# (sh_key, variant) → phase → colour
# SH0: blue family  |  SH*: orange family
PALETTE: dict[tuple[str, str], dict[str, str]] = {
    ("sh0",    "pytorch"): {"prelude_ms": "#c5d9f0", "inverse_ms": "#2a4a7f"},
    ("sh0",    "cuda"):    {"prelude_ms": "#ffd5b5", "inverse_ms": "#a05a2c"},
    ("shstar", "pytorch"): {"prelude_ms": "#b8ddb8", "inverse_ms": "#2d6e3a"},
    ("shstar", "cuda"):    {"prelude_ms": "#e5b5c5", "inverse_ms": "#8b2e32"},
}

# single-SH palette (no SH dimension)
PALETTE_SINGLE = {
    "pytorch": {"prelude_ms": "#aec6e8", "inverse_ms": "#2a4a7f"},
    "cuda":    {"prelude_ms": "#ffc09f", "inverse_ms": "#a05a2c"},
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k != "variant" else v) for k, v in row.items()})
            rows[-1]["frame_id"] = int(rows[-1]["frame_id"])
    return rows


def stats_by_variant(rows: list[dict]) -> tuple[dict, dict]:
    """Returns (means, stds) keyed by variant → phase."""
    by_v: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)
    means = {v: {ph: float(np.mean([r[ph] for r in rl])) for ph in PHASES} for v, rl in by_v.items()}
    stds  = {v: {ph: float(np.std( [r[ph] for r in rl])) for ph in PHASES} for v, rl in by_v.items()}
    return means, stds


# ---------------------------------------------------------------------------
# Combined SH plot
# ---------------------------------------------------------------------------
BAR_KEYS = [("sh0", "pytorch"), ("sh0", "cuda"), ("shstar", "pytorch"), ("shstar", "cuda")]
BAR_LABELS = {
    ("sh0",    "pytorch"): "SH0 PyTorch",
    ("sh0",    "cuda"):    "SH0 CUDA",
    ("shstar", "pytorch"): "SH* PyTorch",
    ("shstar", "cuda"):    "SH* CUDA",
}


def plot_latency_combined(
    sh0_means:    list[dict],
    sh0_stds:     list[dict],
    shstar_means: list[dict],
    shstar_stds:  list[dict],
    seq_labels:   list[str],
    output_folder: str,
    fmt: str,
) -> None:
    """4 stacked bars per sequence: SH0-PyTorch | SH0-CUDA | SH*-PyTorch | SH*-CUDA."""
    n_seq   = len(seq_labels)
    n_bars  = len(BAR_KEYS)          # 4
    bar_width = 0.14
    group_gap = 0.10
    group_width = n_bars * bar_width + group_gap
    x_centers = np.arange(n_seq) * group_width

    fig, ax = plt.subplots(figsize=(max(5, n_seq * 3.0), 5))

    legend_handles: dict[str, object] = {}

    def _means_stds(sh_key: str, variant: str, s_idx: int):
        if sh_key == "sh0":
            return sh0_means[s_idx].get(variant, {}), sh0_stds[s_idx].get(variant, {})
        return shstar_means[s_idx].get(variant, {}), shstar_stds[s_idx].get(variant, {})

    for b_idx, (sh_key, variant) in enumerate(BAR_KEYS):
        offset = (b_idx - (n_bars - 1) / 2.0) * bar_width
        x_pos  = x_centers + offset

        for s_idx in range(n_seq):
            means, _ = _means_stds(sh_key, variant, s_idx)
            if not means:
                continue

            bottom = 0.0
            for phase in PHASES:
                val   = means.get(phase, 0.0)
                color = PALETTE[(sh_key, variant)][phase]
                label = f"{BAR_LABELS[(sh_key, variant)]} – {PHASE_LABELS[phase]}"

                bar = ax.bar(
                    x_pos[s_idx], val, bar_width,
                    bottom=bottom,
                    color=color,
                    edgecolor="white", linewidth=0.4,
                    zorder=3,
                    label=label if label not in legend_handles else "_nolegend_",
                )
                if label not in legend_handles:
                    legend_handles[label] = bar
                bottom += val

            prelude_val = means.get("prelude_ms", 0.0)
            inverse_val = means.get("inverse_ms", 0.0)
            total = sum(means.get(ph, 0.0) for ph in PHASES)
            ax.text(x_pos[s_idx], total + 0.3, f"{total:.0f}",
                    ha="center", va="bottom", fontsize=6.5, fontweight="medium")

            if prelude_val < inverse_val:
                inv_mid = prelude_val + inverse_val / 2
                ax.text(x_pos[s_idx], inv_mid, f"{inverse_val:.0f}",
                        ha="center", va="center", fontsize=5.5,
                        color="white", fontweight="bold")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(seq_labels, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("RAHT Decode Latency: SH0 vs SH*  ×  PyTorch vs CUDA — Prelude / Inverse (Jetson)",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=list(legend_handles.values()), fontsize=7,
              loc="upper right", ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(output_folder, f"raht_latency_sh_combined.{fmt}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_speedup_combined(
    sh0_means:    list[dict],
    shstar_means: list[dict],
    seq_labels:   list[str],
    output_folder: str,
    fmt: str,
) -> None:
    """CUDA speedup over PyTorch, shown for SH0 and SH* side by side per sequence."""
    n_seq     = len(seq_labels)
    bar_width = 0.3
    x         = np.arange(n_seq)

    fig, ax = plt.subplots(figsize=(max(4, n_seq * 2.0), 4))

    for b_idx, (sh_key, sh_label, all_means) in enumerate([
        ("sh0",    "SH0", sh0_means),
        ("shstar", "SH*", shstar_means),
    ]):
        speedups = []
        for s_idx in range(n_seq):
            m_py = all_means[s_idx].get("pytorch", {})
            m_cu = all_means[s_idx].get("cuda",    {})
            tot_py = sum(m_py.get(ph, 0.0) for ph in PHASES)
            tot_cu = sum(m_cu.get(ph, 0.0) for ph in PHASES)
            speedups.append(tot_py / tot_cu if tot_cu > 0 else 0.0)

        color = "#4c72b0" if sh_key == "sh0" else "#55a868"
        offset = (b_idx - 0.5) * bar_width
        bars = ax.bar(x + offset, speedups, bar_width, label=sh_label,
                      color=color, zorder=3)
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}x", ha="center", va="bottom", fontsize=8.5)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, fontsize=10)
    ax.set_ylabel("Speedup (PyTorch / CUDA)", fontsize=11)
    ax.set_title("RAHT Total CUDA Speedup over PyTorch: SH0 vs SH*",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(output_folder, f"raht_speedup_sh_combined.{fmt}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def print_summary_combined(
    sh0_means:    list[dict],
    shstar_means: list[dict],
    seq_labels:   list[str],
) -> None:
    print(f"\n{'Sequence':<18s}  {'SH':<5s}  {'Variant':<10s}  "
          f"{'Prelude':>10s}  {'Forward':>10s}  {'Inverse':>10s}  {'Total':>10s}")
    print("-" * 80)
    for s_idx, label in enumerate(seq_labels):
        for sh_key, sh_label, all_means in [("sh0", "SH0", sh0_means),
                                             ("shstar", "SH*", shstar_means)]:
            for variant in VARIANTS:
                m = all_means[s_idx].get(variant, {})
                if not m:
                    continue
                total = sum(m.get(ph, 0.0) for ph in PHASES)
                print(f"{label:<18s}  {sh_label:<5s}  {VARIANT_LABELS[variant]:<10s}  "
                      f"{m.get('prelude_ms', 0):>10.2f}  {m.get('forward_ms', 0):>10.2f}  "
                      f"{m.get('inverse_ms', 0):>10.2f}  {total:>10.2f}")
            # speedup within this sh
            for sh_key2, all_means2 in [(sh_key, all_means)]:
                m_py = all_means2[s_idx].get("pytorch", {})
                m_cu = all_means2[s_idx].get("cuda",    {})
                if m_py and m_cu:
                    tot_py = sum(m_py.get(ph, 0.0) for ph in PHASES)
                    tot_cu = sum(m_cu.get(ph, 0.0) for ph in PHASES)
                    print(f"{'':18s}  {sh_label:<5s}  {'→ speedup':<10s}  "
                          f"{'':>10s}  {'':>10s}  {'':>10s}  "
                          f"{tot_py / tot_cu if tot_cu > 0 else 0:>9.2f}x")
        print()


# ---------------------------------------------------------------------------
# Single-SH plot (original behaviour)
# ---------------------------------------------------------------------------
def plot_latency_single(
    all_means:    list[dict],
    all_stds:     list[dict],
    seq_labels:   list[str],
    output_folder: str,
    fmt: str,
) -> None:
    n_seq = len(seq_labels)
    n_var = len(VARIANTS)
    bar_width   = 0.35
    group_gap   = 0.2
    group_width = n_var * bar_width + group_gap
    x_centers   = np.arange(n_seq) * group_width

    fig, ax = plt.subplots(figsize=(max(6, n_seq * 3.5), 5))
    legend_handles: dict[str, object] = {}

    for v_idx, variant in enumerate(VARIANTS):
        x_pos = x_centers + (v_idx - (n_var - 1) / 2) * bar_width
        for s_idx in range(n_seq):
            means = all_means[s_idx].get(variant, {})
            if not means:
                continue
            bottom = 0.0
            for phase in PHASES:
                val   = means.get(phase, 0.0)
                color = PALETTE_SINGLE[variant][phase]
                label = f"{VARIANT_LABELS[variant]} – {PHASE_LABELS[phase]}"
                bar = ax.bar(x_pos[s_idx], val, bar_width, bottom=bottom,
                             color=color, edgecolor="white", linewidth=0.5,
                             zorder=3,
                             label=label if label not in legend_handles else "_nolegend_")
                if label not in legend_handles:
                    legend_handles[label] = bar
                bottom += val
            prelude_val = means.get("prelude_ms", 0.0)
            inverse_val = means.get("inverse_ms", 0.0)
            total = sum(means.get(ph, 0.0) for ph in PHASES)
            ax.text(x_pos[s_idx], total + 0.5, f"{total:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="medium")

            if prelude_val < inverse_val:
                inv_mid = prelude_val + inverse_val / 2
                ax.text(x_pos[s_idx], inv_mid, f"{inverse_val:.1f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(seq_labels, fontsize=10)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("RAHT Decode Latency: PyTorch vs CUDA — Prelude / Inverse",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=list(legend_handles.values()), fontsize=8,
              loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(output_folder, f"raht_latency.{fmt}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_speedup_single(
    all_means:    list[dict],
    seq_labels:   list[str],
    output_folder: str,
    fmt: str,
) -> None:
    n_seq     = len(seq_labels)
    n_ph      = len(PHASES)
    bar_width = 0.2
    x         = np.arange(n_seq)

    fig, ax = plt.subplots(figsize=(max(5, n_seq * 2.5), 4))
    for p_idx, phase in enumerate(PHASES):
        speedups = [
            (all_means[i].get("pytorch", {}).get(phase, 0) /
             all_means[i].get("cuda",    {}).get(phase, 1))
            for i in range(n_seq)
        ]
        offset = (p_idx - (n_ph - 1) / 2) * bar_width
        bars = ax.bar(x + offset, speedups, bar_width, label=PHASE_LABELS[phase], zorder=3)
        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}x", ha="center", va="bottom", fontsize=8)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_labels, fontsize=10)
    ax.set_ylabel("Speedup (PyTorch / CUDA)", fontsize=11)
    ax.set_title("RAHT CUDA Speedup over PyTorch", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = os.path.join(output_folder, f"raht_speedup.{fmt}")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Plot RAHT ablation results")
    p.add_argument("--input_csvs",    nargs="+", default=None,
                   help="[Single-SH mode] One ablation_raht.csv per sequence")
    p.add_argument("--sh0_csvs",      nargs="+", default=None,
                   help="[Combined mode] SH0 ablation_raht.csv files, one per sequence")
    p.add_argument("--shstar_csvs",   nargs="+", default=None,
                   help="[Combined mode] SH* ablation_raht.csv files, one per sequence")
    p.add_argument("--seq_labels",    nargs="+", default=None)
    p.add_argument("--output_folder", type=str,  default=None)
    p.add_argument("--format",        type=str,  choices=["pdf", "png"], default="png")
    args = p.parse_args()

    combined_mode = (args.sh0_csvs is not None and args.shstar_csvs is not None)
    single_mode   = (args.input_csvs is not None)

    if not combined_mode and not single_mode:
        p.error("Provide either --input_csvs (single-SH) or --sh0_csvs + --shstar_csvs (combined).")

    if args.output_folder is None:
        ref = (args.sh0_csvs or args.input_csvs)[0]
        args.output_folder = os.path.join(os.path.dirname(ref), "plots")
    os.makedirs(args.output_folder, exist_ok=True)

    if combined_mode:
        if len(args.sh0_csvs) != len(args.shstar_csvs):
            p.error("--sh0_csvs and --shstar_csvs must have the same number of files.")
        seq_labels = args.seq_labels or [Path(c).parts[-4] for c in args.sh0_csvs]

        sh0_means_list, sh0_stds_list     = [], []
        shstar_means_list, shstar_stds_list = [], []

        for csv_path in args.sh0_csvs:
            rows = load_csv(csv_path)
            print(f"Loaded SH0   {len(rows)} rows: {csv_path}")
            m, s = stats_by_variant(rows)
            sh0_means_list.append(m); sh0_stds_list.append(s)

        for csv_path in args.shstar_csvs:
            rows = load_csv(csv_path)
            print(f"Loaded SH*   {len(rows)} rows: {csv_path}")
            m, s = stats_by_variant(rows)
            shstar_means_list.append(m); shstar_stds_list.append(s)

        print_summary_combined(sh0_means_list, shstar_means_list, seq_labels)
        plot_latency_combined(sh0_means_list, sh0_stds_list,
                              shstar_means_list, shstar_stds_list,
                              seq_labels, args.output_folder, args.format)
        plot_speedup_combined(sh0_means_list, shstar_means_list,
                              seq_labels, args.output_folder, args.format)

    else:  # single-SH mode
        seq_labels = args.seq_labels or [Path(c).parts[-4] for c in args.input_csvs]
        all_means, all_stds = [], []
        for csv_path in args.input_csvs:
            rows = load_csv(csv_path)
            print(f"Loaded {len(rows)} rows from {csv_path}")
            m, s = stats_by_variant(rows)
            all_means.append(m); all_stds.append(s)

        plot_latency_single(all_means, all_stds, seq_labels, args.output_folder, args.format)
        plot_speedup_single(all_means, seq_labels, args.output_folder, args.format)

    print("Done!")


if __name__ == "__main__":
    main()


'''
# Single-SH (original)
python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
    --input_csvs \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_raht_sh0/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_raht_sh0/ablation_raht.csv \
    --seq_labels "Flame Salmon" "Sear Steak" \
    --output_folder scripts/livogs_baseline/ablation/plots/raht_sh0

# Combined SH0 + SH*  (Actor1 first, then QUEEN sequences)
python scripts/livogs_baseline/ablation/plot_ablation_raht.py \
    --sh0_csvs \
        /home/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/ablation/livogs_raht_sh0/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_raht_sh0/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_raht_sh0/ablation_raht.csv \
    --shstar_csvs \
        /home/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/ablation/livogs_raht_sh3/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/ablation/livogs_raht_sh2/ablation_raht.csv \
        /home/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/ablation/livogs_raht_sh2/ablation_raht.csv \
    --seq_labels "Actor1" "Flame Salmon" "Sear Steak" \
    --output_folder scripts/livogs_baseline/ablation/plots
'''
