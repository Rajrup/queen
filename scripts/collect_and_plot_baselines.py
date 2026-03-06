#!/usr/bin/env python3
"""Collect baseline outputs and generate per-frame comparison plots for QUEEN."""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_NAME = "Neural_3D_Video"
DATA_PATH = "/synology/rajrup/Queen"

EXPERIMENTS: dict[str, list[int]] = {
    "cook_spinach": [1, 51, 101, 151],
    "coffee_martini": [1, 51, 101, 151],
    "cut_roasted_beef": [1, 51, 101, 151],
    "flame_salmon_1": [1, 51, 101, 151],
    "flame_steak": [1, 51, 101, 151],
    "sear_steak": [1, 51, 101, 151],
}

BASELINES: dict[str, dict[str, Any]] = {
    "DracoGS": {
        "subdir": "dracogs",
        "output_tag": "eg_16_eo_16_et_16_es_16_cl_10",
        "benchmark_csv": "benchmark_dracogs.csv",
    },
    "MesonGS": {
        "subdir": "mesongs",
        "output_tag": "params_default",
        "benchmark_csv": "benchmark_mesongs.csv",
    },
    "VideoGS": {
        "subdir": "videogs",
        "output_tag": "qp_25",
        "benchmark_csv": "benchmark_videogs_pipeline.csv",
    },
}

BASELINE_STYLES: dict[str, dict[str, Any]] = {
    "DracoGS": {"color": "#1f77b4", "marker": "o", "label": "DracoGS"},
    "MesonGS": {"color": "#2ca02c", "marker": "s", "label": "MesonGS"},
    "VideoGS": {"color": "#d62728", "marker": "^", "label": "VideoGS"},
}

LIVOGS_HULL_ENABLED = True
LIVOGS_RD_SUBDIR = "livogs_rd_nvcomp"
LIVOGS_HULL_STYLE = {
    "color": "#ff7f0e",
    "linewidth": 1.8,
    "linestyle": "-",
    "marker": "D",
    "markersize": 4,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "baseline_comparison_res")


CSV_COLUMNS = [
    "sequence_name",
    "baseline",
    "frame_id",
    "compressed_size_bytes",
    "compressed_mb",
    "uncompressed_size_bytes",
    "uncompressed_mb",
    "encode_ms",
    "decode_ms",
    "gt_psnr",
    "gt_ssim",
    "decomp_psnr",
    "decomp_ssim",
    "psnr_drop",
    "ssim_drop",
]


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


def _iter_frame_groups(rows: list[dict[str, Any]]) -> list[tuple[str, int, list[dict[str, Any]]]]:
    by_seq = _group_by(rows, "sequence_name")
    grouped: list[tuple[str, int, list[dict[str, Any]]]] = []
    for seq_name, seq_rows in sorted(by_seq.items()):
        frame_ids = sorted({int(r["frame_id"]) for r in seq_rows})
        for frame_id in frame_ids:
            frame_rows = [r for r in seq_rows if int(r["frame_id"]) == frame_id]
            if frame_rows:
                grouped.append((seq_name, frame_id, frame_rows))
    return grouped


def _first_float(row: dict[str, str], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def _model_root(sequence: str) -> str:
    return os.path.join(
        DATA_PATH,
        "pretrained_output",
        DATASET_NAME,
        f"queen_compressed_{sequence}",
    )


def _output_folder(sequence: str, baseline_key: str) -> str:
    cfg = BASELINES[baseline_key]
    return os.path.join(_model_root(sequence), "compression", cfg["subdir"], cfg["output_tag"])


def _resolve_livogs_hull_csv(sequence: str, frame_id: int) -> Optional[str]:
    plot_dir = os.path.join(_model_root(sequence), "compression", LIVOGS_RD_SUBDIR, "plots")
    exact_name = f"convex_hull_{DATASET_NAME}_{sequence}_frame{frame_id}.csv"
    exact_path = os.path.join(plot_dir, exact_name)
    if os.path.isfile(exact_path):
        return exact_path

    if not os.path.isdir(plot_dir):
        return None

    suffix = f"_frame{frame_id}.csv"
    prefix = f"convex_hull_{DATASET_NAME}_"
    candidates = [
        os.path.join(plot_dir, name)
        for name in os.listdir(plot_dir)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    if not candidates:
        return None

    candidates.sort()
    return candidates[0]


def _load_livogs_hull_points(sequence: str, frame_id: int) -> list[tuple[float, float]]:
    hull_csv = _resolve_livogs_hull_csv(sequence, frame_id)
    if hull_csv is None:
        return []

    points: list[tuple[float, float]] = []
    try:
        with open(hull_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                points.append((float(row["compressed_mb"]), float(row["decomp_psnr"])))
    except (OSError, KeyError, ValueError):
        return []

    points.sort(key=lambda p: p[0])
    return points


def _load_sequence_results(
    output_folder: str,
    sequence: str,
    baseline: str,
    benchmark_csv_name: str,
    frame_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    benchmark_path = os.path.join(output_folder, benchmark_csv_name)
    benchmark_by_frame: dict[int, dict[str, Any]] = {}
    if os.path.isfile(benchmark_path):
        try:
            with open(benchmark_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fid = int(row["frame_id"])
                    benchmark_by_frame[fid] = {
                        "compressed_size_bytes": int(row["compressed_size_bytes"]),
                        "uncompressed_size_bytes": int(row.get("uncompressed_size_bytes", 0)),
                        "encode_ms": _first_float(row, ("total_encode_ms", "encode_time_ms", "encode_ms")),
                        "decode_ms": _first_float(row, ("total_decode_ms", "decode_time_ms", "decode_ms")),
                    }
        except (OSError, KeyError, ValueError) as exc:
            print(f"  [WARN] Failed to read {benchmark_path}: {exc}")
    else:
        print(f"  [WARN] Benchmark CSV not found: {benchmark_path}")

    eval_json_path = os.path.join(output_folder, "evaluation", "evaluation_results.json")
    metrics_by_frame: dict[int, dict[str, float]] = {}
    if os.path.isfile(eval_json_path):
        try:
            with open(eval_json_path, encoding="utf-8") as f:
                eval_data = json.load(f)
            for fr in eval_data.get("per_frame", []):
                fid = int(fr["frame"])
                metrics_by_frame[fid] = {
                    "gt_psnr": float(fr["gt_psnr"]),
                    "gt_ssim": float(fr["gt_ssim"]),
                    "decomp_psnr": float(fr["decomp_psnr"]),
                    "decomp_ssim": float(fr["decomp_ssim"]),
                }
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"  [WARN] Failed to read {eval_json_path}: {exc}")
    else:
        print(f"  [WARN] Evaluation JSON not found: {eval_json_path}")

    for fid in frame_ids:
        if fid not in benchmark_by_frame:
            print(f"  [SKIP] {baseline} | {sequence} | frame {fid} (no benchmark data)")
            continue
        if fid not in metrics_by_frame:
            print(f"  [SKIP] {baseline} | {sequence} | frame {fid} (no evaluation data)")
            continue

        b = benchmark_by_frame[fid]
        m = metrics_by_frame[fid]
        comp = b["compressed_size_bytes"]
        uncomp = b["uncompressed_size_bytes"]
        rows.append(
            {
                "sequence_name": sequence,
                "baseline": baseline,
                "frame_id": fid,
                "compressed_size_bytes": comp,
                "compressed_mb": comp / (1024 * 1024),
                "uncompressed_size_bytes": uncomp,
                "uncompressed_mb": uncomp / (1024 * 1024),
                "encode_ms": b.get("encode_ms"),
                "decode_ms": b.get("decode_ms"),
                "gt_psnr": m["gt_psnr"],
                "gt_ssim": m["gt_ssim"],
                "decomp_psnr": m["decomp_psnr"],
                "decomp_ssim": m["decomp_ssim"],
                "psnr_drop": m["gt_psnr"] - m["decomp_psnr"],
                "ssim_drop": m["gt_ssim"] - m["decomp_ssim"],
            }
        )

    return rows


def collect_all_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence, frame_ids in EXPERIMENTS.items():
        for baseline, cfg in BASELINES.items():
            output_folder = _output_folder(sequence, baseline)
            rows.extend(
                _load_sequence_results(
                    output_folder,
                    sequence,
                    baseline,
                    cfg["benchmark_csv"],
                    frame_ids,
                )
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    print(f"  Wrote {len(rows)} rows to: {path}")


def plot_psnr_size_per_frame(rows: list[dict[str, Any]], plot_dir: str) -> None:
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        fig, ax = plt.subplots(figsize=(9, 6))
        by_baseline = _group_by(frame_rows, "baseline")

        for baseline in BASELINES:
            if baseline not in by_baseline:
                continue
            bl_rows = by_baseline[baseline]
            style = BASELINE_STYLES[baseline]
            xs = [r["compressed_mb"] for r in bl_rows]
            ys = [r["decomp_psnr"] for r in bl_rows]
            x = float(np.mean(xs))
            y = float(np.mean(ys))
            ax.scatter(
                [x],
                [y],
                color=style["color"],
                marker=style["marker"],
                s=120,
                alpha=1.0,
                zorder=4,
                edgecolors="black",
                linewidths=0.8,
                label=f"{style['label']} ({y:.2f} dB, {x:.2f} MB)",
            )

        if LIVOGS_HULL_ENABLED:
            hull_points = _load_livogs_hull_points(seq_name, frame_id)
            if hull_points:
                print(f"  LiVoGS hull: {seq_name} frame {frame_id} ({len(hull_points)} points)")
                hx = [p[0] for p in hull_points]
                hy = [p[1] for p in hull_points]
                ax.plot(
                    hx,
                    hy,
                    color=LIVOGS_HULL_STYLE["color"],
                    linewidth=LIVOGS_HULL_STYLE["linewidth"],
                    linestyle=LIVOGS_HULL_STYLE["linestyle"],
                    marker=LIVOGS_HULL_STYLE["marker"],
                    markersize=LIVOGS_HULL_STYLE["markersize"],
                    alpha=0.95,
                    zorder=3,
                    label="LiVoGS hull",
                )

        gt_psnrs = [r["gt_psnr"] for r in frame_rows if r.get("gt_psnr") is not None]
        if gt_psnrs:
            gt_mean = float(np.mean(gt_psnrs))
            ax.axhline(
                gt_mean,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label=f"Uncompressed ({gt_mean:.2f} dB)",
                zorder=1,
            )

        ax.set_xlabel("Compressed Size (MB)", fontsize=11)
        ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.set_title(f"PSNR-Size | {seq_name} | Frame {frame_id}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"psnr_size_{seq_name}_frame{frame_id}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_latency_method_per_frame(rows: list[dict[str, Any]], plot_dir: str) -> None:
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        by_baseline = _group_by(frame_rows, "baseline")
        baselines_present = [b for b in BASELINES if b in by_baseline]
        if not baselines_present:
            continue

        labels = [BASELINE_STYLES[b]["label"] for b in baselines_present]
        encode_vals: list[float] = []
        decode_vals: list[float] = []
        for bl in baselines_present:
            bl_rows = by_baseline[bl]
            enc = [float(r["encode_ms"]) for r in bl_rows if r.get("encode_ms") is not None]
            dec = [float(r["decode_ms"]) for r in bl_rows if r.get("decode_ms") is not None]
            encode_vals.append(float(np.mean(enc)) if enc else np.nan)
            decode_vals.append(float(np.mean(dec)) if dec else np.nan)

        x = np.arange(len(baselines_present))
        width = 0.34
        fig, ax = plt.subplots(figsize=(9, 5.5))

        enc_plot = [v if np.isfinite(v) else 0.0 for v in encode_vals]
        dec_plot = [v if np.isfinite(v) else 0.0 for v in decode_vals]
        bars_enc = ax.bar(x - width / 2, enc_plot, width, color="#4e79a7", label="Encode")
        bars_dec = ax.bar(x + width / 2, dec_plot, width, color="#f28e2b", label="Decode")

        for i, v in enumerate(encode_vals):
            if np.isfinite(v):
                ax.text(x[i] - width / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                bars_enc[i].set_alpha(0.18)
                ax.text(x[i] - width / 2, 0.0, "N/A", ha="center", va="bottom", fontsize=8)

        for i, v in enumerate(decode_vals):
            if np.isfinite(v):
                ax.text(x[i] + width / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
            else:
                bars_dec[i].set_alpha(0.18)
                ax.text(x[i] + width / 2, 0.0, "N/A", ha="center", va="bottom", fontsize=8)

        finite_vals = [v for v in encode_vals + decode_vals if np.isfinite(v)]
        ymax = max(finite_vals) * 1.18 if finite_vals else 1.0
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Latency (ms/frame)", fontsize=11)
        ax.set_title(f"Latency by Method | {seq_name} | Frame {frame_id}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"latency_method_{seq_name}_frame{frame_id}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_size_method_per_frame(rows: list[dict[str, Any]], plot_dir: str) -> None:
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        by_baseline = _group_by(frame_rows, "baseline")
        baselines_present = [b for b in BASELINES if b in by_baseline]
        if not baselines_present:
            continue

        labels = [BASELINE_STYLES[b]["label"] for b in baselines_present]
        compressed_vals: list[float] = []
        uncompressed_vals: list[float] = []
        colors: list[str] = []
        for bl in baselines_present:
            bl_rows = by_baseline[bl]
            comp = [float(r["compressed_mb"]) for r in bl_rows if r.get("compressed_mb") is not None]
            uncomp = [float(r["uncompressed_mb"]) for r in bl_rows if r.get("uncompressed_mb") is not None]
            compressed_vals.append(float(np.mean(comp)) if comp else np.nan)
            uncompressed_vals.append(float(np.mean(uncomp)) if uncomp else np.nan)
            colors.append(BASELINE_STYLES[bl]["color"])

        x = np.arange(len(baselines_present))
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        comp_plot = [v if np.isfinite(v) else 0.0 for v in compressed_vals]
        bars = ax.bar(x, comp_plot, width=0.55, color=colors, edgecolor="black", linewidth=0.7)

        for i, v in enumerate(compressed_vals):
            if np.isfinite(v):
                ax.text(x[i], v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            else:
                bars[i].set_alpha(0.18)
                ax.text(x[i], 0.0, "N/A", ha="center", va="bottom", fontsize=8)

        finite_comp = [v for v in compressed_vals if np.isfinite(v)]
        ymax = max(finite_comp) * 1.18 if finite_comp else 1.0
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Compressed Size (MB/frame)", fontsize=11)
        ax.set_title(f"Size by Method | {seq_name} | Frame {frame_id}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        legend_handles = []
        legend_labels = []
        for i, bl in enumerate(baselines_present):
            legend_handles.append(bars[i])
            if np.isfinite(uncompressed_vals[i]):
                legend_labels.append(
                    f"{BASELINE_STYLES[bl]['label']} (Uncompressed: {uncompressed_vals[i]:.2f} MB)"
                )
            else:
                legend_labels.append(f"{BASELINE_STYLES[bl]['label']} (Uncompressed: N/A)")
        ax.legend(legend_handles, legend_labels, fontsize=8, loc="upper left")

        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"size_method_{seq_name}_frame{frame_id}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("Baseline Comparison — Collect & Plot (QUEEN)")
    print(f"  Sequences:  {len(EXPERIMENTS)}")
    print(f"  Baselines:  {', '.join(BASELINES.keys())}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(sep)

    print(f"\n{sep}\nStep 1: Collect results\n{sep}")
    rows = collect_all_results()
    print(f"\n  Total results collected: {len(rows)}")

    if not rows:
        print("[ERROR] No results found. Did run_baseline_experiments.py complete?")
        sys.exit(1)

    by_baseline = _group_by(rows, "baseline")
    for bl in BASELINES:
        bl_rows = by_baseline.get(bl, [])
        if bl_rows:
            avg_psnr = np.mean([r["decomp_psnr"] for r in bl_rows])
            avg_size = np.mean([r["compressed_mb"] for r in bl_rows])
            print(
                f"    {bl:10s}: {len(bl_rows):3d} frames, "
                f"avg PSNR={avg_psnr:.2f} dB, avg size={avg_size:.2f} MB"
            )

    csv_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    write_csv(rows, csv_path)

    print(f"\n{sep}\nStep 2: Generate plots\n{sep}")
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for name in os.listdir(plot_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(plot_dir, name))

    plot_psnr_size_per_frame(rows, plot_dir)
    plot_latency_method_per_frame(rows, plot_dir)
    plot_size_method_per_frame(rows, plot_dir)

    print(f"\n{sep}")
    print(f"Done! All outputs in: {OUTPUT_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
