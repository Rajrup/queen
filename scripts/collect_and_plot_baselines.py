#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Collect baseline outputs and generate per-frame comparison plots for QUEEN."""

from __future__ import annotations

import csv
import json
import os
import sys
from functools import lru_cache
from typing import Any, Optional

import matplotlib  # type: ignore[reportMissingImports]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
import numpy as np


DATASET_NAME = "Neural_3D_Video"
DATA_PATH = "/synology/rajrup/Queen"
VIDEOGS_QPS = [25]
VIDEOGS_GROUP_SIZE = 20

EXPERIMENTS: dict[str, list[int]] = {
    "cook_spinach": list(range(1, 201, 20)),
    "coffee_martini": list(range(1, 201, 20)),
    "cut_roasted_beef": list(range(1, 201, 20)),
    "flame_salmon_1": list(range(1, 201, 20)),
    "flame_steak": list(range(1, 201, 20)),
    "sear_steak": list(range(1, 201, 20)),
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
        "output_tags": [f"qp_{qp}" for qp in VIDEOGS_QPS],
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
    "baseline_family",
    "videogs_qp",
    "frame_id",
    "gop_anchor_frame",
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
        anchor_ids = sorted({int(r.get("gop_anchor_frame", r["frame_id"])) for r in seq_rows})
        for anchor_id in anchor_ids:
            anchor_rows = [
                r
                for r in seq_rows
                if int(r.get("gop_anchor_frame", r["frame_id"])) == anchor_id
            ]
            if anchor_rows:
                grouped.append((seq_name, anchor_id, anchor_rows))
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


def _selected_to_span(frame_ids: list[int]) -> tuple[int, int, int]:
    if not frame_ids:
        raise ValueError("Frame list must not be empty")
    sorted_ids = sorted(set(int(v) for v in frame_ids))
    return sorted_ids[0], sorted_ids[-1], 1


@lru_cache(maxsize=None)
def _sequence_max_frame(sequence: str) -> int:
    frames_root = os.path.join(_model_root(sequence), "frames")
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"Frames root not found: {frames_root}")

    frame_ids = sorted(
        int(name)
        for name in os.listdir(frames_root)
        if name.isdigit()
        and os.path.isdir(os.path.join(frames_root, name))
        and (
            os.path.isfile(os.path.join(frames_root, name, "point_cloud.ply"))
            or os.path.isdir(os.path.join(frames_root, name, "point_cloud"))
        )
    )
    if not frame_ids:
        raise FileNotFoundError(
            f"No frame folders with point_cloud data found under {frames_root}"
        )

    return frame_ids[-1]


def _videogs_gop_frame_ids(sequence: str, anchor: int) -> list[int]:
    max_frame = _sequence_max_frame(sequence)
    if anchor > max_frame:
        raise ValueError(
            f"VideoGS anchor frame {anchor} exceeds last available frame {max_frame} "
            f"for sequence {sequence}"
        )

    gop_end = min(int(anchor) + VIDEOGS_GROUP_SIZE - 1, max_frame)
    frame_ids = list(range(int(anchor), gop_end + 1))
    if not frame_ids:
        raise ValueError(
            f"Resolved empty VideoGS GOP for sequence {sequence}: "
            f"anchor={anchor}, end={gop_end}"
        )
    return frame_ids


def _frame_span_tag(frame_start: int, frame_end: int, interval: int) -> str:
    return f"frames_{frame_start}_{frame_end}_int_{interval}"


def _candidate_output_folders(
    sequence: str,
    subdir: str,
    output_tag: str,
    frame_start: int,
    frame_end: int,
    interval: int,
) -> list[str]:
    legacy_root = os.path.join(_model_root(sequence), "compression", subdir, output_tag)
    return [
        os.path.join(legacy_root, _frame_span_tag(frame_start, frame_end, interval)),
        legacy_root,
    ]


def _resolve_output_folder(
    sequence: str,
    subdir: str,
    output_tag: str,
    frame_start: int,
    frame_end: int,
    interval: int,
    benchmark_csv_name: str,
) -> str:
    for folder in _candidate_output_folders(
        sequence,
        subdir,
        output_tag,
        frame_start,
        frame_end,
        interval,
    ):
        benchmark_path = os.path.join(folder, benchmark_csv_name)
        eval_json_path = os.path.join(folder, "evaluation", "evaluation_results.json")
        if os.path.isfile(benchmark_path) or os.path.isfile(eval_json_path):
            return folder

    return _candidate_output_folders(
        sequence,
        subdir,
        output_tag,
        frame_start,
        frame_end,
        interval,
    )[0]


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
    baseline_family: str,
    videogs_qp: Optional[int],
    benchmark_csv_name: str,
    frame_ids: list[int],
    gop_anchor_frame: Optional[int] = None,
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
                "baseline_family": baseline_family,
                "videogs_qp": videogs_qp,
                "frame_id": fid,
                "gop_anchor_frame": int(gop_anchor_frame) if gop_anchor_frame is not None else fid,
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


def _baseline_sort_key(rows_for_baseline: list[dict[str, Any]]) -> tuple[int, float, str]:
    sample = rows_for_baseline[0]
    family = str(sample.get("baseline_family", sample.get("baseline", "")))
    label = str(sample.get("baseline", family))
    family_rank = {"DracoGS": 0, "MesonGS": 1, "VideoGS": 2}.get(family, 99)
    qp = sample.get("videogs_qp")
    qp_sort = float(qp) if isinstance(qp, (int, float)) else -1.0
    return family_rank, qp_sort, label


def _style_for_baseline(rows_for_baseline: list[dict[str, Any]]) -> dict[str, Any]:
    sample = rows_for_baseline[0]
    family = str(sample.get("baseline_family", sample.get("baseline", "")))
    label = str(sample.get("baseline", family))

    base_style = BASELINE_STYLES.get(
        family,
        {"color": "#7f7f7f", "marker": "o", "label": label},
    )
    style = {
        "color": base_style["color"],
        "marker": base_style["marker"],
        "label": label,
    }

    qp = sample.get("videogs_qp")
    if family == "VideoGS" and isinstance(qp, int):
        qp_palette = {
            0: "#1b9e77",   # teal
            4: "#d95f02",   # orange
            10: "#7570b3",  # purple
            15: "#e7298a",  # magenta
            20: "#66a61e",  # green
        }
        qp_markers = {
            0: "o",
            4: "s",
            10: "D",
            15: "^",
            20: "v",
        }
        style["color"] = qp_palette.get(qp, style["color"])
        style["marker"] = qp_markers.get(qp, "o")

    return style


def collect_all_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence, frame_ids in EXPERIMENTS.items():
        frame_start, frame_end, interval = _selected_to_span(frame_ids)
        for baseline_family, cfg in BASELINES.items():
            output_tags = cfg.get("output_tags", [cfg.get("output_tag")])
            for output_tag in output_tags:
                if not output_tag:
                    continue

                videogs_qp: Optional[int] = None
                baseline_label = baseline_family
                if baseline_family == "VideoGS" and str(output_tag).startswith("qp_"):
                    qp_suffix = str(output_tag).split("_", maxsplit=1)[1]
                    try:
                        videogs_qp = int(qp_suffix)
                    except ValueError:
                        videogs_qp = None
                    baseline_label = (
                        f"VideoGS (QP={videogs_qp})"
                        if videogs_qp is not None
                        else f"VideoGS ({output_tag})"
                    )

                if baseline_family == "VideoGS":
                    for anchor in frame_ids:
                        gop_frame_ids = _videogs_gop_frame_ids(sequence, int(anchor))
                        output_folder = _resolve_output_folder(
                            sequence,
                            cfg["subdir"],
                            str(output_tag),
                            gop_frame_ids[0],
                            gop_frame_ids[-1],
                            1,
                            cfg["benchmark_csv"],
                        )
                        rows.extend(
                            _load_sequence_results(
                                output_folder,
                                sequence,
                                baseline_label,
                                baseline_family,
                                videogs_qp,
                                cfg["benchmark_csv"],
                                gop_frame_ids,
                                gop_anchor_frame=int(anchor),
                            )
                        )
                else:
                    output_folder = _resolve_output_folder(
                        sequence,
                        cfg["subdir"],
                        str(output_tag),
                        frame_start,
                        frame_end,
                        interval,
                        cfg["benchmark_csv"],
                    )
                    rows.extend(
                        _load_sequence_results(
                            output_folder,
                            sequence,
                            baseline_label,
                            baseline_family,
                            videogs_qp,
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
        sorted_baselines = sorted(by_baseline.items(), key=lambda item: _baseline_sort_key(item[1]))

        x_values = [float(r["compressed_mb"]) for r in frame_rows if r.get("compressed_mb") is not None]
        if x_values:
            x_min = min(x_values)
            x_max = max(x_values)
            x_range = x_max - x_min
            box_width = max(0.02, x_range * 0.025) if x_range > 0 else max(0.02, x_max * 0.025)
        else:
            box_width = 0.02

        for _, bl_rows in sorted_baselines:
            style = _style_for_baseline(bl_rows)
            family = str(bl_rows[0].get("baseline_family", "")) if bl_rows else ""
            if family == "VideoGS":
                psnr_values = [float(r["decomp_psnr"]) for r in bl_rows if r.get("decomp_psnr") is not None]
                size_values = [float(r["compressed_mb"]) for r in bl_rows if r.get("compressed_mb") is not None]
                if not psnr_values or not size_values:
                    continue
                avg_size = float(np.mean(size_values))
                median_psnr = float(np.median(psnr_values))
                ax.boxplot(
                    [psnr_values],
                    positions=[avg_size],
                    widths=[box_width],
                    vert=True,
                    patch_artist=True,
                    manage_ticks=False,
                    boxprops=dict(facecolor=style["color"], alpha=0.55, edgecolor=style["color"], linewidth=1.5),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color=style["color"], linewidth=1.3),
                    capprops=dict(color=style["color"], linewidth=1.3),
                    flierprops=dict(markerfacecolor=style["color"], markeredgecolor=style["color"], marker="o", markersize=4),
                    zorder=4,
                )
                ax.scatter(
                    [],
                    [],
                    color=style["color"],
                    marker=style["marker"],
                    s=80,
                    label=f"{style['label']} (med {median_psnr:.2f} dB, {avg_size:.2f} MB)",
                )
            else:
                xs = [r["compressed_mb"] for r in bl_rows if r.get("compressed_mb") is not None]
                ys = [r["decomp_psnr"] for r in bl_rows if r.get("decomp_psnr") is not None]
                if not xs or not ys:
                    continue
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

        anchor_gt_candidates = [
            float(r["gt_psnr"])
            for r in frame_rows
            if r.get("gt_psnr") is not None and int(r.get("frame_id", -1)) == int(frame_id)
        ]
        if anchor_gt_candidates:
            gt_anchor = anchor_gt_candidates[0]
            ax.axhline(
                gt_anchor,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label=f"GT anchor frame ({gt_anchor:.2f} dB)",
                zorder=1,
            )

        ax.set_xlabel("Compressed Size (MB)", fontsize=11)
        ax.set_ylabel("PSNR (dB)", fontsize=11)
        ax.set_title(f"PSNR-Size | {seq_name} | Anchor {frame_id}", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"psnr_size_{seq_name}_anchor{frame_id}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_latency_method_per_frame(rows: list[dict[str, Any]], plot_dir: str) -> None:
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        by_baseline = _group_by(frame_rows, "baseline")
        baselines_present = sorted(by_baseline.keys(), key=lambda b: _baseline_sort_key(by_baseline[b]))
        if not baselines_present:
            continue

        labels = [_style_for_baseline(by_baseline[b])["label"] for b in baselines_present]
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
        ax.set_title(f"Latency by Method | {seq_name} | Anchor {frame_id}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_path = os.path.join(plot_dir, f"latency_method_{seq_name}_anchor{frame_id}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_size_method_per_frame(rows: list[dict[str, Any]], plot_dir: str) -> None:
    for seq_name, frame_id, frame_rows in _iter_frame_groups(rows):
        by_baseline = _group_by(frame_rows, "baseline")
        baselines_present = sorted(by_baseline.keys(), key=lambda b: _baseline_sort_key(by_baseline[b]))
        if not baselines_present:
            continue

        labels = [_style_for_baseline(by_baseline[b])["label"] for b in baselines_present]
        compressed_vals: list[float] = []
        uncompressed_vals: list[float] = []
        colors: list[str] = []
        for bl in baselines_present:
            bl_rows = by_baseline[bl]
            comp = [float(r["compressed_mb"]) for r in bl_rows if r.get("compressed_mb") is not None]
            uncomp = [float(r["uncompressed_mb"]) for r in bl_rows if r.get("uncompressed_mb") is not None]
            compressed_vals.append(float(np.mean(comp)) if comp else np.nan)
            uncompressed_vals.append(float(np.mean(uncomp)) if uncomp else np.nan)
            colors.append(_style_for_baseline(bl_rows)["color"])

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
        ax.set_title(f"Size by Method | {seq_name} | Anchor {frame_id}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)

        legend_handles = []
        legend_labels = []
        for i, bl in enumerate(baselines_present):
            bl_label = _style_for_baseline(by_baseline[bl])["label"]
            legend_handles.append(bars[i])
            if np.isfinite(uncompressed_vals[i]):
                legend_labels.append(
                    f"{bl_label} (Uncompressed: {uncompressed_vals[i]:.2f} MB)"
                )
            else:
                legend_labels.append(f"{bl_label} (Uncompressed: N/A)")
        ax.legend(legend_handles, legend_labels, fontsize=8, loc="upper left")

        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"size_method_{seq_name}_anchor{frame_id}.png")
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
    sorted_baselines = sorted(by_baseline.keys(), key=lambda b: _baseline_sort_key(by_baseline[b]))
    for bl in sorted_baselines:
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
