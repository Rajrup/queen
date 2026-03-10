#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Collect baseline outputs and generate sequence-level comparison plots for QUEEN."""

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "baseline_comparison_res")

LARGE_CSV_COLUMNS = [
    "sequence_name",
    "baseline",
    "frame_id",
    "gt_psnr",
    "gt_ssim",
    "decomp_psnr",
    "decomp_ssim",
    "size",
]


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


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

    available_frame_ids = sorted(set(benchmark_by_frame.keys()) & set(metrics_by_frame.keys()))
    selected_frame_ids = list(frame_ids)
    unavailable_frame_ids = sorted(set(selected_frame_ids) - set(available_frame_ids))
    if unavailable_frame_ids:
        print(
            f"  [INFO] {baseline} | {sequence}: unavailable requested frames "
            f"{unavailable_frame_ids}; available frames {available_frame_ids}"
        )
    if (
        selected_frame_ids
        and available_frame_ids
        and not any(fid in available_frame_ids for fid in selected_frame_ids)
    ):
        print(
            f"  [INFO] {baseline} | {sequence}: requested frames {selected_frame_ids} "
            f"not present in both benchmark/eval; using available frames {available_frame_ids}"
        )
        selected_frame_ids = available_frame_ids

    for fid in selected_frame_ids:
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
            0: "#1b9e77",
            4: "#d95f02",
            10: "#7570b3",
            15: "#e7298a",
            20: "#66a61e",
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


def write_large_csv(rows: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LARGE_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence_name": row.get("sequence_name", ""),
                    "baseline": row.get("baseline", ""),
                    "frame_id": row.get("frame_id", ""),
                    "gt_psnr": row.get("gt_psnr", ""),
                    "gt_ssim": row.get("gt_ssim", ""),
                    "decomp_psnr": row.get("decomp_psnr", ""),
                    "decomp_ssim": row.get("decomp_ssim", ""),
                    "size": row.get("compressed_size_bytes", ""),
                }
            )
    print(f"  Wrote {len(rows)} rows to: {path}")


def _values_by_frame(
    seq_rows: list[dict[str, Any]],
    key: str,
    allowed_frame_ids: Optional[set[int]] = None,
    reducer: str = "mean",
) -> list[float]:
    by_frame: dict[int, list[float]] = {}
    for r in seq_rows:
        fid = int(r["frame_id"])
        if allowed_frame_ids is not None and fid not in allowed_frame_ids:
            continue
        v = r.get(key)
        if v is None or v == 0:
            continue
        by_frame.setdefault(fid, []).append(float(v))
    if reducer == "max":
        return [float(np.max(by_frame[fid])) for fid in sorted(by_frame.keys())]
    return [float(np.mean(by_frame[fid])) for fid in sorted(by_frame.keys())]


def _sequence_common_frame_ids(
    seq_by_bl: dict[str, list[dict[str, Any]]], baseline_labels: list[str]
) -> set[int]:
    present_baselines = [
        bl for bl in baseline_labels if seq_by_bl.get(bl)
    ]
    if not present_baselines:
        return set()

    frame_sets = [
        {int(r["frame_id"]) for r in seq_by_bl[bl]}
        for bl in present_baselines
    ]

    common = set.intersection(*frame_sets)
    if common:
        return common

    union: set[int] = set()
    for frame_set in frame_sets:
        union.update(frame_set)
    return union


def _get_ordered_baselines(rows: list[dict[str, Any]]) -> list[str]:
    by_bl = _group_by(rows, "baseline")
    return sorted(by_bl.keys(), key=lambda b: _baseline_sort_key(by_bl[b]))


def _bar_colors(
    baseline_labels: list[str],
    by_baseline: dict[str, list[dict[str, Any]]],
) -> list[str]:
    colors = ["#999999"]
    for bl in baseline_labels:
        style = _style_for_baseline(by_baseline[bl])
        colors.append(style["color"])
    return colors


def plot_size_by_sequence(rows: list[dict[str, Any]], plot_dir: str) -> None:
    sequences = list(dict.fromkeys(r["sequence_name"] for r in rows))
    baseline_labels = _get_ordered_baselines(rows)
    by_seq = _group_by(rows, "sequence_name")
    by_baseline = _group_by(rows, "baseline")

    bar_labels = ["Uncompressed"] + list(baseline_labels)
    x_labels = sequences + ["Average"]
    n_groups = len(x_labels)
    n_bars = len(bar_labels)

    means = np.zeros((n_bars, n_groups))
    stds = np.zeros((n_bars, n_groups))
    aggregate_values: list[list[float]] = [[] for _ in range(n_bars)]

    for j, seq in enumerate(sequences):
        seq_rows = by_seq.get(seq, [])
        seq_by_bl = _group_by(seq_rows, "baseline")
        common_frame_ids = _sequence_common_frame_ids(seq_by_bl, baseline_labels)

        vals = _values_by_frame(seq_rows, "uncompressed_mb", common_frame_ids)
        if vals:
            means[0, j] = float(np.mean(vals))
            stds[0, j] = float(np.std(vals))
            aggregate_values[0].extend(vals)

        for i, bl in enumerate(baseline_labels):
            vals = _values_by_frame(seq_by_bl.get(bl, []), "compressed_mb", common_frame_ids)
            if vals:
                means[i + 1, j] = float(np.mean(vals))
                stds[i + 1, j] = float(np.std(vals))
                aggregate_values[i + 1].extend(vals)

    for i, vals in enumerate(aggregate_values):
        if vals:
            means[i, -1] = float(np.mean(vals))
            stds[i, -1] = float(np.std(vals))

    colors = _bar_colors(baseline_labels, by_baseline)
    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2.2), 7))
    x = np.arange(n_groups, dtype=float)
    width = 0.8 / n_bars

    for i in range(n_bars):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means[i],
            width,
            yerr=stds[i],
            label=bar_labels[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Size (MB, log scale)", fontsize=11)
    ax.set_title("Compressed Size by Sequence", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out_path = os.path.join(plot_dir, "size_by_sequence.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_quality_by_sequence(
    rows: list[dict[str, Any]],
    plot_dir: str,
    decomp_key: str,
    gt_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    sequences = list(dict.fromkeys(r["sequence_name"] for r in rows))
    baseline_labels = _get_ordered_baselines(rows)
    by_seq = _group_by(rows, "sequence_name")
    by_baseline = _group_by(rows, "baseline")

    bar_labels = ["Uncompressed"] + list(baseline_labels)
    x_labels = sequences + ["Average"]
    n_groups = len(x_labels)
    n_bars = len(bar_labels)

    means = np.zeros((n_bars, n_groups))
    stds = np.zeros((n_bars, n_groups))
    aggregate_values: list[list[float]] = [[] for _ in range(n_bars)]

    for j, seq in enumerate(sequences):
        seq_rows = by_seq.get(seq, [])
        seq_by_bl = _group_by(seq_rows, "baseline")
        common_frame_ids = _sequence_common_frame_ids(seq_by_bl, baseline_labels)

        vals = _values_by_frame(seq_rows, gt_key, common_frame_ids, reducer="max")
        if vals:
            means[0, j] = float(np.mean(vals))
            stds[0, j] = float(np.std(vals))
            aggregate_values[0].extend(vals)

        for i, bl in enumerate(baseline_labels):
            vals = _values_by_frame(seq_by_bl.get(bl, []), decomp_key, common_frame_ids)
            if vals:
                means[i + 1, j] = float(np.mean(vals))
                stds[i + 1, j] = float(np.std(vals))
                aggregate_values[i + 1].extend(vals)

    for i, vals in enumerate(aggregate_values):
        if vals:
            means[i, -1] = float(np.mean(vals))
            stds[i, -1] = float(np.std(vals))

    colors = _bar_colors(baseline_labels, by_baseline)
    fig, ax = plt.subplots(figsize=(max(14, n_groups * 2.2), 7))
    x = np.arange(n_groups, dtype=float)
    width = 0.8 / n_bars

    for i in range(n_bars):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(
            x + offset,
            means[i],
            width,
            yerr=stds[i],
            label=bar_labels[i],
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out_path = os.path.join(plot_dir, filename)
    fig.savefig(out_path, dpi=150)
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

    large_csv_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    write_large_csv(rows, large_csv_path)

    print(f"\n{sep}\nStep 2: Generate plots\n{sep}")
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for name in os.listdir(plot_dir):
        if name.endswith(".png"):
            os.remove(os.path.join(plot_dir, name))

    plot_size_by_sequence(rows, plot_dir)
    plot_quality_by_sequence(
        rows,
        plot_dir,
        "decomp_psnr",
        "gt_psnr",
        "PSNR (dB)",
        "PSNR by Sequence",
        "psnr_by_sequence.png",
    )
    plot_quality_by_sequence(
        rows,
        plot_dir,
        "decomp_ssim",
        "gt_ssim",
        "SSIM",
        "SSIM by Sequence",
        "ssim_by_sequence.png",
    )

    print(f"\n{sep}")
    print(f"Done! All outputs in: {OUTPUT_DIR}")
    print(sep)


if __name__ == "__main__":
    main()
