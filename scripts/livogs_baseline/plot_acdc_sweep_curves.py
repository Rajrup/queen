#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.livogs_baseline.collect_rd_results import collect_rd_root
from scripts.livogs_baseline.rd_pipeline import config

# ---------------------------------------------------------------------------
# Plot configuration  (edit here — independent of experiment runner)
# ---------------------------------------------------------------------------

DATA_PATH = config.DATA_PATH
RD_SUBDIR = "livogs_rd_new"
HULL_SOURCE_SUBDIR = "livogs_rd_nvcomp"
FIG_DPI = 150
SWEEP_LABEL_PREFIX = "hullleft_gtloss02"

PLOT_SEQUENCES: list[dict[str, str]] = [
    {
        "dataset_name": "Neural_3D_Video",
        "sequence_name": "sear_steak",
        "qp_dir_name": "DyNeRF_sear_steak",
    },
]
PLOT_FRAME_IDS: list[int] = [1]
PLOT_SEED_INDICES: list[int] | None = None
WRITE_ZOOMED_FIGURE = True
ZOOM_PLOT_SEED_INDICES: list[int] | None = [5, 6, 7, 8]
ZOOM_X_MARGIN_RATIO = 0.08
ZOOM_Y_MARGIN_RATIO = 0.12

DC_QP_SWEEP: list[float] | None = [v / 255.0 for v in (1, 2, 4, 8, 16)]
AC_QP_SWEEP: list[float] | None = [v / 255.0 for v in (0.1, 1, 4, 8, 16, 32, 64, 100, 128)]
SEED_LEFT_INCLUSIVE = True
MAX_SEED_POINTS: int | None = None
MATCH_EPS = 1e-9
NEAR_LOSSLESS_DROP_DB = 0.2
FORCE_RECOLLECT_SUMMARY = True

LABEL_PATTERN = re.compile(
    rf"^{re.escape(SWEEP_LABEL_PREFIX)}_cart_dc_([0-9emp]+)_ac_([0-9emp]+)_d(\d+)_seed(\d+)$"
)


@dataclass
class AnchorPoint:
    depth: int
    qp_sh: float
    qp_quats: float
    qp_scales: float
    qp_opacity: float
    decomp_psnr: float
    compressed_mb: float


def _load_rows(csv_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "step": str(row.get("step", "")),
                    "label": str(row.get("label", "")),
                    "depth": int(row["depth"]) if row.get("depth") not in (None, "") else -1,
                    "seed_idx": int(row["seed_idx"]) if row.get("seed_idx") not in (None, "") else 0,
                    "qp_dc": float(row["qp_dc"]) if row.get("qp_dc") not in (None, "") else float("nan"),
                    "qp_ac": float(row["qp_ac"]) if row.get("qp_ac") not in (None, "") else float("nan"),
                    "qp_quats": float(row["qp_quats"]) if row.get("qp_quats") not in (None, "") else float("nan"),
                    "qp_scales": float(row["qp_scales"]) if row.get("qp_scales") not in (None, "") else float("nan"),
                    "qp_opacity": float(row["qp_opacity"]) if row.get("qp_opacity") not in (None, "") else float("nan"),
                    "beta": float(row["beta"]) if row.get("beta") not in (None, "") else float("nan"),
                    "decomp_psnr": float(row["decomp_psnr"]),
                    "compressed_mb": float(row["compressed_mb"]),
                }
            )
    return rows


def _parse_qp_token(tok: str) -> float:
    return float(tok.replace("m", "-").replace("p", "."))


def _parse_cartesian_label(label: str) -> tuple[float, float, int, int] | None:
    m = LABEL_PATTERN.match(label)
    if m is None:
        return None
    dc_tok, ac_tok, depth_tok, seed_tok = m.groups()
    try:
        return _parse_qp_token(dc_tok), _parse_qp_token(ac_tok), int(depth_tok), int(seed_tok)
    except ValueError:
        return None


def _collect_target_rows(dataset_name: str, sequence_name: str, frame_id: int) -> list[dict[str, Any]]:
    rd_root = config.rd_output_root(
        DATA_PATH,
        dataset_name,
        sequence_name,
        rd_subdir_name=RD_SUBDIR,
    )
    return collect_rd_root(rd_root, sequence_name, frame_ids=[frame_id])


def _build_summary_rows_from_collected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label", ""))
        parsed = _parse_cartesian_label(label)
        if parsed is None:
            continue
        dc_qp, ac_qp, depth_from_label, seed_idx = parsed
        if depth_from_label != int(row["depth"]):
            continue
        summary_rows.append(
            {
                "step": "cartesian_acdc",
                "label": label,
                "depth": int(row["depth"]),
                "seed_idx": int(seed_idx),
                "qp_dc": float(dc_qp),
                "qp_ac": float(ac_qp),
                "qp_quats": float(row.get("qp_quats", math.nan)),
                "qp_scales": float(row.get("qp_scales", math.nan)),
                "qp_opacity": float(row.get("qp_opacity", math.nan)),
                "beta": float(row.get("beta", math.nan)),
                "decomp_psnr": float(row["decomp_psnr"]),
                "compressed_mb": float(row["compressed_mb"]),
            }
        )
    summary_rows.sort(key=lambda x: (x["seed_idx"], x["depth"], x["qp_dc"], x["qp_ac"]))
    return summary_rows


def _write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "step",
        "label",
        "depth",
        "seed_idx",
        "qp_dc",
        "qp_ac",
        "qp_quats",
        "qp_scales",
        "qp_opacity",
        "beta",
        "decomp_psnr",
        "compressed_mb",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find_hull_csv(dataset_name: str, sequence_name: str, frame_id: int) -> str | None:
    rd_root = config.rd_output_root(
        DATA_PATH,
        dataset_name,
        sequence_name,
        rd_subdir_name=HULL_SOURCE_SUBDIR,
    )
    plot_dir = os.path.join(rd_root, "plots")
    expected = [
        os.path.join(plot_dir, f"convex_hull_{dataset_name}_{sequence_name}_frame{frame_id}.csv"),
        os.path.join(plot_dir, f"convex_hull_{dataset_name}_{HULL_SOURCE_SUBDIR}_frame{frame_id}.csv"),
    ]
    for path in expected:
        if os.path.exists(path):
            return path

    matches = sorted(glob.glob(os.path.join(plot_dir, f"convex_hull_*_frame{frame_id}.csv")))
    if len(matches) == 1:
        return matches[0]
    return None


def _load_hull_points(hull_csv_path: str) -> list[AnchorPoint]:
    points: list[AnchorPoint] = []
    with open(hull_csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                points.append(
                    AnchorPoint(
                        depth=int(round(float(row["depth"]))),
                        qp_sh=float(row["qp_sh"]),
                        qp_quats=float(row["qp_quats"]),
                        qp_scales=float(row["qp_scales"]),
                        qp_opacity=float(row["qp_opacity"]),
                        decomp_psnr=float(row["decomp_psnr"]),
                        compressed_mb=float(row["compressed_mb"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    points.sort(key=lambda p: p.compressed_mb)
    return points


def _load_collected_rows(dataset_name: str, sequence_name: str) -> list[dict[str, Any]]:
    rd_root = config.rd_output_root(
        DATA_PATH,
        dataset_name,
        sequence_name,
        rd_subdir_name=HULL_SOURCE_SUBDIR,
    )
    csv_path = os.path.join(rd_root, "collected_rd_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing collected RD CSV: {csv_path}")

    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)
            for key in ("frame_id", "depth"):
                row[key] = int(row[key])
            for key in (
                "qp_sh",
                "beta",
                "qp_quats",
                "qp_scales",
                "qp_opacity",
                "gt_psnr",
                "decomp_psnr",
                "psnr_drop",
                "compressed_mb",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def _is_close(a: float, b: float, eps: float = MATCH_EPS) -> bool:
    return abs(a - b) <= eps


def _match_anchor_row(anchor: AnchorPoint, rows: list[dict[str, Any]], frame_id: int) -> dict[str, Any]:
    exact = [
        r
        for r in rows
        if r["frame_id"] == frame_id
        and int(r["depth"]) == anchor.depth
        and _is_close(float(r["qp_sh"]), anchor.qp_sh)
        and _is_close(float(r["qp_quats"]), anchor.qp_quats)
        and _is_close(float(r["qp_scales"]), anchor.qp_scales)
        and _is_close(float(r["qp_opacity"]), anchor.qp_opacity)
    ]
    if exact:
        exact.sort(key=lambda r: (-float(r["decomp_psnr"]), float(r["compressed_mb"])))
        return exact[0]

    nearest: tuple[float, dict[str, Any]] | None = None
    for r in rows:
        if r["frame_id"] != frame_id:
            continue
        dist = 0.0
        dist += (float(r["depth"]) - float(anchor.depth)) ** 2
        dist += (float(r["qp_sh"]) - anchor.qp_sh) ** 2
        dist += (float(r["qp_quats"]) - anchor.qp_quats) ** 2
        dist += (float(r["qp_scales"]) - anchor.qp_scales) ** 2
        dist += (float(r["qp_opacity"]) - anchor.qp_opacity) ** 2
        if nearest is None or dist < nearest[0]:
            nearest = (dist, r)
    if nearest is None:
        raise RuntimeError("No candidate rows for anchor matching.")
    return nearest[1]


def _pick_global_near_lossless_anchor(
    hull_points: list[AnchorPoint],
    rows: list[dict[str, Any]],
    frame_id: int,
) -> tuple[AnchorPoint, dict[str, Any]]:
    candidates: list[tuple[AnchorPoint, dict[str, Any]]] = []
    for p in hull_points:
        row = _match_anchor_row(p, rows, frame_id)
        psnr_drop = float(row.get("psnr_drop", row.get("gt_psnr", 0.0) - row.get("decomp_psnr", 0.0)))
        if psnr_drop <= NEAR_LOSSLESS_DROP_DB:
            candidates.append((p, row))
    if not candidates:
        raise RuntimeError(
            f"No hull points satisfy near-lossless criterion psnr_drop <= {NEAR_LOSSLESS_DROP_DB} dB"
        )
    candidates.sort(key=lambda x: x[0].compressed_mb)
    return candidates[0]


def _select_seed_hull_points(hull_points: list[AnchorPoint], left_near_lossless: AnchorPoint) -> list[AnchorPoint]:
    anchor_mb = left_near_lossless.compressed_mb
    if SEED_LEFT_INCLUSIVE:
        seeds = [p for p in hull_points if p.compressed_mb <= anchor_mb + MATCH_EPS]
    else:
        seeds = [p for p in hull_points if p.compressed_mb < anchor_mb - MATCH_EPS]
    seeds.sort(key=lambda p: p.compressed_mb)
    if MAX_SEED_POINTS is not None:
        seeds = seeds[: max(0, int(MAX_SEED_POINTS))]
    return seeds


def _load_anchor_qp_config(
    dataset_name: str,
    sequence_name: str,
    frame_id: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    exp_dir = config.experiment_dir(
        DATA_PATH,
        dataset_name,
        sequence_name,
        frame_id,
        int(row["depth"]),
        str(row["label"]),
        rd_subdir_name=HULL_SOURCE_SUBDIR,
    )
    qp_json = os.path.join(exp_dir, "qp_config.json")
    if not os.path.exists(qp_json):
        raise FileNotFoundError(f"Anchor qp_config.json not found: {qp_json}")
    with open(qp_json, encoding="utf-8") as f:
        return json.load(f)


def _build_seed_expectations(
    seed_points: list[AnchorPoint],
    collected_rows: list[dict[str, Any]],
    dataset_name: str,
    sequence_name: str,
    frame_id: int,
) -> dict[int, dict[str, float]]:
    expectations: dict[int, dict[str, float]] = {}
    for seed_idx, seed in enumerate(seed_points):
        row = _match_anchor_row(seed, collected_rows, frame_id)
        seed_cfg = _load_anchor_qp_config(dataset_name, sequence_name, frame_id, row)
        quantize_cfg = seed_cfg.get("quantize_config", {})
        expectations[seed_idx] = {
            "qp_quats": float(seed_cfg.get("qp_quats", quantize_cfg.get("quats", math.nan))),
            "qp_scales": float(seed_cfg.get("qp_scales", quantize_cfg.get("scales", math.nan))),
            "qp_opacity": float(seed_cfg.get("qp_opacity", quantize_cfg.get("opacity", math.nan))),
            "beta": float(seed_cfg.get("beta", row.get("beta", math.nan))),
        }
    return expectations


def _qp_in_sweep(value: float, sweep: list[float] | None, eps: float = 1e-9) -> bool:
    if sweep is None:
        return True
    return any(abs(value - s) <= eps for s in sweep)


def _qp_matches_configured(value: float, sweep: list[float] | None, scaled_eps: float = 1e-3) -> bool:
    if sweep is None:
        return True
    value_255 = float(value) * 255.0
    return any(abs(value_255 - (float(s) * 255.0)) <= scaled_eps for s in sweep)


def _snap_qp_to_configured(value: float, sweep: list[float] | None, scaled_eps: float = 1e-3) -> float:
    if sweep is None:
        return float(value)
    value_255 = float(value) * 255.0
    best: tuple[float, float] | None = None
    for candidate in sweep:
        candidate_255 = float(candidate) * 255.0
        dist = abs(value_255 - candidate_255)
        if best is None or dist < best[0]:
            best = (dist, float(candidate))
    if best is not None and best[0] <= scaled_eps:
        return best[1]
    return float(value)


def _filter_cartesian_rows(
    rows: list[dict[str, Any]],
    seed_points: list[AnchorPoint],
    seed_expectations: dict[int, dict[str, float]],
    eps: float = 1e-9,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for r in rows:
        if r["step"] != "cartesian_acdc":
            continue
        parsed = _parse_cartesian_label(str(r["label"]))
        if parsed is None:
            continue
        dc_from_label, ac_from_label, depth_from_label, seed_idx = parsed
        if seed_idx < 0 or seed_idx >= len(seed_points):
            continue
        if depth_from_label != int(r["depth"]):
            continue
        if int(r["depth"]) != seed_points[seed_idx].depth:
            continue
        dc_value = float(r.get("qp_dc", dc_from_label))
        ac_value = float(r.get("qp_ac", ac_from_label))
        if not _qp_matches_configured(dc_value, DC_QP_SWEEP):
            continue
        if not _qp_matches_configured(ac_value, AC_QP_SWEEP):
            continue
        expected = seed_expectations.get(seed_idx)
        if expected is None:
            continue
        ok = True
        for key in ("qp_quats", "qp_scales", "qp_opacity", "beta"):
            v = float(r[key])
            exp = float(expected[key])
            if math.isnan(exp):
                continue
            if math.isnan(v) or abs(v - exp) > eps:
                ok = False
                break
        if not ok:
            continue
        normalized = dict(r)
        normalized["seed_idx"] = int(seed_idx)
        normalized["depth"] = int(depth_from_label)
        normalized["qp_dc"] = _snap_qp_to_configured(dc_value, DC_QP_SWEEP)
        normalized["qp_ac"] = _snap_qp_to_configured(ac_value, AC_QP_SWEEP)
        selected.append(normalized)
    selected.sort(key=lambda r: (int(r["seed_idx"]), float(r["qp_dc"]), float(r["qp_ac"])))
    return selected


def _resolve_selected_seed_indices(seed_points: list[AnchorPoint]) -> set[int]:
    all_indices = set(range(len(seed_points)))
    if PLOT_SEED_INDICES is None:
        return all_indices
    selected = {int(v) for v in PLOT_SEED_INDICES}
    valid = {idx for idx in selected if idx in all_indices}
    invalid = sorted(selected - valid)
    if invalid:
        print(f"[WARN] Ignoring out-of-range PLOT_SEED_INDICES: {invalid}")
    return valid


def _resolve_zoom_seed_indices(seed_points: list[AnchorPoint], selected_seed_indices: set[int]) -> set[int]:
    if ZOOM_PLOT_SEED_INDICES is None:
        return set(selected_seed_indices)
    all_indices = set(range(len(seed_points)))
    requested = {int(v) for v in ZOOM_PLOT_SEED_INDICES}
    valid = {idx for idx in requested if idx in all_indices}
    invalid = sorted(requested - valid)
    if invalid:
        print(f"[WARN] Ignoring out-of-range ZOOM_PLOT_SEED_INDICES: {invalid}")
    return valid


def _compute_zoom_limits(
    cart_rows: list[dict[str, Any]],
    seed_points: list[AnchorPoint],
    zoom_seed_indices: set[int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not zoom_seed_indices:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for row in cart_rows:
        if int(row["seed_idx"]) in zoom_seed_indices:
            xs.append(float(row["compressed_mb"]))
            ys.append(float(row["decomp_psnr"]))
    for idx in sorted(zoom_seed_indices):
        if 0 <= idx < len(seed_points):
            xs.append(float(seed_points[idx].compressed_mb))
            ys.append(float(seed_points[idx].decomp_psnr))
    if not xs or not ys:
        return None
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_margin = x_span * ZOOM_X_MARGIN_RATIO
    y_margin = y_span * ZOOM_Y_MARGIN_RATIO
    return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)


def _recollect_summary_if_requested(
    dataset_name: str,
    sequence_name: str,
    frame_id: int,
    summary_csv: str,
    seed_points: list[AnchorPoint],
    seed_expectations: dict[int, dict[str, float]],
) -> bool:
    if not FORCE_RECOLLECT_SUMMARY:
        return True
    print(f"[INFO] Re-collecting summary for {sequence_name} frame {frame_id} using current plot configs.")
    collected = _collect_target_rows(dataset_name, sequence_name, frame_id)
    summary_rows = _build_summary_rows_from_collected(collected)
    summary_rows = _filter_cartesian_rows(summary_rows, seed_points, seed_expectations)
    if not summary_rows:
        if os.path.exists(summary_csv):
            os.remove(summary_csv)
        print(f"[WARN] Re-collection produced no valid summary rows for {sequence_name} frame {frame_id}.")
        return False
    _write_summary_csv(summary_csv, summary_rows)
    print(f"[INFO] Wrote re-collected summary: {summary_csv}")
    return True


def _format_qp_list_255(qps: list[float]) -> str:
    if not qps:
        return "[]"
    scaled = [v * 255.0 for v in sorted(set(qps))]
    items = ",".join(f"{v:.6g}" for v in scaled)
    return f"[{items}]"


def _build_qp_note(rows: list[dict[str, Any]]) -> str:
    observed_dc_vals = sorted({float(r["qp_dc"]) for r in rows})
    observed_ac_vals = sorted({float(r["qp_ac"]) for r in rows})
    configured_dc_vals = list(DC_QP_SWEEP) if DC_QP_SWEEP is not None else observed_dc_vals
    configured_ac_vals = list(AC_QP_SWEEP) if AC_QP_SWEEP is not None else observed_ac_vals
    return (
        f"Configured DC={_format_qp_list_255(configured_dc_vals)}\n"
        f"Configured AC={_format_qp_list_255(configured_ac_vals)}\n"
        f"Observed DC={_format_qp_list_255(observed_dc_vals)}\n"
        f"Observed AC={_format_qp_list_255(observed_ac_vals)}\n"
        "(all /255)"
    )


def _make_plot(
    sequence_name: str,
    frame_id: int,
    rows: list[dict[str, Any]],
    hull_points: list[AnchorPoint],
    seed_points: list[AnchorPoint],
    global_anchor: AnchorPoint,
    seed_expectations: dict[int, dict[str, float]],
    output_path: str,
) -> list[str]:
    if not seed_points:
        return []

    selected_seed_indices = _resolve_selected_seed_indices(seed_points)
    if not selected_seed_indices:
        return []

    cart_rows = _filter_cartesian_rows(rows, seed_points, seed_expectations)
    cart_rows = [r for r in cart_rows if int(r["seed_idx"]) in selected_seed_indices]
    if not cart_rows:
        return []

    zoom_seed_indices = _resolve_zoom_seed_indices(seed_points, selected_seed_indices)
    zoom_limits = _compute_zoom_limits(cart_rows, seed_points, zoom_seed_indices)

    def render_to_path(
        save_path: str,
        x_limits: tuple[float, float] | None = None,
        y_limits: tuple[float, float] | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(11.5, 6))

        by_seed: dict[int, list[dict[str, Any]]] = {}
        for r in cart_rows:
            by_seed.setdefault(int(r["seed_idx"]), []).append(r)

        palette = [
            "#1d4ed8",
            "#d97706",
            "#16a34a",
            "#7c3aed",
            "#dc2626",
            "#0f766e",
            "#9333ea",
            "#b45309",
            "#4f46e5",
            "#0284c7",
        ]
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]
        linestyles = ["-", "--", "-.", ":"]

        for seed_idx in sorted(by_seed.keys()):
            if seed_idx < 0 or seed_idx >= len(seed_points) or seed_idx not in selected_seed_indices:
                continue
            seed = seed_points[seed_idx]
            color = palette[seed_idx % len(palette)]
            marker = markers[seed_idx % len(markers)]
            linestyle = linestyles[seed_idx % len(linestyles)]
            items = sorted(by_seed[seed_idx], key=lambda r: r["compressed_mb"])
            x = [r["compressed_mb"] for r in items]
            y = [r["decomp_psnr"] for r in items]
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                markersize=3,
                linewidth=1.0,
                linestyle=linestyle,
                alpha=0.8,
                label=f"Seed #{seed_idx} (depth={seed.depth})",
                zorder=3,
            )
            end_x = x[-1]
            end_y = y[-1]
            x_offset = 6 + (seed_idx % 3) * 8
            y_offset = -10 + (seed_idx % 5) * 5
            ax.annotate(
                f"{seed_idx}",
                (end_x, end_y),
                textcoords="offset points",
                xytext=(x_offset, y_offset),
                fontsize=8,
                color=color,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.85),
                zorder=6,
            )

        hull_sorted = sorted(hull_points, key=lambda p: p.compressed_mb)
        if len(hull_sorted) >= 2:
            ax.plot(
                [p.compressed_mb for p in hull_sorted],
                [p.decomp_psnr for p in hull_sorted],
                color="red",
                linewidth=2.0,
                marker="D",
                markersize=4,
                label=f"Baseline hull ({len(hull_sorted)} pts)",
                zorder=1,
            )
        elif hull_sorted:
            ax.scatter(
                [hull_sorted[0].compressed_mb],
                [hull_sorted[0].decomp_psnr],
                color="red",
                marker="D",
                s=24,
                label="Baseline hull (1 pt)",
                zorder=1,
            )

        seed_label_added = False
        for idx in sorted(selected_seed_indices):
            seed = seed_points[idx]
            color = palette[idx % len(palette)]
            ax.scatter(
                [seed.compressed_mb],
                [seed.decomp_psnr],
                s=42,
                facecolors="white",
                edgecolors=color,
                linewidths=1.4,
                marker="o",
                label=(f"Seed hull points ({len(selected_seed_indices)}/{len(seed_points)})" if not seed_label_added else "_nolegend_"),
                zorder=7,
            )
            seed_label_added = True
            ax.annotate(
                f"{idx}",
                (seed.compressed_mb, seed.decomp_psnr),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=color,
                weight="bold",
                zorder=8,
            )
        ax.scatter(
            [global_anchor.compressed_mb],
            [global_anchor.decomp_psnr],
            s=72,
            color="#111827",
            marker="*",
            label="Near-lossless anchor",
            zorder=5,
        )

        ax.annotate(
            _build_qp_note(cart_rows)
            + f"\nNear-lossless: drop<={NEAR_LOSSLESS_DROP_DB}dB"
            + "\nEach curve: one seed, DC x AC sweep",
            xy=(1.02, 0.98),
            xycoords="axes fraction",
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

        ax.set_xlabel("Compressed size (MB)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"AC/DC Sweep — {sequence_name} frame {frame_id}")
        ax.grid(True, alpha=0.3)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI)
        plt.close(fig)

    generated_paths = [output_path]
    render_to_path(output_path)
    if WRITE_ZOOMED_FIGURE and zoom_limits is not None:
        zoom_path = os.path.splitext(output_path)[0] + "_zoom.png"
        render_to_path(zoom_path, x_limits=zoom_limits[0], y_limits=zoom_limits[1])
        generated_paths.append(zoom_path)
    return generated_paths


def main() -> None:
    sep = "=" * 72
    print(sep)
    print("Plot AC/DC Sweep PSNR-Size Curves (queen)")
    print(f"  RD subdir: {RD_SUBDIR}")
    print(f"  Force recollect summary: {FORCE_RECOLLECT_SUMMARY}")
    print(f"  Plot seed indices: {PLOT_SEED_INDICES if PLOT_SEED_INDICES is not None else 'all'}")
    print(f"  Sequences: {[s['sequence_name'] for s in PLOT_SEQUENCES]}")
    print(f"  Frames: {PLOT_FRAME_IDS}")
    print(sep)

    generated: list[str] = []
    missing: list[str] = []
    skipped: list[str] = []

    for seq in PLOT_SEQUENCES:
        seq_name = seq["sequence_name"]
        dataset_name = seq["dataset_name"]
        rd_root = config.rd_output_root(
            DATA_PATH,
            dataset_name,
            seq_name,
            rd_subdir_name=RD_SUBDIR,
        )
        plot_dir = config.plot_output_dir(
            DATA_PATH,
            dataset_name,
            seq_name,
            rd_subdir_name=RD_SUBDIR,
        )

        for frame_id in PLOT_FRAME_IDS:
            summary_csv = os.path.join(rd_root, f"acdc_hull_sweep_summary_frame{frame_id}.csv")
            hull_csv = _find_hull_csv(dataset_name, seq_name, frame_id)
            if hull_csv is None:
                skipped.append(summary_csv)
                print(f"[WARN] Missing hull CSV for {seq_name} frame {frame_id}; skip.")
                continue

            hull_points = _load_hull_points(hull_csv)
            try:
                collected_rows = _load_collected_rows(dataset_name, seq_name)
                global_anchor, _ = _pick_global_near_lossless_anchor(hull_points, collected_rows, frame_id)
                seed_points = _select_seed_hull_points(hull_points, global_anchor)
                seed_expectations = _build_seed_expectations(
                    seed_points,
                    collected_rows,
                    dataset_name,
                    seq_name,
                    frame_id,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                skipped.append(summary_csv)
                print(f"[WARN] Cannot reproduce runner seed logic for {seq_name} frame {frame_id}: {exc}")
                continue

            recollect_ok = _recollect_summary_if_requested(
                dataset_name,
                seq_name,
                frame_id,
                summary_csv,
                seed_points,
                seed_expectations,
            )
            if not recollect_ok:
                skipped.append(summary_csv)
                continue

            if not os.path.exists(summary_csv):
                missing.append(summary_csv)
                continue

            rows = _load_rows(summary_csv)

            out_png = os.path.join(plot_dir, f"acdc_psnr_size_curve_frame{frame_id}.png")
            generated_paths = _make_plot(
                seq_name,
                frame_id,
                rows,
                hull_points,
                seed_points,
                global_anchor,
                seed_expectations,
                out_png,
            )
            if generated_paths:
                generated.extend(generated_paths)
                for path in generated_paths:
                    print(f"Generated: {path}")
            else:
                skipped.append(summary_csv)
                print(f"[WARN] No valid cartesian rows found in summary: {summary_csv}")

    print(f"\n{sep}")
    print(f"Plots generated: {len(generated)}")
    for path in generated:
        print(f"  - {path}")
    if missing:
        print(f"Missing summaries: {len(missing)}")
        for path in missing:
            print(f"  - {path}")
    if skipped:
        print(f"Skipped summaries (no valid rows): {len(skipped)}")
        for path in skipped:
            print(f"  - {path}")
    print(sep)


if __name__ == "__main__":
    main()
