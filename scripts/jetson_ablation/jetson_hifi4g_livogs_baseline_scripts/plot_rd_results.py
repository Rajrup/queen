#!/usr/bin/env python3
"""Collect LiVoGS RD experiment results and produce RD-curve plots.

Flow:
  1. Collect results from experiment directories into a CSV
     (reuses collect_rd_results.py logic)
  2. Generate RD-curve plots from the collected CSV

Edit the "Global configuration" section below, then run::

    python scripts/livogs_baseline/plot_rd_results.py
"""

import itertools
import json
import os
import sys
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from collect_rd_results import collect_rd_root, _infer_sequence_name
from rd_pipeline.plot import plot_rd


# ---------------------------------------------------------------------------
# Global configuration  (edit here)
# ---------------------------------------------------------------------------

# Each entry is a dict describing one livogs_rd/ root to scan.
# Required key:
#   "path"       — absolute path to a livogs_rd/ directory
# Optional keys:
#   "name"       — sequence name override (default: inferred from path)
#   "frame_ids"  — list of frame IDs to collect (default: all frames)
RD_OUTPUT_ROOTS: list[dict[str, Any]] = [
    # --- VideoGS ---
    {
        "path": "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor1_Greeting/compression/livogs_rd",
        "frame_ids": [0],
    },
    # {
    #     "path": "/synology/rajrup/VideoGS/train_output/HiFi4G_Dataset/4K_Actor3_Violin/compression/livogs_rd",
    #     "frame_ids": [0],
    # },

    # --- Queen ---
    # {
    #     "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/compression/livogs_rd",
    #     "frame_ids": [1],
    # },
    # {
    #     "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/livogs_rd",
    #     "frame_ids": [1],
    # },
]

# Plot specifications.  Each group has an optional psnr_range and a list of
# plot specs.  Each spec defines:
#   "curve_var" — the knob to sweep (separate RD curve per value)
#   "fixed"     — dict of knob=value filters applied before plotting
#
PLOT_GROUPS: list[dict[str, Any]] = [
]
PLOT_CONFIG_JSONS: list[str] = [
    os.path.join(SCRIPT_DIR, "plot_configs", "default.json"),
]

# Output directory for plots.  None = place inside first RD root under plots/.
PLOT_OUTPUT_DIR: Optional[str] = None

# Collected CSV path.  None = auto (first RD root / collected_rd_results.csv).
COLLECTED_CSV: Optional[str] = None
FORCE_COLLECT: bool = False


# ---------------------------------------------------------------------------
# Plot config loading  (same JSON format as the old pipeline)
# ---------------------------------------------------------------------------

def _parse_one_plot_group(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    plots: list[dict[str, Any]] = cfg.get("plots", [])
    raw_range = cfg.get("psnr_range")
    psnr_range: Optional[tuple[float, float]] = None
    if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
        try:
            psnr_range = (float(raw_range[0]), float(raw_range[1]))
        except (TypeError, ValueError):
            pass
    return {"source": source, "plots": plots, "psnr_range": psnr_range}


def load_plot_config(json_path: str) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg, list):
        return [_parse_one_plot_group(e, json_path) for e in cfg if isinstance(e, dict)]
    return [_parse_one_plot_group(cfg, json_path)]


def resolve_plot_groups() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group in PLOT_GROUPS:
        groups.append(_parse_one_plot_group(group, "inline"))
    for path in PLOT_CONFIG_JSONS:
        abs_path = path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)
        if not os.path.isfile(abs_path):
            print(f"[WARN] Plot config JSON not found: {abs_path}")
            continue
        try:
            loaded = load_plot_config(abs_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"[WARN] Failed to load plot config {abs_path}: {exc}")
            continue
        groups.extend(loaded)
        total_specs = sum(len(g["plots"]) for g in loaded)
        print(f"[INFO] Loaded plot config: {abs_path} ({len(loaded)} group(s), {total_specs} specs)")
    if not groups:
        print("[WARN] No plot specs defined.")
    return groups


# ---------------------------------------------------------------------------
# Collection (delegates to collect_rd_results)
# ---------------------------------------------------------------------------

def _default_collected_csv(rd_root: str) -> str:
    return os.path.join(rd_root, "collected_rd_results.csv")


def _default_plot_dir(rd_root: str) -> str:
    return os.path.join(rd_root, "plots")


def collect_all() -> list[dict[str, str]]:
    import csv as _csv
    from collect_rd_results import CSV_COLUMNS

    collected: list[dict[str, str]] = []

    if COLLECTED_CSV is not None and len(RD_OUTPUT_ROOTS) > 1:
        print("[WARN] COLLECTED_CSV is ignored when multiple RD_OUTPUT_ROOTS are configured.")
        print("       Writing one CSV per root instead.")

    for entry in RD_OUTPUT_ROOTS:
        rd_root = entry["path"]
        seq_name = entry.get("name") or _infer_sequence_name(rd_root)
        frame_ids = entry.get("frame_ids")
        csv_path = entry.get("output_csv")
        if csv_path is None:
            if COLLECTED_CSV is not None and len(RD_OUTPUT_ROOTS) == 1:
                csv_path = COLLECTED_CSV
            else:
                csv_path = _default_collected_csv(rd_root)

        if not FORCE_COLLECT and os.path.exists(csv_path):
            print(f"  CSV already exists, skipping collect: {csv_path}")
            collected.append({
                "rd_root": rd_root,
                "sequence_name": seq_name,
                "csv_path": csv_path,
            })
            continue

        print(f"Collecting: {seq_name}  ({rd_root})")
        rows = collect_rd_root(rd_root, seq_name, frame_ids=frame_ids)
        print(f"  {len(rows)} result(s)")

        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})

        print(f"  Wrote {len(rows)} rows to: {csv_path}\n")
        collected.append({
            "rd_root": rd_root,
            "sequence_name": seq_name,
            "csv_path": csv_path,
        })

    if not collected:
        print("[WARN] No RD roots processed.")

    return collected


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

KNOB_NAMES = frozenset({"depth", "qp_sh", "beta", "qp_quats", "qp_scales", "qp_opacity"})
CONFIG_KNOB_NAMES = KNOB_NAMES


def generate_plots(csv_path: str, plot_dir: str, plot_groups: list[dict[str, Any]]) -> None:
    """Generate RD-curve PNGs from the collected CSV."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    # Infer sequence names and frame IDs present in the CSV
    import csv as _csv
    seq_frames: dict[str, set[int]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            sn = row.get("sequence_name", "unknown")
            fid = int(row.get("frame_id", 0))
            seq_frames.setdefault(sn, set()).add(fid)

    for group in plot_groups:
        psnr_range = group.get("psnr_range")
        for plot_spec in group.get("plots", []):
            curve_var = plot_spec.get("curve_var")
            raw_fixed = plot_spec.get("fixed", {})
            curve_values = plot_spec.get("curve_values")

            if curve_var not in CONFIG_KNOB_NAMES:
                print(f"[WARN] Invalid curve_var '{curve_var}'; skipping.")
                continue
            invalid_keys = [k for k in raw_fixed if k not in CONFIG_KNOB_NAMES]
            if invalid_keys:
                print(f"[WARN] Invalid fixed keys {invalid_keys}; skipping.")
                continue

            # Expand list-valued fixed keys into cartesian product
            list_keys = sorted(
                k for k, v in raw_fixed.items() if isinstance(v, (list, tuple))
            )
            scalar_fixed = {
                k: v for k, v in raw_fixed.items() if not isinstance(v, (list, tuple))
            }

            if list_keys:
                combos = list(itertools.product(
                    *(raw_fixed[k] for k in list_keys)
                ))
            else:
                combos = [()]

            for combo in combos:
                fixed = dict(scalar_fixed)
                for k, v in zip(list_keys, combo):
                    fixed[k] = v

                fixed_tag = "_".join(
                    f"{k}{v}" for k, v in sorted(fixed.items())
                )

                for seq_name, frame_ids in sorted(seq_frames.items()):
                    for frame_id in sorted(frame_ids):
                        output_path = os.path.join(
                            plot_dir,
                            f"rd_{seq_name}_frame{frame_id}_{curve_var}_sweep_{fixed_tag}.png",
                        )
                        plot_rd(
                            csv_path=csv_path,
                            curve_var=curve_var,
                            fixed=fixed,
                            output_path=output_path,
                            sequence_name=seq_name,
                            frame_id=frame_id,
                            psnr_range=psnr_range,
                            curve_values=curve_values,
                        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 70
    print(sep)
    print("LiVoGS RD Plot Pipeline")
    print(f"  RD roots:     {len(RD_OUTPUT_ROOTS)}")
    print(f"  Config JSONs: {PLOT_CONFIG_JSONS or '(none)'}")
    print(sep)

    # Step 1: Collect
    print(f"\n{sep}\nStep 1: Collect results\n{sep}")
    collected_infos = collect_all()

    # Step 2: Plot
    print(f"{sep}\nStep 2: Generate plots\n{sep}")
    plot_groups = resolve_plot_groups()
    total_specs = sum(len(g.get("plots", [])) for g in plot_groups)
    print(f"  {len(plot_groups)} group(s), {total_specs} plot spec(s)")

    for info in collected_infos:
        rd_root = info["rd_root"]
        csv_path = info["csv_path"]
        seq_name = info["sequence_name"]
        plot_dir = PLOT_OUTPUT_DIR if PLOT_OUTPUT_DIR is not None else _default_plot_dir(rd_root)

        print(f"  Sequence: {seq_name}")
        print(f"    CSV:    {csv_path}")
        print(f"    Output: {plot_dir}")
        generate_plots(csv_path, plot_dir, plot_groups)

    print(f"\n{sep}")
    print("Done.")
    print(sep)


if __name__ == "__main__":
    main()
