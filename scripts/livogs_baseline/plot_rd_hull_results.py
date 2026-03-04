#!/usr/bin/env python3
"""RD scatter + convex hull pipeline with inline global configuration only."""

import os
import sys
import csv
import itertools
from typing import Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from collect_rd_results import collect_rd_root, _infer_sequence_name
from rd_pipeline.plot import plot_rd_scatter

RD_OUTPUT_ROOTS: list[dict[str, Any]] = [
    {
        "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_flame_salmon_1/compression/livogs_rd",
        "frame_ids": [1],
    },
    {
        "path": "/synology/rajrup/Queen/pretrained_output/Neural_3D_Video/queen_compressed_sear_steak/compression/livogs_rd",
        "frame_ids": [1],
    },
]

SCATTER_PSNR_RANGE: Optional[tuple[float, float]] = None

SCATTER_SPEC: dict[str, Any] = {
    "name": "default_qps",
    "fixed": {
        "depth": [12, 13, 14, 15, 16, 17, 18],
        "qp_sh": [v / 255.0 for v in (0.01, 0.1, 0.5, 1, 2, 4, 8, 16)],
        "qp_opacity": [0.001, 0.01, 0.02, 0.04, 0.06],
        "qp_scales": [0.001, 0.01, 0.02, 0.04, 0.06],
        "qp_quats": [0.001, 0.01, 0.02, 0.04, 0.06],
    },
}

FORCE_COLLECT: bool = False
DEFAULT_PSNR_RANGE: Optional[tuple[float, float]] = None
PLOT_OUTPUT_DIR: Optional[str] = None
COLLECTED_CSV: Optional[str] = None

KNOB_NAMES = frozenset({"depth", "qp_sh", "beta", "qp_quats", "qp_scales", "qp_opacity"})


def _normalize_psnr_range(raw_range: Any) -> Optional[tuple[float, float]]:
    if raw_range is None:
        return None
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        print(f"[WARN] Ignoring invalid psnr_range={raw_range!r}: expected [min, max].")
        return None
    try:
        lo = float(raw_range[0])
        hi = float(raw_range[1])
    except (TypeError, ValueError):
        print(f"[WARN] Ignoring invalid psnr_range={raw_range!r}: values must be numeric.")
        return None
    if lo >= hi:
        print(f"[WARN] Ignoring invalid psnr_range={raw_range!r}: min must be < max.")
        return None
    return lo, hi


def _expand_numeric_range(raw_range: dict[str, Any]) -> list[float]:
    try:
        start = float(raw_range["start"])
        stop = float(raw_range["stop"])
        step = float(raw_range.get("step", 1.0))
    except (KeyError, TypeError, ValueError):
        print(f"[WARN] Invalid numeric range {raw_range!r}; expected start/stop/(optional)step.")
        return []

    if step == 0:
        print(f"[WARN] Invalid numeric range {raw_range!r}; step must be non-zero.")
        return []
    if step > 0 and start > stop:
        print(f"[WARN] Invalid numeric range {raw_range!r}; positive step with start>stop.")
        return []
    if step < 0 and start < stop:
        print(f"[WARN] Invalid numeric range {raw_range!r}; negative step with start<stop.")
        return []

    values: list[float] = []
    cur = start
    eps = abs(step) * 1e-9
    max_iters = 100000
    for _ in range(max_iters):
        if step > 0 and cur > stop + eps:
            break
        if step < 0 and cur < stop - eps:
            break
        values.append(round(cur, 12))
        cur += step
    else:
        print(f"[WARN] Range expansion exceeded {max_iters} values for {raw_range!r}; truncating.")

    return values


def _normalize_fixed_value(value: Any) -> Any:
    if isinstance(value, dict) and "start" in value and "stop" in value:
        return _expand_numeric_range(value)
    return value


def _normalize_fixed(raw_fixed: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw_fixed, dict):
        return None
    fixed: dict[str, Any] = {}
    for key, value in raw_fixed.items():
        fixed[key] = _normalize_fixed_value(value)
    return fixed


def _parse_scatter_spec(spec: dict[str, Any], source: str) -> Optional[dict[str, Any]]:
    fixed = _normalize_fixed(spec.get("fixed", {}))
    if fixed is None:
        print(f"[WARN] Invalid fixed field in {source}: {spec!r}")
        return None

    invalid_keys = [k for k in fixed if k not in KNOB_NAMES]
    if invalid_keys:
        print(f"[WARN] Invalid fixed keys in {source}: {invalid_keys}. Skipping spec.")
        return None

    name = spec.get("name")
    return {"name": str(name) if name is not None else None, "fixed": fixed}


def resolve_scatter_config() -> tuple[dict[str, Any], Optional[tuple[float, float]]]:
    parsed = _parse_scatter_spec(SCATTER_SPEC, "inline")
    if parsed is None:
        print("[WARN] Invalid SCATTER_SPEC; using empty fixed filter.")
        parsed = {"name": "all_points", "fixed": {}}

    psnr_range = _normalize_psnr_range(SCATTER_PSNR_RANGE)
    if psnr_range is None:
        psnr_range = DEFAULT_PSNR_RANGE

    return parsed, psnr_range


def _infer_dataset_name(rd_root: str) -> str:
    parts = os.path.normpath(rd_root).split(os.sep)
    for i, part in enumerate(parts):
        if part == "pretrained_output" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown_dataset"


def _default_collected_csv(rd_root: str) -> str:
    return os.path.join(rd_root, "collected_rd_results.csv")


def _default_plot_dir(rd_root: str) -> str:
    return os.path.join(rd_root, "plots")


def _round6(value: float) -> float:
    return round(float(value), 6)


def _matches_fixed(row: dict[str, Any], fixed: dict[str, Any]) -> bool:
    for key, target in fixed.items():
        row_value = row.get(key)
        if row_value is None:
            return False
        if isinstance(target, (list, tuple)):
            if not any(_round6(float(row_value)) == _round6(float(tv)) for tv in target):
                return False
        else:
            if _round6(float(row_value)) != _round6(float(target)):
                return False
    return True


def _expected_missing_combos(
    frame_rows: list[dict[str, Any]],
    fixed: dict[str, Any],
) -> tuple[list[str], set[tuple[float, ...]], set[tuple[float, ...]], set[tuple[float, ...]]]:
    combo_keys = ["depth", "qp_sh", "qp_opacity", "qp_scales", "qp_quats"]
    if not all(k in fixed for k in combo_keys):
        missing_keys = [k for k in combo_keys if k not in fixed]
        return missing_keys, set(), set(), set()

    def value_list(v: Any) -> list[float]:
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return [float(v)]

    expected = set(
        tuple(_round6(v) for v in combo)
        for combo in itertools.product(*(value_list(fixed[k]) for k in combo_keys))
    )

    observed: set[tuple[float, ...]] = set()
    for row in frame_rows:
        if not _matches_fixed(row, fixed):
            continue
        try:
            observed.add(tuple(_round6(float(row[k])) for k in combo_keys))
        except (TypeError, ValueError, KeyError):
            continue

    missing = expected - observed
    return [], expected, observed, missing


def _write_missing_report_csv(
    output_path: str,
    sequence_name: str,
    dataset_name: str,
    frame_id: int,
    missing: set[tuple[float, ...]],
) -> None:
    fieldnames = [
        "dataset_name",
        "sequence_name",
        "frame_id",
        "depth",
        "qp_sh",
        "sh_times_255",
        "qp_opacity",
        "qp_scales",
        "qp_quats",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for depth, qp_sh, qp_opacity, qp_scales, qp_quats in sorted(missing):
            writer.writerow(
                {
                    "dataset_name": dataset_name,
                    "sequence_name": sequence_name,
                    "frame_id": frame_id,
                    "depth": int(round(depth)),
                    "qp_sh": f"{qp_sh:.6g}",
                    "sh_times_255": f"{(qp_sh * 255.0):.6g}",
                    "qp_opacity": f"{qp_opacity:.6g}",
                    "qp_scales": f"{qp_scales:.6g}",
                    "qp_quats": f"{qp_quats:.6g}",
                }
            )


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
        dataset_name = entry.get("dataset") or _infer_dataset_name(rd_root)
        frame_ids = entry.get("frame_ids")
        csv_path = entry.get("output_csv")
        if csv_path is None:
            if COLLECTED_CSV is not None and len(RD_OUTPUT_ROOTS) == 1:
                csv_path = COLLECTED_CSV
            else:
                csv_path = _default_collected_csv(rd_root)

        if not FORCE_COLLECT and os.path.exists(csv_path):
            print(f"  CSV already exists, skipping collect: {csv_path}")
            collected.append({"rd_root": rd_root, "sequence_name": seq_name, "dataset_name": dataset_name, "csv_path": csv_path})
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
        collected.append({"rd_root": rd_root, "sequence_name": seq_name, "dataset_name": dataset_name, "csv_path": csv_path})

    if not collected:
        print("[WARN] No RD roots processed.")

    return collected


def generate_scatter_plots(
    csv_path: str,
    plot_dir: str,
    sequence_name: str,
    dataset_name: str,
    fixed: dict[str, Any],
    psnr_range: Optional[tuple[float, float]],
) -> None:
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    all_rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in KNOB_NAMES:
                raw = parsed.get(key)
                if key == "qp_sh" and raw in (None, ""):
                    raw = parsed.get("qp_sh", parsed.get("qp_sh"))
                if raw in (None, ""):
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(raw)
                    except (TypeError, ValueError):
                        parsed[key] = None
            all_rows.append(parsed)

    frame_ids: set[int] = set()
    for row in all_rows:
        if row.get("sequence_name") == sequence_name:
            try:
                frame_ids.add(int(row["frame_id"]))
            except (TypeError, ValueError, KeyError):
                continue

    if not frame_ids:
        print(f"[WARN] No frame IDs found in CSV for sequence: {sequence_name}")
        return

    for frame_id in sorted(frame_ids):
        frame_rows = [
            row for row in all_rows
            if row.get("sequence_name") == sequence_name
            and str(row.get("frame_id")) == str(frame_id)
        ]

        scatter_path = os.path.join(plot_dir, f"scatter_{dataset_name}_{sequence_name}_frame{frame_id}.png")
        hull_csv_path = os.path.join(plot_dir, f"convex_hull_{dataset_name}_{sequence_name}_frame{frame_id}.csv")
        missing_csv_path = os.path.join(plot_dir, f"missing_{dataset_name}_{sequence_name}_frame{frame_id}.csv")

        plot_rd_scatter(
            csv_path=csv_path,
            output_path=scatter_path,
            sequence_name=sequence_name,
            frame_id=frame_id,
            hull_csv_path=hull_csv_path,
            psnr_range=psnr_range,
            fixed=fixed,
            deduplicate=True,
        )

        missing_keys, expected, observed, missing = _expected_missing_combos(frame_rows, fixed)
        if missing_keys:
            print(
                f"[WARN] Missing-report skipped for {sequence_name} frame {frame_id}: "
                f"fixed does not include {missing_keys}"
            )
        else:
            print(
                f"  Coverage {sequence_name} frame {frame_id}: "
                f"observed {len(observed)}/{len(expected)}, missing {len(missing)}"
            )
            _write_missing_report_csv(
                output_path=missing_csv_path,
                sequence_name=sequence_name,
                dataset_name=dataset_name,
                frame_id=frame_id,
                missing=missing,
            )
            print(f"  Missing report: {missing_csv_path}")


def main() -> None:
    sep = "=" * 70
    print(sep)
    print("LiVoGS RD Hull Plot Pipeline (QUEEN)")
    print(f"  RD roots:       {len(RD_OUTPUT_ROOTS)}")
    print("  Scatter specs:  1")
    print(sep)

    print(f"\n{sep}\nStep 1: Collect results\n{sep}")
    collected_infos = collect_all()

    print(f"{sep}\nStep 2: Resolve scatter spec\n{sep}")
    scatter_spec, psnr_range = resolve_scatter_config()
    spec_name = scatter_spec.get("name") or "unnamed"
    print(f"  Using spec: {spec_name}")

    print(f"{sep}\nStep 3: Generate scatter plots + hull CSVs\n{sep}")
    for info in collected_infos:
        rd_root = info["rd_root"]
        csv_path = info["csv_path"]
        seq_name = info["sequence_name"]
        dataset_name = info["dataset_name"]
        plot_dir = PLOT_OUTPUT_DIR if PLOT_OUTPUT_DIR is not None else _default_plot_dir(rd_root)

        print(f"  Sequence: {seq_name}  (dataset: {dataset_name})")
        print(f"    CSV:    {csv_path}")
        print(f"    Output: {plot_dir}")
        generate_scatter_plots(
            csv_path=csv_path,
            plot_dir=plot_dir,
            sequence_name=seq_name,
            dataset_name=dataset_name,
            fixed=scatter_spec.get("fixed", {}),
            psnr_range=psnr_range,
        )

    print(f"\n{sep}")
    print("Done.")
    print(sep)


if __name__ == "__main__":
    main()
