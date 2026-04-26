#!/usr/bin/env python3
"""Latency and resource benchmark for 4 compression pipelines across HiFi4G sequences.

Runs LiVoGS, VideoGS, DracoGS, MesonGS sequentially while monitoring
CPU, RAM, and GPU resources at 30ms intervals. No quality evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import pynvml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_NAME = "HiFi4G_Dataset"
SH_DEGREE = 0
RESOLUTION = 2
SAMPLING_INTERVAL_SEC = 0.030  # 30 ms

SEQUENCES = [
    "4K_Actor1_Greeting",
    "4K_Actor2_Dancing",
    "4K_Actor3_Violin",
    "4K_Actor4_Dancing",
    "4K_Actor5_Oil-paper_Umbrella",
    "4K_Actor6_Changing_Clothes",
    "4K_Actor7_Nunchaku",
]

PIPELINES = ["livogs", "videogs", "dracogs", "mesongs"]

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
VIDEOGS_ROOT = SCRIPTS_DIR.parent
MESONGS_ROOT = VIDEOGS_ROOT / "MesonGS"


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
class PipelineConfig:
    """Holds the command template and defaults for one pipeline."""

    def __init__(
        self,
        name: str,
        conda_env: str,
        script: Path,
        cwd: Path,
        default_interval: int,
        build_args_fn,
    ):
        self.name = name
        self.conda_env = conda_env
        self.script = script
        self.cwd = cwd
        self.default_interval = default_interval
        self.build_args_fn = build_args_fn


LIVOGS_SEQ_PARAMS: dict[str, dict] = {
    "4K_Actor1_Greeting": {
        "J": "12",
        "quantize_step_scales": "0.0001",
        "quantize_step_quats": "0.01",
        "quantize_step_opacity": "0.1",
        "quantize_step_sh_dc": "0.02",
        "quantize_step_sh_rest": "0.06",
        "rlgr_block_size": "512",
    },
}

_LIVOGS_DEFAULTS = {
    "J": "12",
    "quantize_step": "0.0001",
    "rlgr_block_size": "4096",
}


def _livogs_args(ply_path: str, output_folder: str, seq: str,
                 frame_start: int, frame_end: int, interval: int) -> list[str]:
    params = {**_LIVOGS_DEFAULTS, **LIVOGS_SEQ_PARAMS.get(seq, {})}
    args = [
        "--ply_path", ply_path,
        "--output_folder", output_folder,
        "--frame_start", str(frame_start),
        "--frame_end", str(frame_end),
        "--interval", str(interval),
        "--sh_degree", str(SH_DEGREE),
        "--sh_color_space", "klt",
        "--nvcomp_algorithm", "None",
    ]
    for key, val in params.items():
        args.extend([f"--{key}", val])
    return args


def _videogs_args(ply_path: str, output_folder: str, seq: str,
                  frame_start: int, frame_end: int, interval: int) -> list[str]:
    return [
        "--ply_path", ply_path,
        "--output_folder", output_folder,
        "--frame_start", str(frame_start),
        "--frame_end", str(frame_end),
        "--interval", str(interval),
        "--group_size", "20",
        "--sh_degree", str(SH_DEGREE),
        "--qp", "25",
    ]


def _dracogs_args(ply_path: str, output_folder: str, seq: str,
                  frame_start: int, frame_end: int, interval: int) -> list[str]:
    return [
        "--ply_path", ply_path,
        "--output_folder", output_folder,
        "--frame_start", str(frame_start),
        "--frame_end", str(frame_end),
        "--interval", str(interval),
        "--sh_degree", str(SH_DEGREE),
        "--scene_name", seq,
        "--eg", "16", "--eo", "16", "--et", "16", "--es", "16",
        "--cl", "10",
    ]


def _mesongs_args(ply_path: str, output_folder: str, seq: str,
                  frame_start: int, frame_end: int, interval: int) -> list[str]:
    dataset_path = str(
        Path(DATA_PATH_GLOBAL) / f"{DATASET_NAME}_processed" / seq
    )
    return [
        "--ply_path", ply_path,
        "--dataset_path", dataset_path,
        "--output_folder", output_folder,
        "--frame_start", str(frame_start),
        "--frame_end", str(frame_end),
        "--interval", str(interval),
        "--sh_degree", str(SH_DEGREE),
        "--resolution", str(RESOLUTION),
        "--scene_name", seq,
    ]


DATA_PATH_GLOBAL = "/home/rajrup/VideoGS"


def build_pipeline_configs() -> dict[str, PipelineConfig]:
    return {
        "livogs": PipelineConfig(
            "livogs", "videogs",
            VIDEOGS_ROOT / "scripts" / "livogs_baseline" / "compress_decompress_pipeline.py",
            VIDEOGS_ROOT, 10, _livogs_args,
        ),
        "videogs": PipelineConfig(
            "videogs", "videogs",
            VIDEOGS_ROOT / "scripts" / "videogs_baseline" / "compress_decompress_pipeline.py",
            VIDEOGS_ROOT, 1, _videogs_args,
        ),
        "dracogs": PipelineConfig(
            "dracogs", "videogs",
            VIDEOGS_ROOT / "scripts" / "dracogs_baseline" / "compress_decompress_pipeline.py",
            VIDEOGS_ROOT, 10, _dracogs_args,
        ),
        "mesongs": PipelineConfig(
            "mesongs", "mesongs",
            VIDEOGS_ROOT / "scripts" / "mesongs_baseline" / "compress_decompress_pipeline.py",
            MESONGS_ROOT, 10, _mesongs_args,
        ),
    }


# ---------------------------------------------------------------------------
# ResourceMonitor
# ---------------------------------------------------------------------------
class ResourceMonitor:
    """Background thread that samples CPU/RAM/GPU metrics at a fixed interval."""

    _JETSON_GPU_LOAD_PATH = Path("/sys/devices/platform/gpu.0/load")

    def __init__(self, gpu_index: int = 0, interval: float = SAMPLING_INTERVAL_SEC):
        self._interval = interval
        self._gpu_index = gpu_index
        self._stop_event = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._t0: float = 0.0
        self._thread: threading.Thread | None = None
        self._num_cores: int = psutil.cpu_count(logical=True) or 1

        pynvml.nvmlInit()
        self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        self._is_tegra = self._detect_tegra()
        if self._is_tegra:
            print("  [ResourceMonitor] Tegra/Jetson detected — using sysfs fallbacks for GPU metrics")

    @classmethod
    def _detect_tegra(cls) -> bool:
        """Return True when running on an NVIDIA Jetson (Tegra) platform."""
        if cls._JETSON_GPU_LOAD_PATH.exists():
            return True
        try:
            return Path("/etc/nv_tegra_release").exists()
        except OSError:
            return False

    def _jetson_gpu_util_pct(self) -> float:
        """Read GPU utilisation from sysfs (0-1000 scale -> 0-100%)."""
        try:
            raw = self._JETSON_GPU_LOAD_PATH.read_text().strip()
            return round(int(raw) / 10.0, 1)
        except (OSError, ValueError):
            return 0.0

    def _collect_sample(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self._t0

        cpu_overall = psutil.cpu_percent(percpu=False)
        cpu_per_core = psutil.cpu_percent(percpu=True)
        cpu_cores_active = sum(1 for c in cpu_per_core if c > 5.0)
        ram_used_mb = psutil.virtual_memory().used / (1024 * 1024)

        if self._is_tegra:
            gpu_util_pct = self._jetson_gpu_util_pct()
            gpu_mem_used_mb = 0
            gpu_enc_util_pct = 0
            gpu_dec_util_pct = 0
            try:
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_mem_used_mb = gpu_mem.used // (1024 * 1024)
            except pynvml.NVMLError:
                pass
        else:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            gpu_enc = pynvml.nvmlDeviceGetEncoderUtilization(self._gpu_handle)
            gpu_dec = pynvml.nvmlDeviceGetDecoderUtilization(self._gpu_handle)
            gpu_util_pct = gpu_util.gpu
            gpu_mem_used_mb = gpu_mem.used // (1024 * 1024)
            gpu_enc_util_pct = gpu_enc[0]
            gpu_dec_util_pct = gpu_dec[0]

        sample: dict[str, Any] = {
            "elapsed_sec": round(elapsed, 4),
            "cpu_pct": cpu_overall,
            "cpu_cores_active": cpu_cores_active,
            "ram_used_mb": round(ram_used_mb, 1),
            "gpu_util_pct": gpu_util_pct,
            "gpu_mem_used_mb": gpu_mem_used_mb,
            "gpu_enc_util_pct": gpu_enc_util_pct,
            "gpu_dec_util_pct": gpu_dec_util_pct,
        }
        for i, val in enumerate(cpu_per_core):
            sample[f"cpu_core_{i}_pct"] = val

        return sample

    def _run(self) -> None:
        psutil.cpu_percent(percpu=True)
        time.sleep(0.05)

        while not self._stop_event.is_set():
            t_start = time.perf_counter()
            sample = self._collect_sample()
            self._samples.append(sample)
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0, self._interval - elapsed)
            self._stop_event.wait(sleep_time)

    def start(self) -> None:
        self._samples.clear()
        self._stop_event.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return list(self._samples)

    def record_baseline(self, duration: float = 5.0) -> list[dict[str, Any]]:
        """Record idle system metrics for the given duration."""
        self.start()
        time.sleep(duration)
        return self.stop()

    @property
    def num_cores(self) -> int:
        return self._num_cores

    def shutdown(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {}
    keys = ["cpu_pct", "cpu_cores_active", "ram_used_mb",
            "gpu_util_pct", "gpu_mem_used_mb", "gpu_enc_util_pct", "gpu_dec_util_pct"]
    summary: dict[str, float] = {}
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        if vals:
            summary[f"avg_{k}"] = round(sum(vals) / len(vals), 2)
            summary[f"peak_{k}"] = round(max(vals), 2)
    return summary


def save_timeseries_csv(samples: list[dict[str, Any]], path: str, num_cores: int) -> None:
    if not samples:
        return
    base_cols = ["elapsed_sec", "cpu_pct", "cpu_cores_active"]
    core_cols = [f"cpu_core_{i}_pct" for i in range(num_cores)]
    tail_cols = ["ram_used_mb", "gpu_util_pct", "gpu_mem_used_mb",
                 "gpu_enc_util_pct", "gpu_dec_util_pct"]
    header = base_cols + core_cols + tail_cols

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for s in samples:
            w.writerow([s.get(c, "") for c in header])


def log_header(msg: str) -> None:
    print(f"\n{'=' * 70}\n  {msg}\n{'=' * 70}")


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ts_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Latency & resource benchmark for 4 compression pipelines"
    )
    p.add_argument("--data_path", type=str, default="/home/rajrup/VideoGS")
    p.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=PIPELINES)
    p.add_argument("--sequences", nargs="+", default=SEQUENCES)
    p.add_argument("--frame_start", type=int, default=None,
                   help="Override frame start for all pipelines (default: 0)")
    p.add_argument("--frame_end", type=int, default=None,
                   help="Override frame end for all pipelines (default: 200)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose summary.json already exists")
    p.add_argument("--gpu_id", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global DATA_PATH_GLOBAL
    DATA_PATH_GLOBAL = args.data_path

    configs = build_pipeline_configs()
    gpu_id = args.gpu_id

    frame_start_override = args.frame_start if args.frame_start is not None else 0
    frame_end_override = args.frame_end if args.frame_end is not None else 200

    monitor = None if args.dry_run else ResourceMonitor(gpu_index=gpu_id)

    run_start = time.time()
    log_header("Latency & Resource Benchmark")
    print(f"  Started:    {timestamp()}")
    print(f"  Data path:  {args.data_path}")
    print(f"  Pipelines:  {', '.join(args.pipelines)}")
    print(f"  Sequences:  {len(args.sequences)}")
    print(f"  Frames:     {frame_start_override} to {frame_end_override}")
    print(f"  GPU:        {gpu_id}")
    print(f"  Sampling:   {SAMPLING_INTERVAL_SEC * 1000:.0f}ms")
    if args.dry_run:
        print("  Mode:       DRY RUN")
    if args.skip_existing:
        print("  Skip:       existing outputs")
    print("=" * 70)

    failed_runs: list[tuple[str, str]] = []

    for seq in args.sequences:
        log_header(f"Sequence: {seq}")

        ply_path = str(
            Path(args.data_path) / "train_output" / DATASET_NAME / seq / "checkpoint"
        )

        for pipeline_name in args.pipelines:
            cfg = configs[pipeline_name]
            interval = cfg.default_interval
            frame_start = frame_start_override
            frame_end = frame_end_override
            
            output_dir = str(
                Path(args.data_path) / "train_output" / DATASET_NAME / seq
                / "latency_benchmark" / pipeline_name
            )
            if pipeline_name == "livogs":
                    output_dir = str(
                    Path(args.data_path) / "train_output" / DATASET_NAME / seq
                    / "latency_benchmark" / f"{pipeline_name}_sh_degree_{SH_DEGREE}"
                )

            if args.skip_existing and Path(output_dir, "summary.json").is_file():
                print(f"  SKIP (exists): {pipeline_name} | {seq}")
                continue

            pipeline_args = cfg.build_args_fn(
                ply_path, output_dir, seq, frame_start, frame_end, interval
            )
            cmd = [
                "conda", "run", "-n", cfg.conda_env,
                "python", str(cfg.script), *pipeline_args,
            ]

            num_frames = len(range(frame_start, frame_end, interval))
            print(f"\n--- {pipeline_name.upper()} | {seq} | "
                  f"frames {frame_start}-{frame_end - 1} (interval={interval}, "
                  f"n={num_frames}) | {timestamp()}")

            if args.dry_run:
                print(f"  [DRY RUN] cwd={cfg.cwd}")
                print(f"  [DRY RUN] CUDA_VISIBLE_DEVICES={gpu_id}")
                print(f"  [DRY RUN] {shlex.join(cmd)}")
                continue

            os.makedirs(output_dir, exist_ok=True)

            # Baseline recording
            print(f"  Recording baseline (5s)...")
            baseline_samples = monitor.record_baseline(duration=5.0)
            baseline_summary = summarize_samples(baseline_samples)
            print(f"  Baseline: CPU={baseline_summary.get('avg_cpu_pct', 0):.1f}%, "
                  f"GPU={baseline_summary.get('avg_gpu_util_pct', 0):.1f}%, "
                  f"RAM={baseline_summary.get('avg_ram_used_mb', 0):.0f}MB")

            # Start monitoring and run pipeline
            monitor.start()
            log_path = os.path.join(output_dir, f"output_{ts_filename()}.log")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            step_start = time.time()
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cfg.cwd),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                returncode = proc.wait()

            wall_time = time.time() - step_start
            run_samples = monitor.stop()

            if returncode != 0:
                print(f"  FAILED (exit {returncode}). See: {log_path}")
                failed_runs.append((pipeline_name, seq))
                continue

            # Save timeseries CSV
            csv_path = os.path.join(output_dir, "resource_timeseries.csv")
            save_timeseries_csv(run_samples, csv_path, monitor.num_cores)

            # Build and save summary JSON
            run_summary = summarize_samples(run_samples)
            pipeline_config_info = {
                "livogs": {"J": 12, "qstep": 0.0001, "sh_color_space": "klt", "nvcomp": "ANS"},
                "videogs": {"qp": 25, "group_size": 20},
                "dracogs": {"eg": 16, "eo": 16, "et": 16, "es": 16, "cl": 10},
                "mesongs": {"resolution": RESOLUTION},
            }

            summary = {
                "pipeline": pipeline_name,
                "sequence": seq,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "interval": interval,
                "num_frames": num_frames,
                "wall_time_sec": round(wall_time, 2),
                "gpu_device": gpu_id,
                "sampling_interval_ms": round(SAMPLING_INTERVAL_SEC * 1000),
                "num_samples": len(run_samples),
                "config": pipeline_config_info.get(pipeline_name, {}),
                "baseline": {
                    "num_samples": len(baseline_samples),
                    **baseline_summary,
                },
                "metrics": run_summary,
            }

            summary_path = os.path.join(output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"  Done in {wall_time:.1f}s | "
                  f"CPU={run_summary.get('avg_cpu_pct', 0):.1f}% | "
                  f"GPU={run_summary.get('avg_gpu_util_pct', 0):.1f}% | "
                  f"Samples={len(run_samples)}")
            print(f"  Summary: {summary_path}")
            print(f"  Log:     {log_path}")

    if monitor is not None:
        monitor.shutdown()

    total_sec = int(time.time() - run_start)
    log_header("Benchmark Complete")
    print(f"  Finished:   {timestamp()}")
    print(f"  Total time: {total_sec // 3600}h {(total_sec % 3600) // 60}m {total_sec % 60}s")

    if failed_runs:
        print(f"\n  FAILED RUNS ({len(failed_runs)}):")
        for p, s in failed_runs:
            print(f"    - {p} | {s}")
    else:
        print("  All runs succeeded.")
    print("=" * 70)


if __name__ == "__main__":
    main()

'''
conda activate videogs
python scripts/run_latency_benchmark.py --gpu_id 0

# Single sequence
python scripts/run_latency_benchmark.py --pipelines livogs --sequences 4K_Actor1_Greeting
'''
