#!/usr/bin/env python3
"""
Experiment orchestration for bounded real-time LLM inference.

Runs multiple iterations of benchmarks and aggregates results for
statistical analysis of variance, worst-case behavior, and reproducibility.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics


@dataclass
class ExperimentConfig:
    """Configuration for an experiment (multiple runs)."""
    name: str
    model_path: str
    prompt: str
    context_length: int = 2048
    output_tokens: int = 128
    threads: int = 4
    sampling: str = "greedy"
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    seed: int = 42
    mlock: bool = False
    mmap: bool = True

    # Experiment parameters
    iterations: int = 50
    warmup_runs: int = 3
    cooldown_seconds: float = 1.0

    # Environment controls (configuration-only)
    pin_threads: bool = False      # Use taskset for CPU pinning
    cpu_cores: str = "0-3"         # Cores to pin to
    disable_turbo: bool = False    # Reminder only - needs root

    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class AggregatedResults:
    """Aggregated statistics across all runs."""
    experiment_name: str
    timestamp: str
    total_runs: int
    successful_runs: int
    failed_runs: int

    # Per-run total time
    total_time_mean_ms: float = 0.0
    total_time_stddev_ms: float = 0.0
    total_time_min_ms: float = 0.0
    total_time_max_ms: float = 0.0

    # Per-token latency (aggregated across runs)
    token_latency_mean_ms: float = 0.0
    token_latency_stddev_ms: float = 0.0
    token_latency_min_ms: float = 0.0
    token_latency_max_ms: float = 0.0
    token_latency_p50_ms: float = 0.0
    token_latency_p95_ms: float = 0.0
    token_latency_p99_ms: float = 0.0

    # Per-run max latency (worst token per run)
    run_max_latency_mean_ms: float = 0.0
    run_max_latency_stddev_ms: float = 0.0
    run_max_latency_min_ms: float = 0.0
    run_max_latency_max_ms: float = 0.0  # Worst of the worst

    # Per-run p99 latency
    run_p99_latency_mean_ms: float = 0.0
    run_p99_latency_stddev_ms: float = 0.0
    run_p99_latency_min_ms: float = 0.0
    run_p99_latency_max_ms: float = 0.0

    # Memory
    peak_rss_mean_mb: float = 0.0
    peak_rss_stddev_mb: float = 0.0
    peak_rss_min_mb: float = 0.0
    peak_rss_max_mb: float = 0.0

    # Throughput
    tokens_per_second_mean: float = 0.0
    tokens_per_second_stddev: float = 0.0
    tokens_per_second_min: float = 0.0
    tokens_per_second_max: float = 0.0

    # Coefficient of variation (CV) for key metrics
    total_time_cv: float = 0.0
    token_latency_cv: float = 0.0
    peak_rss_cv: float = 0.0


def compute_cv(mean: float, stddev: float) -> float:
    """Compute coefficient of variation."""
    if mean == 0:
        return 0.0
    return (stddev / mean) * 100


def percentile(data: list, p: float) -> float:
    """Compute percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if c == f:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def run_single_benchmark(config: ExperimentConfig, run_id: str,
                         output_dir: Path, binary_path: str) -> dict:
    """Run a single benchmark iteration."""

    script_dir = Path(__file__).parent
    benchmark_script = script_dir / "benchmark.py"

    cmd = [
        sys.executable, str(benchmark_script),
        "-m", config.model_path,
        "-p", config.prompt,
        "-c", str(config.context_length),
        "-n", str(config.output_tokens),
        "-t", str(config.threads),
        "--sampling", config.sampling,
        "--temp", str(config.temperature),
        "--top-k", str(config.top_k),
        "--top-p", str(config.top_p),
        "--seed", str(config.seed),
        "--run-id", run_id,
        "-o", str(output_dir),
        "--binary", binary_path,
        "-q",  # Quiet mode
    ]

    if config.mlock:
        cmd.append("--mlock")
    if not config.mmap:
        cmd.append("--no-mmap")

    # Optional CPU pinning
    if config.pin_threads:
        cmd = ["taskset", "-c", config.cpu_cores] + cmd

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Load the result file
    result_file = output_dir / f"run_{run_id}.json"
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)
    else:
        return {
            "error": f"Result file not found: {result_file}",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }


def aggregate_results(results: list, experiment_name: str) -> AggregatedResults:
    """Compute aggregate statistics from individual runs."""

    successful = [r for r in results if r.get("error") is None]
    failed = [r for r in results if r.get("error") is not None]

    agg = AggregatedResults(
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
        total_runs=len(results),
        successful_runs=len(successful),
        failed_runs=len(failed),
    )

    if not successful:
        return agg

    # Collect metrics from successful runs
    total_times = [r["total_time_ms"] for r in successful if r.get("total_time_ms")]
    peak_rss = [r["peak_rss_mb"] for r in successful if r.get("peak_rss_mb")]
    tps = [r["tokens_per_second"] for r in successful if r.get("tokens_per_second")]

    # Per-token latencies (flattened)
    all_token_latencies = []
    for r in successful:
        if r.get("per_token_latencies_ms"):
            all_token_latencies.extend(r["per_token_latencies_ms"])

    # Per-run max and p99 latencies
    run_max_latencies = []
    run_p99_latencies = []
    for r in successful:
        if r.get("latency_max_ms"):
            run_max_latencies.append(r["latency_max_ms"])
        if r.get("latency_p99_ms"):
            run_p99_latencies.append(r["latency_p99_ms"])

    # Total time stats
    if total_times:
        agg.total_time_mean_ms = statistics.mean(total_times)
        agg.total_time_stddev_ms = statistics.stdev(total_times) if len(total_times) > 1 else 0
        agg.total_time_min_ms = min(total_times)
        agg.total_time_max_ms = max(total_times)
        agg.total_time_cv = compute_cv(agg.total_time_mean_ms, agg.total_time_stddev_ms)

    # Token latency stats (across all tokens from all runs)
    if all_token_latencies:
        agg.token_latency_mean_ms = statistics.mean(all_token_latencies)
        agg.token_latency_stddev_ms = statistics.stdev(all_token_latencies) if len(all_token_latencies) > 1 else 0
        agg.token_latency_min_ms = min(all_token_latencies)
        agg.token_latency_max_ms = max(all_token_latencies)
        agg.token_latency_p50_ms = percentile(all_token_latencies, 50)
        agg.token_latency_p95_ms = percentile(all_token_latencies, 95)
        agg.token_latency_p99_ms = percentile(all_token_latencies, 99)
        agg.token_latency_cv = compute_cv(agg.token_latency_mean_ms, agg.token_latency_stddev_ms)

    # Per-run max latency stats
    if run_max_latencies:
        agg.run_max_latency_mean_ms = statistics.mean(run_max_latencies)
        agg.run_max_latency_stddev_ms = statistics.stdev(run_max_latencies) if len(run_max_latencies) > 1 else 0
        agg.run_max_latency_min_ms = min(run_max_latencies)
        agg.run_max_latency_max_ms = max(run_max_latencies)

    # Per-run p99 latency stats
    if run_p99_latencies:
        agg.run_p99_latency_mean_ms = statistics.mean(run_p99_latencies)
        agg.run_p99_latency_stddev_ms = statistics.stdev(run_p99_latencies) if len(run_p99_latencies) > 1 else 0
        agg.run_p99_latency_min_ms = min(run_p99_latencies)
        agg.run_p99_latency_max_ms = max(run_p99_latencies)

    # Memory stats
    if peak_rss:
        agg.peak_rss_mean_mb = statistics.mean(peak_rss)
        agg.peak_rss_stddev_mb = statistics.stdev(peak_rss) if len(peak_rss) > 1 else 0
        agg.peak_rss_min_mb = min(peak_rss)
        agg.peak_rss_max_mb = max(peak_rss)
        agg.peak_rss_cv = compute_cv(agg.peak_rss_mean_mb, agg.peak_rss_stddev_mb)

    # Throughput stats
    if tps:
        agg.tokens_per_second_mean = statistics.mean(tps)
        agg.tokens_per_second_stddev = statistics.stdev(tps) if len(tps) > 1 else 0
        agg.tokens_per_second_min = min(tps)
        agg.tokens_per_second_max = max(tps)

    return agg


def print_aggregated_summary(agg: AggregatedResults):
    """Print formatted summary of aggregated results."""

    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUMMARY: {agg.experiment_name}")
    print("=" * 70)

    print(f"\nRuns: {agg.successful_runs}/{agg.total_runs} successful")
    if agg.failed_runs > 0:
        print(f"      {agg.failed_runs} failed")

    print(f"\n{'METRIC':<35} {'MEAN':>10} {'STDDEV':>10} {'MIN':>10} {'MAX':>10}")
    print("-" * 70)

    # Total time
    print(f"{'Total Time (ms)':<35} {agg.total_time_mean_ms:>10.2f} {agg.total_time_stddev_ms:>10.2f} "
          f"{agg.total_time_min_ms:>10.2f} {agg.total_time_max_ms:>10.2f}")

    # Token latency
    print(f"{'Token Latency (ms)':<35} {agg.token_latency_mean_ms:>10.2f} {agg.token_latency_stddev_ms:>10.2f} "
          f"{agg.token_latency_min_ms:>10.2f} {agg.token_latency_max_ms:>10.2f}")

    # Per-run max latency
    print(f"{'Per-Run Max Latency (ms)':<35} {agg.run_max_latency_mean_ms:>10.2f} {agg.run_max_latency_stddev_ms:>10.2f} "
          f"{agg.run_max_latency_min_ms:>10.2f} {agg.run_max_latency_max_ms:>10.2f}")

    # Memory
    print(f"{'Peak RSS (MB)':<35} {agg.peak_rss_mean_mb:>10.2f} {agg.peak_rss_stddev_mb:>10.2f} "
          f"{agg.peak_rss_min_mb:>10.2f} {agg.peak_rss_max_mb:>10.2f}")

    # Throughput
    print(f"{'Tokens/Second':<35} {agg.tokens_per_second_mean:>10.2f} {agg.tokens_per_second_stddev:>10.2f} "
          f"{agg.tokens_per_second_min:>10.2f} {agg.tokens_per_second_max:>10.2f}")

    print("-" * 70)

    print(f"\nPERCENTILES (Token Latency):")
    print(f"  P50: {agg.token_latency_p50_ms:>10.2f} ms")
    print(f"  P95: {agg.token_latency_p95_ms:>10.2f} ms")
    print(f"  P99: {agg.token_latency_p99_ms:>10.2f} ms")

    print(f"\nVARIANCE INDICATORS (Coefficient of Variation):")
    print(f"  Total Time CV:    {agg.total_time_cv:>6.2f}%")
    print(f"  Token Latency CV: {agg.token_latency_cv:>6.2f}%")
    print(f"  Peak RSS CV:      {agg.peak_rss_cv:>6.2f}%")

    print("=" * 70)


def run_experiment(config: ExperimentConfig, binary_path: str,
                   output_base: Path) -> AggregatedResults:
    """Run a complete experiment with multiple iterations."""

    # Create experiment output directory
    exp_dir = output_base / config.name / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = exp_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Save config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\nExperiment: {config.name}")
    print(f"Output: {exp_dir}")
    print(f"Iterations: {config.iterations} (+ {config.warmup_runs} warmup)")
    print("-" * 50)

    all_results = []

    # Warmup runs
    if config.warmup_runs > 0:
        print(f"\nWarmup runs ({config.warmup_runs})...")
        for i in range(config.warmup_runs):
            run_id = f"warmup_{i:03d}"
            print(f"  Warmup {i+1}/{config.warmup_runs}...", end=" ", flush=True)
            result = run_single_benchmark(config, run_id, runs_dir, binary_path)
            if result.get("error"):
                print(f"FAILED: {result['error']}")
            else:
                print(f"OK ({result.get('total_time_ms', 0):.0f}ms)")
            time.sleep(config.cooldown_seconds)

    # Main runs
    print(f"\nBenchmark runs ({config.iterations})...")
    for i in range(config.iterations):
        run_id = f"run_{i:03d}"
        print(f"  Run {i+1}/{config.iterations}...", end=" ", flush=True)

        result = run_single_benchmark(config, run_id, runs_dir, binary_path)
        all_results.append(result)

        if result.get("error"):
            print(f"FAILED: {result['error']}")
        else:
            print(f"OK ({result.get('total_time_ms', 0):.0f}ms, "
                  f"p99={result.get('latency_p99_ms', 0):.1f}ms, "
                  f"max={result.get('latency_max_ms', 0):.1f}ms)")

        # Cooldown between runs
        if i < config.iterations - 1:
            time.sleep(config.cooldown_seconds)

    # Aggregate results
    aggregated = aggregate_results(all_results, config.name)

    # Save aggregated results
    with open(exp_dir / "aggregated.json", "w") as f:
        json.dump(asdict(aggregated), f, indent=2)

    # Save all individual results summary
    with open(exp_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Run bounded real-time LLM inference experiment"
    )

    parser.add_argument("-c", "--config", help="Path to experiment config JSON")
    parser.add_argument("-m", "--model", help="Path to GGUF model file")
    parser.add_argument("-n", "--name", default="baseline", help="Experiment name")
    parser.add_argument("-i", "--iterations", type=int, default=50, help="Number of runs")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("-p", "--prompt", default="Once upon a time in a land far away, there lived a wise old wizard who",
                        help="Input prompt")
    parser.add_argument("--tokens", type=int, default=128, help="Output tokens")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Thread count")
    parser.add_argument("--context", type=int, default=2048, help="Context length")

    # Sampling strategies
    parser.add_argument("--sampling", choices=["greedy", "temperature", "top_k", "top_p"],
                        default="greedy", help="Sampling strategy")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Memory options
    parser.add_argument("--mlock", action="store_true", help="Lock memory")
    parser.add_argument("--no-mmap", action="store_true", help="Disable memory mapping")

    # CPU pinning
    parser.add_argument("--pin-cpu", action="store_true", help="Pin to specific CPUs")
    parser.add_argument("--cpu-cores", default="0-3", help="CPU cores to pin to")

    # Output and binary
    parser.add_argument("-o", "--output", default="results", help="Output directory")
    parser.add_argument("--binary", help="Path to llama-cli binary")

    args = parser.parse_args()

    # Determine binary path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    if args.binary:
        binary_path = args.binary
    else:
        binary_path = str(project_root / "llama.cpp" / "build" / "bin" / "llama-cli")

    # Build config
    if args.config:
        config = ExperimentConfig.from_file(args.config)
    else:
        if not args.model:
            parser.error("Either --config or --model is required")

        config = ExperimentConfig(
            name=args.name,
            model_path=args.model,
            prompt=args.prompt,
            context_length=args.context,
            output_tokens=args.tokens,
            threads=args.threads,
            sampling=args.sampling,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            mlock=args.mlock,
            mmap=not args.no_mmap,
            iterations=args.iterations,
            warmup_runs=args.warmup,
            pin_threads=args.pin_cpu,
            cpu_cores=args.cpu_cores,
        )

    output_base = Path(args.output)

    # Run experiment
    aggregated = run_experiment(config, binary_path, output_base)

    # Print summary
    print_aggregated_summary(aggregated)

    print(f"\nResults saved to: {output_base / config.name}")


if __name__ == "__main__":
    main()
