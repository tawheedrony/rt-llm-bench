#!/usr/bin/env python3
"""
Analysis tools for bounded real-time LLM inference experiments.

Provides:
- Cross-experiment comparison
- Variance analysis
- Visualization (histograms, time series)
- Report generation
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_experiment_results(exp_dir: Path) -> dict:
    """Load aggregated results from experiment directory."""
    agg_file = exp_dir / "aggregated.json"
    if not agg_file.exists():
        raise FileNotFoundError(f"No aggregated.json in {exp_dir}")

    with open(agg_file) as f:
        return json.load(f)


def load_individual_runs(exp_dir: Path) -> list:
    """Load all individual run results."""
    runs_dir = exp_dir / "runs"
    results = []

    if runs_dir.exists():
        for run_file in sorted(runs_dir.glob("run_*.json")):
            with open(run_file) as f:
                results.append(json.load(f))

    return results


def compare_experiments(exp_dirs: list, metric: str = "token_latency_p99_ms"):
    """Compare a metric across multiple experiments."""
    print(f"\nComparing: {metric}")
    print("-" * 70)
    print(f"{'Experiment':<30} {'Value':>15} {'vs Baseline':>15}")
    print("-" * 70)

    baseline_value = None

    for i, exp_dir in enumerate(exp_dirs):
        try:
            agg = load_experiment_results(Path(exp_dir))
            name = agg.get("experiment_name", Path(exp_dir).name)
            value = agg.get(metric, 0)

            if i == 0:
                baseline_value = value
                print(f"{name:<30} {value:>15.2f} {'(baseline)':>15}")
            else:
                if baseline_value and baseline_value > 0:
                    delta_pct = ((value - baseline_value) / baseline_value) * 100
                    sign = "+" if delta_pct >= 0 else ""
                    print(f"{name:<30} {value:>15.2f} {sign}{delta_pct:>14.1f}%")
                else:
                    print(f"{name:<30} {value:>15.2f} {'N/A':>15}")

        except Exception as e:
            print(f"{exp_dir:<30} ERROR: {e}")

    print("-" * 70)


def analyze_variance(exp_dir: Path):
    """Detailed variance analysis for an experiment."""
    agg = load_experiment_results(exp_dir)
    runs = load_individual_runs(exp_dir)

    print("\n" + "=" * 70)
    print(f"VARIANCE ANALYSIS: {agg['experiment_name']}")
    print("=" * 70)

    # Key metrics and their CVs
    print("\nCoefficient of Variation (lower = more deterministic):")
    print("-" * 50)

    cv_metrics = [
        ("Total Time", agg.get("total_time_cv", 0)),
        ("Token Latency", agg.get("token_latency_cv", 0)),
        ("Peak RSS", agg.get("peak_rss_cv", 0)),
    ]

    for name, cv in cv_metrics:
        status = "GOOD" if cv < 5 else "MODERATE" if cv < 15 else "HIGH"
        bar = "█" * int(min(cv, 50) / 2)
        print(f"  {name:<20} {cv:>6.2f}%  [{status:<8}] {bar}")

    # Worst-case analysis
    print("\nWorst-Case Bounds:")
    print("-" * 50)

    print(f"  Token Latency:")
    print(f"    Max observed:     {agg.get('token_latency_max_ms', 0):>10.2f} ms")
    print(f"    P99:              {agg.get('token_latency_p99_ms', 0):>10.2f} ms")
    print(f"    P95:              {agg.get('token_latency_p95_ms', 0):>10.2f} ms")
    print(f"    Ratio max/mean:   {agg.get('token_latency_max_ms', 0) / max(agg.get('token_latency_mean_ms', 1), 0.001):>10.2f}x")

    print(f"\n  Per-Run Max Latency:")
    print(f"    Worst-of-worst:   {agg.get('run_max_latency_max_ms', 0):>10.2f} ms")
    print(f"    Mean of max:      {agg.get('run_max_latency_mean_ms', 0):>10.2f} ms")
    print(f"    Stddev of max:    {agg.get('run_max_latency_stddev_ms', 0):>10.2f} ms")

    print(f"\n  Memory (Peak RSS):")
    print(f"    Max:              {agg.get('peak_rss_max_mb', 0):>10.2f} MB")
    print(f"    Min:              {agg.get('peak_rss_min_mb', 0):>10.2f} MB")
    print(f"    Range:            {agg.get('peak_rss_max_mb', 0) - agg.get('peak_rss_min_mb', 0):>10.2f} MB")

    # Stability across runs
    if runs:
        print("\nRun-to-Run Stability:")
        print("-" * 50)

        max_latencies = [r.get("latency_max_ms", 0) for r in runs if r.get("latency_max_ms")]
        if max_latencies:
            sorted_max = sorted(max_latencies)
            outliers = [m for m in max_latencies if m > sorted_max[int(len(sorted_max) * 0.95)]]
            print(f"  Runs with outlier max latency: {len(outliers)}/{len(max_latencies)}")

        p99_latencies = [r.get("latency_p99_ms", 0) for r in runs if r.get("latency_p99_ms")]
        if p99_latencies:
            import statistics
            p99_cv = (statistics.stdev(p99_latencies) / statistics.mean(p99_latencies)) * 100 if len(p99_latencies) > 1 else 0
            print(f"  P99 consistency (CV): {p99_cv:.2f}%")

    print("=" * 70)


def generate_histogram(exp_dir: Path, output_path: Optional[str] = None):
    """Generate latency histogram."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    runs = load_individual_runs(exp_dir)
    agg = load_experiment_results(exp_dir)

    # Collect all per-token latencies
    all_latencies = []
    for run in runs:
        if run.get("per_token_latencies_ms"):
            all_latencies.extend(run["per_token_latencies_ms"])

    if not all_latencies:
        print("No per-token latency data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(all_latencies, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(agg.get("token_latency_mean_ms", 0), color='r', linestyle='--',
                label=f'Mean: {agg.get("token_latency_mean_ms", 0):.2f}ms')
    ax1.axvline(agg.get("token_latency_p99_ms", 0), color='g', linestyle='--',
                label=f'P99: {agg.get("token_latency_p99_ms", 0):.2f}ms')
    ax1.axvline(agg.get("token_latency_max_ms", 0), color='orange', linestyle='--',
                label=f'Max: {agg.get("token_latency_max_ms", 0):.2f}ms')
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Token Latency Distribution\n{agg["experiment_name"]}')
    ax1.legend()

    # CDF
    ax2 = axes[1]
    sorted_lat = sorted(all_latencies)
    cdf = [i / len(sorted_lat) for i in range(1, len(sorted_lat) + 1)]
    ax2.plot(sorted_lat, cdf, linewidth=2)
    ax2.axhline(0.99, color='g', linestyle='--', alpha=0.7, label='P99')
    ax2.axhline(0.95, color='orange', linestyle='--', alpha=0.7, label='P95')
    ax2.set_xlabel('Latency (ms)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Latency CDF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to: {output_path}")
    else:
        plt.show()


def generate_run_comparison(exp_dir: Path, output_path: Optional[str] = None):
    """Generate run-by-run comparison plot."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available")
        return

    runs = load_individual_runs(exp_dir)
    agg = load_experiment_results(exp_dir)

    if not runs:
        print("No run data available")
        return

    # Extract metrics per run
    run_nums = list(range(len(runs)))
    max_latencies = [r.get("latency_max_ms", 0) for r in runs]
    p99_latencies = [r.get("latency_p99_ms", 0) for r in runs]
    mean_latencies = [r.get("latency_mean_ms", 0) for r in runs]
    peak_rss = [r.get("peak_rss_mb", 0) for r in runs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Max latency over runs
    ax1 = axes[0, 0]
    ax1.plot(run_nums, max_latencies, 'o-', markersize=4)
    ax1.axhline(agg.get("run_max_latency_mean_ms", 0), color='r', linestyle='--',
                label=f'Mean: {agg.get("run_max_latency_mean_ms", 0):.2f}ms')
    ax1.set_xlabel('Run')
    ax1.set_ylabel('Max Latency (ms)')
    ax1.set_title('Per-Run Maximum Token Latency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # P99 latency over runs
    ax2 = axes[0, 1]
    ax2.plot(run_nums, p99_latencies, 'o-', markersize=4, color='green')
    ax2.axhline(agg.get("run_p99_latency_mean_ms", 0), color='r', linestyle='--',
                label=f'Mean: {agg.get("run_p99_latency_mean_ms", 0):.2f}ms')
    ax2.set_xlabel('Run')
    ax2.set_ylabel('P99 Latency (ms)')
    ax2.set_title('Per-Run P99 Token Latency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mean latency over runs
    ax3 = axes[1, 0]
    ax3.plot(run_nums, mean_latencies, 'o-', markersize=4, color='purple')
    ax3.set_xlabel('Run')
    ax3.set_ylabel('Mean Latency (ms)')
    ax3.set_title('Per-Run Mean Token Latency')
    ax3.grid(True, alpha=0.3)

    # Peak RSS over runs
    ax4 = axes[1, 1]
    ax4.plot(run_nums, peak_rss, 'o-', markersize=4, color='orange')
    ax4.axhline(agg.get("peak_rss_mean_mb", 0), color='r', linestyle='--',
                label=f'Mean: {agg.get("peak_rss_mean_mb", 0):.2f}MB')
    ax4.set_xlabel('Run')
    ax4.set_ylabel('Peak RSS (MB)')
    ax4.set_title('Per-Run Peak Memory Usage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Run-by-Run Analysis: {agg["experiment_name"]}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved run comparison to: {output_path}")
    else:
        plt.show()


def generate_text_report(exp_dir: Path, output_path: Optional[str] = None):
    """Generate markdown report."""
    agg = load_experiment_results(exp_dir)
    config_file = exp_dir / "config.json"

    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    report = []
    report.append(f"# Experiment Report: {agg['experiment_name']}")
    report.append(f"\nGenerated: {agg['timestamp']}")
    report.append(f"\n## Configuration\n")

    for key, value in config.items():
        report.append(f"- **{key}**: {value}")

    report.append(f"\n## Results Summary\n")
    report.append(f"| Metric | Mean | Stddev | Min | Max |")
    report.append(f"|--------|------|--------|-----|-----|")

    metrics = [
        ("Total Time (ms)", "total_time"),
        ("Token Latency (ms)", "token_latency"),
        ("Peak RSS (MB)", "peak_rss"),
    ]

    for label, prefix in metrics:
        mean = agg.get(f"{prefix}_mean_ms", agg.get(f"{prefix}_mean_mb", 0))
        std = agg.get(f"{prefix}_stddev_ms", agg.get(f"{prefix}_stddev_mb", 0))
        min_v = agg.get(f"{prefix}_min_ms", agg.get(f"{prefix}_min_mb", 0))
        max_v = agg.get(f"{prefix}_max_ms", agg.get(f"{prefix}_max_mb", 0))
        report.append(f"| {label} | {mean:.2f} | {std:.2f} | {min_v:.2f} | {max_v:.2f} |")

    report.append(f"\n## Percentiles (Token Latency)\n")
    report.append(f"- P50: {agg.get('token_latency_p50_ms', 0):.2f} ms")
    report.append(f"- P95: {agg.get('token_latency_p95_ms', 0):.2f} ms")
    report.append(f"- P99: {agg.get('token_latency_p99_ms', 0):.2f} ms")
    report.append(f"- Max: {agg.get('token_latency_max_ms', 0):.2f} ms")

    report.append(f"\n## Variance Analysis\n")
    report.append(f"- Total Time CV: {agg.get('total_time_cv', 0):.2f}%")
    report.append(f"- Token Latency CV: {agg.get('token_latency_cv', 0):.2f}%")
    report.append(f"- Peak RSS CV: {agg.get('peak_rss_cv', 0):.2f}%")

    report.append(f"\n## Worst-Case Analysis\n")
    report.append(f"- Worst-of-worst max latency: {agg.get('run_max_latency_max_ms', 0):.2f} ms")
    report.append(f"- Max/Mean ratio: {agg.get('token_latency_max_ms', 0) / max(agg.get('token_latency_mean_ms', 1), 0.001):.2f}x")

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)
        print(f"Saved report to: {output_path}")
    else:
        print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")

    subparsers = parser.add_subparsers(dest="command", help="Analysis command")

    # Compare experiments
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiments", nargs="+", help="Experiment directories")
    compare_parser.add_argument("-m", "--metric", default="token_latency_p99_ms",
                                help="Metric to compare")

    # Variance analysis
    variance_parser = subparsers.add_parser("variance", help="Variance analysis")
    variance_parser.add_argument("experiment", help="Experiment directory")

    # Generate histogram
    hist_parser = subparsers.add_parser("histogram", help="Generate histogram")
    hist_parser.add_argument("experiment", help="Experiment directory")
    hist_parser.add_argument("-o", "--output", help="Output file path")

    # Run comparison plot
    runs_parser = subparsers.add_parser("runs", help="Run-by-run plot")
    runs_parser.add_argument("experiment", help="Experiment directory")
    runs_parser.add_argument("-o", "--output", help="Output file path")

    # Text report
    report_parser = subparsers.add_parser("report", help="Generate text report")
    report_parser.add_argument("experiment", help="Experiment directory")
    report_parser.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()

    if args.command == "compare":
        compare_experiments(args.experiments, args.metric)
    elif args.command == "variance":
        analyze_variance(Path(args.experiment))
    elif args.command == "histogram":
        generate_histogram(Path(args.experiment), args.output)
    elif args.command == "runs":
        generate_run_comparison(Path(args.experiment), args.output)
    elif args.command == "report":
        generate_text_report(Path(args.experiment), args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
