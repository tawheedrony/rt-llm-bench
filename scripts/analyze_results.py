#!/usr/bin/env python3
"""
Analysis tools for bounded real-time LLM inference experiments.

Provides:
- Cross-experiment comparison (single metric or all metrics)
- Model vs sampling matrix views
- Variance analysis
- Visualization (histograms, time series)
- Report generation
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Key metrics to compare
ALL_METRICS = [
    ("token_latency_mean_ms", "Latency Mean (ms)"),
    ("token_latency_p50_ms", "Latency P50 (ms)"),
    ("token_latency_p95_ms", "Latency P95 (ms)"),
    ("token_latency_p99_ms", "Latency P99 (ms)"),
    ("token_latency_max_ms", "Latency Max (ms)"),
    ("token_latency_cv", "Latency CV (%)"),
    ("total_time_mean_ms", "Total Time (ms)"),
    ("total_time_cv", "Total Time CV (%)"),
    ("peak_rss_mean_mb", "Memory (MB)"),
    ("peak_rss_cv", "Memory CV (%)"),
    ("throughput_mean", "Throughput (tok/s)"),
]


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


def parse_experiment_name(name: str) -> tuple:
    """Parse experiment name into (model, sampling) tuple."""
    # Expected format: model_sampling (e.g., gemma3-270m_greedy, gemma3-270m_top_k)
    # Known sampling strategies
    sampling_strategies = ["greedy", "temperature", "top_k", "top_p"]

    for strategy in sampling_strategies:
        if name.endswith(f"_{strategy}"):
            model = name[:-len(strategy)-1]
            return model, strategy

    # Fallback: split on last underscore
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, "unknown"


def load_all_experiments(exp_dirs: List[str]) -> Dict[str, dict]:
    """Load all experiments and return dict keyed by experiment name."""
    experiments = {}
    for exp_dir in exp_dirs:
        try:
            path = Path(exp_dir)
            agg = load_experiment_results(path)
            name = agg.get("experiment_name", path.name)
            experiments[name] = agg
        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}", file=sys.stderr)
    return experiments


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


def compare_all_metrics(exp_dirs: list, output_path: Optional[str] = None):
    """Compare all metrics across multiple experiments."""
    experiments = load_all_experiments(exp_dirs)

    if not experiments:
        print("No valid experiments found.")
        return

    # Sort experiments by name
    sorted_names = sorted(experiments.keys())

    # Build output
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("COMPREHENSIVE METRICS COMPARISON")
    lines.append("=" * 100)
    lines.append(f"Experiments: {len(experiments)}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # For each metric, show comparison
    for metric_key, metric_label in ALL_METRICS:
        lines.append(f"\n{'─' * 100}")
        lines.append(f"  {metric_label} ({metric_key})")
        lines.append(f"{'─' * 100}")
        lines.append(f"{'Experiment':<35} {'Value':>12} {'vs Best':>12} {'vs Worst':>12}")
        lines.append("-" * 75)

        # Get all values for this metric
        values = {}
        for name in sorted_names:
            val = experiments[name].get(metric_key, 0)
            values[name] = val

        if not values:
            continue

        # Determine if lower is better (for most metrics, lower is better except throughput)
        lower_is_better = "throughput" not in metric_key.lower()

        valid_values = [v for v in values.values() if v > 0]
        if not valid_values:
            continue

        best_val = min(valid_values) if lower_is_better else max(valid_values)
        worst_val = max(valid_values) if lower_is_better else min(valid_values)

        for name in sorted_names:
            val = values[name]
            if val > 0:
                vs_best = ((val - best_val) / best_val) * 100 if best_val else 0
                vs_worst = ((val - worst_val) / worst_val) * 100 if worst_val else 0

                best_marker = " ★" if val == best_val else ""
                worst_marker = " ✗" if val == worst_val else ""

                sign_best = "+" if vs_best >= 0 else ""
                sign_worst = "+" if vs_worst >= 0 else ""

                lines.append(f"{name:<35} {val:>12.2f}{best_marker:<2} {sign_best}{vs_best:>10.1f}% {sign_worst}{vs_worst:>10.1f}%{worst_marker}")
            else:
                lines.append(f"{name:<35} {'N/A':>12}")

    lines.append("\n" + "=" * 100)
    lines.append("★ = Best   ✗ = Worst")
    lines.append("=" * 100)

    output = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Saved comparison to: {output_path}")
    else:
        print(output)


def compare_matrix(exp_dirs: list, metric: str = "token_latency_p99_ms", output_path: Optional[str] = None):
    """Generate model x sampling matrix comparison."""
    experiments = load_all_experiments(exp_dirs)

    if not experiments:
        print("No valid experiments found.")
        return

    # Parse into model/sampling groups
    models = set()
    samplings = set()
    data = {}

    for name, agg in experiments.items():
        model, sampling = parse_experiment_name(name)
        models.add(model)
        samplings.add(sampling)
        data[(model, sampling)] = agg.get(metric, 0)

    models = sorted(models)
    samplings = sorted(samplings)

    # Find metric label
    metric_label = metric
    for key, label in ALL_METRICS:
        if key == metric:
            metric_label = label
            break

    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"MODEL x SAMPLING MATRIX: {metric_label}")
    lines.append(f"{'=' * 80}\n")

    # Header
    header = f"{'Model':<20}"
    for s in samplings:
        header += f" {s:>12}"
    header += f" {'Best':>12} {'Worst':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for model in models:
        row_values = []
        row = f"{model:<20}"
        for s in samplings:
            val = data.get((model, s), 0)
            row_values.append(val)
            if val > 0:
                row += f" {val:>12.2f}"
            else:
                row += f" {'N/A':>12}"

        # Best/worst for this model
        valid = [v for v in row_values if v > 0]
        if valid:
            lower_is_better = "throughput" not in metric.lower()
            best = min(valid) if lower_is_better else max(valid)
            worst = max(valid) if lower_is_better else min(valid)
            row += f" {best:>12.2f} {worst:>12.2f}"

        lines.append(row)

    # Summary row
    lines.append("-" * len(header))
    summary = f"{'Column Best':<20}"
    for s in samplings:
        col_values = [data.get((m, s), 0) for m in models]
        valid = [v for v in col_values if v > 0]
        if valid:
            lower_is_better = "throughput" not in metric.lower()
            best = min(valid) if lower_is_better else max(valid)
            summary += f" {best:>12.2f}"
        else:
            summary += f" {'N/A':>12}"
    lines.append(summary)

    lines.append(f"\n{'=' * 80}")

    output = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Saved matrix to: {output_path}")
    else:
        print(output)


def compare_all_matrices(exp_dirs: list, output_path: Optional[str] = None):
    """Generate model x sampling matrix for ALL metrics."""
    experiments = load_all_experiments(exp_dirs)

    if not experiments:
        print("No valid experiments found.")
        return

    # Parse into model/sampling groups
    models = set()
    samplings = set()
    data = defaultdict(dict)  # metric -> (model, sampling) -> value

    for name, agg in experiments.items():
        model, sampling = parse_experiment_name(name)
        models.add(model)
        samplings.add(sampling)
        for metric_key, _ in ALL_METRICS:
            data[metric_key][(model, sampling)] = agg.get(metric_key, 0)

    models = sorted(models)
    samplings = sorted(samplings)

    lines = []
    lines.append("=" * 100)
    lines.append("COMPREHENSIVE MODEL x SAMPLING COMPARISON")
    lines.append("=" * 100)
    lines.append(f"Models: {', '.join(models)}")
    lines.append(f"Sampling: {', '.join(samplings)}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for metric_key, metric_label in ALL_METRICS:
        metric_data = data[metric_key]

        lines.append(f"\n{'─' * 100}")
        lines.append(f"  {metric_label}")
        lines.append(f"{'─' * 100}")

        # Header
        header = f"{'Model':<20}"
        for s in samplings:
            header += f" {s:>12}"
        lines.append(header)
        lines.append("-" * (20 + 13 * len(samplings)))

        # Find global best/worst for coloring
        all_values = [v for v in metric_data.values() if v > 0]
        lower_is_better = "throughput" not in metric_key.lower()
        global_best = min(all_values) if (all_values and lower_is_better) else (max(all_values) if all_values else 0)
        global_worst = max(all_values) if (all_values and lower_is_better) else (min(all_values) if all_values else 0)

        # Data rows
        for model in models:
            row = f"{model:<20}"
            for s in samplings:
                val = metric_data.get((model, s), 0)
                if val > 0:
                    marker = ""
                    if val == global_best:
                        marker = "★"
                    elif val == global_worst:
                        marker = "✗"
                    row += f" {val:>11.2f}{marker}"
                else:
                    row += f" {'N/A':>12}"
            lines.append(row)

        lines.append("")

    lines.append("=" * 100)
    lines.append("★ = Best across all   ✗ = Worst across all")
    lines.append("=" * 100)

    output = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Saved comprehensive comparison to: {output_path}")
    else:
        print(output)


def generate_markdown_report(exp_dirs: list, output_path: Optional[str] = None):
    """Generate comprehensive markdown report for all experiments."""
    experiments = load_all_experiments(exp_dirs)

    if not experiments:
        print("No valid experiments found.")
        return

    # Parse into model/sampling groups
    models = set()
    samplings = set()
    data = defaultdict(dict)

    for name, agg in experiments.items():
        model, sampling = parse_experiment_name(name)
        models.add(model)
        samplings.add(sampling)
        for metric_key, _ in ALL_METRICS:
            data[metric_key][(model, sampling)] = agg.get(metric_key, 0)

    models = sorted(models)
    samplings = sorted(samplings)

    lines = []
    lines.append("# RT-LLM-Bench Comprehensive Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"## Overview\n")
    lines.append(f"- **Models tested**: {len(models)}")
    lines.append(f"- **Sampling strategies**: {len(samplings)}")
    lines.append(f"- **Total experiments**: {len(experiments)}\n")

    lines.append("### Models\n")
    for m in models:
        lines.append(f"- {m}")

    lines.append("\n### Sampling Strategies\n")
    for s in samplings:
        lines.append(f"- {s}")

    # Generate table for each metric
    for metric_key, metric_label in ALL_METRICS:
        metric_data = data[metric_key]

        lines.append(f"\n## {metric_label}\n")

        # Header
        header = "| Model |"
        separator = "|-------|"
        for s in samplings:
            header += f" {s} |"
            separator += "-----:|"
        lines.append(header)
        lines.append(separator)

        # Find best/worst
        all_values = [v for v in metric_data.values() if v > 0]
        lower_is_better = "throughput" not in metric_key.lower()
        global_best = min(all_values) if (all_values and lower_is_better) else (max(all_values) if all_values else 0)

        # Data rows
        for model in models:
            row = f"| {model} |"
            for s in samplings:
                val = metric_data.get((model, s), 0)
                if val > 0:
                    if val == global_best:
                        row += f" **{val:.2f}** |"
                    else:
                        row += f" {val:.2f} |"
                else:
                    row += " N/A |"
            lines.append(row)

    # Key findings
    lines.append("\n## Key Findings\n")

    # Best model for latency
    p99_data = data["token_latency_p99_ms"]
    if p99_data:
        best_p99 = min((v, k) for k, v in p99_data.items() if v > 0)
        lines.append(f"- **Lowest P99 Latency**: {best_p99[1][0]} with {best_p99[1][1]} sampling ({best_p99[0]:.2f} ms)")

    # Best throughput
    throughput_data = data["throughput_mean"]
    if throughput_data:
        valid_throughput = [(v, k) for k, v in throughput_data.items() if v > 0]
        if valid_throughput:
            best_throughput = max(valid_throughput)
            lines.append(f"- **Highest Throughput**: {best_throughput[1][0]} with {best_throughput[1][1]} sampling ({best_throughput[0]:.2f} tok/s)")

    # Lowest memory
    mem_data = data["peak_rss_mean_mb"]
    if mem_data:
        best_mem = min((v, k) for k, v in mem_data.items() if v > 0)
        lines.append(f"- **Lowest Memory**: {best_mem[1][0]} ({best_mem[0]:.0f} MB)")

    output = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Saved markdown report to: {output_path}")
    else:
        print(output)


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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare single metric across experiments
  %(prog)s compare results/*/*/2*/ -m token_latency_p99_ms

  # Compare ALL metrics across experiments
  %(prog)s compare-all results/*/*/2*/

  # Generate model x sampling matrix for one metric
  %(prog)s matrix results/*/*/2*/ -m token_latency_mean_ms

  # Generate matrices for ALL metrics
  %(prog)s matrix-all results/*/*/2*/

  # Generate comprehensive markdown report
  %(prog)s report-all results/*/*/2*/ -o analysis/report.md

  # Variance analysis for single experiment
  %(prog)s variance results/gemma3-270m_greedy/*/2*/
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis command")

    # Compare experiments (single metric)
    compare_parser = subparsers.add_parser("compare", help="Compare single metric across experiments")
    compare_parser.add_argument("experiments", nargs="+", help="Experiment directories")
    compare_parser.add_argument("-m", "--metric", default="token_latency_p99_ms",
                                help="Metric to compare")

    # Compare ALL metrics
    compare_all_parser = subparsers.add_parser("compare-all", help="Compare ALL metrics across experiments")
    compare_all_parser.add_argument("experiments", nargs="+", help="Experiment directories")
    compare_all_parser.add_argument("-o", "--output", help="Output file path")

    # Matrix comparison (single metric)
    matrix_parser = subparsers.add_parser("matrix", help="Model x Sampling matrix for single metric")
    matrix_parser.add_argument("experiments", nargs="+", help="Experiment directories")
    matrix_parser.add_argument("-m", "--metric", default="token_latency_p99_ms",
                               help="Metric to compare")
    matrix_parser.add_argument("-o", "--output", help="Output file path")

    # Matrix ALL metrics
    matrix_all_parser = subparsers.add_parser("matrix-all", help="Model x Sampling matrices for ALL metrics")
    matrix_all_parser.add_argument("experiments", nargs="+", help="Experiment directories")
    matrix_all_parser.add_argument("-o", "--output", help="Output file path")

    # Comprehensive markdown report
    report_all_parser = subparsers.add_parser("report-all", help="Generate comprehensive markdown report")
    report_all_parser.add_argument("experiments", nargs="+", help="Experiment directories")
    report_all_parser.add_argument("-o", "--output", help="Output file path")

    # Variance analysis
    variance_parser = subparsers.add_parser("variance", help="Variance analysis for single experiment")
    variance_parser.add_argument("experiment", help="Experiment directory")

    # Generate histogram
    hist_parser = subparsers.add_parser("histogram", help="Generate histogram")
    hist_parser.add_argument("experiment", help="Experiment directory")
    hist_parser.add_argument("-o", "--output", help="Output file path")

    # Run comparison plot
    runs_parser = subparsers.add_parser("runs", help="Run-by-run plot")
    runs_parser.add_argument("experiment", help="Experiment directory")
    runs_parser.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args()

    if args.command == "compare":
        compare_experiments(args.experiments, args.metric)
    elif args.command == "compare-all":
        compare_all_metrics(args.experiments, args.output)
    elif args.command == "matrix":
        compare_matrix(args.experiments, args.metric, args.output)
    elif args.command == "matrix-all":
        compare_all_matrices(args.experiments, args.output)
    elif args.command == "report-all":
        generate_markdown_report(args.experiments, args.output)
    elif args.command == "variance":
        analyze_variance(Path(args.experiment))
    elif args.command == "histogram":
        generate_histogram(Path(args.experiment), args.output)
    elif args.command == "runs":
        generate_run_comparison(Path(args.experiment), args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
