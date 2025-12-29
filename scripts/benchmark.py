#!/usr/bin/env python3
"""
Benchmark harness for bounded real-time LLM inference evaluation.

This harness runs llama.cpp with controlled parameters and collects:
- Per-token latency
- End-to-end latency
- Peak RSS (memory)
- Timing statistics (mean, stddev, p50, p95, p99, max)

Configuration-only study: no llama.cpp source modifications.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import threading
import resource
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import psutil for better memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    # Model settings
    model_path: str

    # Context and generation
    prompt: str
    context_length: int = 2048
    output_tokens: int = 128

    # Threading
    threads: int = 4

    # Sampling strategy - key for execution determinism study
    sampling: str = "greedy"  # greedy, top_k, top_p, temperature
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    repeat_penalty: float = 1.0

    # Seed for reproducibility (when using stochastic sampling)
    seed: int = 42

    # Memory settings
    mlock: bool = False  # Lock memory to prevent paging
    mmap: bool = True    # Memory-map the model

    # Batch size (512 is a reasonable default for determinism study)
    batch_size: int = 512

    # llama.cpp binary path
    binary_path: str = ""

    def __post_init__(self):
        if not self.binary_path:
            # Default path relative to project
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            self.binary_path = str(project_root / "llama.cpp" / "build" / "bin" / "llama-cli")


@dataclass
class TokenTiming:
    """Timing for a single token generation."""
    token_index: int
    latency_ms: float
    cumulative_ms: float


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    # Configuration
    config: dict

    # Run metadata
    run_id: str
    timestamp: str

    # Timing results
    per_token_latencies_ms: list = field(default_factory=list)
    total_tokens: int = 0
    prompt_eval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Memory results
    peak_rss_bytes: int = 0
    peak_rss_mb: float = 0.0

    # Statistics (computed after run)
    latency_mean_ms: float = 0.0
    latency_stddev_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Tokens per second
    tokens_per_second: float = 0.0

    # Raw output
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0

    # Errors
    error: Optional[str] = None


class MemoryMonitor:
    """Monitor peak RSS of a subprocess."""

    def __init__(self, pid: int, interval_ms: int = 10):
        self.pid = pid
        self.interval = interval_ms / 1000.0
        self.peak_rss = 0
        self.samples = []
        self._stop = False
        self._thread = None

    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor(self):
        if HAS_PSUTIL:
            self._monitor_psutil()
        else:
            self._monitor_proc()

    def _monitor_psutil(self):
        try:
            proc = psutil.Process(self.pid)
            while not self._stop:
                try:
                    mem_info = proc.memory_info()
                    rss = mem_info.rss
                    self.samples.append(rss)
                    self.peak_rss = max(self.peak_rss, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(self.interval)
        except Exception:
            pass

    def _monitor_proc(self):
        """Fallback using /proc filesystem."""
        statm_path = f"/proc/{self.pid}/statm"
        page_size = os.sysconf('SC_PAGE_SIZE')

        while not self._stop:
            try:
                with open(statm_path, 'r') as f:
                    parts = f.read().split()
                    rss_pages = int(parts[1])
                    rss = rss_pages * page_size
                    self.samples.append(rss)
                    self.peak_rss = max(self.peak_rss, rss)
            except (FileNotFoundError, ProcessLookupError, IndexError):
                break
            time.sleep(self.interval)


def build_command(config: BenchmarkConfig) -> list:
    """Build llama.cpp command line from config."""
    cmd = [
        config.binary_path,
        "-m", config.model_path,
        "-p", config.prompt,
        "-c", str(config.context_length),
        "-n", str(config.output_tokens),
        "-t", str(config.threads),
        "-b", str(config.batch_size),
        "-s", str(config.seed),
        "--repeat-penalty", str(config.repeat_penalty),
        "-no-cnv",           # Disable conversation mode (use text completion)
        "-ngl", "0",         # Force CPU-only (disable GPU layers)
    ]

    # Sampling strategy
    if config.sampling == "greedy":
        cmd.extend(["--temp", "0"])
    elif config.sampling == "temperature":
        cmd.extend(["--temp", str(config.temperature)])
    elif config.sampling == "top_k":
        cmd.extend(["--temp", str(config.temperature), "--top-k", str(config.top_k)])
    elif config.sampling == "top_p":
        cmd.extend(["--temp", str(config.temperature), "--top-p", str(config.top_p)])

    # Memory options
    if config.mlock:
        cmd.append("--mlock")
    if not config.mmap:
        cmd.append("--no-mmap")

    # Enable timing output
    cmd.append("--verbose-prompt")

    return cmd


def parse_llama_output(stdout: str, stderr: str) -> dict:
    """Parse llama.cpp output to extract timing information."""
    result = {
        "prompt_eval_time_ms": 0.0,
        "generation_time_ms": 0.0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "tokens_per_second": 0.0,
        "ms_per_token": 0.0,
        "per_token_times": [],
    }

    combined = stdout + "\n" + stderr

    # Parse timing stats from llama.cpp output
    # Format varies by version, try multiple patterns

    # Prompt evaluation time
    prompt_match = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
        combined, re.IGNORECASE
    )
    if prompt_match:
        result["prompt_eval_time_ms"] = float(prompt_match.group(1))
        result["prompt_tokens"] = int(prompt_match.group(2))

    # Generation time (not prompt eval - use negative lookbehind)
    # Format: "eval time = 1061.53 ms / 31 runs (34.24 ms per token, 29.20 tokens per second)"
    gen_match = re.search(
        r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        combined, re.IGNORECASE
    )
    if gen_match:
        result["generation_time_ms"] = float(gen_match.group(1))
        result["generated_tokens"] = int(gen_match.group(2))
        result["ms_per_token"] = float(gen_match.group(3))
        result["tokens_per_second"] = float(gen_match.group(4))

    # Fallback for tokens per second if not captured above
    if result["tokens_per_second"] == 0:
        # Look for the eval line specifically (context_print)
        tps_match = re.search(
            r"context_print.*?eval time.*?([\d.]+)\s*tokens per second",
            combined, re.IGNORECASE
        )
        if tps_match:
            result["tokens_per_second"] = float(tps_match.group(1))

    # Try to extract per-token timing if available
    # This depends on llama.cpp verbosity settings
    token_times = re.findall(
        r"token\s*(\d+).*?(\d+\.?\d*)\s*ms",
        combined, re.IGNORECASE
    )
    if token_times:
        result["per_token_times"] = [
            {"index": int(idx), "time_ms": float(t)}
            for idx, t in token_times
        ]

    return result


def compute_statistics(latencies: list) -> dict:
    """Compute statistical metrics from latency list."""
    if not latencies:
        return {
            "mean": 0.0,
            "stddev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    import statistics

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    mean = statistics.mean(latencies)
    stddev = statistics.stdev(latencies) if n > 1 else 0.0

    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]

    return {
        "mean": mean,
        "stddev": stddev,
        "min": sorted_latencies[0],
        "max": sorted_latencies[-1],
        "p50": percentile(sorted_latencies, 50),
        "p95": percentile(sorted_latencies, 95),
        "p99": percentile(sorted_latencies, 99),
    }


def run_benchmark(config: BenchmarkConfig, run_id: str = None) -> BenchmarkResult:
    """Execute a single benchmark run."""

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    result = BenchmarkResult(
        config=asdict(config),
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
    )

    # Build command
    cmd = build_command(config)

    # Verify binary exists
    if not Path(config.binary_path).exists():
        result.error = f"Binary not found: {config.binary_path}"
        return result

    # Verify model exists
    if not Path(config.model_path).exists():
        result.error = f"Model not found: {config.model_path}"
        return result

    try:
        # Start process
        start_time = time.perf_counter()

        # Set up environment with correct library path and locale
        env = os.environ.copy()
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        lib_path = str(project_root / "llama.cpp" / "build" / "src")
        ggml_lib_path = str(project_root / "llama.cpp" / "build" / "ggml" / "src")
        # Use only our build's library paths (avoid system llama.cpp conflicts)
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{ggml_lib_path}"
        # Ensure proper UTF-8 encoding for output
        env["LANG"] = "en_US.UTF-8"
        env["LC_ALL"] = "en_US.UTF-8"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        # Start memory monitoring
        mem_monitor = MemoryMonitor(proc.pid)
        mem_monitor.start()

        # Wait for completion
        stdout, stderr = proc.communicate()

        end_time = time.perf_counter()

        # Stop memory monitoring
        mem_monitor.stop()

        # Record results
        result.stdout = stdout
        result.stderr = stderr
        result.return_code = proc.returncode
        result.total_time_ms = (end_time - start_time) * 1000

        # Memory stats
        result.peak_rss_bytes = mem_monitor.peak_rss
        result.peak_rss_mb = mem_monitor.peak_rss / (1024 * 1024)

        # Parse llama.cpp output
        parsed = parse_llama_output(stdout, stderr)
        result.prompt_eval_time_ms = parsed["prompt_eval_time_ms"]
        result.generation_time_ms = parsed["generation_time_ms"]
        result.total_tokens = parsed["generated_tokens"]
        result.tokens_per_second = parsed["tokens_per_second"]

        # Per-token latencies
        if parsed["per_token_times"]:
            result.per_token_latencies_ms = [t["time_ms"] for t in parsed["per_token_times"]]
        elif result.total_tokens > 0:
            # Use ms_per_token from llama.cpp output if available, else calculate
            if parsed["ms_per_token"] > 0:
                avg_latency = parsed["ms_per_token"]
            elif result.generation_time_ms > 0:
                avg_latency = result.generation_time_ms / result.total_tokens
            else:
                avg_latency = 0
            if avg_latency > 0:
                result.per_token_latencies_ms = [avg_latency] * result.total_tokens

        # Compute statistics
        if result.per_token_latencies_ms:
            stats = compute_statistics(result.per_token_latencies_ms)
            result.latency_mean_ms = stats["mean"]
            result.latency_stddev_ms = stats["stddev"]
            result.latency_min_ms = stats["min"]
            result.latency_max_ms = stats["max"]
            result.latency_p50_ms = stats["p50"]
            result.latency_p95_ms = stats["p95"]
            result.latency_p99_ms = stats["p99"]

        if proc.returncode != 0:
            result.error = f"Process exited with code {proc.returncode}"

    except Exception as e:
        result.error = str(e)

    return result


def save_result(result: BenchmarkResult, output_dir: Path):
    """Save benchmark result to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"run_{result.run_id}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    return filepath


def print_summary(result: BenchmarkResult):
    """Print summary of benchmark result."""
    print("\n" + "=" * 60)
    print(f"Benchmark Run: {result.run_id}")
    print("=" * 60)

    if result.error:
        print(f"ERROR: {result.error}")
        return

    print(f"\nTiming:")
    print(f"  Total time:        {result.total_time_ms:>10.2f} ms")
    print(f"  Prompt eval:       {result.prompt_eval_time_ms:>10.2f} ms")
    print(f"  Generation:        {result.generation_time_ms:>10.2f} ms")
    print(f"  Tokens generated:  {result.total_tokens:>10d}")
    print(f"  Tokens/sec:        {result.tokens_per_second:>10.2f}")

    print(f"\nLatency Statistics (per-token):")
    print(f"  Mean:              {result.latency_mean_ms:>10.2f} ms")
    print(f"  Stddev:            {result.latency_stddev_ms:>10.2f} ms")
    print(f"  Min:               {result.latency_min_ms:>10.2f} ms")
    print(f"  Max:               {result.latency_max_ms:>10.2f} ms")
    print(f"  P50:               {result.latency_p50_ms:>10.2f} ms")
    print(f"  P95:               {result.latency_p95_ms:>10.2f} ms")
    print(f"  P99:               {result.latency_p99_ms:>10.2f} ms")

    print(f"\nMemory:")
    print(f"  Peak RSS:          {result.peak_rss_mb:>10.2f} MB")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark harness for bounded real-time LLM inference"
    )

    # Required
    parser.add_argument("-m", "--model", required=True, help="Path to GGUF model file")

    # Optional with defaults
    parser.add_argument("-p", "--prompt", default="Hello, world!", help="Input prompt")
    parser.add_argument("-c", "--context", type=int, default=2048, help="Context length")
    parser.add_argument("-n", "--tokens", type=int, default=128, help="Output tokens")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Thread count")

    # Sampling strategy
    parser.add_argument("--sampling", choices=["greedy", "temperature", "top_k", "top_p"],
                       default="greedy", help="Sampling strategy")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (for non-greedy)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K value")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Memory options
    parser.add_argument("--mlock", action="store_true", help="Lock memory")
    parser.add_argument("--no-mmap", action="store_true", help="Disable memory mapping")

    # Output
    parser.add_argument("-o", "--output-dir", default="results", help="Output directory")
    parser.add_argument("--run-id", help="Custom run ID")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")

    # llama.cpp binary
    parser.add_argument("--binary", help="Path to llama-cli binary")

    args = parser.parse_args()

    # Build config
    config = BenchmarkConfig(
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
    )

    if args.binary:
        config.binary_path = args.binary

    # Run benchmark
    result = run_benchmark(config, run_id=args.run_id)

    # Save result
    output_dir = Path(args.output_dir)
    filepath = save_result(result, output_dir)

    # Print summary
    if not args.quiet:
        print_summary(result)
        print(f"\nResults saved to: {filepath}")

    # Exit with appropriate code
    sys.exit(0 if result.error is None else 1)


if __name__ == "__main__":
    main()
