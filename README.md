# Bounded Real-Time LLM Inference Benchmark

A reproducible benchmark harness for evaluating bounded, real-time inference behavior of small LLMs using llama.cpp. This is a **configuration-only study** - no llama.cpp source modifications.

## Project Goal

Gather measurable, systems-level evidence that bounded LLM inference is achievable for edge devices by:
- Characterizing sources of latency variance in llama.cpp
- Measuring worst-case (p99, max) behavior across repeated runs
- Evaluating trade-offs across model size, quantization, and context length
- Comparing isolated vs. contended environments

## Quick Start

```bash
# 1. Setup llama.cpp (pins to specific commit)
./scripts/setup_llamacpp.sh

# 2. Download a test model
./scripts/download_models.sh recommended

# 3. Run baseline experiment
python scripts/run_experiment.py \
    -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -n baseline \
    -i 50

# 4. Analyze results
python scripts/analyze_results.py variance results/baseline/<timestamp>
```

## Directory Structure

```
rt-llm-bench/
├── scripts/
│   ├── setup_llamacpp.sh     # Build llama.cpp at pinned version
│   ├── download_models.sh    # Download GGUF models
│   ├── benchmark.py          # Single benchmark run
│   ├── run_experiment.py     # Multi-run experiment orchestration
│   ├── analyze_results.py    # Analysis and visualization
│   └── prepare_system.sh     # System configuration for isolated tests
├── configs/
│   ├── baseline_greedy.json  # Greedy sampling baseline
│   ├── sampling_*.json       # Different sampling strategies
│   ├── isolated_pinned.json  # CPU-pinned, memory-locked config
│   └── experiment_matrix.json # Full experiment plan
├── results/                  # Benchmark output (gitignored)
├── models/                   # GGUF models (gitignored)
├── analysis/                 # Generated reports and plots
├── docs/                     # Provenance and documentation
└── llama.cpp/               # Cloned llama.cpp (gitignored)
```

## Benchmark Harness

### Single Run

```bash
python scripts/benchmark.py \
    -m <model.gguf> \
    -p "Your prompt here" \
    -n 128 \              # output tokens
    -c 2048 \             # context length
    -t 4 \                # threads
    --sampling greedy     # sampling strategy
```

### Experiment (Multiple Runs)

```bash
python scripts/run_experiment.py \
    -m <model.gguf> \
    -n experiment_name \
    -i 50 \               # iterations
    --warmup 3 \          # warmup runs
    --sampling greedy \
    --pin-cpu \           # CPU pinning (taskset)
    --mlock               # memory locking
```

### Sampling Strategies

All strategies are tested to verify execution determinism is independent of sampling:

| Strategy | Flag | Description |
|----------|------|-------------|
| Greedy | `--sampling greedy` | Always pick highest probability token |
| Temperature | `--sampling temperature --temp 0.8` | Softmax temperature sampling |
| Top-K | `--sampling top_k --top-k 40` | Sample from top K tokens |
| Top-P | `--sampling top_p --top-p 0.9` | Nucleus sampling |

## System Configuration

### Isolated Environment (Recommended for Phase 1)

```bash
# Show current system state
./scripts/prepare_system.sh status

# Configure for isolated testing (requires root for full effect)
sudo ./scripts/prepare_system.sh isolate

# Run experiment with CPU pinning
python scripts/run_experiment.py -m <model> --pin-cpu --cpu-cores "0-3"

# Restore defaults
sudo ./scripts/prepare_system.sh restore
```

### Contended Environment (Phase 2)

```bash
# Show background load options
./scripts/prepare_system.sh contended

# Example: 50% CPU load
stress-ng --cpu 2 --cpu-load 50 &
python scripts/run_experiment.py -m <model> -n contended_50pct
pkill stress-ng
```

## Analysis Tools

```bash
# Variance analysis
python scripts/analyze_results.py variance results/baseline/<timestamp>

# Compare experiments (first is baseline)
python scripts/analyze_results.py compare \
    results/baseline/<ts> \
    results/isolated/<ts>

# Generate histogram
python scripts/analyze_results.py histogram results/baseline/<ts> -o hist.png

# Run-by-run plot
python scripts/analyze_results.py runs results/baseline/<ts> -o runs.png

# Markdown report
python scripts/analyze_results.py report results/baseline/<ts> -o report.md
```

## Metrics Collected

| Metric | Description | Bounded Goal |
|--------|-------------|--------------|
| Per-token latency | Time to generate each token | Low variance (CV < 10%) |
| P99 latency | 99th percentile token latency | Stable across runs |
| Max latency | Worst-case token latency | Bounded, predictable |
| Peak RSS | Maximum resident set size | Minimal jitter |
| Total time | End-to-end inference time | Predictable |

## Key Measurements

- **Coefficient of Variation (CV)**: stddev/mean as percentage. Lower = more deterministic.
- **Max/Mean Ratio**: How much worse is the worst case vs average?
- **P99 Stability**: Does p99 vary significantly run-to-run?
- **RSS Jitter**: Memory variance between identical runs

## Phase 1 Experiments

1. **Baseline Characterization**: 50+ runs with default settings, measure variance
2. **Sampling Comparison**: Verify execution time is sampling-independent
3. **Isolated Environment**: CPU pinning, mlock, governor=performance
4. **Parameter Sweeps**: Context length, thread count, output length

## Recommended Models

| Model | Parameters | Size (Q4) | Use Case |
|-------|------------|-----------|----------|
| TinyLlama 1.1B | 1.1B | ~640MB | Primary test model, fastest |
| Qwen2 0.5B | 0.5B | ~530MB | Extremely small, edge devices |
| Phi-2 | 2.7B | ~1.6GB | Upper bound of target range |

## Requirements

- Linux (tested on Ubuntu 22.04+)
- Python 3.8+
- CMake 3.14+
- GCC/G++ with C++17 support
- Optional: matplotlib (for visualization), psutil (for memory monitoring)

```bash
# Install Python dependencies
pip install matplotlib psutil  # optional but recommended
```

## Configuration-Only Controls

This study uses only external configuration, not llama.cpp modifications:

| Control | Method |
|---------|--------|
| CPU frequency | `cpupower frequency-set -g performance` |
| Turbo boost | `/sys/devices/system/cpu/intel_pstate/no_turbo` |
| CPU pinning | `taskset -c 0-3` |
| Memory locking | `--mlock` flag |
| Swappiness | `/proc/sys/vm/swappiness` |
| Thread count | `-t` flag |
| Context pre-allocation | `-c` flag (sets max context) |

## Success Criteria

Phase 1 is successful if we can show:

1. **Repeatable worst-case bounds**: p99 and max latency are stable across runs (CV < 15%)
2. **Memory stability**: peak RSS varies minimally between runs (CV < 5%)
3. **Sampling independence**: execution variance is consistent across sampling strategies
4. **Clear baseline metrics**: documented variance for comparison with optimizations
