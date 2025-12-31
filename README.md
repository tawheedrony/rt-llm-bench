# RT-LLM-Bench: Bounded Real-Time LLM Inference Benchmark

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

# 2. Download test models
./scripts/download_models.sh recommended

# 3. Run comprehensive experiments (all models x all sampling strategies)
./scripts/run_all_experiments.sh

# 4. Analyze and compare results
python scripts/analyze_results.py compare results/*_greedy/*/ results/*_temperature/*/
```

## Directory Structure

```
rt-llm-bench/
├── scripts/
│   ├── setup_llamacpp.sh        # Build llama.cpp at pinned version
│   ├── download_models.sh       # Download GGUF models
│   ├── benchmark.py             # Single benchmark run
│   ├── run_experiment.py        # Multi-run experiment orchestration
│   ├── run_all_experiments.sh   # Run all models x all sampling strategies
│   ├── run_all_models.sh        # Run all models with single config
│   ├── analyze_results.py       # Analysis and visualization
│   └── prepare_system.sh        # System configuration for isolated tests
├── configs/
│   ├── baseline_greedy.json     # Greedy sampling baseline
│   ├── sampling_temperature.json # Temperature-based sampling
│   ├── sampling_top_k.json      # Top-K sampling
│   ├── sampling_top_p.json      # Top-P (nucleus) sampling
│   ├── isolated_pinned.json     # CPU-pinned, memory-locked config
│   └── experiment_matrix.json   # Full experiment plan
├── results/                     # Benchmark output (gitignored)
├── models/                      # GGUF models (gitignored)
├── analysis/                    # Generated reports and plots
├── docs/                        # Provenance and documentation
└── llama.cpp/                   # Cloned llama.cpp (gitignored)
```

## Running Experiments

### Comprehensive Experiments (All Models x All Sampling)

The `run_all_experiments.sh` script runs a full matrix of experiments across all available models and sampling strategies:

```bash
# Run all models with all sampling strategies (default: 30 iterations each)
./scripts/run_all_experiments.sh

# Run with more iterations
./scripts/run_all_experiments.sh -i 50

# Run specific models only
./scripts/run_all_experiments.sh -m gemma3-270m,llama3.2-1b

# Run specific sampling strategies only
./scripts/run_all_experiments.sh -s greedy,temperature

# Run with isolated environment settings
./scripts/run_all_experiments.sh --pin-cpu --mlock

# Quick test run
./scripts/run_all_experiments.sh -m gemma3-270m -s greedy -i 5

# Dry run to see what would be executed
./scripts/run_all_experiments.sh --dry-run
```

**Available Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-m, --models` | Comma-separated model list | all |
| `-s, --sampling` | Comma-separated sampling strategies | all |
| `-i, --iterations` | Runs per experiment | 30 |
| `-w, --warmup` | Warmup runs | 3 |
| `-t, --threads` | Thread count | 4 |
| `-n, --tokens` | Output tokens | 128 |
| `-c, --context` | Context length | 2048 |
| `-o, --output` | Output directory | results/ |
| `--pin-cpu` | Enable CPU pinning | off |
| `--mlock` | Enable memory locking | off |
| `--dry-run` | Preview without executing | - |

All runs are automatically logged to `results/log_run_YYYYMMDD_HHMMSS.log`.

### Available Models

| Model | Size | Description |
|-------|------|-------------|
| `gemma3-270m` | 279MB | Smallest, edge device focused |
| `gemma3-1b` | 1.0GB | Google's efficient 1B |
| `llama3.2-1b` | 1.3GB | Meta's efficient 1B |
| `deepseek-r1-1.5b` | 1.8GB | Reasoning-focused distilled model |
| `smollm2-1.7b` | ~1.8GB | Efficient 1.7B model |

### Sampling Strategies

| Strategy | Flag | Parameters | Description |
|----------|------|------------|-------------|
| `greedy` | `--sampling greedy` | temp=0.0 | Deterministic, always picks max probability |
| `temperature` | `--sampling temperature` | temp=0.8 | Softmax temperature sampling |
| `top_k` | `--sampling top_k` | temp=0.8, k=40 | Sample from top K tokens |
| `top_p` | `--sampling top_p` | temp=0.8, p=0.9 | Nucleus sampling |

All sampling strategies are tested to verify execution time variance is sampling-independent.

### Single Model Experiments

```bash
# Run all models with a single configuration
./scripts/run_all_models.sh 30  # 30 iterations per model
```

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

### Multi-Run Experiment

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

### Config-Based Experiments

```bash
# Run from config file
python scripts/run_experiment.py -c configs/baseline_greedy.json

# Available configs
ls configs/
# baseline_greedy.json      - Deterministic baseline
# sampling_temperature.json - Temperature sampling
# sampling_top_k.json       - Top-K sampling
# sampling_top_p.json       - Top-P (nucleus) sampling
# isolated_pinned.json      - CPU-pinned with mlock
```

## System Configuration

### Isolated Environment (Recommended)

```bash
# Show current system state
./scripts/prepare_system.sh status

# Configure for isolated testing (requires root for full effect)
sudo ./scripts/prepare_system.sh isolate

# Run experiment with CPU pinning
./scripts/run_all_experiments.sh --pin-cpu --cpu-cores "0-3" --mlock

# Restore defaults
sudo ./scripts/prepare_system.sh restore
```

### Contended Environment

```bash
# Show background load options
./scripts/prepare_system.sh contended

# Example: 50% CPU load
stress-ng --cpu 2 --cpu-load 50 &
./scripts/run_all_experiments.sh -m gemma3-270m -s greedy
pkill stress-ng
```

## Analysis Tools

```bash
# Variance analysis for a single experiment
python scripts/analyze_results.py variance results/gemma3-270m_greedy/<timestamp>

# Compare multiple experiments (first is baseline)
python scripts/analyze_results.py compare \
    results/gemma3-270m_greedy/*/ \
    results/gemma3-270m_temperature/*/ \
    results/gemma3-270m_top_k/*/

# Compare all models with greedy sampling
python scripts/analyze_results.py compare results/*_greedy/*/

# Compare sampling strategies for a single model
python scripts/analyze_results.py compare \
    results/llama3.2-1b_greedy/*/ \
    results/llama3.2-1b_temperature/*/ \
    results/llama3.2-1b_top_k/*/ \
    results/llama3.2-1b_top_p/*/

# Generate histogram
python scripts/analyze_results.py histogram results/gemma3-270m_greedy/*/ -o hist.png

# Run-by-run plot
python scripts/analyze_results.py runs results/gemma3-270m_greedy/*/ -o runs.png

# Markdown report
python scripts/analyze_results.py report results/gemma3-270m_greedy/*/ -o report.md
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

## Experiment Matrix

The full experiment plan supports these sweeps:

| Experiment Type | Variable | Values |
|----------------|----------|--------|
| Baseline | - | Default settings, 50 runs |
| Sampling | Strategy | greedy, temperature, top_k, top_p |
| Context | Length | 512, 1024, 2048, 4096 |
| Threads | Count | 1, 2, 4, 8 |
| Output | Tokens | 32, 64, 128, 256 |
| Environment | Isolation | default, pinned, mlock |

## Example Workflow

```bash
# 1. Setup environment
./scripts/setup_llamacpp.sh
./scripts/download_models.sh recommended

# 2. Run baseline experiments (all models, greedy only)
./scripts/run_all_experiments.sh -s greedy -i 50

# 3. Run sampling comparison
./scripts/run_all_experiments.sh -s temperature,top_k,top_p -i 50

# 4. Run isolated environment experiments
sudo ./scripts/prepare_system.sh isolate
./scripts/run_all_experiments.sh -s greedy --pin-cpu --mlock -i 50
sudo ./scripts/prepare_system.sh restore

# 5. Analyze results
python scripts/analyze_results.py compare results/*_greedy/*/
python scripts/analyze_results.py compare results/gemma3-270m_*/*/
```

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

## Results Naming Convention

Experiments are saved with the naming pattern: `{model}_{sampling}/`

Examples:
- `results/gemma3-270m_greedy/` - Gemma 270M with greedy sampling
- `results/llama3.2-1b_temperature/` - Llama 3.2 1B with temperature sampling
- `results/deepseek-r1-1.5b_top_p/` - DeepSeek R1 1.5B with nucleus sampling

Each experiment directory contains:
- `config.json` - Exact configuration used
- `aggregated.json` - Summary statistics
- `all_results.json` - All run data
- `runs/run_*.json` - Individual run results
