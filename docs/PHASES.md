# Project Phases: Bounded Real-Time LLM Inference

## Overview

This project investigates whether bounded, real-time LLM inference is achievable on edge devices using llama.cpp with configuration-only controls (no source modifications).

---

## Phase 1: Environment Setup & Baseline Characterization (CURRENT)

**Goal:** Establish reproducible measurement infrastructure and quantify current non-determinism.

### Tasks

| # | Task | Status | Deliverable |
|---|------|--------|-------------|
| 1.1 | Pin llama.cpp version | ✅ Done | `scripts/setup_llamacpp.sh` |
| 1.2 | Select models | ✅ Done | `scripts/download_models.sh` |
| 1.3 | Create benchmark harness | ✅ Done | `scripts/benchmark.py` |
| 1.4 | Instrument per-token timing | ✅ Done | Integrated in harness |
| 1.5 | Create experiment orchestration | ✅ Done | `scripts/run_experiment.py` |
| 1.6 | Statistical analysis tools | ✅ Done | `scripts/analyze_results.py` |
| 1.7 | System preparation scripts | ✅ Done | `scripts/prepare_system.sh` |
| 1.8 | Configuration files | ✅ Done | `configs/*.json` |

### Next: Run Experiments

1. Build llama.cpp: `./scripts/setup_llamacpp.sh`
2. Download model: `./scripts/download_models.sh recommended`
3. Run baseline: `python scripts/run_experiment.py -m models/tinyllama*.gguf -n baseline -i 50`
4. Analyze: `python scripts/analyze_results.py variance results/baseline/*`

### Key Measurements

- Per-token latency distribution (histogram)
- Token position vs. latency (positional effects)
- RSS over time (memory growth curve)
- Variance across identical runs (CV metrics)

---

## Phase 2: Source Analysis & Non-Determinism Taxonomy

**Goal:** Identify root causes of variance in llama.cpp execution without modifying source.

### Tasks

| # | Task | Deliverable |
|---|------|-------------|
| 2.1 | Profile with perf/flamegraph | Hotspot identification |
| 2.2 | Trace memory allocations | Malloc patterns analysis |
| 2.3 | Analyze KV-cache behavior | Growth pattern documentation |
| 2.4 | Identify control-flow variance | Branching analysis |
| 2.5 | Document threading behavior | Sync point identification |

### Taxonomy Categories

1. **Algorithmic:** KV-cache growth, speculative decoding paths
2. **Memory:** Dynamic allocation, fragmentation, page faults
3. **Concurrency:** Thread scheduling, lock contention
4. **Hardware:** Cache misses, NUMA effects, frequency scaling

---

## Phase 3: Apply Bounded Constraints (Configuration-Only)

**Goal:** Apply system-level mitigations using only external configuration.

### Available Controls

| Constraint | Implementation | Script |
|------------|----------------|--------|
| Thread pinning | `taskset` / `--pin-cpu` | `run_experiment.py` |
| Fixed CPU frequency | `cpupower frequency-set` | `prepare_system.sh` |
| Disable turbo boost | sysfs interface | `prepare_system.sh` |
| Memory locking | `--mlock` flag | `benchmark.py` |
| Pre-allocate context | `-c` flag | `benchmark.py` |
| Disable mmap | `--no-mmap` flag | `benchmark.py` |
| Reduce swappiness | sysctl | `prepare_system.sh` |

### Experiments

1. Baseline (no constraints)
2. CPU pinning only
3. CPU pinning + fixed frequency
4. Full isolation (all constraints)
5. Compare improvement in CV metrics

---

## Phase 4: Comparative Evaluation

**Goal:** Quantify improvement and characterize trade-offs.

### Experiment Matrix

| Experiment | Variables |
|------------|-----------|
| Baseline vs. Constrained | Same workload, compare distributions |
| Model size sweep | TinyLlama 1.1B vs Phi-2 2.7B |
| Quantization sweep | Q8_0 → Q5_K → Q4_K → Q4_0 |
| Context length sweep | 512 → 1024 → 2048 → 4096 tokens |
| Thread count sweep | 1, 2, 4, 8 threads |

### Separate Environment Phases

**Phase 4a: Isolated Environment**
- CPU governor: performance
- Turbo boost: disabled
- CPU pinning: enabled
- Memory locked: yes
- Background load: none

**Phase 4b: Contended Environment**
- Same as isolated but with background load
- Test levels: 25%, 50%, 75% CPU utilization
- Use `stress-ng` for controlled load generation
- Measure degradation characteristics

---

## Phase 5: Analysis & Reporting

**Goal:** Synthesize findings into actionable conclusions.

### Deliverables

| Deliverable | Content |
|-------------|---------|
| Technical report | Methodology, results, trade-off analysis |
| Reproducibility package | All scripts, configs, raw data |
| Recommendations | Best practices for edge deployment |
| Limitations | What bounds couldn't be achieved, why |

### Key Questions to Answer

1. What p99/max latency bounds are achievable for 1B models at Q4?
2. How much does memory pre-allocation reduce RSS variance?
3. What is the latency cost of each quantization level?
4. At what background load does bounded behavior break down?
5. Is execution variance truly independent of sampling strategy?

---

## Success Criteria

The project is successful if we can demonstrate:

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Repeatable worst-case bounds | p99/max stable across runs | CV of p99 < 15% |
| Memory stability | Minimal RSS jitter | CV of peak RSS < 5% |
| Controlled degradation | Predictable under load | Linear degradation curve |
| Sampling independence | Same variance for all strategies | CV difference < 5% |
| Clear documentation | Reproducible results | Full provenance chain |

---

## Risk Factors & Mitigations

| Risk | Mitigation |
|------|------------|
| llama.cpp API changes | Pinned to specific commit |
| Hardware variance | Dedicated test machine, reboot between runs |
| Insufficient baseline variance | Longer sequences, more diverse prompts |
| Cannot achieve bounds | Document limitations, identify root causes |
