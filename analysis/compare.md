    Comparing: token_latency_mean_ms
    ----------------------------------------------------------------------
    Experiment                               Value     vs Baseline
    ----------------------------------------------------------------------
    baseline_greedy                          33.98      (baseline)
    sampling_temp                            46.71 +          37.5%
    sampling_topk                            46.68 +          37.4%
    ----------------------------------------------------------------------

    === Token Latency Max ===

    Comparing: token_latency_max_ms
    ----------------------------------------------------------------------
    Experiment                               Value     vs Baseline
    ----------------------------------------------------------------------
    baseline_greedy                          34.64      (baseline)
    sampling_temp                            64.03 +          84.8%
    sampling_topk                            63.78 +          84.1%
    ----------------------------------------------------------------------

    === Latency CV ===

    Comparing: token_latency_cv
    ----------------------------------------------------------------------
    Experiment                               Value     vs Baseline
    ----------------------------------------------------------------------
    baseline_greedy                           0.73      (baseline)
    sampling_temp                            16.47 +        2161.2%
    sampling_topk                            16.40 +        2151.8%
    ----------------------------------------------------------------------

    === Peak RSS ===

    Comparing: peak_rss_mean_mb
    ----------------------------------------------------------------------
    Experiment                               Value     vs Baseline
    ----------------------------------------------------------------------
    baseline_greedy                        2149.97      (baseline)
    sampling_temp                          2149.91           -0.0%
    sampling_topk                          2149.95           -0.0%
    ----------------------------------------------------------------------

Significant Finding: Sampling Strategy DOES Affect Execution Variance     
                       
  | Metric       | Greedy     | Temperature     | Top-K           |
  |--------------|------------|-----------------|-----------------|
  | Mean Latency | 33.98 ms   | 46.71 ms (+37%) | 46.68 ms (+37%) |
  | Max Latency  | 34.64 ms   | 64.03 ms (+85%) | 63.78 ms (+84%) |
  | Latency CV   | 0.73%      | 16.47%          | 16.40%          |
  | Peak RSS     | 2149.97 MB | 2149.91 MB      | 2149.95 MB      |

  Key Insight

  The initial hypothesis was wrong. Sampling strategy significantly impacts execution variance:

  - Greedy: Deterministic token selection → same compute path every run → CV < 1%
  - Stochastic: Different tokens each run → different vocabulary lookups and probability calculations → CV ~16%

  Memory is identical, so the variance comes from:
  1. Different tokens requiring different amounts of computation
  2. Variable-length sequences being generated
  3. Different softmax/sampling code paths

  Implications for Real-Time Inference

  For bounded real-time systems, greedy/deterministic decoding is strongly preferred - it provides:
  - 37% lower mean latency
  - 85% lower worst-case latency
  - 22x more predictable behavior (0.73% vs 16.4% CV)

---

## Multi-Model Comparison (CPU-Only, Greedy Sampling)

**Config:** 4 threads, 128 tokens, 30 iterations, `-ngl 0 -no-cnv`

| Model | Latency (mean) | Latency (p99) | CV | Peak RSS | Tokens/sec |
|-------|---------------|---------------|-----|----------|------------|
| Gemma3-270M | 6.43 ms | 6.80 ms | 1.62% | 356 MB | 155.5 |
| Gemma3-1B | 21.46 ms | 23.23 ms | 2.44% | 1110 MB | 46.6 |
| Llama3.2-1B | 24.33 ms | 25.49 ms | 1.48% | 1388 MB | 41.1 |
| DeepSeek-R1-1.5B | 31.78 ms | 32.92 ms | 1.31% | 1919 MB | 31.5 |
| SmolLM2-1.7B | 35.18 ms | 38.52 ms | 2.51% | 2153 MB | 28.4 |

**Key Findings:**
- All models achieve CV < 3% with greedy sampling (bounded latency)
- DeepSeek-R1 most stable (1.31% CV), SmolLM2 most variable (2.51% CV)
- Latency scales ~0.02 ms per million parameters
- Memory efficiency: 1.1-1.4 MB per billion parameters (Q8 quantization)

**Recommendations:**
- Ultra-low latency: Gemma3-270M (<7 ms/token, 356 MB)
- Balanced: Gemma3-1B or Llama3.2-1B (~22-25 ms/token, ~1.2 GB)
- Maximum stability: DeepSeek-R1-1.5B (lowest CV)
