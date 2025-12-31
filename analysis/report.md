# RT-LLM-Bench Comprehensive Results

Generated: 2025-12-31 00:41:43

## Overview

- **Models tested**: 5
- **Sampling strategies**: 4
- **Total experiments**: 20

### Models

- deepseek-r1-1.5b
- gemma3-1b
- gemma3-270m
- llama3.2-1b
- smollm2-1.7b

### Sampling Strategies

- greedy
- temperature
- top_k
- top_p

## Latency Mean (ms)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 30.43 | 32.06 | 31.99 | 31.13 |
| gemma3-1b | 20.88 | 20.94 | 20.90 | 20.88 |
| gemma3-270m | 6.56 | 6.56 | 6.57 | **6.56** |
| llama3.2-1b | 24.07 | 23.96 | 24.08 | 24.40 |
| smollm2-1.7b | 33.48 | 33.16 | 33.14 | 33.19 |

## Latency P50 (ms)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 30.43 | 32.04 | 31.98 | 30.52 |
| gemma3-1b | 20.78 | 20.91 | 20.84 | 20.79 |
| gemma3-270m | 6.56 | 6.56 | **6.55** | 6.56 |
| llama3.2-1b | 23.96 | 23.84 | 24.02 | 24.19 |
| smollm2-1.7b | 33.37 | 33.15 | 33.13 | 33.16 |

## Latency P95 (ms)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 30.53 | 32.28 | 32.16 | 32.56 |
| gemma3-1b | 21.43 | 21.38 | 21.18 | 21.36 |
| gemma3-270m | 6.62 | 6.65 | 6.66 | **6.61** |
| llama3.2-1b | 24.72 | 24.45 | 24.52 | 25.38 |
| smollm2-1.7b | 34.50 | 33.29 | 33.25 | 33.54 |

## Latency P99 (ms)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 30.54 | 32.56 | 32.21 | 33.45 |
| gemma3-1b | 21.54 | 21.41 | 21.63 | 21.51 |
| gemma3-270m | 6.66 | 6.69 | 6.67 | **6.61** |
| llama3.2-1b | 24.95 | 25.08 | 24.97 | 25.97 |
| smollm2-1.7b | 34.70 | 33.29 | 33.27 | 33.78 |

## Latency Max (ms)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 30.54 | 32.56 | 32.21 | 33.45 |
| gemma3-1b | 21.54 | 21.41 | 21.63 | 21.51 |
| gemma3-270m | 6.66 | 6.69 | 6.67 | **6.61** |
| llama3.2-1b | 24.95 | 25.08 | 24.97 | 25.97 |
| smollm2-1.7b | 34.70 | 33.29 | 33.27 | 33.78 |

## Latency CV (%)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 0.17 | 0.42 | 0.31 | 2.81 |
| gemma3-1b | 1.16 | 0.73 | 0.82 | 1.04 |
| gemma3-270m | 0.63 | 0.68 | 0.72 | 0.45 |
| llama3.2-1b | 1.33 | 1.17 | 0.95 | 2.02 |
| smollm2-1.7b | 1.10 | 0.17 | **0.15** | 0.45 |

## Total Time (ms)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 4385.89 | 4628.19 | 4616.64 | 4487.92 |
| gemma3-1b | 3149.59 | 3171.67 | 3164.80 | 3160.59 |
| gemma3-270m | 1220.15 | 1221.60 | **1218.04** | 1219.01 |
| llama3.2-1b | 3572.31 | 3557.94 | 3576.98 | 3623.11 |
| smollm2-1.7b | 4828.27 | 4787.15 | 4785.06 | 4790.94 |

## Total Time CV (%)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 0.29 | 0.46 | 0.30 | 2.91 |
| gemma3-1b | 1.20 | 0.69 | 0.80 | 1.33 |
| gemma3-270m | 0.81 | 0.75 | 0.64 | 0.68 |
| llama3.2-1b | 1.45 | 1.18 | 1.09 | 2.01 |
| smollm2-1.7b | 1.01 | 0.15 | **0.14** | 0.44 |

## Memory (MB)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 1919.53 | 1919.27 | 1919.23 | 1919.47 |
| gemma3-1b | 1110.47 | 1110.44 | 1110.39 | 1110.41 |
| gemma3-270m | **355.76** | 355.79 | 355.88 | 355.86 |
| llama3.2-1b | 1388.53 | 1388.51 | 1388.53 | 1388.44 |
| smollm2-1.7b | 2153.57 | 2153.67 | 2153.61 | 2153.56 |

## Memory CV (%)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | 0.01 | 0.01 | 0.01 | 0.01 |
| gemma3-1b | 0.02 | 0.02 | 0.02 | 0.03 |
| gemma3-270m | 0.07 | 0.05 | 0.07 | 0.07 |
| llama3.2-1b | 0.02 | 0.01 | 0.01 | 0.01 |
| smollm2-1.7b | 0.01 | 0.01 | 0.01 | **0.01** |

## Throughput (tok/s)

| Model | greedy | temperature | top_k | top_p |
|-------|-----:|-----:|-----:|-----:|
| deepseek-r1-1.5b | N/A | N/A | N/A | N/A |
| gemma3-1b | N/A | N/A | N/A | N/A |
| gemma3-270m | N/A | N/A | N/A | N/A |
| llama3.2-1b | N/A | N/A | N/A | N/A |
| smollm2-1.7b | N/A | N/A | N/A | N/A |

## Key Findings

- **Lowest P99 Latency**: gemma3-270m with top_p sampling (6.61 ms)
- **Lowest Memory**: gemma3-270m (356 MB)