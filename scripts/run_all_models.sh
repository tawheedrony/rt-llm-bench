#!/bin/bash
# Run experiments on all downloaded models
# Each model gets its own results folder

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Experiment parameters
ITERATIONS=${1:-30}
WARMUP=3
TOKENS=128
THREADS=4
PROMPT="Once upon a time in a land far away, there lived a wise old wizard who"

log() {
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "============================================================"
}

run_experiment() {
    local model_path="$1"
    local model_name="$2"
    local results_folder="${RESULTS_DIR}/${model_name}"

    log "Running experiment: $model_name"
    echo "Model: $model_path"
    echo "Results: $results_folder"
    echo "Iterations: $ITERATIONS"

    python "${SCRIPT_DIR}/run_experiment.py" \
        -m "$model_path" \
        -n "$model_name" \
        -i "$ITERATIONS" \
        --warmup "$WARMUP" \
        --tokens "$TOKENS" \
        -t "$THREADS" \
        -p "$PROMPT" \
        --sampling greedy \
        -o "$results_folder"

    echo ""
}

# Define models to test
declare -A MODELS=(
    ["smollm2-1.7b"]="${MODELS_DIR}/SmolLM2.q8.gguf"
    ["gemma3-270m"]="${MODELS_DIR}/gemma-3-270m-it-Q8_0.gguf"
    ["gemma3-1b"]="${MODELS_DIR}/google_gemma-3-1b-it-Q8_0.gguf"
    ["llama3.2-1b"]="${MODELS_DIR}/Llama-3.2-1B-Instruct-Q8_0.gguf"
    ["deepseek-r1-1.5b"]="${MODELS_DIR}/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
)

main() {
    log "Starting multi-model benchmark suite"
    echo "Models to test: ${!MODELS[@]}"
    echo ""

    local start_time=$(date +%s)

    for model_name in "${!MODELS[@]}"; do
        model_path="${MODELS[$model_name]}"

        if [ -f "$model_path" ] || [ -L "$model_path" ]; then
            run_experiment "$model_path" "$model_name"
        else
            echo "WARNING: Model not found: $model_path"
        fi
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log "All experiments complete!"
    echo "Total time: $((duration / 60)) minutes $((duration % 60)) seconds"
    echo ""
    echo "Results saved to: $RESULTS_DIR"
    echo ""
    echo "To analyze results:"
    echo "  python scripts/analyze_results.py compare \\"
    for model_name in "${!MODELS[@]}"; do
        echo "    results/${model_name}/*/ \\"
    done
    echo ""
}

main "$@"
