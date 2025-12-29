#!/bin/bash
# Download models for bounded real-time inference experiments
# All models are <3B parameters in GGUF format
#
# Usage:
#   ./download_models.sh list              # List available models
#   ./download_models.sh <model-tag>       # Download specific model
#   ./download_models.sh all               # Download all models
#   ./download_models.sh recommended       # Download recommended starter set
#
# To add a new model:
#   1. Add entry to MODELS array: ["tag"]="URL"
#   2. Add entry to MODEL_SIZES array: ["tag"]="size"
#   3. Optionally add to MODEL_DESCRIPTIONS array

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"

mkdir -p "$MODELS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Check for wget or curl (prefer these for direct URL downloads)
check_download_tool() {
    if command -v wget &> /dev/null; then
        echo "wget"
    elif command -v curl &> /dev/null; then
        echo "curl"
    else
        echo "none"
    fi
}

download_with_wget() {
    local url="$1"
    local output="$2"

    if [ -f "$output" ]; then
        log "Already exists: $output"
        return 0
    fi

    log "Downloading: $(basename "$output")"
    log "URL: $url"
    wget -q --show-progress -O "$output" "$url"
}

download_with_curl() {
    local url="$1"
    local output="$2"

    if [ -f "$output" ]; then
        log "Already exists: $output"
        return 0
    fi

    log "Downloading: $(basename "$output")"
    log "URL: $url"
    curl -L -# -o "$output" "$url"
}

# ============================================================================
# MODEL REGISTRY - Add new models here
# ============================================================================

declare -A MODELS=(
    # --- SmolLM2 Family ---
    ["smollm2-1.7b-q8"]="https://huggingface.co/TheBloke/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct.Q8_0.gguf"
    ["smollm2-1.7b-q4"]="https://huggingface.co/TheBloke/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct.Q4_K_M.gguf"

    # --- TinyLlama 1.1B ---
    ["tinyllama-1.1b-q8"]="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    ["tinyllama-1.1b-q4"]="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    # --- DeepSeek R1 Distill Qwen 1.5B ---
    ["deepseek-r1-qwen-1.5b-q8"]="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
    ["deepseek-r1-qwen-1.5b-q4"]="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"

    # --- Gemma 3 270M (smallest) ---
    ["gemma3-270m-q8"]="https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf"
    ["gemma3-270m-q4"]="https://huggingface.co/bartowski/google_gemma-3-270m-it-GGUF/resolve/main/google_gemma-3-270m-it-Q4_K_M.gguf"

    # --- Gemma 3 1B ---
    ["gemma3-1b-q8"]="https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q8_0.gguf"
    ["gemma3-1b-q4"]="https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf"

    # --- Llama 3.2 1B ---
    ["llama3.2-1b-q8"]="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf"
    ["llama3.2-1b-q4"]="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    # --- Qwen2 0.5B (extremely small) ---
    ["qwen2-0.5b-q8"]="https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q8_0.gguf"

    # --- Phi-2 2.7B ---
    ["phi2-2.7b-q4"]="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
)

declare -A MODEL_SIZES=(
    ["smollm2-1.7b-q8"]="1.7G"
    ["smollm2-1.7b-q4"]="1.0G"
    ["tinyllama-1.1b-q8"]="1.1G"
    ["tinyllama-1.1b-q4"]="637M"
    ["deepseek-r1-qwen-1.5b-q8"]="1.6G"
    ["deepseek-r1-qwen-1.5b-q4"]="934M"
    ["gemma3-270m-q8"]="292M"
    ["gemma3-270m-q4"]="190M"
    ["gemma3-1b-q8"]="1.1G"
    ["gemma3-1b-q4"]="640M"
    ["llama3.2-1b-q8"]="1.3G"
    ["llama3.2-1b-q4"]="756M"
    ["qwen2-0.5b-q8"]="531M"
    ["phi2-2.7b-q4"]="1.6G"
)

declare -A MODEL_DESCRIPTIONS=(
    ["smollm2-1.7b-q8"]="SmolLM2 1.7B Instruct - good all-rounder"
    ["smollm2-1.7b-q4"]="SmolLM2 1.7B Instruct Q4 - smaller, faster"
    ["tinyllama-1.1b-q8"]="TinyLlama 1.1B - fast baseline model"
    ["tinyllama-1.1b-q4"]="TinyLlama 1.1B Q4 - smallest practical"
    ["deepseek-r1-qwen-1.5b-q8"]="DeepSeek R1 Distill - reasoning focused"
    ["deepseek-r1-qwen-1.5b-q4"]="DeepSeek R1 Distill Q4"
    ["gemma3-270m-q8"]="Gemma 3 270M - ultra small, edge device"
    ["gemma3-270m-q4"]="Gemma 3 270M Q4 - smallest model"
    ["gemma3-1b-q8"]="Gemma 3 1B - Google's efficient 1B"
    ["gemma3-1b-q4"]="Gemma 3 1B Q4"
    ["llama3.2-1b-q8"]="Llama 3.2 1B - Meta's efficient 1B"
    ["llama3.2-1b-q4"]="Llama 3.2 1B Q4"
    ["qwen2-0.5b-q8"]="Qwen2 0.5B - extremely small"
    ["phi2-2.7b-q4"]="Phi-2 2.7B - upper bound of target range"
)

# Models recommended for the benchmark study
RECOMMENDED_MODELS=(
    "gemma3-270m-q8"
    "gemma3-1b-q8"
    "llama3.2-1b-q8"
    "deepseek-r1-qwen-1.5b-q8"
)

# ============================================================================
# Functions
# ============================================================================

list_models() {
    echo ""
    echo "Available models for download:"
    echo "=============================="
    echo ""
    printf "%-28s %-8s %s\n" "TAG" "SIZE" "DESCRIPTION"
    printf "%-28s %-8s %s\n" "---" "----" "-----------"

    # Sort by size (approximate)
    for model in gemma3-270m-q4 gemma3-270m-q8 qwen2-0.5b-q8 tinyllama-1.1b-q4 tinyllama-1.1b-q8 \
                 gemma3-1b-q4 gemma3-1b-q8 llama3.2-1b-q4 llama3.2-1b-q8 \
                 deepseek-r1-qwen-1.5b-q4 deepseek-r1-qwen-1.5b-q8 \
                 smollm2-1.7b-q4 smollm2-1.7b-q8 phi2-2.7b-q4; do
        if [ -n "${MODELS[$model]+x}" ]; then
            local size="${MODEL_SIZES[$model]:-???}"
            local desc="${MODEL_DESCRIPTIONS[$model]:-}"
            printf "%-28s %-8s %s\n" "$model" "$size" "$desc"
        fi
    done

    echo ""
    echo "Usage: $0 <model-tag> | all | recommended"
    echo ""
    echo "Examples:"
    echo "  $0 gemma3-270m-q8           # Download Gemma 3 270M"
    echo "  $0 llama3.2-1b-q8           # Download Llama 3.2 1B"
    echo "  $0 recommended              # Download recommended set for benchmarks"
    echo "  $0 all                      # Download all models"
    echo ""
    echo "Recommended models for benchmark study:"
    for m in "${RECOMMENDED_MODELS[@]}"; do
        echo "  - $m (${MODEL_SIZES[$m]:-?})"
    done
    echo ""
}

download_model() {
    local model_tag="$1"

    if [ -z "${MODELS[$model_tag]+x}" ]; then
        log "ERROR: Unknown model tag: $model_tag"
        echo "Use '$0 list' to see available models"
        return 1
    fi

    local url="${MODELS[$model_tag]}"
    local filename=$(basename "$url")
    local output="${MODELS_DIR}/${filename}"

    local tool=$(check_download_tool)

    case "$tool" in
        wget)
            download_with_wget "$url" "$output"
            ;;
        curl)
            download_with_curl "$url" "$output"
            ;;
        *)
            log "ERROR: No download tool available. Install wget or curl."
            exit 1
            ;;
    esac

    if [ -f "$output" ]; then
        log "Complete: $output"
        log "Size: $(du -h "$output" | cut -f1)"

        # Create a symlink with the tag name for easier reference
        local link_name="${MODELS_DIR}/${model_tag}.gguf"
        ln -sf "$(basename "$output")" "$link_name" 2>/dev/null || true
    fi
}

download_recommended() {
    log "Downloading recommended models for benchmark study..."
    echo ""

    for model in "${RECOMMENDED_MODELS[@]}"; do
        log ">>> $model (${MODEL_SIZES[$model]:-?})"
        download_model "$model"
        echo ""
    done
}

download_all() {
    log "Downloading all models..."
    echo ""

    for model in "${!MODELS[@]}"; do
        log ">>> $model (${MODEL_SIZES[$model]:-?})"
        download_model "$model"
        echo ""
    done
}

show_downloaded() {
    echo ""
    echo "Downloaded models in $MODELS_DIR:"
    echo "=================================="
    if ls "$MODELS_DIR"/*.gguf &>/dev/null; then
        ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null | awk '{print $NF, $5}' | while read f s; do
            printf "  %-50s %s\n" "$(basename "$f")" "$s"
        done
    else
        echo "  (no models downloaded yet)"
    fi
    echo ""
}

main() {
    local target="${1:-}"

    case "$target" in
        ""|"list")
            list_models
            show_downloaded
            ;;
        "recommended")
            download_recommended
            show_downloaded
            ;;
        "all")
            download_all
            show_downloaded
            ;;
        "status"|"downloaded")
            show_downloaded
            ;;
        *)
            if [ -n "${MODELS[$target]+x}" ]; then
                download_model "$target"
                show_downloaded
            else
                echo "Unknown model: $target"
                list_models
                exit 1
            fi
            ;;
    esac
}

main "$@"
