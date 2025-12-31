#!/bin/bash
# Run comprehensive experiments: all models x all sampling strategies
# Generates complete benchmark data for cross-comparison analysis

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_ROOT}/models"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Default experiment parameters
ITERATIONS=${ITERATIONS:-30}
WARMUP=${WARMUP:-3}
TOKENS=${TOKENS:-128}
THREADS=${THREADS:-4}
CONTEXT=${CONTEXT:-2048}
PROMPT="Once upon a time in a land far away, there lived a wise old wizard who"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Setup logging
LOG_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${RESULTS_DIR}/log_run_${LOG_TIMESTAMP}.log"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Setup tee for logging (both terminal and file)
exec > >(tee -a "$LOG_FILE") 2>&1

# Strip ANSI colors for log file
strip_colors() {
    sed -r 's/\x1b\[[0-9;]*m//g' "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
}

# Trap to clean up colors in log file on exit
trap strip_colors EXIT

# Available models (name -> path mapping)
declare -A MODELS=(
    ["gemma3-270m"]="${MODELS_DIR}/gemma-3-270m-it-Q8_0.gguf"
    ["gemma3-1b"]="${MODELS_DIR}/google_gemma-3-1b-it-Q8_0.gguf"
    ["llama3.2-1b"]="${MODELS_DIR}/Llama-3.2-1B-Instruct-Q8_0.gguf"
    ["deepseek-r1-1.5b"]="${MODELS_DIR}/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
    ["smollm2-1.7b"]="${MODELS_DIR}/SmolLM2.q8.gguf"
)

# Sampling strategies with their parameters
declare -A SAMPLING_CONFIGS=(
    ["greedy"]="--sampling greedy --temp 0.0"
    ["temperature"]="--sampling temperature --temp 0.8"
    ["top_k"]="--sampling top_k --temp 0.8 --top-k 40"
    ["top_p"]="--sampling top_p --temp 0.8 --top-p 0.9"
)

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run comprehensive benchmark experiments across all models and sampling strategies.

OPTIONS:
    -h, --help              Show this help message
    -m, --models MODELS     Comma-separated list of models to test (default: all)
                            Available: ${!MODELS[@]}
    -s, --sampling TYPES    Comma-separated list of sampling strategies (default: all)
                            Available: ${!SAMPLING_CONFIGS[@]}
    -i, --iterations N      Number of iterations per experiment (default: $ITERATIONS)
    -w, --warmup N          Number of warmup runs (default: $WARMUP)
    -t, --threads N         Number of threads (default: $THREADS)
    -n, --tokens N          Output tokens to generate (default: $TOKENS)
    -c, --context N         Context length (default: $CONTEXT)
    --pin-cpu               Enable CPU pinning (taskset)
    --cpu-cores CORES       CPU cores to use with pinning (default: 0-3)
    --mlock                 Enable memory locking
    --dry-run               Show what would be run without executing
    -o, --output DIR        Output directory (default: $RESULTS_DIR)

EXAMPLES:
    # Run all models with all sampling strategies (30 iterations each)
    $(basename "$0")

    # Run specific models with greedy sampling
    $(basename "$0") -m gemma3-270m,llama3.2-1b -s greedy

    # Run all models with temperature and top_k, 50 iterations
    $(basename "$0") -s temperature,top_k -i 50

    # Run with CPU pinning and memory locking (isolated environment)
    $(basename "$0") --pin-cpu --mlock

    # Quick test run
    $(basename "$0") -m gemma3-270m -s greedy -i 5

ENVIRONMENT VARIABLES:
    ITERATIONS      Override default iterations
    WARMUP          Override default warmup runs
    TOKENS          Override default output tokens
    THREADS         Override default thread count
    CONTEXT         Override default context length

EOF
    exit 0
}

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_section() {
    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $*${NC}"
    echo -e "${GREEN}============================================================${NC}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Parse command line arguments
SELECTED_MODELS=""
SELECTED_SAMPLING=""
PIN_CPU=""
CPU_CORES="0-3"
MLOCK=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -m|--models)
            SELECTED_MODELS="$2"
            shift 2
            ;;
        -s|--sampling)
            SELECTED_SAMPLING="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -n|--tokens)
            TOKENS="$2"
            shift 2
            ;;
        -c|--context)
            CONTEXT="$2"
            shift 2
            ;;
        --pin-cpu)
            PIN_CPU="--pin-cpu"
            shift
            ;;
        --cpu-cores)
            CPU_CORES="$2"
            shift 2
            ;;
        --mlock)
            MLOCK="--mlock"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -o|--output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Convert comma-separated lists to arrays
if [[ -n "$SELECTED_MODELS" ]]; then
    IFS=',' read -ra MODEL_LIST <<< "$SELECTED_MODELS"
else
    MODEL_LIST=("${!MODELS[@]}")
fi

if [[ -n "$SELECTED_SAMPLING" ]]; then
    IFS=',' read -ra SAMPLING_LIST <<< "$SELECTED_SAMPLING"
else
    SAMPLING_LIST=("${!SAMPLING_CONFIGS[@]}")
fi

# Validate selected models
for model in "${MODEL_LIST[@]}"; do
    if [[ ! -v "MODELS[$model]" ]]; then
        log_error "Unknown model: $model"
        echo "Available models: ${!MODELS[@]}"
        exit 1
    fi
done

# Validate selected sampling strategies
for sampling in "${SAMPLING_LIST[@]}"; do
    if [[ ! -v "SAMPLING_CONFIGS[$sampling]" ]]; then
        log_error "Unknown sampling strategy: $sampling"
        echo "Available strategies: ${!SAMPLING_CONFIGS[@]}"
        exit 1
    fi
done

run_experiment() {
    local model_name="$1"
    local model_path="$2"
    local sampling_name="$3"
    local sampling_args="$4"
    local experiment_name="${model_name}_${sampling_name}"
    local results_folder="${RESULTS_DIR}/${experiment_name}"

    log "Running: $experiment_name"
    echo "  Model: $model_path"
    echo "  Sampling: $sampling_name ($sampling_args)"
    echo "  Iterations: $ITERATIONS, Warmup: $WARMUP"
    echo "  Tokens: $TOKENS, Context: $CONTEXT, Threads: $THREADS"
    [[ -n "$PIN_CPU" ]] && echo "  CPU Pinning: enabled (cores: $CPU_CORES)"
    [[ -n "$MLOCK" ]] && echo "  Memory Lock: enabled"
    echo "  Results: $results_folder"
    echo ""

    if [[ -n "$DRY_RUN" ]]; then
        echo "  [DRY RUN - skipping execution]"
        return 0
    fi

    # Build the command
    local cmd="python ${SCRIPT_DIR}/run_experiment.py"
    cmd+=" -m \"$model_path\""
    cmd+=" -n \"$experiment_name\""
    cmd+=" -i $ITERATIONS"
    cmd+=" --warmup $WARMUP"
    cmd+=" --tokens $TOKENS"
    cmd+=" --context $CONTEXT"
    cmd+=" -t $THREADS"
    cmd+=" -p \"$PROMPT\""
    cmd+=" $sampling_args"
    [[ -n "$PIN_CPU" ]] && cmd+=" $PIN_CPU --cpu-cores $CPU_CORES"
    [[ -n "$MLOCK" ]] && cmd+=" $MLOCK"
    cmd+=" -o \"$results_folder\""

    # Execute
    if ! eval "$cmd"; then
        log_warning "Experiment $experiment_name failed"
        return 1
    fi
}

count_available_models() {
    local count=0
    for model in "${MODEL_LIST[@]}"; do
        local path="${MODELS[$model]}"
        if [[ -f "$path" ]] || [[ -L "$path" ]]; then
            ((count++)) || true
        fi
    done
    echo "$count"
}

main() {
    log_section "RT-LLM-Bench: Comprehensive Experiment Suite"

    echo ""
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Configuration:"
    echo "  Models to test: ${MODEL_LIST[*]}"
    echo "  Sampling strategies: ${SAMPLING_LIST[*]}"
    echo "  Iterations per experiment: $ITERATIONS"
    echo "  Warmup runs: $WARMUP"
    echo "  Output tokens: $TOKENS"
    echo "  Context length: $CONTEXT"
    echo "  Threads: $THREADS"
    [[ -n "$PIN_CPU" ]] && echo "  CPU Pinning: enabled (cores: $CPU_CORES)"
    [[ -n "$MLOCK" ]] && echo "  Memory Lock: enabled"
    [[ -n "$DRY_RUN" ]] && echo "  Mode: DRY RUN"
    echo "  Results directory: $RESULTS_DIR"
    echo ""

    # Count experiments
    local available_models=$(count_available_models)
    local total_experiments=$((available_models * ${#SAMPLING_LIST[@]}))
    local total_runs=$((total_experiments * ITERATIONS))

    echo "Experiments to run: $total_experiments ($available_models models x ${#SAMPLING_LIST[@]} sampling strategies)"
    echo "Total benchmark runs: $total_runs"
    echo ""

    # Check for missing models
    for model in "${MODEL_LIST[@]}"; do
        local path="${MODELS[$model]}"
        if [[ ! -f "$path" ]] && [[ ! -L "$path" ]]; then
            log_warning "Model not found: $model ($path)"
        fi
    done
    echo ""

    local start_time=$(date +%s)
    local experiment_count=0
    local skipped_count=0

    # Run all combinations
    for model_name in "${MODEL_LIST[@]}"; do
        local model_path="${MODELS[$model_name]}"

        # Check if model exists
        if [[ ! -f "$model_path" ]] && [[ ! -L "$model_path" ]]; then
            log_warning "Skipping $model_name - model not found"
            ((skipped_count++)) || true
            continue
        fi

        for sampling_name in "${SAMPLING_LIST[@]}"; do
            local sampling_args="${SAMPLING_CONFIGS[$sampling_name]}"

            log_section "Experiment $((experiment_count + 1))/$total_experiments: ${model_name} + ${sampling_name}"

            if run_experiment "$model_name" "$model_path" "$sampling_name" "$sampling_args"; then
                ((experiment_count++)) || true
            else
                ((skipped_count++)) || true
            fi

            # Small delay between experiments
            if [[ -z "$DRY_RUN" ]] && [[ $experiment_count -lt $total_experiments ]]; then
                log "Cooldown before next experiment..."
                sleep 2
            fi
        done
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_section "All Experiments Complete!"
    echo ""
    echo "Summary:"
    echo "  Experiments completed: $experiment_count"
    echo "  Experiments skipped: $skipped_count"
    echo "  Total time: $((duration / 60)) minutes $((duration % 60)) seconds"
    echo "  Results saved to: $RESULTS_DIR"
    echo "  Log file: $LOG_FILE"
    echo ""

    # Generate comparison command
    if [[ $experiment_count -gt 0 ]] && [[ -z "$DRY_RUN" ]]; then
        echo "To compare results across experiments:"
        echo ""
        echo "  python scripts/analyze_results.py compare \\"

        local first=true
        for model_name in "${MODEL_LIST[@]}"; do
            local model_path="${MODELS[$model_name]}"
            if [[ -f "$model_path" ]] || [[ -L "$model_path" ]]; then
                for sampling_name in "${SAMPLING_LIST[@]}"; do
                    local exp_name="${model_name}_${sampling_name}"
                    if $first; then
                        echo "    results/${exp_name}/*/ \\"
                        first=false
                    else
                        echo "    results/${exp_name}/*/ \\"
                    fi
                done
            fi
        done
        echo ""

        echo "To generate a full report:"
        echo "  python scripts/analyze_results.py report results/<experiment>/*/ -o analysis/report.md"
        echo ""
    fi
}

main "$@"
