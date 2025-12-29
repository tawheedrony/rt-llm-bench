#!/bin/bash
# Setup script for llama.cpp - pins to specific commit for reproducibility
# Configuration-only study: no source modifications

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP_DIR="${PROJECT_ROOT}/llama.cpp"

# Pin to specific commit for reproducibility
# Update this commit hash when upgrading llama.cpp version
# LLAMA_CPP_COMMIT="b4382"  # Tag b4382, Dec 2024 - too old for Gemma3/DeepSeek-R1
LLAMA_CPP_COMMIT="master"  # Use latest for newer model support
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"

# Build configuration - CPU only, no GPU backends
BUILD_TYPE="Release"
THREADS=$(nproc)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

check_dependencies() {
    log "Checking dependencies..."
    local missing=()

    for cmd in git cmake make g++; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing[*]}"
    fi

    log "All dependencies found"
}

clone_or_update() {
    if [ -d "$LLAMA_CPP_DIR" ]; then
        log "llama.cpp directory exists, checking commit..."
        cd "$LLAMA_CPP_DIR"

        current_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        if [[ "$current_commit" == "${LLAMA_CPP_COMMIT}"* ]]; then
            log "Already at correct commit: $current_commit"
            return 0
        fi

        log "Fetching and checking out commit ${LLAMA_CPP_COMMIT}..."
        git fetch origin
        git checkout "$LLAMA_CPP_COMMIT"
    else
        log "Cloning llama.cpp..."
        git clone "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
        cd "$LLAMA_CPP_DIR"
        git checkout "$LLAMA_CPP_COMMIT"
    fi

    log "llama.cpp at commit: $(git rev-parse --short HEAD)"
}

build_llamacpp() {
    log "Building llama.cpp (CPU-only)..."
    cd "$LLAMA_CPP_DIR"

    # Clean previous build
    rm -rf build
    mkdir build
    cd build

    # Configure - CPU only, no CUDA/Metal/etc
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_SYCL=OFF \
        -DGGML_OPENMP=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=OFF

    # Build
    make -j"$THREADS"

    log "Build complete"
}

verify_build() {
    log "Verifying build..."

    local main_binary="${LLAMA_CPP_DIR}/build/bin/llama-cli"
    if [ ! -x "$main_binary" ]; then
        # Try alternate location
        main_binary="${LLAMA_CPP_DIR}/build/bin/main"
    fi

    if [ ! -x "$main_binary" ]; then
        error "llama-cli binary not found or not executable"
    fi

    log "Binary found: $main_binary"
    "$main_binary" --version 2>/dev/null || "$main_binary" --help | head -5

    log "Build verified successfully"
}

record_provenance() {
    log "Recording build provenance..."

    local provenance_file="${PROJECT_ROOT}/docs/llama_cpp_provenance.txt"

    cat > "$provenance_file" << EOF
llama.cpp Build Provenance
==========================
Date: $(date -Iseconds)
Commit: $(cd "$LLAMA_CPP_DIR" && git rev-parse HEAD)
Short Commit: $(cd "$LLAMA_CPP_DIR" && git rev-parse --short HEAD)
Branch/Tag: ${LLAMA_CPP_COMMIT}
Build Type: ${BUILD_TYPE}
Compiler: $(g++ --version | head -1)
CMake: $(cmake --version | head -1)
Host: $(uname -a)
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
Cores: $(nproc)

Build Flags:
- GGML_CUDA=OFF
- GGML_METAL=OFF
- GGML_VULKAN=OFF
- GGML_OPENMP=ON
- CPU-only build
EOF

    log "Provenance recorded to: $provenance_file"
}

main() {
    log "=== llama.cpp Setup Script ==="
    log "Project root: $PROJECT_ROOT"

    check_dependencies
    clone_or_update
    build_llamacpp
    verify_build
    record_provenance

    log "=== Setup Complete ==="
    log "llama.cpp location: $LLAMA_CPP_DIR"
    log "Binary: ${LLAMA_CPP_DIR}/build/bin/llama-cli"
}

main "$@"
