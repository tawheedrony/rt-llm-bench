#!/bin/bash
# System preparation for bounded real-time inference benchmarks
# Configuration-only: no kernel modifications, uses available system controls
#
# Usage:
#   ./prepare_system.sh status    - Show current system state
#   ./prepare_system.sh isolate   - Configure for isolated benchmarks
#   ./prepare_system.sh restore   - Restore default settings
#   ./prepare_system.sh contended - Configure for contended environment tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/../.system_state_backup"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Check if running as root for certain operations
check_root() {
    if [ "$EUID" -ne 0 ]; then
        return 1
    fi
    return 0
}

# Get current CPU governor
get_governor() {
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    else
        echo "unavailable"
    fi
}

# Get current CPU frequency range
get_freq_range() {
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq ]; then
        local min=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
        local max=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
        echo "$((min/1000))MHz - $((max/1000))MHz"
    else
        echo "unavailable"
    fi
}

# Get turbo boost status
get_turbo_status() {
    # Intel
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        if [ "$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)" = "1" ]; then
            echo "disabled"
        else
            echo "enabled"
        fi
    # AMD
    elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
        if [ "$(cat /sys/devices/system/cpu/cpufreq/boost)" = "0" ]; then
            echo "disabled"
        else
            echo "enabled"
        fi
    else
        echo "unavailable"
    fi
}

# Show current system status
show_status() {
    echo "========================================"
    echo "SYSTEM STATUS FOR BENCHMARKING"
    echo "========================================"
    echo ""

    echo "CPU Information:"
    echo "  Model: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
    echo "  Cores: $(nproc)"
    echo "  Governor: $(get_governor)"
    echo "  Frequency: $(get_freq_range)"
    echo "  Turbo Boost: $(get_turbo_status)"
    echo ""

    echo "Memory Information:"
    echo "  Total: $(free -h | awk '/^Mem:/ {print $2}')"
    echo "  Available: $(free -h | awk '/^Mem:/ {print $7}')"
    echo "  Swap: $(free -h | awk '/^Swap:/ {print $2}')"
    echo ""

    echo "Process Priority:"
    echo "  Max nice (user): $(ulimit -e)"
    echo "  RT priority available: $(if [ -f /proc/sys/kernel/sched_rt_runtime_us ]; then echo "yes"; else echo "no"; fi)"
    echo ""

    echo "Kernel Parameters:"
    if [ -f /proc/sys/kernel/sched_rt_runtime_us ]; then
        echo "  sched_rt_runtime_us: $(cat /proc/sys/kernel/sched_rt_runtime_us)"
    fi
    if [ -f /proc/sys/vm/swappiness ]; then
        echo "  vm.swappiness: $(cat /proc/sys/vm/swappiness)"
    fi
    echo ""

    echo "Potential Noise Sources:"
    echo "  Background processes: $(ps aux | wc -l)"

    # Check for common noise sources
    local noise_sources=""
    pgrep -x "snapd" > /dev/null 2>&1 && noise_sources+="snapd "
    pgrep -x "packagekitd" > /dev/null 2>&1 && noise_sources+="packagekitd "
    pgrep -x "tracker-miner" > /dev/null 2>&1 && noise_sources+="tracker-miner "
    pgrep -x "baloo" > /dev/null 2>&1 && noise_sources+="baloo "

    if [ -n "$noise_sources" ]; then
        echo "  Known noise sources running: $noise_sources"
    else
        echo "  Known noise sources: none detected"
    fi

    echo ""
    echo "========================================"
}

# Save current state for restoration
save_state() {
    log_info "Saving current system state..."

    local state=""
    state+="GOVERNOR=$(get_governor)\n"
    state+="SWAPPINESS=$(cat /proc/sys/vm/swappiness 2>/dev/null || echo 60)\n"

    echo -e "$state" > "$STATE_FILE"
    log_info "State saved to $STATE_FILE"
}

# Configure for isolated benchmarks
configure_isolated() {
    log_info "Configuring system for isolated benchmarks..."
    echo ""

    # Save state first
    save_state

    echo "Recommended actions for isolated benchmarks:"
    echo ""

    # CPU Governor
    echo "1. CPU Governor (requires root):"
    if check_root; then
        if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
            for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
                echo "performance" > "$cpu" 2>/dev/null || true
            done
            log_info "   Set governor to 'performance'"
        fi
    else
        echo "   Run as root: cpupower frequency-set -g performance"
        echo "   Or: echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
    fi
    echo ""

    # Disable turbo boost
    echo "2. Disable Turbo Boost (requires root):"
    if check_root; then
        if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
            echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
            log_info "   Disabled Intel turbo boost"
        elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
            echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
            log_info "   Disabled AMD boost"
        fi
    else
        echo "   Intel: echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"
        echo "   AMD: echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost"
    fi
    echo ""

    # Reduce swappiness
    echo "3. Reduce Swappiness (requires root):"
    if check_root; then
        if [ -f /proc/sys/vm/swappiness ]; then
            echo 10 > /proc/sys/vm/swappiness 2>/dev/null || true
            log_info "   Set swappiness to 10"
        fi
    else
        echo "   Run: echo 10 | sudo tee /proc/sys/vm/swappiness"
    fi
    echo ""

    # Stop noisy services
    echo "4. Stop Background Services (requires root):"
    local services="snapd packagekit tracker-miner-fs"
    for svc in $services; do
        if check_root; then
            systemctl stop "$svc" 2>/dev/null && log_info "   Stopped $svc" || true
        else
            echo "   sudo systemctl stop $svc"
        fi
    done
    echo ""

    # CPU pinning advice
    echo "5. Use CPU pinning when running benchmarks:"
    echo "   taskset -c 0-3 python run_experiment.py ..."
    echo "   Or use --pin-cpu flag in run_experiment.py"
    echo ""

    # Memory locking
    echo "6. Memory locking (for real-time):"
    echo "   Use --mlock flag when running benchmarks"
    echo "   May require: ulimit -l unlimited (needs /etc/security/limits.conf)"
    echo ""

    log_info "Isolated environment configuration complete"
    echo ""
    show_status
}

# Restore default settings
restore_defaults() {
    log_info "Restoring default system settings..."

    if [ ! -f "$STATE_FILE" ]; then
        log_warn "No saved state found, using defaults"
    fi

    # Restore governor
    if check_root; then
        if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
            for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
                echo "powersave" > "$cpu" 2>/dev/null || \
                echo "ondemand" > "$cpu" 2>/dev/null || true
            done
            log_info "Restored governor to powersave/ondemand"
        fi

        # Re-enable turbo
        if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
            echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
            log_info "Re-enabled Intel turbo boost"
        elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
            echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
            log_info "Re-enabled AMD boost"
        fi

        # Restore swappiness
        if [ -f /proc/sys/vm/swappiness ]; then
            echo 60 > /proc/sys/vm/swappiness 2>/dev/null || true
            log_info "Restored swappiness to 60"
        fi
    else
        log_warn "Run as root to restore system settings"
        echo "Commands to restore:"
        echo "  sudo cpupower frequency-set -g powersave"
        echo "  echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo"
        echo "  echo 60 | sudo tee /proc/sys/vm/swappiness"
    fi

    rm -f "$STATE_FILE" 2>/dev/null || true
    log_info "Restore complete"
}

# Configure for contended environment testing
configure_contended() {
    log_info "Configuring for contended environment testing..."
    echo ""

    echo "Contended environment tests simulate real-world edge device conditions"
    echo "with background load competing for resources."
    echo ""

    echo "Background load options:"
    echo ""
    echo "1. CPU load with stress-ng (install: apt install stress-ng):"
    echo "   25% load: stress-ng --cpu 1 --cpu-load 25 &"
    echo "   50% load: stress-ng --cpu 2 --cpu-load 50 &"
    echo "   75% load: stress-ng --cpu 3 --cpu-load 75 &"
    echo ""
    echo "2. Memory pressure:"
    echo "   stress-ng --vm 1 --vm-bytes 1G &"
    echo ""
    echo "3. I/O load:"
    echo "   stress-ng --io 1 &"
    echo ""
    echo "4. Combined realistic load:"
    echo "   stress-ng --cpu 2 --cpu-load 30 --vm 1 --vm-bytes 512M --io 1 &"
    echo ""

    echo "To stop stress-ng:"
    echo "   pkill stress-ng"
    echo ""

    # Check if stress-ng is installed
    if command -v stress-ng &> /dev/null; then
        log_info "stress-ng is available"
    else
        log_warn "stress-ng not installed. Install with: sudo apt install stress-ng"
    fi
}

# Main
case "${1:-status}" in
    status)
        show_status
        ;;
    isolate|isolated)
        configure_isolated
        ;;
    restore)
        restore_defaults
        ;;
    contended|contend)
        configure_contended
        ;;
    *)
        echo "Usage: $0 {status|isolate|restore|contended}"
        echo ""
        echo "Commands:"
        echo "  status    - Show current system state"
        echo "  isolate   - Configure for isolated benchmarks (reduces variance)"
        echo "  restore   - Restore default system settings"
        echo "  contended - Show options for contended environment testing"
        exit 1
        ;;
esac
