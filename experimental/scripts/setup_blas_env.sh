#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 TriexDev

# DeepZig V3 BLAS Environment Setup
# Optimizes BLAS libraries for maximum performance on detected hardware

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 DeepZig V3 BLAS Environment Setup${NC}"
echo "=================================="

# Detect system information
detect_system() {
    local cpu_count=$(nproc)
    local cpu_info=$(lscpu)
    local os_name=$(uname -s)

    echo -e "${GREEN}📊 System Information:${NC}"
    echo "  OS: $os_name"
    echo "  CPU Cores: $cpu_count"

    # Extract CPU model
    if echo "$cpu_info" | grep -q "AMD"; then
        local cpu_vendor="AMD"
        local cpu_model=$(echo "$cpu_info" | grep "Model name" | cut -d: -f2 | xargs)
    elif echo "$cpu_info" | grep -q "Intel"; then
        local cpu_vendor="Intel"
        local cpu_model=$(echo "$cpu_info" | grep "Model name" | cut -d: -f2 | xargs)
    else
        local cpu_vendor="Unknown"
        local cpu_model="Unknown"
    fi

    echo "  CPU: $cpu_vendor - $cpu_model"

    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "  GPU: $gpu_info"
    else
        echo "  GPU: Not detected or not NVIDIA"
    fi

    echo ""

    # Return values for use in optimization
    export DETECTED_CPU_COUNT=$cpu_count
    export DETECTED_CPU_VENDOR=$cpu_vendor
    export DETECTED_CPU_MODEL="$cpu_model"
}

# Setup OpenBLAS optimization
setup_openblas() {
    echo -e "${GREEN}🧮 Configuring OpenBLAS:${NC}"

    # Use all available threads
    export OPENBLAS_NUM_THREADS=$DETECTED_CPU_COUNT
    echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

    # Optimize for the detected CPU
    case "$DETECTED_CPU_VENDOR" in
        "AMD")
            # AMD-specific optimizations
            export OPENBLAS_CORETYPE="ZEN"
            echo "  OPENBLAS_CORETYPE=ZEN (AMD optimization)"
            ;;
        "Intel")
            # Intel-specific optimizations
            export OPENBLAS_CORETYPE="HASWELL"
            echo "  OPENBLAS_CORETYPE=HASWELL (Intel optimization)"
            ;;
    esac

    # Memory allocation optimization
    export OPENBLAS_MAIN_FREE=1
    echo "  OPENBLAS_MAIN_FREE=1 (memory optimization)"

    # Disable OpenMP nested parallelism to avoid conflicts
    export OMP_NUM_THREADS=1
    echo "  OMP_NUM_THREADS=1 (avoid nested parallelism)"
}

# Setup Intel MKL optimization (if available)
setup_mkl() {
    echo -e "${GREEN}🧮 Configuring Intel MKL:${NC}"

    # Use all available threads
    export MKL_NUM_THREADS=$DETECTED_CPU_COUNT
    echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"

    # Optimize threading
    export MKL_DYNAMIC=FALSE
    echo "  MKL_DYNAMIC=FALSE (consistent performance)"

    # Memory optimization
    export MKL_ENABLE_INSTRUCTIONS=AVX2
    echo "  MKL_ENABLE_INSTRUCTIONS=AVX2"
}

# Setup general optimizations
setup_general() {
    echo -e "${GREEN}⚡ General Performance Optimizations:${NC}"

    # Disable CPU frequency scaling for consistent benchmarks
    echo "  Setting CPU governor to 'performance' (requires sudo)"
    if command -v cpupower &> /dev/null; then
        sudo cpupower frequency-set -g performance 2>/dev/null || echo "    Warning: Could not set CPU governor"
    else
        echo "    cpupower not available, skipping CPU governor setup"
    fi

    # Memory allocation optimization
    export MALLOC_ARENA_MAX=4
    echo "  MALLOC_ARENA_MAX=4 (reduce memory fragmentation)"

    # Disable address space randomization for consistent performance
    echo "  Disabling ASLR for this session (performance optimization)"
    echo 0 | sudo tee /proc/sys/kernel/randomize_va_space > /dev/null 2>&1 || echo "    Warning: Could not disable ASLR"
}

# Create environment file
create_env_file() {
    local env_file="$1"
    echo -e "${GREEN}📝 Creating environment file: $env_file${NC}"

    cat > "$env_file" << EOF
# DeepZig V3 BLAS Environment Configuration
# Generated on $(date)
# System: $DETECTED_CPU_VENDOR $DETECTED_CPU_MODEL ($DETECTED_CPU_COUNT cores)

# OpenBLAS Configuration
export OPENBLAS_NUM_THREADS=$DETECTED_CPU_COUNT
export OPENBLAS_MAIN_FREE=1
export OMP_NUM_THREADS=1

# Intel MKL Configuration (if using MKL)
export MKL_NUM_THREADS=$DETECTED_CPU_COUNT
export MKL_DYNAMIC=FALSE
export MKL_ENABLE_INSTRUCTIONS=AVX2

# General Performance
export MALLOC_ARENA_MAX=4

# CPU-specific optimizations
EOF

    case "$DETECTED_CPU_VENDOR" in
        "AMD")
            echo "export OPENBLAS_CORETYPE=ZEN" >> "$env_file"
            ;;
        "Intel")
            echo "export OPENBLAS_CORETYPE=HASWELL" >> "$env_file"
            ;;
    esac

    echo ""
    echo -e "${YELLOW}💡 To use these settings in future sessions:${NC}"
    echo "   source $env_file"
    echo "   # or add to your ~/.bashrc or ~/.zshrc"
}

# Verify BLAS installation
verify_blas() {
    echo -e "${GREEN}🔍 Verifying BLAS Installation:${NC}"

    # Check for OpenBLAS
    if ldconfig -p | grep -q openblas; then
        echo "  ✅ OpenBLAS found"
        local openblas_version=$(pkg-config --modversion openblas 2>/dev/null || echo "unknown")
        echo "     Version: $openblas_version"
    else
        echo -e "  ${RED}❌ OpenBLAS not found${NC}"
        echo "     Install with: sudo apt install libopenblas-dev (Ubuntu/Debian)"
        echo "                   sudo dnf install openblas-devel (Fedora/RHEL)"
    fi

    # Check for Intel MKL
    if ldconfig -p | grep -q mkl; then
        echo "  ✅ Intel MKL found"
    else
        echo "  ℹ️  Intel MKL not found (optional)"
    fi

    echo ""
}

# Performance recommendations
show_recommendations() {
    echo -e "${BLUE}🎯 Performance Recommendations:${NC}"
    echo ""

    case "$DETECTED_CPU_VENDOR" in
        "AMD")
            echo "  For AMD Ryzen processors:"
            echo "  • OpenBLAS with ZEN optimization should give ~1000-1500 GFLOPS"
            echo "  • Consider AMD BLIS for even better performance"
            echo "  • Your Ryzen 9 3900X should achieve excellent performance"
            ;;
        "Intel")
            echo "  For Intel processors:"
            echo "  • Intel MKL typically provides best performance (~20-30% faster)"
            echo "  • OpenBLAS is a good free alternative"
            echo "  • Consider Intel oneAPI MKL for latest optimizations"
            ;;
    esac

    echo ""
    echo "  Build recommendations:"
    echo "  • Use release mode: zig build -Doptimize=ReleaseFast"
    echo "  • Set log level: ZIG_LOG_LEVEL=info (to reduce debug spam)"
    echo "  • Monitor with: htop or nvidia-smi (for GPU usage)"
    echo ""
}

# Main execution
main() {
    detect_system
    verify_blas
    setup_openblas
    setup_mkl
    setup_general

    # Create environment file
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local env_file="$script_dir/../.blas_env"
    create_env_file "$env_file"

    show_recommendations

    echo -e "${GREEN}✅ BLAS environment setup complete!${NC}"
    echo ""
    echo -e "${YELLOW}🚀 Ready to run DeepZig V3 with optimized performance:${NC}"
    echo "   cd experimental"
    echo "   source scripts/../.blas_env"
    echo "   zig build -Doptimize=ReleaseFast run -- --model models/deepzig-medium-model"
    echo ""
}

# Run main function
main "$@"
