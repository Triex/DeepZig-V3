#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 TriexDev

# DeepZig V3 Environment Setup (No Sudo Required)
# Optimizes BLAS libraries for maximum performance

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 DeepZig V3 Environment Setup${NC}"
echo "=================================="

# Detect system information
CPU_COUNT=$(nproc)
CPU_INFO=$(lscpu)

echo -e "${GREEN}📊 System Information:${NC}"
echo "  CPU Cores: $CPU_COUNT"

# Extract CPU vendor
if echo "$CPU_INFO" | grep -q "AMD"; then
    CPU_VENDOR="AMD"
    CPU_MODEL=$(echo "$CPU_INFO" | grep "Model name" | cut -d: -f2 | xargs)
elif echo "$CPU_INFO" | grep -q "Intel"; then
    CPU_VENDOR="Intel"
    CPU_MODEL=$(echo "$CPU_INFO" | grep "Model name" | cut -d: -f2 | xargs)
else
    CPU_VENDOR="Unknown"
    CPU_MODEL="Unknown"
fi

echo "  CPU: $CPU_VENDOR - $CPU_MODEL"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "  GPU: $GPU_INFO"
else
    echo "  GPU: Not detected or not NVIDIA"
fi

echo ""

# Setup OpenBLAS optimization
echo -e "${GREEN}🧮 Configuring OpenBLAS Environment:${NC}"

# Use all available threads
export OPENBLAS_NUM_THREADS=$CPU_COUNT
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

# Optimize for the detected CPU
case "$CPU_VENDOR" in
    "AMD")
        export OPENBLAS_CORETYPE="ZEN"
        echo "  OPENBLAS_CORETYPE=ZEN (AMD optimization)"
        ;;
    "Intel")
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

# Intel MKL Configuration (if using MKL)
export MKL_NUM_THREADS=$CPU_COUNT
export MKL_DYNAMIC=FALSE
export MKL_ENABLE_INSTRUCTIONS=AVX2
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS (if using Intel MKL)"

# General Performance
export MALLOC_ARENA_MAX=4
echo "  MALLOC_ARENA_MAX=4 (reduce memory fragmentation)"

# Reduce debug log spam
export ZIG_LOG_LEVEL=info
echo "  ZIG_LOG_LEVEL=info (reduce debug spam)"

echo ""

# Create environment file for future use
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.blas_env"

echo -e "${GREEN}📝 Creating environment file: $ENV_FILE${NC}"

cat > "$ENV_FILE" << EOF
# DeepZig V3 BLAS Environment Configuration
# Generated on $(date)
# System: $CPU_VENDOR $CPU_MODEL ($CPU_COUNT cores)

# OpenBLAS Configuration
export OPENBLAS_NUM_THREADS=$CPU_COUNT
export OPENBLAS_MAIN_FREE=1
export OMP_NUM_THREADS=1

# Intel MKL Configuration (if using MKL)
export MKL_NUM_THREADS=$CPU_COUNT
export MKL_DYNAMIC=FALSE
export MKL_ENABLE_INSTRUCTIONS=AVX2

# General Performance
export MALLOC_ARENA_MAX=4

# Reduce debug spam
export ZIG_LOG_LEVEL=info

# CPU-specific optimizations
EOF

case "$CPU_VENDOR" in
    "AMD")
        echo "export OPENBLAS_CORETYPE=ZEN" >> "$ENV_FILE"
        ;;
    "Intel")
        echo "export OPENBLAS_CORETYPE=HASWELL" >> "$ENV_FILE"
        ;;
esac

echo ""
echo -e "${BLUE}🎯 Performance Recommendations:${NC}"
echo ""

case "$CPU_VENDOR" in
    "AMD")
        echo "  For AMD Ryzen processors:"
        echo "  • OpenBLAS with ZEN optimization should give ~1000-1500 GFLOPS"
        echo "  • Your Ryzen 9 3900X should achieve excellent performance"
        ;;
    "Intel")
        echo "  For Intel processors:"
        echo "  • Intel MKL typically provides best performance (~20-30% faster)"
        echo "  • OpenBLAS is a good free alternative"
        ;;
esac

echo ""
echo "  Build recommendations:"
echo "  • Use release mode: zig build -Doptimize=ReleaseFast"
echo "  • Environment is already set for reduced debug spam"
echo "  • Monitor with: htop (for CPU usage)"

echo ""
echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""
echo -e "${YELLOW}🚀 Ready to run DeepZig V3 with optimized performance:${NC}"
echo "   # Environment is already active in this shell"
echo "   zig build -Doptimize=ReleaseFast run -- --model models/deepzig-medium-model"
echo ""
echo -e "${YELLOW}💡 To use these settings in future sessions:${NC}"
echo "   source $ENV_FILE"
echo "   # or add to your ~/.bashrc or ~/.zshrc"
echo ""
