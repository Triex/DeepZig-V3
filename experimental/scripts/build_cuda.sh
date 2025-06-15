#!/bin/bash

# Build script for CUDA libraries for DeepZig V3
# Compiles CUDA C++ code into shared libraries for Zig linkage

set -e

echo "🔨 Building CUDA libraries for DeepZig V3"

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC not found. Install CUDA Toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -o "release [0-9]*\.[0-9]*" | cut -d' ' -f2)
echo "✅ Found CUDA version: $CUDA_VERSION"

# Detect GPU
GPU_ARCH="sm_75"  # Default Turing
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "✅ Detected GPU: $GPU_NAME"

    case "$GPU_NAME" in
        *"RTX 5080"*|*"RTX 5090"*) GPU_ARCH="sm_90" ;;
        *"RTX 4090"*|*"RTX 4080"*) GPU_ARCH="sm_89" ;;
        *"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*) GPU_ARCH="sm_86" ;;
        *"RTX 2080"*|*"RTX 2070"*) GPU_ARCH="sm_75" ;;
        *"GTX 1080"*|*"GTX 1070"*) GPU_ARCH="sm_61" ;;
    esac
fi

echo "✅ Using GPU architecture: $GPU_ARCH"

# Build directories
mkdir -p build/cuda
BUILD_DIR="build/cuda"
CUDA_SRC_DIR="src/backends/cuda"

# Compile flags
NVCC_FLAGS="-shared -Xcompiler -fPIC -O3 -use_fast_math"
NVCC_FLAGS="$NVCC_FLAGS -gencode arch=compute_${GPU_ARCH#sm_},code=sm_${GPU_ARCH#sm_}"
NVCC_FLAGS="$NVCC_FLAGS -I/usr/local/cuda/include -lcublas -lcudart"

# Compile library
OUTPUT_LIB="$BUILD_DIR/libdeepzig_cuda.so"
echo "🔨 Compiling CUDA library..."

if nvcc $NVCC_FLAGS -o "$OUTPUT_LIB" "$CUDA_SRC_DIR/cuda_implementation.cu"; then
    echo "✅ CUDA library compiled: $OUTPUT_LIB"
    ln -sf "$OUTPUT_LIB" "libdeepzig_cuda.so"
    echo "✅ Created symlink: libdeepzig_cuda.so"
    echo "🎉 CUDA build completed successfully!"
    echo "💡 Set: export LD_LIBRARY_PATH=$BUILD_DIR:\$LD_LIBRARY_PATH"
else
    echo "❌ CUDA compilation failed"
    exit 1
fi
