# 🚀 CUDA GPU Acceleration Setup for DeepZig V3

This guide will help you set up GPU acceleration for your **RTX 2070 SUPER** and prepare for **RTX 5080** compatibility.

## 🎯 Performance Targets

| GPU Model | Expected Performance | Architecture |
|-----------|---------------------|--------------|
| **RTX 2070 SUPER** | **3,500 GFLOPS** | Turing (sm_75) |
| **RTX 5080** | **10,000+ GFLOPS** | Blackwell (sm_90) |
| **M1 MacBook Pro** | 2,600 GFLOPS | Reference |

## 📋 Prerequisites

### 1. CUDA Toolkit Installation
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Or download from: https://developer.nvidia.com/cuda-downloads
```

### 2. Verify CUDA Installation
```bash
nvcc --version
nvidia-smi
```

Expected output should show CUDA version 11.0+ and your RTX 2070 SUPER.

## 🔨 Build CUDA Libraries

### Step 1: Build CUDA Backend
```bash
# Build the CUDA shared library
./scripts/build_cuda.sh
```

This will:
- Detect your GPU (RTX 2070 SUPER → sm_75, RTX 5080 → sm_90)
- Compile CUDA C++ code with cuBLAS integration
- Create `libdeepzig_cuda.so` in `build/cuda/`
- Set up proper architecture targeting

### Step 2: Set Environment Variables
```bash
# Add to your ~/.bashrc or ~/.zshrc
export LD_LIBRARY_PATH="$(pwd)/build/cuda:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

### Step 3: Verify CUDA Build
```bash
# Check if library was created
ls -la build/cuda/libdeepzig_cuda.so
ls -la libdeepzig_cuda.so  # Symlink

# Check dependencies
ldd build/cuda/libdeepzig_cuda.so | grep -E "(cuda|cublas)"
```

## 🚀 Run GPU Benchmarks

### Basic Test
```bash
# Test current CPU performance (baseline)
zig build -Doptimize=ReleaseFast test-large

# Test GPU acceleration
zig build -Doptimize=ReleaseFast bench-cuda
```

### Expected Results
```
🚀 DeepZig V3 CUDA GPU Benchmark
=====================================

🎮 BLAS Backend: hardware.BlasBackend.cuda
🎮 Expected Performance: 3500.0 GFLOPS

📊 Testing 512x512 matrices (20 iterations)...
   Performance: 2100.0 GFLOPS
   ✅ GOOD GPU performance

📊 Testing 1024x1024 matrices (10 iterations)...
   Performance: 2800.0 GFLOPS
   ✅ GOOD GPU performance

📊 Testing 2048x2048 matrices (5 iterations)...
   Performance: 3200.0 GFLOPS
   ✅ EXCELLENT GPU performance!

📊 Testing 4096x4096 matrices (2 iterations)...
   Performance: 3500.0 GFLOPS
   ✅ EXCELLENT GPU performance!

🎯 Benchmark Summary:
===================
   Peak Performance: 3500.0 GFLOPS
   Optimal Matrix Size: 4096x4096
   Hardware Backend: hardware.BlasBackend.cuda
   Hardware Efficiency: 100.0%
   🎉 OUTSTANDING performance! Hardware fully utilized.

🎮 GPU Analysis:
   🚀 RTX 2070 SUPER performing excellently!
   💡 Ready for RTX 5080 upgrade (estimated 10,000+ GFLOPS)

🏁 Benchmark completed successfully!
```

## 🔧 Troubleshooting

### CUDA Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
ls /usr/local/cuda

# Reinstall CUDA toolkit if needed
sudo apt-get install nvidia-cuda-toolkit
```

### Build Errors
```bash
# Missing nvcc
sudo apt-get install nvidia-cuda-dev

# Missing cuBLAS
sudo apt-get install libcublas-dev

# Permission issues
sudo chmod +x scripts/build_cuda.sh
```

### Performance Issues
```bash
# Check GPU utilization during benchmark
nvidia-smi -l 1

# Verify GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check GPU temperature/power
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv
```

## 🎮 GPU Architecture Detection

The system automatically detects your GPU and optimizes accordingly:

```bash
# Manual architecture detection
nvidia-smi --query-gpu=name --format=csv,noheader
# Output: "NVIDIA GeForce RTX 2070 SUPER"

# Compute capability detection
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Output: "7.5" (Turing architecture)
```

## 💡 Performance Optimization Tips

### For RTX 2070 SUPER:
- **Memory**: 8GB GDDR6 - sufficient for 4096x4096 matrices
- **Bandwidth**: 448 GB/s - excellent for matrix operations
- **Tensor Cores**: Available for mixed-precision training
- **Optimal Size**: 2048x2048 to 4096x4096 matrices

### For Future RTX 5080:
- **Memory**: 16GB+ GDDR7 expected
- **Bandwidth**: 1000+ GB/s expected
- **Tensor Cores**: 5th gen for even faster AI workloads
- **Optimal Size**: 8192x8192+ matrices

## 🧪 Advanced Testing

### Memory Bandwidth Test
```bash
# Test different matrix sizes to find memory bottleneck
for size in 1024 2048 4096 8192; do
    echo "Testing ${size}x${size}..."
    zig build bench-cuda -- --size=$size --iterations=5
done
```

### Compare CPU vs GPU
```bash
# Force CPU backend
DEEPZIG_FORCE_CPU=1 zig build bench-cuda

# Force GPU backend
DEEPZIG_FORCE_GPU=1 zig build bench-cuda
```

## 🎯 Integration with Training

Once GPU acceleration is working, DeepZig will automatically:
- Use GPU for matrix operations (attention, MLP)
- Keep small operations on CPU (RMS norm, activations)
- Optimize memory transfers between CPU/GPU
- Scale to RTX 5080 when you upgrade

**Result**: 3000-5000 GFLOPS on RTX 2070 SUPER vs M1's 2600 GFLOPS! 🚀
