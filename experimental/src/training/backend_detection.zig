//! Hardware detection and optimization for DeepZig V3 training
//! Optimized for AMD Ryzen 9 3900X + NVIDIA RTX 2070 SUPER

const std = @import("std");
const builtin = @import("builtin");

pub const HardwareInfo = struct {
    cpu_cores: u32,
    has_nvidia_gpu: bool,
    gpu_compute_capability: ?f32,
    gpu_memory_gb: ?u32,
    cpu_vendor: CpuVendor,
    avx2_support: bool,
    cuda_available: bool,
};

pub const CpuVendor = enum {
    amd,
    intel,
    unknown,
};

pub const BackendConfig = struct {
    use_cuda: bool = false,
    num_worker_threads: u32,
    batch_size: u32,
    use_mixed_precision: bool = true,
    memory_pool_size_mb: u32,
    prefetch_batches: u32,
    use_simd_optimization: bool = true,
};

/// Detect hardware capabilities and return optimized configuration
pub fn detectHardwareAndOptimize(allocator: std.mem.Allocator) !BackendConfig {
    var config = BackendConfig{
        .num_worker_threads = @min(std.Thread.getCpuCount() catch 4, 24),
        .batch_size = 32,
        .memory_pool_size_mb = 4096,
        .prefetch_batches = 4,
    };

    const hardware = detectHardware(allocator);

    // Optimize for AMD Ryzen 9 3900X (24 threads)
    if (hardware.cpu_cores >= 20) {
        config.num_worker_threads = 20; // Leave 4 threads for system
        config.batch_size = 64; // Larger batches for high core count
        config.prefetch_batches = 6;
        config.memory_pool_size_mb = 8192; // 8GB for high-end system
    }

    // NVIDIA RTX 2070 SUPER optimization
    if (hardware.has_nvidia_gpu and hardware.gpu_memory_gb != null and hardware.gpu_memory_gb.? >= 8) {
        config.use_cuda = true;
        config.batch_size = 128; // GPU can handle larger batches
        config.use_mixed_precision = true; // RTX 2070 SUPER supports Tensor Cores
    }

    // AVX2 optimization for AMD
    if (hardware.avx2_support and hardware.cpu_vendor == .amd) {
        config.use_simd_optimization = true;
    }

    return config;
}

pub fn detectHardware(allocator: std.mem.Allocator) !HardwareInfo {
    var info = HardwareInfo{
        .cpu_cores = @intCast(std.Thread.getCpuCount() catch 4),
        .has_nvidia_gpu = false,
        .gpu_compute_capability = null,
        .gpu_memory_gb = null,
        .cpu_vendor = .unknown,
        .avx2_support = false,
        .cuda_available = false,
    };

    // Detect CPU vendor
    info.cpu_vendor = detectCpuVendor();

    // Detect AVX2 support
    info.avx2_support = detectAvx2Support();

        // Detect NVIDIA GPU
    detectNvidiaGpu(allocator, &info);

    return info;
}

fn detectCpuVendor() CpuVendor {
    // Check for AMD by looking at CPU info
    if (builtin.cpu.arch == .x86_64) {
        // Simple heuristic - in a real implementation, we'd use CPUID
        return .amd; // Assuming AMD for this optimization
    }
    return .unknown;
}

fn detectAvx2Support() bool {
    // For x86_64 systems, assume AVX2 is available on modern CPUs
    if (builtin.cpu.arch == .x86_64) {
        return true;
    }
    return false;
}

fn detectNvidiaGpu(allocator: std.mem.Allocator, info: *HardwareInfo) void {
    // Try to run nvidia-smi to detect GPU
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits" },
    }) catch return;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term.Exited == 0) {
        // Parse output for RTX 2070 SUPER
        var lines = std.mem.splitSequence(u8, result.stdout, "\n");
        while (lines.next()) |line| {
            if (std.mem.indexOf(u8, line, "RTX 2070 SUPER") != null or
                std.mem.indexOf(u8, line, "GeForce RTX 2070 SUPER") != null) {
                info.has_nvidia_gpu = true;
                info.gpu_memory_gb = 8; // RTX 2070 SUPER has 8GB
                info.gpu_compute_capability = 7.5; // RTX 2070 SUPER is compute capability 7.5
                info.cuda_available = true;
                break;
            }
        }
    }
}

/// Get optimal configuration for detected hardware
pub fn getOptimalConfig(hardware: HardwareInfo) BackendConfig {
    var config = BackendConfig{
        .num_worker_threads = @min(hardware.cpu_cores, 24),
        .batch_size = 32,
        .memory_pool_size_mb = 4096,
        .prefetch_batches = 4,
    };

    // Optimize for AMD Ryzen 9 3900X (24 threads)
    if (hardware.cpu_cores >= 20) {
        config.num_worker_threads = 20; // Leave 4 threads for system
        config.batch_size = 64; // Larger batches for high core count
        config.prefetch_batches = 8;
        config.memory_pool_size_mb = 8192; // 8GB for high-end system
    }

    // NVIDIA RTX 2070 SUPER optimization
    if (hardware.has_nvidia_gpu and hardware.gpu_memory_gb != null and hardware.gpu_memory_gb.? >= 8) {
        config.use_cuda = true;
        config.batch_size = 128; // GPU can handle larger batches
        config.use_mixed_precision = true; // RTX 2070 SUPER supports Tensor Cores
    }

    // AVX2 optimization for AMD
    if (hardware.avx2_support and hardware.cpu_vendor == .amd) {
        config.use_simd_optimization = true;
    }

    return config;
}
