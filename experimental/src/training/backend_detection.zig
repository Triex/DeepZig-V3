//! Hardware detection and optimization for DeepZig V3 training
//! Optimized for AMD Ryzen 9 3900X + NVIDIA RTX 2070 SUPER

const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const Allocator = std.mem.Allocator;

/// CPU vendor detection for optimization
pub const CpuVendor = enum {
    intel,
    amd,
    apple,
    unknown,
};

/// Comprehensive hardware information with detailed capabilities
pub const HardwareInfo = struct {
    // CPU Information
    cpu_cores: u32,
    cpu_threads: u32,
    cpu_model: []const u8,
    cpu_vendor: CpuVendor,
    avx2_support: bool,
    avx512_support: bool,
    cpu_cache_l3_mb: u32,

    // Memory Information
    total_ram_gb: f32,
    available_ram_gb: f32,
    memory_bandwidth_gbps: f32,
    swap_available_gb: f32,

    // GPU Information
    cuda_available: bool,
    gpu_count: u32,
    gpu_memory_gb: ?f32,
    gpu_compute_capability: ?f32,
    gpu_name: ?[]const u8,
    gpu_memory_bandwidth_gbps: ?f32,
    tensor_cores_available: bool,

    // Storage Information
    primary_storage_type: StorageType,
    available_storage_gb: f32,

    // Performance Characteristics
    estimated_memory_bandwidth: f32,
    estimated_compute_throughput: f32,
    thermal_headroom: ThermalProfile,

    pub const StorageType = enum {
        hdd,
        ssd,
        nvme,
        unknown,
    };

    pub const ThermalProfile = enum {
        conservative,
        balanced,
        aggressive,
        datacenter,
    };

    /// Get recommended performance profile based on hardware capabilities
    pub fn getPerformanceProfile(self: *const HardwareInfo) PerformanceProfile {
        // Datacenter-class hardware
        if (self.total_ram_gb >= 64 and self.gpu_memory_gb != null and self.gpu_memory_gb.? >= 16) {
            return .datacenter;
        }

        // High-end workstation
        if (self.total_ram_gb >= 32 and self.cpu_cores >= 12) {
            return .workstation;
        }

        // Gaming/enthusiast system
        if (self.total_ram_gb >= 16 and self.gpu_memory_gb != null and self.gpu_memory_gb.? >= 8) {
            return .gaming;
        }

        // Budget/laptop system
        return .budget;
    }

    pub const PerformanceProfile = enum {
        budget,
        gaming,
        workstation,
        datacenter,
    };

    /// Clean up allocated memory
    pub fn deinit(self: *HardwareInfo, allocator: std.mem.Allocator) void {
        // Free CPU model string if it was allocated
        if (!std.mem.eql(u8, self.cpu_model, "Unknown")) {
            allocator.free(self.cpu_model);
        }

        // Free GPU name if it was allocated
        if (self.gpu_name) |name| {
            allocator.free(name);
        }
    }
};

/// Intelligent training configuration based on hardware capabilities
pub const OptimalConfig = struct {
    // Model Configuration
    recommended_model_size: ModelSize,
    max_safe_model_size: ModelSize,
    memory_budget_mb: u32,

    // Training Configuration
    optimal_batch_size: u32,
    max_batch_size: u32,
    recommended_micro_batch_size: u32,
    gradient_accumulation_steps: u32,

    // Resource Utilization
    dataloader_workers: u32,
    optimizer_threads: u32,
    enable_mixed_precision: bool,
    enable_gradient_checkpointing: bool,

    // Performance Features
    enable_cuda: bool,
    enable_tensor_cores: bool,
    enable_simd: bool,
    pin_memory: bool,

    // Memory Management
    memory_safety_margin: f32,  // Fraction of memory to keep free
    prefetch_batches: u32,
    cache_datasets: bool,

    // Performance Estimates
    estimated_throughput_samples_per_sec: f32,
    estimated_training_time_hours: f32,
    bottleneck_analysis: BottleneckAnalysis,

    // Recommendations
    performance_recommendations: []const []const u8,

    pub const ModelSize = enum {
        tiny,    // <1M params
        testing, // <5M params (renamed from 'test' which is reserved)
        small,   // <25M params
        medium,  // <100M params
        large,   // <500M params
        xlarge,  // <2B params
        xxlarge, // 2B+ params
    };

    pub const BottleneckAnalysis = struct {
        primary_bottleneck: Bottleneck,
        memory_utilization: f32,
        compute_utilization: f32,
        io_utilization: f32,

        pub const Bottleneck = enum {
            memory_bandwidth,
            compute_bound,
            storage_io,
            cpu_threads,
            gpu_memory,
            network_io,
            balanced,
        };
    };
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

pub fn detectHardware(allocator: Allocator) !HardwareInfo {
    log.info("🔍 Performing comprehensive hardware detection...", .{});

    var info = HardwareInfo{
        .cpu_cores = 1,
        .cpu_threads = 1,
        .cpu_model = "Unknown",
        .cpu_vendor = .unknown,
        .avx2_support = false,
        .avx512_support = false,
        .cpu_cache_l3_mb = 0,
        .total_ram_gb = 4.0,
        .available_ram_gb = 2.0,
        .memory_bandwidth_gbps = 25.0,
        .swap_available_gb = 0.0,
        .cuda_available = false,
        .gpu_count = 0,
        .gpu_memory_gb = null,
        .gpu_compute_capability = null,
        .gpu_name = null,
        .gpu_memory_bandwidth_gbps = null,
        .tensor_cores_available = false,
        .primary_storage_type = .unknown,
        .available_storage_gb = 100.0,
        .estimated_memory_bandwidth = 25.0,
        .estimated_compute_throughput = 100.0,
        .thermal_headroom = .balanced,
    };

    // Detect CPU information
    try detectCPUInfo(&info, allocator);

    // Detect memory information
    try detectMemoryInfo(&info);

    // Detect GPU information
    try detectGPUInfo(&info, allocator);

    // Detect storage information
    try detectStorageInfo(&info);

    // Analyze thermal and performance characteristics
    analyzePerformanceCharacteristics(&info);

    logHardwareDetection(&info);

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

/// Legacy function - use getOptimalConfig instead
pub fn getLegacyConfig(hardware: HardwareInfo) BackendConfig {
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

    return config;
}

/// Detect detailed CPU information including architecture and capabilities
fn detectCPUInfo(info: *HardwareInfo, allocator: Allocator) !void {
    // Get CPU core count
    info.cpu_cores = @intCast(std.Thread.getCpuCount() catch 4);
    info.cpu_threads = info.cpu_cores; // Assume 1:1 for safety, detect HT later

    // Detect SIMD capabilities
    if (builtin.cpu.arch == .x86_64) {
        // Most modern x86_64 CPUs have AVX2
        info.avx2_support = true;

        // Try to detect AVX-512 (more complex detection needed in practice)
        info.avx512_support = false; // Conservative default
    }

    // Try to get CPU model from /proc/cpuinfo on Linux
    if (builtin.os.tag == .linux) {
        detectLinuxCPUInfo(info, allocator) catch {
            log.warn("Could not read /proc/cpuinfo, using defaults", .{});
        };
    }

    // Detect CPU vendor
    info.cpu_vendor = detectCpuVendor();

    // Estimate L3 cache (typical values)
    if (info.cpu_cores >= 16) {
        info.cpu_cache_l3_mb = 32; // High-end desktop/server
    } else if (info.cpu_cores >= 8) {
        info.cpu_cache_l3_mb = 16; // Mid-range desktop
    } else {
        info.cpu_cache_l3_mb = 8;  // Budget/mobile
    }
}

/// Parse Linux /proc/cpuinfo for detailed CPU information
fn detectLinuxCPUInfo(info: *HardwareInfo, allocator: Allocator) !void {
    const file = std.fs.cwd().openFile("/proc/cpuinfo", .{}) catch return;
    defer file.close();

    const content = try file.readToEndAlloc(allocator, 32 * 1024);
    defer allocator.free(content);

    var lines = std.mem.splitSequence(u8, content, "\n");
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "model name")) {
            if (std.mem.indexOf(u8, line, ":")) |colon_pos| {
                const model_name = std.mem.trim(u8, line[colon_pos + 1..], " \t");
                info.cpu_model = try allocator.dupe(u8, model_name);
            }
        } else if (std.mem.startsWith(u8, line, "flags") or std.mem.startsWith(u8, line, "Features")) {
            // Check for specific instruction sets
            if (std.mem.indexOf(u8, line, "avx2") != null) {
                info.avx2_support = true;
            }
            if (std.mem.indexOf(u8, line, "avx512") != null) {
                info.avx512_support = true;
            }
        } else if (std.mem.startsWith(u8, line, "cache size")) {
            // Extract cache size (this is L3 cache on most systems)
            if (std.mem.indexOf(u8, line, ":")) |colon_pos| {
                const cache_str = std.mem.trim(u8, line[colon_pos + 1..], " \t");
                if (std.mem.indexOf(u8, cache_str, "KB")) |kb_pos| {
                    const kb_str = cache_str[0..kb_pos];
                    if (std.fmt.parseInt(u32, kb_str, 10)) |kb| {
                        info.cpu_cache_l3_mb = kb / 1024;
                    } else |_| {}
                }
            }
        }
    }
}

/// Detect system memory information including available memory
fn detectMemoryInfo(info: *HardwareInfo) !void {
    if (builtin.os.tag == .linux) {
        detectLinuxMemoryInfo(info) catch {
            log.warn("Could not read /proc/meminfo, using estimates", .{});
            info.total_ram_gb = 16.0; // Conservative estimate
            info.available_ram_gb = 12.0;
        };
    } else {
        // Fallback estimates for other platforms
        info.total_ram_gb = 16.0;
        info.available_ram_gb = 12.0;
    }

    // Estimate memory bandwidth based on system type
    if (info.total_ram_gb >= 64) {
        info.memory_bandwidth_gbps = 100.0; // Server/workstation
    } else if (info.total_ram_gb >= 32) {
        info.memory_bandwidth_gbps = 50.0;  // High-end desktop
    } else {
        info.memory_bandwidth_gbps = 25.0;  // Standard desktop
    }
}

/// Parse Linux /proc/meminfo for memory details
fn detectLinuxMemoryInfo(info: *HardwareInfo) !void {
    const file = std.fs.cwd().openFile("/proc/meminfo", .{}) catch return;
    defer file.close();

    const content = try file.readToEndAlloc(std.heap.page_allocator, 4096);
    defer std.heap.page_allocator.free(content);

    var lines = std.mem.splitSequence(u8, content, "\n");
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "MemTotal:")) {
            if (parseMemoryLine(line)) |kb| {
                info.total_ram_gb = @as(f32, @floatFromInt(kb)) / (1024.0 * 1024.0);
            }
        } else if (std.mem.startsWith(u8, line, "MemAvailable:")) {
            if (parseMemoryLine(line)) |kb| {
                info.available_ram_gb = @as(f32, @floatFromInt(kb)) / (1024.0 * 1024.0);
            }
        } else if (std.mem.startsWith(u8, line, "SwapTotal:")) {
            if (parseMemoryLine(line)) |kb| {
                info.swap_available_gb = @as(f32, @floatFromInt(kb)) / (1024.0 * 1024.0);
            }
        }
    }
}

/// Parse memory value from /proc/meminfo line
fn parseMemoryLine(line: []const u8) ?u64 {
    var tokens = std.mem.tokenizeAny(u8, line, " \t");
    _ = tokens.next(); // Skip field name
    if (tokens.next()) |value_str| {
        return std.fmt.parseInt(u64, value_str, 10) catch null;
    }
    return null;
}

/// Detect GPU information including CUDA capabilities and VRAM
fn detectGPUInfo(info: *HardwareInfo, allocator: Allocator) !void {
    // Try nvidia-smi first for NVIDIA GPUs
    if (detectNVIDIAGPU(info, allocator)) {
        info.cuda_available = true;
        info.gpu_count = 1; // Simplify to single GPU for now

        // Detect Tensor Core availability based on compute capability
        if (info.gpu_compute_capability) |cc| {
            info.tensor_cores_available = cc >= 7.0; // Volta and newer
        }
    } else |_| {
        info.cuda_available = false;
        info.gpu_count = 0;
    }
}

/// Detect NVIDIA GPU using nvidia-smi
fn detectNVIDIAGPU(info: *HardwareInfo, allocator: Allocator) !void {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{"nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"},
    }) catch return error.NoNVIDIAGPU;

    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term.Exited != 0) return error.NoNVIDIAGPU;

    // Parse nvidia-smi output
    var lines = std.mem.splitSequence(u8, result.stdout, "\n");
    if (lines.next()) |line| {
        var parts = std.mem.splitSequence(u8, line, ",");

        // GPU name
        if (parts.next()) |name| {
            info.gpu_name = try allocator.dupe(u8, std.mem.trim(u8, name, " "));
        }

        // GPU memory in MB
        if (parts.next()) |memory_str| {
            const memory_mb = std.fmt.parseInt(u32, std.mem.trim(u8, memory_str, " "), 10) catch 0;
            info.gpu_memory_gb = @as(f32, @floatFromInt(memory_mb)) / 1024.0;
        }

        // Compute capability
        if (parts.next()) |cc_str| {
            info.gpu_compute_capability = std.fmt.parseFloat(f32, std.mem.trim(u8, cc_str, " ")) catch null;
        }
    }

    // Estimate memory bandwidth based on GPU generation
    if (info.gpu_compute_capability) |cc| {
        if (cc >= 8.0) {
            info.gpu_memory_bandwidth_gbps = 900.0; // RTX 30/40 series
        } else if (cc >= 7.5) {
            info.gpu_memory_bandwidth_gbps = 450.0; // RTX 20 series
        } else {
            info.gpu_memory_bandwidth_gbps = 250.0; // GTX 10 series
        }
    }
}

/// Detect storage type and available space
fn detectStorageInfo(info: *HardwareInfo) !void {
    // Simple heuristic: check if /sys/block exists (Linux)
    if (builtin.os.tag == .linux) {
        detectLinuxStorageInfo(info) catch {
            info.primary_storage_type = .unknown;
            info.available_storage_gb = 100.0; // Conservative estimate
        };
    } else {
        info.primary_storage_type = .ssd; // Modern systems likely have SSD
        info.available_storage_gb = 500.0;
    }
}

/// Detect Linux storage information
fn detectLinuxStorageInfo(info: *HardwareInfo) !void {
    // Check for NVMe (fastest)
    if (std.fs.cwd().access("/sys/block/nvme0n1", .{})) {
        info.primary_storage_type = .nvme;
    } else |_| {
        // Check for SSD vs HDD (more complex detection needed)
        info.primary_storage_type = .ssd; // Assume SSD by default
    }

    // Get available disk space
    const stat = std.fs.cwd().statFile(".") catch return;
    _ = stat; // Use statvfs in real implementation
    info.available_storage_gb = 500.0; // Placeholder
}

/// Analyze performance characteristics and thermal profile
fn analyzePerformanceCharacteristics(info: *HardwareInfo) void {
    // Estimate compute throughput
    info.estimated_compute_throughput = @as(f32, @floatFromInt(info.cpu_cores)) * 100.0;
    if (info.gpu_memory_gb != null) {
        info.estimated_compute_throughput += info.gpu_memory_gb.? * 500.0; // GPU boost
    }

    // Estimate memory bandwidth
    info.estimated_memory_bandwidth = info.memory_bandwidth_gbps;

    // Determine thermal profile
    if (info.cpu_cores >= 24 and info.total_ram_gb >= 64) {
        info.thermal_headroom = .datacenter;
    } else if (info.cpu_cores >= 12 and info.gpu_memory_gb != null and info.gpu_memory_gb.? >= 8) {
        info.thermal_headroom = .aggressive;
    } else if (info.cpu_cores >= 8) {
        info.thermal_headroom = .balanced;
    } else {
        info.thermal_headroom = .conservative;
    }
}

/// Log comprehensive hardware detection results
fn logHardwareDetection(info: *const HardwareInfo) void {
    log.info("=== HARDWARE DETECTION COMPLETE ===", .{});
    log.info("CPU: {s} ({} cores, {} threads)", .{info.cpu_model, info.cpu_cores, info.cpu_threads});
    log.info("    SIMD: AVX2={}, AVX512={}, L3 Cache={}MB", .{info.avx2_support, info.avx512_support, info.cpu_cache_l3_mb});
    log.info("Memory: {d:.1}GB total, {d:.1}GB available ({d:.0}GB/s bandwidth)", .{info.total_ram_gb, info.available_ram_gb, info.memory_bandwidth_gbps});

    if (info.cuda_available and info.gpu_name != null) {
        log.info("GPU: {s} ({d:.1}GB VRAM, CC {d:.1})", .{info.gpu_name.?, info.gpu_memory_gb.?, info.gpu_compute_capability.?});
        log.info("     Tensor Cores: {}, Memory BW: {d:.0}GB/s", .{info.tensor_cores_available, info.gpu_memory_bandwidth_gbps.?});
    } else {
        log.info("GPU: Not detected or not available", .{});
    }

    log.info("Storage: {} ({d:.0}GB available)", .{info.primary_storage_type, info.available_storage_gb});
    log.info("Profile: {} (thermal: {})", .{info.getPerformanceProfile(), info.thermal_headroom});
    log.info("=====================================", .{});
}

/// Calculate optimal training configuration based on hardware capabilities
pub fn getOptimalConfig(hardware: *const HardwareInfo, allocator: Allocator) !OptimalConfig {
    log.info("🧮 Calculating optimal training configuration...", .{});

    var config = OptimalConfig{
        .recommended_model_size = .testing,
        .max_safe_model_size = .small,
        .memory_budget_mb = 1000,
        .optimal_batch_size = 16,
        .max_batch_size = 32,
        .recommended_micro_batch_size = 8,
        .gradient_accumulation_steps = 2,
        .dataloader_workers = 4,
        .optimizer_threads = 4,
        .enable_mixed_precision = true,
        .enable_gradient_checkpointing = true,
        .enable_cuda = hardware.cuda_available,
        .enable_tensor_cores = hardware.tensor_cores_available,
        .enable_simd = hardware.avx2_support,
        .pin_memory = hardware.cuda_available,
        .memory_safety_margin = 0.2, // Keep 20% memory free
        .prefetch_batches = 2,
        .cache_datasets = false,
        .estimated_throughput_samples_per_sec = 10.0,
        .estimated_training_time_hours = 1.0,
        .bottleneck_analysis = .{
            .primary_bottleneck = .balanced,
            .memory_utilization = 0.5,
            .compute_utilization = 0.7,
            .io_utilization = 0.3,
        },
        .performance_recommendations = &[_][]const u8{},
    };

    // Calculate memory budget (use 70% of available RAM for safety)
    const usable_memory_gb = hardware.available_ram_gb * (1.0 - config.memory_safety_margin);
    config.memory_budget_mb = @intFromFloat(usable_memory_gb * 1024.0);

    // Determine optimal model size based on memory budget
    config.recommended_model_size = selectOptimalModelSize(config.memory_budget_mb);
    config.max_safe_model_size = selectMaxSafeModelSize(hardware.total_ram_gb);

    // Calculate optimal batch sizes
    try calculateOptimalBatchSizes(&config, hardware);

    // Configure worker threads
    config.dataloader_workers = @max(2, @min(hardware.cpu_cores - 2, 12)); // Leave 2 cores free
    config.optimizer_threads = @max(2, @min(hardware.cpu_cores / 2, 8));

    // Configure advanced features
    config.enable_mixed_precision = hardware.cuda_available and (hardware.gpu_compute_capability orelse 0.0) >= 7.0;
    config.enable_tensor_cores = config.enable_mixed_precision and hardware.tensor_cores_available;
    config.pin_memory = hardware.cuda_available and hardware.total_ram_gb >= 16;
    config.cache_datasets = hardware.available_storage_gb >= 100 and hardware.primary_storage_type != .hdd;

    // Performance estimates
    try calculatePerformanceEstimates(&config, hardware);

    // Generate recommendations
    config.performance_recommendations = try generatePerformanceRecommendations(&config, hardware, allocator);

    logOptimalConfiguration(&config);

    return config;
}

/// Select optimal model size based on memory budget
fn selectOptimalModelSize(memory_budget_mb: u32) OptimalConfig.ModelSize {
    if (memory_budget_mb >= 8000) {
        return .large;    // ~500M params need ~1GB
    } else if (memory_budget_mb >= 2000) {
        return .medium;   // ~100M params need ~200MB
    } else if (memory_budget_mb >= 500) {
        return .small;    // ~25M params need ~50MB
    } else if (memory_budget_mb >= 100) {
        return .testing;     // ~5M params need ~10MB
    } else {
        return .tiny;     // ~1M params need ~2MB
    }
}

/// Select maximum safe model size
fn selectMaxSafeModelSize(total_ram_gb: f32) OptimalConfig.ModelSize {
    if (total_ram_gb >= 64) {
        return .xxlarge;  // Can handle 2B+ param models
    } else if (total_ram_gb >= 32) {
        return .xlarge;   // Can handle up to 2B param models
    } else if (total_ram_gb >= 16) {
        return .large;    // Can handle up to 500M param models
    } else if (total_ram_gb >= 8) {
        return .medium;   // Can handle up to 100M param models
    } else {
        return .small;    // Stick to smaller models
    }
}

/// Get model memory requirements in MB
fn getModelMemoryMb(model_size: OptimalConfig.ModelSize) u32 {
    return switch (model_size) {
        .tiny => 10,
        .testing => 20,
        .small => 100,
        .medium => 400,
        .large => 1000,
        .xlarge => 3000,
        .xxlarge => 8000,
    };
}

/// Get memory per sample in MB
fn getMemoryPerSample(model_size: OptimalConfig.ModelSize) f32 {
    return switch (model_size) {
        .tiny, .testing => 1.0,
        .small => 2.0,
        .medium => 4.0,
        .large => 8.0,
        .xlarge => 16.0,
        .xxlarge => 32.0,
    };
}

/// Calculate optimal batch sizes for GPU utilization
fn calculateOptimalBatchSizes(config: *OptimalConfig, hardware: *const HardwareInfo) !void {
    // Calculate model memory requirements
    const model_memory_mb = getModelMemoryMb(config.recommended_model_size);

    if (hardware.gpu_memory_gb) |vram_gb| {
        // VRAM-based batch size calculation
        const vram_mb = @as(u32, @intFromFloat(vram_gb * 1024.0));

        // Use 60% of VRAM for model + activations, rest for batches
        const available_for_batches = @as(f32, @floatFromInt(vram_mb - model_memory_mb)) * 0.6;
        const mb_per_sample = getMemoryPerSample(config.recommended_model_size);

        config.max_batch_size = @intFromFloat(@max(1.0, available_for_batches / mb_per_sample));
        config.optimal_batch_size = @max(8, @min(config.max_batch_size, 128)); // Reasonable range

        // Tensor Core optimization (multiples of 8)
        if (hardware.tensor_cores_available) {
            config.optimal_batch_size = (config.optimal_batch_size / 8) * 8;
        }
    } else {
        // CPU-only batch size (based on RAM)
        config.optimal_batch_size = @max(4, @min(32, hardware.cpu_cores * 2));
        config.max_batch_size = config.optimal_batch_size * 2;
    }

    config.recommended_micro_batch_size = @max(1, config.optimal_batch_size / 4);
    config.gradient_accumulation_steps = config.optimal_batch_size / config.recommended_micro_batch_size;
}

/// Calculate performance estimates and bottleneck analysis
fn calculatePerformanceEstimates(config: *OptimalConfig, hardware: *const HardwareInfo) !void {
    // Estimate throughput based on hardware
    var base_throughput: f32 = @as(f32, @floatFromInt(hardware.cpu_cores)) * 2.0; // CPU baseline

    if (hardware.cuda_available and hardware.gpu_memory_gb != null) {
        // GPU acceleration boost
        const gpu_factor: f32 = if (hardware.tensor_cores_available) 50.0 else 20.0;
        base_throughput += hardware.gpu_memory_gb.? * gpu_factor;
    }

    // Apply batch size scaling
    const batch_factor = @as(f32, @floatFromInt(config.optimal_batch_size)) / 16.0;
    config.estimated_throughput_samples_per_sec = base_throughput * batch_factor;

    // Estimate training time (rough approximation)
    const samples_to_train = 100000; // Typical training set size
    const total_samples = samples_to_train * 3; // 3 epochs
    config.estimated_training_time_hours = @as(f32, @floatFromInt(total_samples)) / (config.estimated_throughput_samples_per_sec * 3600.0);

    // Bottleneck analysis
    analyzeBottlenecks(config, hardware);
}

/// Analyze system bottlenecks
fn analyzeBottlenecks(config: *OptimalConfig, hardware: *const HardwareInfo) void {
    var bottleneck = OptimalConfig.BottleneckAnalysis{
        .primary_bottleneck = .balanced,
        .memory_utilization = 0.6,
        .compute_utilization = 0.7,
        .io_utilization = 0.3,
    };

    // Memory utilization
    const memory_needed_gb = @as(f32, @floatFromInt(config.memory_budget_mb)) / 1024.0;
    bottleneck.memory_utilization = memory_needed_gb / hardware.available_ram_gb;

    // Determine primary bottleneck
    if (bottleneck.memory_utilization > 0.8) {
        bottleneck.primary_bottleneck = .memory_bandwidth;
    } else if (!hardware.cuda_available) {
        bottleneck.primary_bottleneck = .cpu_threads;
    } else if (hardware.gpu_memory_gb != null and hardware.gpu_memory_gb.? < 8.0) {
        bottleneck.primary_bottleneck = .gpu_memory;
    } else if (hardware.primary_storage_type == .hdd) {
        bottleneck.primary_bottleneck = .storage_io;
    } else {
        bottleneck.primary_bottleneck = .balanced;
    }

    config.bottleneck_analysis = bottleneck;
}

/// Generate performance recommendations
fn generatePerformanceRecommendations(config: *const OptimalConfig, hardware: *const HardwareInfo, allocator: Allocator) ![]const[]const u8 {
    var recommendations = std.ArrayList([]const u8).init(allocator);

    // Model size recommendations
    if (config.max_safe_model_size != config.recommended_model_size) {
        const rec = try std.fmt.allocPrint(allocator,
            "💡 Your system can handle up to '{}' model size (currently using '{}')",
            .{config.max_safe_model_size, config.recommended_model_size});
        try recommendations.append(rec);
    }

    // Batch size recommendations
    if (config.max_batch_size > @as(u32, @intFromFloat(@as(f32, @floatFromInt(config.optimal_batch_size)) * 1.5))) {
        const rec = try std.fmt.allocPrint(allocator,
            "⚡ Increase batch size to {} for better GPU utilization (currently {})",
            .{@min(config.max_batch_size, config.optimal_batch_size * 2), config.optimal_batch_size});
        try recommendations.append(rec);
    }

    // Hardware feature recommendations
    if (hardware.tensor_cores_available and !config.enable_tensor_cores) {
        try recommendations.append(try allocator.dupe(u8, "🚀 Enable Tensor Cores for 2x training speedup"));
    }

    if (hardware.avx2_support and !config.enable_simd) {
        try recommendations.append(try allocator.dupe(u8, "🔧 Enable AVX2 SIMD for CPU optimization"));
    }

    // Memory recommendations
    if (config.bottleneck_analysis.memory_utilization > 0.9) {
        try recommendations.append(try allocator.dupe(u8, "⚠️ High memory usage - consider smaller model or batch size"));
    }

    // Performance recommendations
    if (config.estimated_throughput_samples_per_sec < 10.0) {
        try recommendations.append(try allocator.dupe(u8, "🐌 Low throughput detected - check GPU utilization and batch size"));
    }

    return recommendations.toOwnedSlice();
}

/// Log optimal configuration results
fn logOptimalConfiguration(config: *const OptimalConfig) void {
    log.info("=== OPTIMAL TRAINING CONFIGURATION ===", .{});
    log.info("Model: {} (max safe: {}, budget: {}MB)", .{config.recommended_model_size, config.max_safe_model_size, config.memory_budget_mb});
    log.info("Batch: {} optimal, {} max (micro: {}, accum: {})", .{config.optimal_batch_size, config.max_batch_size, config.recommended_micro_batch_size, config.gradient_accumulation_steps});
    log.info("Workers: {} dataloader, {} optimizer", .{config.dataloader_workers, config.optimizer_threads});
    log.info("Features: CUDA={}, Precision={}, TensorCores={}, SIMD={}", .{config.enable_cuda, config.enable_mixed_precision, config.enable_tensor_cores, config.enable_simd});
    log.info("Performance: {d:.1} samples/sec, {d:.1}h training time", .{config.estimated_throughput_samples_per_sec, config.estimated_training_time_hours});
    log.info("Bottleneck: {} (mem: {d:.0}%, compute: {d:.0}%)", .{config.bottleneck_analysis.primary_bottleneck, config.bottleneck_analysis.memory_utilization * 100, config.bottleneck_analysis.compute_utilization * 100});

    log.info("Recommendations:", .{});
    for (config.performance_recommendations) |rec| {
        log.info("  {s}", .{rec});
    }
    log.info("=====================================", .{});
}

/// Backwards compatibility type aliases
pub const BackendConfig = OptimalConfig;
