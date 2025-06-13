// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

// High-Performance BLAS Integration for DeepZig V3
// Automatically detects and uses the fastest BLAS implementation per platform
//
// Performance targets:
// - Apple Silicon (M1/M2/M3/M4): Accelerate.framework (~2600 GFLOPS)
// - Intel/AMD x86_64: Intel MKL or OpenBLAS (~1000+ GFLOPS)
// - ARM64 Linux: OpenBLAS with NEON (~500+ GFLOPS)
// - Fallback: Naive implementation (~10 GFLOPS)

const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.Random;
const builtin = @import("builtin");

// Import our comprehensive hardware detection
const hardware = @import("hardware.zig");
const HardwareInfo = hardware.HardwareInfo;
const BlasConfig = hardware.BlasConfig;
const BlasBackend = hardware.BlasBackend;

// CUDA Backend Implementation
const CudaBackend = struct {
    allocator: Allocator,
    device_id: i32,
    context: ?*anyopaque,

    const Self = @This();

    // Use the main MatrixLayout and Transpose types
    pub const CudaMatrixLayout = MatrixLayout;
    pub const CudaTranspose = Transpose;

    pub fn init(allocator: Allocator) !Self {
        // Try to initialize CUDA
        if (cuInit(0) != 0) {
            return error.CudaNotAvailable;
        }

        var device_count: i32 = 0;
        if (cuDeviceGetCount(&device_count) != 0 or device_count == 0) {
            return error.CudaNotAvailable;
        }

        var device: i32 = 0;
        if (cuDeviceGet(&device, 0) != 0) {
            return error.CudaNotAvailable;
        }

        var context: ?*anyopaque = null;
        if (cuCtxCreate(&context, 0, device) != 0) {
            return error.CudaNotAvailable;
        }

        return Self{
            .allocator = allocator,
            .device_id = device,
            .context = context,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.context) |ctx| {
            _ = cuCtxDestroy(ctx);
        }
    }

    pub fn sgemm(
        self: *const Self,
        layout: MatrixLayout,
        transa: Transpose,
        transb: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: CudaBuffer,
        lda: usize,
        b: CudaBuffer,
        ldb: usize,
        beta: f32,
        c: CudaBuffer,
        ldc: usize,
    ) !void {
        _ = self;
        _ = layout;
        _ = transa;
        _ = transb;
        _ = m;
        _ = n;
        _ = k;
        _ = alpha;
        _ = a;
        _ = lda;
        _ = b;
        _ = ldb;
        _ = beta;
        _ = c;
        _ = ldc;
        // For now, return error to trigger fallback
        return error.CudaNotImplemented;
    }

    pub fn synchronize(self: *const Self) !void {
        _ = self;
        if (cuCtxSynchronize() != 0) {
            return error.CudaSyncFailed;
        }
    }
};

const CudaBuffer = struct {
    ptr: ?*anyopaque,
    size: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, size: usize) !CudaBuffer {
        var ptr: ?*anyopaque = null;
        if (cuMemAlloc(&ptr, size) != 0) {
            return error.CudaMemoryAllocationFailed;
        }
        return CudaBuffer{
            .ptr = ptr,
            .size = size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CudaBuffer) void {
        if (self.ptr) |ptr| {
            _ = cuMemFree(ptr);
        }
    }

    pub fn copyFromHost(self: *const CudaBuffer, comptime T: type, data: []const T) !void {
        const byte_size = data.len * @sizeOf(T);
        if (byte_size > self.size) return error.BufferTooSmall;
        if (cuMemcpyHtoD(self.ptr, data.ptr, byte_size) != 0) {
            return error.CudaCopyFailed;
        }
    }

    pub fn copyToHost(self: *const CudaBuffer, comptime T: type, data: []T) !void {
        const byte_size = data.len * @sizeOf(T);
        if (byte_size > self.size) return error.BufferTooSmall;
        if (cuMemcpyDtoH(data.ptr, self.ptr, byte_size) != 0) {
            return error.CudaCopyFailed;
        }
    }
};

// CUDA Driver API - CPU-only stub implementations for seamless fallback
// Force CPU-only mode for maximum compatibility
const cuda_disabled = true;

// Stub implementations that return error codes (non-zero) to trigger CPU fallback
fn cuInit(flags: u32) i32 {
    _ = flags;
    return 1;
}
fn cuDeviceGetCount(count: *i32) i32 {
    count.* = 0;
    return 1;
}
fn cuDeviceGet(device: *i32, ordinal: i32) i32 {
    _ = device;
    _ = ordinal;
    return 1;
}
fn cuCtxCreate(ctx: *?*anyopaque, flags: u32, device: i32) i32 {
    _ = ctx;
    _ = flags;
    _ = device;
    return 1;
}
fn cuCtxDestroy(ctx: *anyopaque) i32 {
    _ = ctx;
    return 1;
}
fn cuCtxSynchronize() i32 {
    return 1;
}
fn cuMemAlloc(ptr: *?*anyopaque, size: usize) i32 {
    _ = ptr;
    _ = size;
    return 1;
}
fn cuMemFree(ptr: *anyopaque) i32 {
    _ = ptr;
    return 1;
}
fn cuMemcpyHtoD(dst: ?*anyopaque, src: *const anyopaque, size: usize) i32 {
    _ = dst;
    _ = src;
    _ = size;
    return 1;
}
fn cuMemcpyDtoH(dst: *anyopaque, src: ?*anyopaque, size: usize) i32 {
    _ = dst;
    _ = src;
    _ = size;
    return 1;
}

/// Re-export MatrixDims for public use
pub const MatrixDims = hardware.MatrixDims;

/// Conditional logging that's disabled in release mode for optimal performance
inline fn logInfo(comptime fmt: []const u8, args: anytype) void {
    // Always show essential benchmark results
    std.log.info(fmt, args);
}

inline fn logDebug(comptime fmt: []const u8, args: anytype) void {
    if (builtin.mode == .Debug) {
        std.log.debug(fmt, args);
    }
}

/// Simple Apple Silicon detection for BLAS optimization
fn isAppleSilicon() bool {
    return builtin.os.tag == .macos and builtin.target.cpu.arch == .aarch64;
}

/// Enhanced CPU detection for better performance estimates
const CpuInfo = struct {
    cores: u32,
    threads: u32,
    has_avx2: bool,
    has_avx512: bool,
    has_fma: bool,
    vendor: CpuVendor,

    const CpuVendor = enum {
        intel,
        amd,
        apple,
        unknown,
    };

    fn detect() CpuInfo {
        const cpu_count = std.Thread.getCpuCount() catch 1;

        // Detect CPU vendor and features
        const vendor = switch (builtin.cpu.arch) {
            .aarch64 => if (builtin.os.tag == .macos) CpuVendor.apple else CpuVendor.unknown,
            .x86_64 => detectX86Vendor(),
            else => CpuVendor.unknown,
        };

        return CpuInfo{
            .cores = @intCast(@max(1, cpu_count / 2)), // Assume hyperthreading
            .threads = @intCast(cpu_count),
            .has_avx2 = std.Target.x86.featureSetHas(builtin.cpu.features, .avx2),
            .has_avx512 = std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f),
            .has_fma = std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
            .vendor = vendor,
        };
    }

    fn detectX86Vendor() CpuVendor {
        // Simple heuristic based on CPU model string patterns
        // In a real implementation, you'd use CPUID instruction
        return switch (builtin.cpu.model) {
            .generic => CpuVendor.unknown,
            else => blk: {
                // Check common AMD models
                const model_name = @tagName(builtin.cpu.model);
                if (std.mem.indexOf(u8, model_name, "zen") != null or
                    std.mem.indexOf(u8, model_name, "ryzen") != null)
                {
                    break :blk CpuVendor.amd;
                }
                // Default to Intel for x86_64
                break :blk CpuVendor.intel;
            },
        };
    }
};

pub const BlasPerformanceInfo = struct {
    peak_gflops: f32,
    memory_bandwidth_gb_s: f32,
    supports_mixed_precision: bool,
    simd_width: u32,
    num_threads: u32,
};

/// Memory layout for matrices
pub const MatrixLayout = enum {
    row_major, // C-style (row by row)
    column_major, // Fortran-style (column by column)
};

/// Transpose operations
pub const Transpose = enum {
    no_trans,
    trans,
    conj_trans, // For complex numbers

    fn toCblas(self: Transpose) c_int {
        return switch (self) {
            .no_trans => 111, // CblasNoTrans
            .trans => 112, // CblasTrans
            .conj_trans => 113, // CblasConjTrans
        };
    }
};

// Platform-specific FFI declarations
const blas_c = switch (builtin.os.tag) {
    .macos => struct {
        // macOS Accelerate.framework
        extern "c" fn cblas_sgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: [*]const f32,
            lda: c_int,
            b: [*]const f32,
            ldb: c_int,
            beta: f32,
            result: [*]f32,
            ldc: c_int,
        ) void;

        extern "c" fn cblas_dgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f64,
            a: [*]const f64,
            lda: c_int,
            b: [*]const f64,
            ldb: c_int,
            beta: f64,
            result: [*]f64,
            ldc: c_int,
        ) void;
    },
    else => struct {
        // OpenBLAS or Intel MKL (same CBLAS interface)
        extern "c" fn cblas_sgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: [*]const f32,
            lda: c_int,
            b: [*]const f32,
            ldb: c_int,
            beta: f32,
            result: [*]f32,
            ldc: c_int,
        ) void;

        extern "c" fn cblas_dgemm(
            order: c_int,
            transa: c_int,
            transb: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f64,
            a: [*]const f64,
            lda: c_int,
            b: [*]const f64,
            ldb: c_int,
            beta: f64,
            result: [*]f64,
            ldc: c_int,
        ) void;

        // OpenBLAS threading control
        extern "c" fn openblas_set_num_threads(num_threads: c_int) void;
        extern "c" fn openblas_get_num_threads() c_int;
    },
};

/// Setup optimal threading for BLAS libraries using hardware detection
fn setupBlasThreading(blas_config: BlasConfig) void {
    switch (blas_config.backend) {
        .cuda => {
            // CUDA/cuBLAS setup
            logInfo("🎮 CUDA/cuBLAS detected - GPU acceleration enabled", .{});
            logInfo("🎮 Expected GPU performance: {d:.0} GFLOPS", .{blas_config.estimated_gflops});
        },
        .openblas => {
            if (builtin.os.tag != .macos) {
                // Set OpenBLAS to use optimal number of threads
                blas_c.openblas_set_num_threads(@intCast(blas_config.num_threads));
                logInfo("🧵 OpenBLAS configured to use {} threads", .{blas_config.num_threads});

                if (blas_config.blas_coretype) |coretype| {
                    logInfo("🧵 Recommended: OPENBLAS_CORETYPE={s}", .{coretype});
                }
            }
        },
        .accelerate => {
            // Accelerate framework automatically uses all cores
            logInfo("🧵 Accelerate framework will use all available cores automatically", .{});
        },
        .intel_mkl => {
            // MKL threading is typically controlled via environment variables
            logInfo("🧵 Intel MKL threading controlled via MKL_NUM_THREADS environment variable", .{});
            logInfo("🧵 Recommended: MKL_NUM_THREADS={}", .{blas_config.num_threads});
        },
        .naive => {
            // Naive implementation is single-threaded
        },
    }
}

/// Convert BlasConfig to BlasPerformanceInfo for compatibility
fn configToPerformanceInfo(config: BlasConfig) BlasPerformanceInfo {
    return BlasPerformanceInfo{
        .peak_gflops = config.estimated_gflops,
        .memory_bandwidth_gb_s = config.memory_bandwidth_gb_s,
        .supports_mixed_precision = config.backend != .naive,
        .simd_width = config.simd_width,
        .num_threads = config.num_threads,
    };
}

/// High-level BLAS interface - automatically chooses optimal implementation
pub const Blas = struct {
    backend: BlasBackend,
    performance_info: BlasPerformanceInfo,
    allocator: Allocator,
    hardware_info: ?*HardwareInfo = null,
    cuda_backend: ?CudaBackend = null,

    /// Global singleton BLAS instance
    var global_instance: ?Blas = null;
    var global_hw_info: ?*HardwareInfo = null;
    var global_allocator: ?Allocator = null;
    var init_mutex: std.Thread.Mutex = .{};

    /// Get or create the global BLAS instance (thread-safe) with hardware detection
    pub fn global(allocator: Allocator) !*const Blas {
        init_mutex.lock();
        defer init_mutex.unlock();

        if (global_instance == null) {
            // Detect hardware and get optimal configuration
            var hw_info = hardware.detectHardware(allocator) catch |err| {
                logInfo("⚠️ Hardware detection failed: {}, using defaults", .{err});
                global_instance = try initWithDefaults(allocator);
                global_allocator = allocator;
                return &global_instance.?;
            };

            // Allocate hardware info on heap for global persistence
            const hw_info_ptr = try allocator.create(HardwareInfo);
            hw_info_ptr.* = hw_info;
            global_hw_info = hw_info_ptr;
            global_allocator = allocator;

            const blas_config = hw_info.getOptimalBlasConfig();

            // Initialize CUDA if selected - with better error handling
            var cuda_backend_instance: ?CudaBackend = null;
            var final_backend = blas_config.backend;
            var final_gflops = blas_config.estimated_gflops;

            if (blas_config.backend == .cuda) {
                if (CudaBackend.init(allocator)) |cuda_instance| {
                    cuda_backend_instance = cuda_instance;
                    logInfo("🎮 CUDA backend successfully initialized!", .{});
                    final_gflops = blas_config.estimated_gflops; // Use GPU estimate
                } else |err| {
                    logInfo("⚠️ CUDA backend initialization failed: {}", .{err});
                    logInfo("🔧 Falling back to OpenBLAS with optimized CPU settings", .{});
                    // Fall back to OpenBLAS but keep high performance estimate
                    final_backend = .openblas;
                    final_gflops = 3500.0; // High-performance CPU estimate for 24-core Ryzen
                }
            }

            // Setup optimal threading with final backend
            const final_config = BlasConfig{
                .backend = final_backend,
                .num_threads = blas_config.num_threads,
                .estimated_gflops = final_gflops,
                .memory_bandwidth_gb_s = blas_config.memory_bandwidth_gb_s,
                .simd_width = blas_config.simd_width,
                .cache_line_size = blas_config.cache_line_size,
                .blas_coretype = blas_config.blas_coretype,
            };
            setupBlasThreading(final_config);

            // Only log once for the global instance
            logInfo("🚀 BLAS initialized: {} backend, {d:.1} GFLOPS estimated", .{ final_backend, final_gflops });

            global_instance = Blas{
                .backend = final_backend,
                .performance_info = configToPerformanceInfo(final_config),
                .allocator = allocator,
                .hardware_info = hw_info_ptr,
                .cuda_backend = cuda_backend_instance,
            };
        }

        return &global_instance.?;
    }

    /// Clean up global BLAS resources - call this at program shutdown
    pub fn globalCleanup() void {
        init_mutex.lock();
        defer init_mutex.unlock();

        if (global_instance) |*instance| {
            if (instance.cuda_backend) |*cuda_ctx| {
                cuda_ctx.deinit();
            }
        }

        if (global_hw_info) |hw_info| {
            if (global_allocator) |allocator| {
                hw_info.deinit();
                allocator.destroy(hw_info);
                global_hw_info = null;
                global_allocator = null;
            }
        }

        global_instance = null;
    }

    /// Initialize BLAS with hardware detection (deprecated - use global() instead)
    pub fn init(allocator: Allocator) !Blas {
        // Detect hardware and get optimal configuration
        var hw_info = hardware.detectHardware(allocator) catch |err| {
            logInfo("⚠️ Hardware detection failed: {}, using defaults", .{err});
            return try initWithDefaults(allocator);
        };
        // Note: hw_info will be cleaned up by caller or when Blas.deinit() is called

        const blas_config = hw_info.getOptimalBlasConfig();

        // Initialize CUDA if selected
        var cuda_backend_instance: ?CudaBackend = null;
        if (blas_config.backend == .cuda) {
            cuda_backend_instance = CudaBackend.init(allocator) catch |err| {
                logInfo("⚠️ CUDA initialization failed: {}, falling back to OpenBLAS", .{err});
                // Create fallback configuration
                const fallback_hw_info = try allocator.create(HardwareInfo);
                fallback_hw_info.* = hw_info;
                return Blas{
                    .backend = .openblas,
                    .performance_info = BlasPerformanceInfo{
                        .peak_gflops = 1000.0,
                        .memory_bandwidth_gb_s = blas_config.memory_bandwidth_gb_s,
                        .supports_mixed_precision = true,
                        .simd_width = blas_config.simd_width,
                        .num_threads = blas_config.num_threads,
                    },
                    .allocator = allocator,
                    .hardware_info = fallback_hw_info,
                    .cuda_backend = null,
                };
            };
        }

        // Setup optimal threading
        setupBlasThreading(blas_config);

        // Minimal logging for debugging
        logDebug("BLAS initialized with {} backend", .{blas_config.backend});

        // Allocate hardware info on heap so it persists
        const hw_info_ptr = try allocator.create(HardwareInfo);
        hw_info_ptr.* = hw_info;

        return Blas{
            .backend = blas_config.backend,
            .performance_info = configToPerformanceInfo(blas_config),
            .allocator = allocator,
            .hardware_info = hw_info_ptr,
            .cuda_backend = cuda_backend_instance,
        };
    }

    /// Clean up BLAS resources
    pub fn deinit(self: *Blas) void {
        if (self.cuda_backend) |*cuda_ctx| {
            cuda_ctx.deinit();
        }
        if (self.hardware_info) |hw_info| {
            hw_info.deinit();
            self.allocator.destroy(hw_info);
            self.hardware_info = null;
        }
    }

    /// Fallback initialization when hardware detection fails
    fn initWithDefaults(allocator: Allocator) !Blas {
        const backend = switch (builtin.os.tag) {
            .macos => BlasBackend.accelerate,
            .linux, .windows => BlasBackend.openblas,
            else => BlasBackend.naive,
        };

        const performance_info = BlasPerformanceInfo{
            .peak_gflops = 800.0, // Conservative estimate
            .memory_bandwidth_gb_s = 80.0,
            .supports_mixed_precision = backend != .naive,
            .simd_width = 256,
            .num_threads = @intCast(std.Thread.getCpuCount() catch 1),
        };

        return Blas{
            .backend = backend,
            .performance_info = performance_info,
            .allocator = allocator,
            .hardware_info = null,
            .cuda_backend = null,
        };
    }

    /// Single-precision matrix multiplication: C = alpha * A * B + beta * C
    pub fn sgemm(
        self: *const Blas,
        layout: MatrixLayout,
        transa: Transpose,
        transb: Transpose,
        dims: MatrixDims,
        alpha: f32,
        a: []const f32,
        b: []const f32,
        beta: f32,
        result: []f32,
    ) void {
        switch (self.backend) {
            .cuda => {
                if (self.cuda_backend) |*cuda_ctx| {
                    self.cudasgemm(cuda_ctx, layout, transa, transb, dims, alpha, a, b, beta, result) catch |err| {
                        logInfo("🎮 CUDA SGEMM failed: {}, falling back to naive", .{err});
                        naiveSgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
                    };
                } else {
                    logInfo("🎮 CUDA backend not initialized, falling back to naive", .{});
                    naiveSgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
                }
            },
            .accelerate, .intel_mkl, .openblas => {
                // Validate input parameters to prevent segfaults
                if (dims.m == 0 or dims.n == 0 or dims.k == 0) {
                    logDebug("⚠️ Invalid matrix dimensions: {}x{}x{}, using naive fallback", .{ dims.m, dims.n, dims.k });
                    naiveSgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
                    return;
                }

                // Check for null pointers
                if (a.len == 0 or b.len == 0 or result.len == 0) {
                    logDebug("⚠️ Empty arrays passed to BLAS, using naive fallback", .{});
                    naiveSgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
                    return;
                }

                // Validate array sizes
                const expected_a_size = dims.m * dims.k;
                const expected_b_size = dims.k * dims.n;
                const expected_c_size = dims.m * dims.n;

                if (a.len < expected_a_size or b.len < expected_b_size or result.len < expected_c_size) {
                    logDebug("⚠️ Array size mismatch: a={} (need {}), b={} (need {}), c={} (need {}), using naive fallback", .{ a.len, expected_a_size, b.len, expected_b_size, result.len, expected_c_size });
                    naiveSgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
                    return;
                }

                // Proceed with validated BLAS call
                const order: c_int = if (layout == .row_major) 101 else 102; // CblasRowMajor : CblasColMajor

                // Calculate leading dimensions based on matrix layout and transpose
                const lda = if (layout == .row_major) blk: {
                    if (transa == .no_trans) break :blk @as(c_int, @intCast(dims.k));
                    break :blk @as(c_int, @intCast(dims.m));
                } else blk: {
                    if (transa == .no_trans) break :blk @as(c_int, @intCast(dims.m));
                    break :blk @as(c_int, @intCast(dims.k));
                };

                const ldb = if (layout == .row_major) blk: {
                    if (transb == .no_trans) break :blk @as(c_int, @intCast(dims.n));
                    break :blk @as(c_int, @intCast(dims.k)); // For transpose: use k instead of n
                } else blk: {
                    if (transb == .no_trans) break :blk @as(c_int, @intCast(dims.k));
                    break :blk @as(c_int, @intCast(dims.n));
                };

                const ldc = if (layout == .row_major) @as(c_int, @intCast(dims.n)) else @as(c_int, @intCast(dims.m));

                // Silent BLAS operations - only log on actual errors
                blas_c.cblas_sgemm(
                    order,
                    transa.toCblas(),
                    transb.toCblas(),
                    @intCast(dims.m),
                    @intCast(dims.n),
                    @intCast(dims.k),
                    alpha,
                    a.ptr,
                    lda,
                    b.ptr,
                    ldb,
                    beta,
                    result.ptr,
                    ldc,
                );
            },
            .naive => {
                naiveSgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
            },
        }
    }

    /// CUDA SGEMM implementation
    fn cudasgemm(
        self: *const Blas,
        cuda_ctx: *const CudaBackend,
        layout: MatrixLayout,
        transa: Transpose,
        transb: Transpose,
        dims: MatrixDims,
        alpha: f32,
        a: []const f32,
        b: []const f32,
        beta: f32,
        result: []f32,
    ) !void {
        // Calculate matrix sizes
        const size_a = dims.m * dims.k;
        const size_b = dims.k * dims.n;
        const size_c = dims.m * dims.n;

        // Allocate GPU memory
        var gpu_a = try CudaBuffer.init(self.allocator, size_a * @sizeOf(f32));
        var gpu_b = try CudaBuffer.init(self.allocator, size_b * @sizeOf(f32));
        var gpu_c = try CudaBuffer.init(self.allocator, size_c * @sizeOf(f32));
        defer gpu_a.deinit();
        defer gpu_b.deinit();
        defer gpu_c.deinit();

        // Copy data to GPU
        try gpu_a.copyFromHost(f32, a);
        try gpu_b.copyFromHost(f32, b);
        try gpu_c.copyFromHost(f32, result); // For beta != 0 case

        // Convert layout and transpose
        const cuda_layout: CudaBackend.CudaMatrixLayout = switch (layout) {
            .row_major => .row_major,
            .column_major => .column_major,
        };

        const cuda_transa: CudaBackend.CudaTranspose = switch (transa) {
            .no_trans => .no_trans,
            .trans => .trans,
            .conj_trans => .conj_trans,
        };

        const cuda_transb: CudaBackend.CudaTranspose = switch (transb) {
            .no_trans => .no_trans,
            .trans => .trans,
            .conj_trans => .conj_trans,
        };

        // Perform matrix multiplication on GPU
        try cuda_ctx.sgemm(
            cuda_layout,
            cuda_transa,
            cuda_transb,
            dims.m,
            dims.n,
            dims.k,
            alpha,
            gpu_a,
            dims.k,
            gpu_b,
            dims.n,
            beta,
            gpu_c,
            dims.n,
        );

        // Synchronize and copy result back
        try cuda_ctx.synchronize();
        try gpu_c.copyToHost(f32, result);
    }

    /// Double-precision matrix multiplication: C = alpha * A * B + beta * C
    pub fn dgemm(
        self: *const Blas,
        layout: MatrixLayout,
        transa: Transpose,
        transb: Transpose,
        dims: MatrixDims,
        alpha: f64,
        a: []const f64,
        b: []const f64,
        beta: f64,
        result: []f64,
    ) void {
        switch (self.backend) {
            .accelerate, .intel_mkl, .openblas => {
                const order: c_int = if (layout == .row_major) 101 else 102;
                const lda = if (layout == .row_major) @as(c_int, @intCast(dims.k)) else @as(c_int, @intCast(dims.m));
                const ldb = if (layout == .row_major) @as(c_int, @intCast(dims.n)) else @as(c_int, @intCast(dims.k));
                const ldc = if (layout == .row_major) @as(c_int, @intCast(dims.n)) else @as(c_int, @intCast(dims.m));

                blas_c.cblas_dgemm(
                    order,
                    transa.toCblas(),
                    transb.toCblas(),
                    @intCast(dims.m),
                    @intCast(dims.n),
                    @intCast(dims.k),
                    alpha,
                    a.ptr,
                    lda,
                    b.ptr,
                    ldb,
                    beta,
                    result.ptr,
                    ldc,
                );
            },
            .cuda => {
                // TODO: Implement cuBLAS dgemm - for now fall back to naive
                naiveDgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
            },
            .naive => {
                naiveDgemm(layout, transa, transb, dims, alpha, a, b, beta, result);
            },
        }
    }

    /// Generic matrix multiplication (chooses sgemm or dgemm based on type)
    pub fn matmul(self: *const Blas, comptime T: type, a: []const T, b: []const T, result: []T, dims: MatrixDims) void {
        switch (T) {
            f32 => self.sgemm(.row_major, .no_trans, .no_trans, dims, 1.0, a, b, 0.0, result),
            f64 => self.dgemm(.row_major, .no_trans, .no_trans, dims, 1.0, a, b, 0.0, result),
            else => @compileError("BLAS matmul only supports f32 and f64"),
        }
    }

    /// Synchronize GPU device operations (ensures all GPU work is complete)
    /// This forces the GPU to finish all pending operations, making GPU fans spin up
    pub fn synchronizeDevice(self: *const Blas) !void {
        switch (self.backend) {
            .cuda => {
                if (self.cuda_backend) |*cuda_ctx| {
                    try cuda_ctx.synchronize();
                } else {
                    // No-op if CUDA backend not initialized
                }
            },
            else => {
                // No-op for CPU backends - they are synchronous by default
            },
        }
    }

    /// Get hardware information if available
    pub fn getHardwareInfo(self: *const Blas) ?*const HardwareInfo {
        return self.hardware_info;
    }
};

// Naive BLAS implementations for fallback - with proper bounds checking
fn naiveSgemm(
    layout: MatrixLayout,
    transa: Transpose,
    transb: Transpose,
    dims: MatrixDims,
    alpha: f32,
    a: []const f32,
    b: []const f32,
    beta: f32,
    result: []f32,
) void {
    _ = layout; // TODO: Handle row/column major properly
    _ = transa;
    _ = transb; // TODO: Handle transposes properly

    // Handle edge cases
    const m = dims.m;
    const n = dims.n;
    const k = dims.k;

    if (m == 0 or n == 0 or k == 0) {
        logDebug("🔧 Naive SGEMM: Zero dimension detected, zeroing result", .{});
        @memset(result, 0.0);
        return;
    }

    const expected_result_size = m * n;
    if (result.len < expected_result_size) {
        logDebug("🔧 Naive SGEMM: Result array too small ({} < {}), skipping", .{ result.len, expected_result_size });
        return;
    }

    const expected_a_size = m * k;
    const expected_b_size = k * n;
    if (a.len < expected_a_size or b.len < expected_b_size) {
        logDebug("🔧 Naive SGEMM: Input arrays too small (a:{}<{}, b:{}<{}), zeroing result", .{ a.len, expected_a_size, b.len, expected_b_size });
        @memset(result[0..expected_result_size], 0.0);
        return;
    }

    logDebug("🔧 Naive SGEMM: Computing {}x{}x{} matrix multiplication", .{ m, n, k });

    // Scale existing C by beta
    for (result[0..expected_result_size]) |*val| {
        val.* *= beta;
    }

    // Add alpha * A * B (simple row-major implementation)
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0.0;
            for (0..k) |l| {
                const a_idx = i * k + l;
                const b_idx = l * n + j;
                if (a_idx < a.len and b_idx < b.len) {
                    sum += a[a_idx] * b[b_idx];
                }
            }
            const result_idx = i * n + j;
            if (result_idx < result.len) {
                result[result_idx] += alpha * sum;
            }
        }
    }

    logDebug("✅ Naive SGEMM completed successfully", .{});
}

fn naiveDgemm(
    layout: MatrixLayout,
    transa: Transpose,
    transb: Transpose,
    dims: MatrixDims,
    alpha: f64,
    a: []const f64,
    b: []const f64,
    beta: f64,
    result: []f64,
) void {
    _ = layout;
    _ = transa;
    _ = transb; // TODO: Handle these properly

    const m = dims.m;
    const n = dims.n;
    const k = dims.k;

    // Scale existing C by beta
    for (result) |*val| {
        val.* *= beta;
    }

    // Add alpha * A * B
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0.0;
            for (0..k) |l| {
                sum += a[i * k + l] * b[l * n + j];
            }
            result[i * n + j] += alpha * sum;
        }
    }
}

/// Helper function to create matrix and fill with test data
pub fn createMatrix(comptime T: type, allocator: Allocator, rows: usize, cols: usize) ![]T {
    return try allocator.alloc(T, rows * cols);
}

/// Benchmark BLAS performance
pub fn benchmarkBlas(allocator: Allocator) !void {
    const size = 1024;
    const iterations = 10;
    const warmup_iterations = 2;

    logInfo("🚀 Benchmarking BLAS operations ({}x{} matrices, {} iterations)...", .{ size, size, iterations });

    // Initialize BLAS
    var blas = try Blas.init(allocator);
    defer blas.deinit();

    // Create test matrices
    const matrix_a = try createMatrix(f32, allocator, size, size);
    const matrix_b = try createMatrix(f32, allocator, size, size);
    const matrix_c = try createMatrix(f32, allocator, size, size);
    defer allocator.free(matrix_a);
    defer allocator.free(matrix_b);
    defer allocator.free(matrix_c);

    // Fill with random data
    var prng = Random.DefaultPrng.init(42);
    const random = prng.random();
    for (matrix_a) |*val| val.* = random.float(f32);
    for (matrix_b) |*val| val.* = random.float(f32);
    @memset(matrix_c, 0.0);

    // Warmup runs to stabilize performance
    for (0..warmup_iterations) |w| {
        _ = w; // Mark unused
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
        @memset(matrix_c, 0.0); // Reset for next iteration
    }

    // Small delay to let system stabilize
    std.time.sleep(100_000_000); // 100ms

    // Benchmark matrix multiplication
    var timer = try std.time.Timer.start();
    for (0..iterations) |i| {
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
        @memset(matrix_c, 0.0); // Reset for next iteration

        // Small pause between iterations to prevent system pressure
        if (i < iterations - 1) {
            std.time.sleep(1_000_000); // 1ms pause
        }
    }
    const elapsed_ns = timer.read();

    const ops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(iterations));
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const gflops = ops / elapsed_s / 1e9;

    logInfo("✅ BLAS Matrix Multiplication Results:", .{});
    logInfo("  Time: {d:.3} ms", .{elapsed_s * 1000.0});
    logInfo("  Performance: {d:.1} GFLOPS", .{gflops});
    logInfo("  Backend: {}", .{blas.backend});

    const efficiency = gflops / blas.performance_info.peak_gflops * 100.0;
    logInfo("  Efficiency: {d:.1}% of peak BLAS performance", .{efficiency});

    // Additional system info for debugging
    logInfo("  Matrix size: {}x{}, {} total operations", .{ size, size, @as(u64, @intCast(iterations)) });
    logInfo("  Average time per iteration: {d:.3} ms", .{elapsed_s * 1000.0 / @as(f64, @floatFromInt(iterations))});
}

// Basic tests
test "BLAS initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const blas = try Blas.init(allocator);
    try std.testing.expect(blas.performance_info.peak_gflops > 0);
}

test "matrix multiplication correctness" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const blas = try Blas.init(allocator);

    // Test 2x2 matrix multiplication
    var matrix_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var matrix_b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var matrix_c = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    blas.matmul(f32, &matrix_a, &matrix_b, &matrix_c, .{ .m = 2, .n = 2, .k = 2 });

    // Expected result: C = [[19, 22], [43, 50]]
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), matrix_c[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), matrix_c[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 43.0), matrix_c[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), matrix_c[3], 1e-6);
}
