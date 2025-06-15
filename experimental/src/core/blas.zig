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
const root = @import("root");
const builtin = @import("builtin");

const Allocator = std.mem.Allocator;
const Random = std.Random;

// Import our comprehensive hardware detection
const hardware = @import("hardware.zig");
const HardwareInfo = hardware.HardwareInfo;
const BlasConfig = hardware.BlasConfig;
const BlasBackend = hardware.BlasBackend;

// cuBLAS types
const cublasStatus_t = c_int;
pub const cublasHandle_t = *opaque {};

// cuBLAS status codes
const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 2;
const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 3;
const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = 8; // Keep this for backward compatibility
const CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = 7;
const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 8;
const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 9;

// cuBLAS operation types
const CUBLAS_OP_N: c_int = 0; // non-transpose
const CUBLAS_OP_T: c_int = 1; // transpose
const CUBLAS_OP_C: c_int = 2; // conjugate transpose

// CUDA event flags
const cudaEventDefault: c_uint = 0;
const cudaEventBlockingSync: c_uint = 1;
const cudaEventDisableTiming: c_uint = 2;
const cudaEventInterprocess: c_uint = 4;

// CUDA Backend Implementation
pub const CudaBackend = struct {
    allocator: Allocator,
    context: ?*anyopaque, // CUDA context (CUcontext)
    cublas_handle: ?cublasHandle_t,
    stream: ?cudaStream_t,
    mempool: ?cudaMemPool_t, // Memory pool for efficient allocation
    event: ?cudaEvent_t,  // CUDA event for synchronization between operations

    const Self = @This();

    // Use the main MatrixLayout and Transpose types
    pub const CudaMatrixLayout = MatrixLayout;
    pub const CudaTranspose = Transpose;

    pub fn init(allocator: Allocator) !Self {
        // Try to initialize CUDA with detailed error logging
        var status: i32 = undefined;

        logInfo("🔍 Attempting CUDA initialization...", .{});

        // Initialize CUDA driver
        status = cuInit(0);
        if (status != 0) {
            logInfo("❌ CUDA driver initialization failed with status code: {}", .{status});
            return error.CudaDriverInitFailed;
        }
        logInfo("✅ CUDA driver initialized successfully", .{});

        // Get device count
        var device_count: c_int = 0;
        status = cuDeviceGetCount(&device_count);
        if (status != 0) {
            logInfo("❌ Failed to get CUDA device count: status code {}", .{status});
            return error.CudaNoDevicesFound;
        }
        if (device_count == 0) {
            logInfo("❌ No CUDA devices found", .{});
            return error.CudaNoDevicesFound;
        }
        logInfo("✅ Found {} CUDA device(s)", .{device_count});

        // Get first CUDA device
        var device: c_int = 0;
        status = cuDeviceGet(&device, 0);
        if (status != 0) {
            logInfo("❌ Failed to get CUDA device 0: status code {}", .{status});
            return error.CudaDeviceAccessFailed;
        }

        // Get device name for logging
        var device_name_buffer: [256]u8 = undefined;
        var device_name: []u8 = device_name_buffer[0..];
        status = cuDeviceGetName(device_name.ptr, @intCast(device_name.len), device);
        if (status == 0) {
            // Find null terminator
            var name_len: usize = 0;
            while (name_len < device_name.len and device_name[name_len] != 0) : (name_len += 1) {}
            logInfo("✅ Using CUDA device: {s}", .{device_name[0..name_len]});
        } else {
            logInfo("✅ Using CUDA device ID: {} (unable to get name)", .{device});
        }

        // Create CUDA context
        var context: ?*anyopaque = null;
        status = cuCtxCreate_v2(&context, 0, device);
        if (status != 0) {
            logInfo("❌ Failed to create CUDA context: status code {}", .{status});
            return error.CudaContextCreationFailed;
        }
        logInfo("✅ CUDA context created successfully", .{});

        // Set context as current for this thread before initializing cuBLAS
        logInfo("🔍 Setting CUDA context as current for this thread...", .{});
        const set_current_result = cuCtxSetCurrent(context);
        if (set_current_result != 0) {
            logInfo("❌ Failed to set CUDA context as current, error code: {}", .{set_current_result});
            _ = cuCtxDestroy_v2(context);
            return error.CudaContextSetCurrentFailed;
        }
        logInfo("✅ CUDA context set as current successfully", .{});

        // Try direct library loading for cuBLAS
        logInfo("🔍 Attempting to directly detect cuBLAS library...", .{});
        const lib_names = [_][]const u8{ "/usr/lib/x86_64-linux-gnu/libcublas.so", "/usr/local/cuda/lib64/libcublas.so" };

        var lib_found = false;
        for (lib_names) |lib_name| {
            const lib_exists = std.fs.cwd().access(lib_name, .{}) catch |err| {
                logInfo("ℹ️ cuBLAS library path check: {s} - {}", .{ lib_name, err });
                continue;
            };
            _ = lib_exists;
            lib_found = true;
            logInfo("✅ Found cuBLAS library at: {s}", .{lib_name});
            break;
        }

        if (!lib_found) {
            logInfo("❌ Could not find cuBLAS library in standard locations", .{});
        }

        // Initialize CUDA Runtime API (required before cuBLAS)
        logInfo("🔍 Initializing CUDA Runtime API with cudaFree(null)...", .{});
        const runtime_result = cudaFree(null);
        if (runtime_result != 0) {
            logInfo("❌ Failed to initialize CUDA Runtime API: error code {}", .{runtime_result});
            _ = cuCtxDestroy_v2(context);
            return error.CudaRuntimeInitFailed;
        }
        logInfo("✅ CUDA Runtime API initialized successfully", .{});

        // Create cuBLAS handle
        logInfo("🔍 Creating cuBLAS handle (with library diagnostics)...", .{});
        var cublas_handle: ?cublasHandle_t = null;
        const cublas_status = cublasCreate_v2(&cublas_handle);
        if (cublas_status != CUBLAS_STATUS_SUCCESS) {
            logInfo("❌ cuBLAS handle creation failed with status code: {}", .{cublas_status});
            // Provide more detailed error information based on status code
            switch (cublas_status) {
                CUBLAS_STATUS_NOT_INITIALIZED => logInfo("❌ ERROR: CUBLAS_STATUS_NOT_INITIALIZED - cuBLAS library not initialized", .{}),
                CUBLAS_STATUS_ALLOC_FAILED => logInfo("❌ ERROR: CUBLAS_STATUS_ALLOC_FAILED - Resource allocation failed", .{}),
                CUBLAS_STATUS_INVALID_VALUE => logInfo("❌ ERROR: CUBLAS_STATUS_INVALID_VALUE - Invalid parameter", .{}),
                CUBLAS_STATUS_ARCH_MISMATCH => logInfo("❌ ERROR: CUBLAS_STATUS_ARCH_MISMATCH - Device compute capability mismatch", .{}),
                else => logInfo("❌ ERROR: Unknown cuBLAS error code: {}", .{cublas_status}),
            }
            _ = cuCtxDestroy_v2(context);
            return error.CublasInitFailed;
        }
        logInfo("✅ cuBLAS handle created successfully", .{});

        logInfo("🚀 CUDA initialization complete!", .{});

        // Initialize a CUDA stream for operations
        var stream: cudaStream_t = null; // Default stream
        const stream_status = cudaStreamCreate(&stream);
        if (stream_status != 0) {
            logInfo("⚠️ Failed to create CUDA stream, using default stream (status: {})", .{stream_status});
            stream = null; // Use default stream
        } else {
            logInfo("✅ Created CUDA stream for operations", .{});
        }

        // Setup memory pool
        var mempool: cudaMemPool_t = null;
        const pool_status = cudaDeviceGetDefaultMemPool(&mempool, device);
        if (pool_status != 0) {
            logInfo("⚠️ Failed to get device memory pool, using default allocation (status: {})", .{pool_status});
        } else {
            logInfo("✅ Got device memory pool for efficient allocation", .{});
            
            // Configure memory pool to retain memory between operations
            // Set threshold to max value to keep all memory in the pool
            const max_value: u64 = std.math.maxInt(u64);
            const attr_status = cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &max_value);
            if (attr_status != 0) {
                logInfo("⚠️ Failed to configure memory pool retention (status: {})", .{attr_status});
            } else {
                logInfo("✅ Configured memory pool for maximum retention", .{});
            }
        }

        // Create CUDA event for synchronization
        var event: cudaEvent_t = null;
        const event_status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        if (event_status != 0) {
            logInfo("⚠️ Failed to create CUDA event for synchronization (status: {})", .{event_status});
            event = null;
        } else {
            logInfo("✅ Created CUDA event for precise synchronization", .{});
        }
        
        logInfo("🚀 CUDA initialization complete with stream-ordered memory allocator and event synchronization!", .{});

        return Self{
            .allocator = allocator,
            .context = context,
            .cublas_handle = cublas_handle,
            .stream = stream,
            .mempool = mempool,
            .event = event,
        };
    }

    pub fn deinit(self: *Self) void {
        // Destroy cuBLAS handle first
        if (self.cublas_handle) |handle| {
            _ = cublasDestroy_v2(handle);
            self.cublas_handle = null;
        }
        
        // Trim the memory pool to free any unused memory back to the system
        if (self.mempool != null) {
            logInfo("🧹 Trimming memory pool to free unused GPU memory", .{});
            // Free all unused memory (keep 0 bytes in the pool)
            const trim_status = cudaMemPoolTrimTo(self.mempool.?, 0);
            if (trim_status != 0) {
                logInfo("⚠️ Failed to trim memory pool (status: {})", .{trim_status});
            }
        }
        
        // Destroy the CUDA stream if it's not the default stream
        if (self.stream != null) {
            logInfo("🔄 Destroying CUDA stream", .{});
            const stream_status = cudaStreamDestroy(self.stream.?);
            if (stream_status != 0) {
                logInfo("⚠️ Failed to destroy CUDA stream (status: {})", .{stream_status});
            }
            self.stream = null;
        }
        
        // Destroy CUDA event if it exists
        if (self.event != null) {
            logInfo("🔄 Destroying CUDA event", .{});
            const event_status = cudaEventDestroy(self.event.?);
            if (event_status != 0) {
                logInfo("⚠️ Failed to destroy CUDA event (status: {})", .{event_status});
            }
            self.event = null;
        }
        
        // Finally, destroy CUDA context
        if (self.context) |ctx| {
            logInfo("🔄 Destroying CUDA context", .{});
            _ = cuCtxDestroy_v2(ctx);
            self.context = null;
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
        a: anytype,
        lda: usize,
        b: anytype,
        ldb: usize,
        beta: f32,
        c: anytype,
        ldc: usize,
    ) !void {
        if (self.cublas_handle == null) {
            return error.CublasHandleNotInitialized;
        }

        // Convert parameters to appropriate types for cuBLAS
        const handle = self.cublas_handle.?;
        const rows = @as(c_int, @intCast(m));
        const cols = @as(c_int, @intCast(n));
        const shared_dim = @as(c_int, @intCast(k));
        const lda_int = @as(c_int, @intCast(lda));
        const ldb_int = @as(c_int, @intCast(ldb));
        const ldc_int = @as(c_int, @intCast(ldc));

        // Map transpose operations to cuBLAS constants
        const cublas_trans_a = switch (transa) {
            .no_trans => CUBLAS_OP_N,
            .trans => CUBLAS_OP_T,
            .conj_trans => CUBLAS_OP_C,
        };

        const cublas_trans_b = switch (transb) {
            .no_trans => CUBLAS_OP_N,
            .trans => CUBLAS_OP_T,
            .conj_trans => CUBLAS_OP_C,
        };

        // For cuBLAS, we need to convert row_major to col_major with appropriate transpose adjustments
        // cuBLAS uses column-major format natively
        const adj_trans_a = cublas_trans_a;
        const adj_trans_b = cublas_trans_b;
        const adj_m = rows;
        const adj_n = cols;
        const adj_k = shared_dim;
        const adj_lda = lda_int;
        const adj_ldb = ldb_int;
        const adj_ldc = ldc_int;

        // If input is row-major, we need to adjust for column-major cuBLAS
        if (layout == .row_major) {
            // For row-major, we compute C^T = B^T * A^T
            // Swap A and B, m and n, and adjust transposes
            // Handle both regular CudaBuffer and PooledCudaBuffer
            // Get the device pointers using inline switch for different buffer types
            const b_ptr = switch (@TypeOf(b)) {
                CudaBuffer => b.ptr,
                PooledCudaBuffer => @intFromPtr(b.devicePtr(f32)),
                else => @compileError("Buffer type must be CudaBuffer or PooledCudaBuffer"),
            };
            
            const a_ptr = switch (@TypeOf(a)) {
                CudaBuffer => a.ptr, 
                PooledCudaBuffer => @intFromPtr(a.devicePtr(f32)),
                else => @compileError("Buffer type must be CudaBuffer or PooledCudaBuffer"),
            };
            
            // No actual swap needed, we use the pointers directly

            // Call cuBLAS SGEMM with swapped and adjusted parameters and associate with our CUDA stream
            // Properly unwrap the optional stream to avoid ??*anyopaque type errors
            const status = cublasSetStream_v2(handle, if (self.stream != null) self.stream.? else null);
            if (status != CUBLAS_STATUS_SUCCESS) {
                logInfo("❌ Failed to set cuBLAS stream: {}", .{status});
                return error.CublasStreamSetFailed;
            }
            
            // Execute the operation on our configured stream
            const sgemm_status = cublasSgemm_v2(handle, adj_trans_b, // Was B's transpose
                adj_trans_a, // Was A's transpose
                adj_n, // Was n, now m for the swapped operation
                adj_m, // Was m, now n for the swapped operation
                adj_k, // k stays the same
                &alpha, @ptrFromInt(b_ptr), // Was B - convert device ptr to f32 pointer
                adj_ldb, // Leading dimension of B
                @ptrFromInt(a_ptr), // Was A - convert device ptr to f32 pointer
                adj_lda, // Leading dimension of A
                &beta, @ptrFromInt(switch (@TypeOf(c)) { // Handle both buffer types for C
                    CudaBuffer => c.ptr,
                    PooledCudaBuffer => @intFromPtr(c.devicePtr(f32)),
                    else => @compileError("Buffer type must be CudaBuffer or PooledCudaBuffer"),
                }), // Convert device ptr to f32 pointer
                adj_ldc // Leading dimension of C
            );

            if (sgemm_status != CUBLAS_STATUS_SUCCESS) {
                logInfo("❌ cuBLAS SGEMM failed with status: {}", .{sgemm_status});
                return error.CublasSgemmFailed;
            }
        } else {
            // For column-major, standard cuBLAS call but first set the stream
            // Properly unwrap the optional stream to avoid ??*anyopaque type errors
            const status = cublasSetStream_v2(handle, if (self.stream != null) self.stream.? else null);
            if (status != CUBLAS_STATUS_SUCCESS) {
                logInfo("❌ Failed to set cuBLAS stream: {}", .{status});
                return error.CublasStreamSetFailed;
            }
            
            // Execute the operation on our configured stream
            // Get device pointers with correct type handling for both buffer types  
            const a_ptr = switch (@TypeOf(a)) {
                CudaBuffer => a.ptr,
                PooledCudaBuffer => @intFromPtr(a.devicePtr(f32)),
                else => @compileError("Buffer type must be CudaBuffer or PooledCudaBuffer"),
            };
            
            const b_ptr = switch (@TypeOf(b)) {
                CudaBuffer => b.ptr, 
                PooledCudaBuffer => @intFromPtr(b.devicePtr(f32)),
                else => @compileError("Buffer type must be CudaBuffer or PooledCudaBuffer"),
            };
            
            const c_ptr = switch (@TypeOf(c)) {
                CudaBuffer => c.ptr,
                PooledCudaBuffer => @intFromPtr(c.devicePtr(f32)),
                else => @compileError("Buffer type must be CudaBuffer or PooledCudaBuffer"),
            };
            
            const sgemm_status = cublasSgemm_v2(handle, adj_trans_a, adj_trans_b, adj_m, adj_n, adj_k, &alpha, 
                                            @ptrFromInt(a_ptr), adj_lda, 
                                            @ptrFromInt(b_ptr), adj_ldb, 
                                            &beta, @ptrFromInt(c_ptr), adj_ldc);

            if (sgemm_status != CUBLAS_STATUS_SUCCESS) {
                logInfo("❌ cuBLAS SGEMM failed with status: {}", .{sgemm_status});
                return error.CublasSgemmFailed;
            }
            
            // Record event for better synchronization in column-major path
            if (self.event != null and self.stream != null) {
                const record_status = cudaEventRecord(self.event.?, self.stream.?);
                if (record_status != 0) {
                    logInfo("⚠️ Failed to record CUDA event after SGEMM (status: {})", .{record_status});
                }
            }
        }

        // Record event for better synchronization regardless of matrix layout path
        if (self.event != null and self.stream != null) {
            const record_status = cudaEventRecord(self.event.?, self.stream.?);
            if (record_status != 0) {
                logInfo("⚠️ Failed to record CUDA event after SGEMM (status: {})", .{record_status});
            }
        }
        
        logDebug("✅ cuBLAS SGEMM completed successfully", .{});
    }

    pub fn synchronize(self: *const Self) !void {
        // First try to synchronize with event if available (more precise)
        if (self.event != null) {
            const event_status = cudaEventSynchronize(self.event.?);
            if (event_status == 0) {
                // Event synchronization succeeded
                return;
            } else {
                // Event synchronization failed, log and fall back to stream sync
                logInfo("⚠️ CUDA event synchronize failed with status: {}, falling back to stream sync", .{event_status});
                // Clear error state before trying stream sync
                _ = cudaGetLastError();
            }
        }
        
        // If we have a stream, use stream synchronize instead of context synchronize
        if (self.stream != null) {
            const status = cudaStreamSynchronize(self.stream.?);
            if (status != 0) {
                logInfo("❌ CUDA stream synchronize failed with status: {}", .{status});
                
                // Enhanced error handling with recovery attempt
                // For CUDA error details, we just log the numeric value
                // since the string functions need special handling in Zig
                logInfo("❌ CUDA error details: Error code {}", .{status});
                
                // Reset the error state first before returning
                _ = cudaGetLastError(); // Clear the last error
                
                // If this is a temporary error, we could retry synchronization
                // For now, just return the error with better diagnostics
                return error.CudaSyncFailed;
            }
        } else {
            // Fall back to context sync if no stream
            const ctx_status = cuCtxSynchronize();
            if (ctx_status != 0) {
                logInfo("❌ CUDA context synchronize failed with status: {}", .{ctx_status});
                return error.CudaSyncFailed;
            }
        }
    }
    
    /// Wait for event completion to ensure proper operation ordering
    /// This helps prevent race conditions between operations
    pub fn waitForEvent(self: *const Self, event: cudaEvent_t) !void {
        if (event == null) return;
        if (self.stream == null) return;
        
        const status = cudaStreamWaitEvent(self.stream, event, 0);
        if (status != 0) {
            logInfo("❌ CUDA stream wait event failed with status: {}", .{status});
            return error.CudaEventWaitFailed;
        }
    }
    
    /// Create a pooled buffer allocation using the stream-ordered memory allocator
    /// This is much more efficient than regular CUDA allocations during training
    pub fn createPooledBuffer(self: *const Self, size: usize) !PooledCudaBuffer {
        if (self.stream == null) {
            logInfo("⚠️ No CUDA stream available for pooled buffer, using default stream", .{});
        }
        
        // Unwrap the optional stream - pass null directly if stream is null
        return PooledCudaBuffer.init(self.allocator, size, if (self.stream != null) self.stream.? else null);
    }
};

pub const CudaBuffer = struct {
    ptr: u64, // CUDA device pointer (CUdeviceptr)
    size: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, size: usize) !CudaBuffer {
        var device_ptr: u64 = 0; // Use u64 for CUDA device pointer as required by API
        const result = cuMemAlloc_v2(&device_ptr, size);
        if (result != 0) {
            return error.CudaMemoryAllocationFailed;
        }
        return CudaBuffer{
            .ptr = device_ptr,
            .size = size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CudaBuffer) void {
        if (self.ptr != 0) {
            _ = cuMemFree_v2(self.ptr);
            self.ptr = 0;
        }
    }

    pub fn copyFromHost(self: *const CudaBuffer, comptime T: type, data: []const T) !void {
        const byte_size = data.len * @sizeOf(T);
        if (byte_size > self.size) return error.BufferTooSmall;
        const result = cuMemcpyHtoD_v2(self.ptr, data.ptr, byte_size);
        if (result != 0) {
            return error.CudaCopyFailed;
        }
    }

    pub fn copyToHost(self: *const CudaBuffer, comptime T: type, data: []T) !void {
        const byte_size = data.len * @sizeOf(T);
        if (byte_size > self.size) return error.BufferTooSmall;
        const result = cuMemcpyDtoH_v2(data.ptr, self.ptr, byte_size);
        if (result != 0) {
            return error.CudaCopyFailed;
        }
    }
};

// CUDA Driver API - Real FFI declarations when CUDA is available, stubs otherwise
// When build.zig sets CUDA_ENABLED=1, link against real CUDA libraries
// Otherwise, fall back to stub implementations
const cuda_disabled = false;

// CUDA acceleration is enabled for this build
const enable_real_cuda = true;

// CUDA and cuBLAS FFI declarations using direct extern "c" syntax for maximum compatibility

// CUDA Driver API functions
pub extern "c" fn cuInit(flags: c_uint) c_int;
pub extern "c" fn cuDeviceGetCount(count: *c_int) c_int;
pub extern "c" fn cuDeviceGet(device: *c_int, ordinal: c_int) c_int;
pub extern "c" fn cuDeviceGetName(name: [*c]u8, len: c_int, dev: c_int) c_int;
pub extern "c" fn cuDeviceGetAttribute(pi: *c_int, attrib: c_int, dev: c_int) c_int;
pub extern "c" fn cuCtxCreate_v2(pctx: *?*anyopaque, flags: c_uint, dev: c_int) c_int;
pub extern "c" fn cuCtxSetCurrent(ctx: ?*anyopaque) c_int;
pub extern "c" fn cuCtxDestroy_v2(ctx: ?*anyopaque) c_int;
pub extern "c" fn cuCtxSynchronize() c_int;
pub extern "c" fn cuMemAlloc_v2(dptr: *u64, bytesize: usize) c_int;
pub extern "c" fn cuMemcpyHtoD_v2(dstDevice: u64, srcHost: *const anyopaque, ByteCount: usize) c_int;
pub extern "c" fn cuMemcpyDtoH_v2(dstHost: *anyopaque, srcDevice: u64, ByteCount: usize) c_int;
pub extern "c" fn cuMemFree_v2(dptr: u64) c_int;

// CUDA Runtime API functions
pub extern "c" fn cudaFree(ptr: ?*anyopaque) c_int;

// Stream and memory pool types
pub const cudaStream_t = ?*anyopaque;
pub const cudaMemPool_t = ?*anyopaque;
pub const cudaEvent_t = ?*anyopaque;

// Memory pool attributes
pub const cudaMemPoolAttr = c_int;
pub const cudaMemPoolAttrReleaseThreshold: cudaMemPoolAttr = 1;

// Stream-ordered memory management API
pub extern "c" fn cudaDeviceGetDefaultMemPool(memPool: *cudaMemPool_t, device: c_int) c_int;
pub extern "c" fn cudaMemPoolSetAttribute(memPool: cudaMemPool_t, attr: cudaMemPoolAttr, value: *const anyopaque) c_int;
pub extern "c" fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) c_int;
pub extern "c" fn cudaMallocAsync(devPtr: *?*anyopaque, size: usize, stream: cudaStream_t) c_int;
pub extern "c" fn cudaFreeAsync(devPtr: ?*anyopaque, stream: cudaStream_t) c_int;
pub extern "c" fn cudaStreamSynchronize(stream: cudaStream_t) c_int;
pub extern "c" fn cudaStreamCreate(pStream: *cudaStream_t) c_int;
pub extern "c" fn cudaStreamDestroy(stream: cudaStream_t) c_int;

// CUDA error handling functions
pub extern "c" fn cudaGetErrorName(err_code: c_int) [*]u8;
pub extern "c" fn cudaGetErrorString(err_code: c_int) [*]u8;
pub extern "c" fn cudaGetLastError() c_int;

// CUDA event management
pub extern "c" fn cudaEventCreate(event: *cudaEvent_t) c_int;
pub extern "c" fn cudaEventCreateWithFlags(event: *cudaEvent_t, flags: c_uint) c_int;
pub extern "c" fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) c_int;
pub extern "c" fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_uint) c_int;
pub extern "c" fn cudaEventSynchronize(event: cudaEvent_t) c_int;
pub extern "c" fn cudaEventDestroy(event: cudaEvent_t) c_int;

// cuBLAS API functions
pub extern "c" fn cublasCreate_v2(handle: *?cublasHandle_t) c_int;
pub extern "c" fn cublasDestroy_v2(handle: cublasHandle_t) c_int;
pub extern "c" fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) c_int;
pub extern "c" fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: *const f32,
    lda: c_int,
    B: *const f32,
    ldb: c_int,
    beta: *const f32,
    C: *f32,
    ldc: c_int,
) c_int;
pub extern "c" fn cublasDgemm_v2(
    handle: cublasHandle_t,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    A: *const f64,
    lda: c_int,
    B: *const f64,
    ldb: c_int,
    beta: *const f64,
    C: *f64,
    ldc: c_int,
) c_int;

// Wrapper functions have been removed as they've been replaced by direct extern "c" calls

// All wrapper functions have been removed and replaced with direct extern "c" calls
// Use cuCtxSynchronize(), cuMemAlloc_v2(), cuMemFree_v2(), and cuMemcpyHtoD_v2() directly

fn cuMemcpyDtoH_wrapper(dst: ?*anyopaque, src: ?*anyopaque, size: usize) i32 {
    if (comptime enable_real_cuda) {
        return cuMemcpyDtoH_v2(dst, src, size);
    } else {
        // Stub implementation - no-op memory copy
        return 1; // Ignore parameters in stub implementation
    }
}

/// Re-export MatrixDims for public use
pub const MatrixDims = hardware.MatrixDims;

/// Conditional logging that's disabled in release mode for optimal performance
inline fn logInfo(comptime fmt: []const u8, args: anytype) void {
    // Only show essential info - benchmark results and critical events
    // In release mode, most BLAS logs are suppressed unless they're critical
    if (builtin.mode != .Debug) {
        // In release mode, only show messages containing these important keywords
        const is_important = isImportantLogMessage(fmt);

        if (is_important) {
            std.log.info(fmt, args);
        }
    } else {
        // In debug mode, show all info logs
        std.log.info(fmt, args);
    }
}

inline fn logDebug(comptime fmt: []const u8, args: anytype) void {
    if (builtin.mode == .Debug) {
        std.log.debug(fmt, args);
    }
}

/// Helper function to determine if a log message is important enough to show in release mode
fn isImportantLogMessage(message: []const u8) bool {
    // Keywords that make a message important enough to keep
    const important_keywords = [_][]const u8{
        "Benchmarking",
        "Performance:",
        "GFLOPS",
        "failed",
        "error",
        "Results",
        "CUDA", // CUDA-related messages are important
        "critical",
        "warning",
        "initialized", // Initialization messages
        "detected", // Hardware detection messages
    };

    // Keywords that indicate routine operations that should be filtered
    const filter_keywords = [_][]const u8{ "Large BLAS operation", "operation", "matrix", "tensor" };

    // Check if it contains any words that indicate it should be filtered
    for (filter_keywords) |keyword| {
        if (std.mem.indexOf(u8, message, keyword) != null) {
            // It contains a filter keyword, don't show unless it also contains an important keyword
            for (important_keywords) |important| {
                if (std.mem.indexOf(u8, message, important) != null) {
                    return true; // Contains both filter and important keywords
                }
            }
            return false; // Contains only filter keywords
        }
    }

    // For all other messages, check if they contain important keywords
    for (important_keywords) |keyword| {
        if (std.mem.indexOf(u8, message, keyword) != null) {
            return true;
        }
    }

    // Default: don't show general messages in release mode
    return false;
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
                errdefer {
                    // If any error occurs after creating fallback_hw_info, ensure it's properly freed
                    fallback_hw_info.deinit();
                    allocator.destroy(fallback_hw_info);
                }
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

    /// CUDA SGEMM implementation with stream-ordered memory allocator
    fn cudasgemm(
        _: *const Blas,  // self not used directly, use _ to avoid unused param warning
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
        
        // Use pooled memory allocation for better efficiency
        // This avoids expensive malloc/free cycles during training
        logDebug("💡 Using stream-ordered memory allocator for CUDA matrices", .{});
        
        // Allocate GPU memory using memory pool
        var gpu_a = try cuda_ctx.createPooledBuffer(size_a * @sizeOf(f32));
        var gpu_b = try cuda_ctx.createPooledBuffer(size_b * @sizeOf(f32));
        var gpu_c = try cuda_ctx.createPooledBuffer(size_c * @sizeOf(f32));
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

    var blas = try Blas.init(allocator);
    defer blas.deinit();
    try std.testing.expect(blas.performance_info.peak_gflops > 0);
}

test "matrix multiplication correctness" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var blas = try Blas.init(allocator);
    defer blas.deinit();

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

/// Stream-ordered memory buffer implementation using CUDA memory pools
pub const PooledCudaBuffer = struct {
    ptr: ?*anyopaque, // CUDA device pointer from cudaMallocAsync
    size: usize,
    allocator: Allocator,
    stream: cudaStream_t, // Associated CUDA stream

    /// Initialize a pooled CUDA buffer using the stream-ordered memory allocator
    /// This uses the CUDA memory pool system to efficiently reuse memory
    pub fn init(allocator: Allocator, size: usize, stream: cudaStream_t) !PooledCudaBuffer {
        var device_ptr: ?*anyopaque = null;
        const result = cudaMallocAsync(&device_ptr, size, stream);
        if (result != 0) {
            logInfo("❌ cudaMallocAsync failed with status: {}", .{result});
            return error.CudaMemoryAllocationFailed;
        }
        
        return PooledCudaBuffer{
            .ptr = device_ptr,
            .size = size,
            .allocator = allocator,
            .stream = stream,
        };
    }

    /// Return memory to the pool rather than immediately freeing it
    /// This allows faster reuse in subsequent allocations
    pub fn deinit(self: *PooledCudaBuffer) void {
        if (self.ptr != null) {
            _ = cudaFreeAsync(self.ptr, self.stream);
            self.ptr = null;
        }
    }

    /// Copy data from host to device
    pub fn copyFromHost(self: *const PooledCudaBuffer, comptime T: type, data: []const T) !void {
        const byte_size = data.len * @sizeOf(T);
        if (byte_size > self.size) return error.BufferTooSmall;
        
        // Use synchronous copy for now (we can add async copy later)
        const result = cuMemcpyHtoD_v2(@intFromPtr(self.ptr.?), data.ptr, byte_size);
        if (result != 0) {
            return error.CudaCopyFailed;
        }
    }

    /// Copy data from device to host
    pub fn copyToHost(self: *const PooledCudaBuffer, comptime T: type, data: []T) !void {
        const byte_size = data.len * @sizeOf(T);
        if (byte_size > self.size) return error.BufferTooSmall;
        
        // Use synchronous copy for now (we can add async copy later)
        const result = cuMemcpyDtoH_v2(data.ptr, @intFromPtr(self.ptr.?), byte_size);
        if (result != 0) {
            return error.CudaCopyFailed;
        }
    }

    // Get the raw device pointer cast to a specific type
    pub fn devicePtr(self: *const PooledCudaBuffer, comptime T: type) [*c]T {
        // Fix alignment issues by using both @alignCast and @ptrCast
        return @ptrCast(@alignCast(self.ptr.?));
    }
};
