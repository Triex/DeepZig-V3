// Simple cuBLAS test program
const std = @import("std");
const builtin = @import("builtin");

const CudaError = struct {
    code: c_int,
    name: []const u8,
};

const cuda_error_codes = [_]CudaError{
    .{ .code = 0, .name = "CUDA_SUCCESS" },
    .{ .code = 1, .name = "CUDA_ERROR_INVALID_VALUE" },
    .{ .code = 2, .name = "CUDA_ERROR_OUT_OF_MEMORY" },
    .{ .code = 3, .name = "CUDA_ERROR_NOT_INITIALIZED" },
    .{ .code = 4, .name = "CUDA_ERROR_DEINITIALIZED" },
    .{ .code = 100, .name = "CUDA_ERROR_NO_DEVICE" },
    .{ .code = 101, .name = "CUDA_ERROR_INVALID_DEVICE" },
    .{ .code = 999, .name = "CUDA_ERROR_UNKNOWN" },
};

const cublas_error_codes = [_]CudaError{
    .{ .code = 0, .name = "CUBLAS_STATUS_SUCCESS" },
    .{ .code = 1, .name = "CUBLAS_STATUS_NOT_INITIALIZED" },
    .{ .code = 2, .name = "CUBLAS_STATUS_ALLOC_FAILED" },
    .{ .code = 3, .name = "CUBLAS_STATUS_INVALID_VALUE" },
    .{ .code = 7, .name = "CUBLAS_STATUS_MAPPING_ERROR" },
    .{ .code = 8, .name = "CUBLAS_STATUS_EXECUTION_FAILED" },
    .{ .code = 9, .name = "CUBLAS_STATUS_INTERNAL_ERROR" },
};

// CUDA Driver API functions
pub extern "c" fn cuInit(flags: c_uint) c_int;
pub extern "c" fn cuDeviceGetCount(count: *c_int) c_int;
pub extern "c" fn cuDeviceGet(device: *c_int, ordinal: c_int) c_int;
pub extern "c" fn cuDeviceGetName(name: [*c]u8, len: c_int, dev: c_int) c_int;
pub extern "c" fn cuDeviceGetAttribute(pi: *c_int, attrib: c_int, dev: c_int) c_int;
pub extern "c" fn cuCtxCreate_v2(pctx: *?*anyopaque, flags: c_uint, dev: c_int) c_int;
pub extern "c" fn cuCtxSetCurrent(ctx: ?*anyopaque) c_int;
pub extern "c" fn cuCtxDestroy_v2(ctx: ?*anyopaque) c_int;

// CUDA Runtime API function (to initialize runtime)
pub extern "c" fn cudaFree(ptr: ?*anyopaque) c_int;

// cuBLAS types and functions
pub const cublasHandle_t = *opaque{};
pub extern "c" fn cublasCreate_v2(handle: *?cublasHandle_t) c_int;
pub extern "c" fn cublasDestroy_v2(handle: cublasHandle_t) c_int;

fn getCudaErrorString(error_code: c_int) []const u8 {
    for (cuda_error_codes) |code_info| {
        if (code_info.code == error_code) {
            return code_info.name;
        }
    }
    return "Unknown CUDA error";
}

fn getCublasErrorString(error_code: c_int) []const u8 {
    for (cublas_error_codes) |code_info| {
        if (code_info.code == error_code) {
            return code_info.name;
        }
    }
    return "Unknown cuBLAS error";
}

pub fn main() !void {
    std.debug.print("cuBLAS Test Program\n", .{});
    std.debug.print("===================\n\n", .{});
    
    // Initialize CUDA
    const result = cuInit(0);
    std.debug.print("cuInit(0) result: {} ({s})\n", .{result, getCudaErrorString(result)});

    if (result != 0) {
        std.debug.print("❌ CUDA initialization failed with code: {}\n", .{result});
        return error.CudaInitFailed;
    }
    std.debug.print("✅ CUDA initialized successfully\n", .{});

    // Get device count
    var device_count: c_int = 0;
    const count_result = cuDeviceGetCount(&device_count);
    
    if (count_result != 0) {
        std.debug.print("❌ Failed to get device count: {}\n", .{count_result});
        return error.CudaDeviceCountFailed;
    }
    
    std.debug.print("Found {} CUDA device(s)\n", .{device_count});
    
    if (device_count > 0) {
        // Get first device
        var device: c_int = 0;
        const dev_result = cuDeviceGet(&device, 0);
        if (dev_result != 0) {
            std.debug.print("❌ Failed to get device: {}\n", .{dev_result});
            return error.CudaDeviceGetFailed;
        }
        
        // Get device name
        var name_buf: [256]u8 = undefined;
        _ = cuDeviceGetName(@ptrCast(&name_buf), @intCast(name_buf.len), device);
        
        // Find null terminator
        var name_len: usize = 0;
        while (name_len < name_buf.len and name_buf[name_len] != 0) : (name_len += 1) {}
        
        std.debug.print("Device 0: {s}\n", .{name_buf[0..name_len]});
        
        // Create CUDA context
        var context: ?*anyopaque = null;
        const ctx_result = cuCtxCreate_v2(&context, 0, device);
        if (ctx_result != 0) {
            std.debug.print("❌ Failed to create CUDA context: {}\n", .{ctx_result});
            return error.CudaContextCreateFailed;
        }
        std.debug.print("✅ CUDA context created successfully\n", .{});
        
        // Set context as current
        const current_result = cuCtxSetCurrent(context);
        if (current_result != 0) {
            std.debug.print("❌ Failed to set CUDA context as current: {}\n", .{current_result});
            _ = cuCtxDestroy_v2(context);
            return error.CudaContextSetCurrentFailed;
        }
        std.debug.print("✅ CUDA context set as current\n", .{});
        
        // Optional: Try initializing CUDA Runtime
        const runtime_result = cudaFree(null);
        std.debug.print("cudaFree(null) result: {}\n", .{runtime_result});
        
        // Create cuBLAS handle
        std.debug.print("\n🔍 Attempting to create cuBLAS handle...\n", .{});
        var cublas_handle: ?cublasHandle_t = null;
        const cublas_result = cublasCreate_v2(&cublas_handle);
        
        if (cublas_result != 0) {
            std.debug.print("❌ cuBLAS handle creation failed: {} ({s})\n", .{
                cublas_result, getCublasErrorString(cublas_result)
            });
        } else {
            std.debug.print("✅ cuBLAS handle created successfully\n", .{});
            
            // Clean up cuBLAS
            _ = cublasDestroy_v2(cublas_handle.?);
            std.debug.print("✅ cuBLAS handle destroyed\n", .{});
        }
        
        // Clean up CUDA context
        _ = cuCtxDestroy_v2(context);
        std.debug.print("✅ CUDA context destroyed\n", .{});
    }
    
    std.debug.print("\n✅ cuBLAS test completed\n", .{});
}
