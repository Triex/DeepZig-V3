// Simple CUDA test program
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
    .{ .code = 5, .name = "CUDA_ERROR_PROFILER_DISABLED" },
    .{ .code = 6, .name = "CUDA_ERROR_PROFILER_NOT_INITIALIZED" },
    .{ .code = 7, .name = "CUDA_ERROR_PROFILER_ALREADY_STARTED" },
    .{ .code = 8, .name = "CUDA_ERROR_PROFILER_ALREADY_STOPPED" },
    .{ .code = 100, .name = "CUDA_ERROR_NO_DEVICE" },
    .{ .code = 101, .name = "CUDA_ERROR_INVALID_DEVICE" },
    .{ .code = 999, .name = "CUDA_ERROR_UNKNOWN" },
};

pub extern "c" fn cuInit(flags: c_uint) c_int;
pub extern "c" fn cuDeviceGetCount(count: *c_int) c_int;
pub extern "c" fn cuDeviceGet(device: *c_int, ordinal: c_int) c_int;
pub extern "c" fn cuDeviceGetName(name: [*c]u8, len: c_int, dev: c_int) c_int;
pub extern "c" fn cuDeviceGetAttribute(pi: *c_int, attrib: c_int, dev: c_int) c_int;

fn getCudaErrorString(error_code: c_int) []const u8 {
    for (cuda_error_codes) |code_info| {
        if (code_info.code == error_code) {
            return code_info.name;
        }
    }
    return "Unknown CUDA error";
}

pub fn main() !void {
    std.debug.print("CUDA Test Program\n", .{});
    std.debug.print("=================\n\n", .{});
    
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
        // Get first device properties
        var device: c_int = 0;
        _ = cuDeviceGet(&device, 0);
        
        var name_buf: [256]u8 = undefined;
        _ = cuDeviceGetName(@ptrCast(&name_buf), @intCast(name_buf.len), device);
        
        // Find null terminator
        var name_len: usize = 0;
        while (name_len < name_buf.len and name_buf[name_len] != 0) : (name_len += 1) {}
        
        std.debug.print("Device 0: {s}\n", .{name_buf[0..name_len]});
        
        // Get compute capability
        var major: c_int = 0;
        var minor: c_int = 0;
        _ = cuDeviceGetAttribute(&major, 75, device); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        _ = cuDeviceGetAttribute(&minor, 76, device); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
        
        std.debug.print("Compute capability: {}.{}\n", .{major, minor});
    }
    
    std.debug.print("\n✅ CUDA test completed successfully\n", .{});
}
