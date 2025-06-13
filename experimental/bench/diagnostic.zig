// Minimal diagnostic to isolate performance issues
// Tests core BLAS functionality without complex MoE overhead

const std = @import("std");
const builtin = @import("builtin");
const print = std.debug.print;
const Timer = std.time.Timer;

const deepseek_core = @import("deepseek_core");
const Blas = deepseek_core.blas.Blas;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🔍 DeepZig Performance Diagnostic\n", .{});
    print("=================================\n\n", .{});

    // 1. Check BLAS initialization
    print("1️⃣ Testing BLAS Initialization...\n", .{});
    var blas = Blas.init(allocator) catch |err| {
        print("❌ BLAS initialization FAILED: {}\n", .{err});
        return;
    };
    defer blas.deinit();

    print("✅ BLAS Backend: {}\n", .{blas.backend});
    print("✅ Expected GFLOPS: {d:.1}\n", .{blas.performance_info.peak_gflops});
    print("✅ Threads: {}\n", .{blas.performance_info.num_threads});
    print("✅ SIMD Width: {} bits\n\n", .{blas.performance_info.simd_width});

    // 2. Check OpenBLAS threading (if Linux)
    if (builtin.os.tag == .linux and blas.backend == .openblas) {
        print("2️⃣ Testing OpenBLAS Threading...\n", .{});

        // Try to call openblas_get_num_threads
        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{"python3", "-c", "import numpy as np; print('OpenBLAS threads:', np.__config__.show())"},
        }) catch {
            print("⚠️ Could not check OpenBLAS threads\n\n", .{});
            return;
        };
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);

        if (result.term.Exited == 0) {
            print("✅ OpenBLAS Info: {s}\n\n", .{result.stdout});
        } else {
            print("⚠️ OpenBLAS check failed\n\n", .{});
        }
    }

    // 3. Minimal matrix multiplication test
    print("3️⃣ Testing Minimal Matrix Multiplication...\n", .{});

    const sizes = [_]u32{ 64, 256, 512 }; // Small sizes for quick testing

    for (sizes) |size| {
        print("Testing {}x{} matrices...\n", .{ size, size });

        // Create matrices
        const matrix_a = try allocator.alloc(f32, size * size);
        const matrix_b = try allocator.alloc(f32, size * size);
        const matrix_c = try allocator.alloc(f32, size * size);
        defer allocator.free(matrix_a);
        defer allocator.free(matrix_b);
        defer allocator.free(matrix_c);

        // Fill with simple data
        for (matrix_a, 0..) |*val, i| val.* = @floatFromInt(i % 100);
        for (matrix_b, 0..) |*val, i| val.* = @floatFromInt((i + 1) % 100);
        @memset(matrix_c, 0.0);

        // Single operation timing
        var timer = try Timer.start();
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
        const elapsed_ns = timer.read();

        const ops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size));
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        const gflops = ops / elapsed_s / 1e9;

        print("  {}x{}: {d:.3} ms, {d:.1} GFLOPS\n", .{ size, size, elapsed_s * 1000.0, gflops });

        // Check result validity
        var sum: f32 = 0;
        for (matrix_c[0..@min(100, matrix_c.len)]) |val| sum += val;
        print("  Result checksum (first 100): {d:.1}\n", .{sum});

        if (gflops < 1.0) {
            print("  ❌ EXTREMELY LOW PERFORMANCE - Likely using naive implementation\n", .{});
        } else if (gflops < 10.0) {
            print("  ⚠️ LOW PERFORMANCE - BLAS may not be working correctly\n", .{});
        } else {
            print("  ✅ REASONABLE PERFORMANCE\n", .{});
        }
        print("\n", .{});
    }

    // 4. CPU usage test
    print("4️⃣ CPU Usage Test (5 seconds of computation)...\n", .{});
    print("Monitor with: htop or top in another terminal\n", .{});

    const size = 1024;
    const matrix_a = try allocator.alloc(f32, size * size);
    const matrix_b = try allocator.alloc(f32, size * size);
    const matrix_c = try allocator.alloc(f32, size * size);
    defer allocator.free(matrix_a);
    defer allocator.free(matrix_b);
    defer allocator.free(matrix_c);

    for (matrix_a, 0..) |*val, i| val.* = @floatFromInt(i % 1000);
    for (matrix_b, 0..) |*val, i| val.* = @floatFromInt((i + 1) % 1000);

    var total_gflops: f64 = 0;
    var iterations: u32 = 0;
    var timer = try Timer.start();

    while (timer.read() < 5_000_000_000) { // 5 seconds
        @memset(matrix_c, 0.0);
        const start = timer.read();
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
        const end = timer.read();

        const ops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size));
        const elapsed_s = @as(f64, @floatFromInt(end - start)) / 1e9;
        const gflops = ops / elapsed_s / 1e9;

        total_gflops += gflops;
        iterations += 1;

        if (iterations % 10 == 0) {
            print("  {} iterations, avg: {d:.1} GFLOPS\n", .{ iterations, total_gflops / @as(f64, @floatFromInt(iterations)) });
        }
    }

    const avg_gflops = total_gflops / @as(f64, @floatFromInt(iterations));
    print("✅ Final Average: {d:.1} GFLOPS over {} iterations\n\n", .{ avg_gflops, iterations });

    // 5. Summary and recommendations
    print("🎯 DIAGNOSTIC SUMMARY\n", .{});
    print("=====================\n", .{});

    if (avg_gflops < 1.0) {
        print("❌ CRITICAL: Performance extremely poor - likely using naive BLAS\n", .{});
        print("   • Check if OpenBLAS is properly installed and linked\n", .{});
        print("   • Verify hardware detection is working\n", .{});
        print("   • Check build configuration\n", .{});
    } else if (avg_gflops < 100.0) {
        print("⚠️ WARNING: Performance below expectations\n", .{});
        print("   • BLAS may be using single-threaded mode\n", .{});
        print("   • Check thread configuration\n", .{});
        print("   • Verify optimization flags\n", .{});
    } else {
        print("✅ GOOD: Performance in acceptable range\n", .{});
        print("   • BLAS appears to be working correctly\n", .{});
        print("   • Performance issue may be in higher-level code\n", .{});
    }
}
