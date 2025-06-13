// CUDA GPU Acceleration Benchmark
// Tests performance of GPU vs CPU matrix operations

const std = @import("std");
const print = std.debug.print;
const Timer = std.time.Timer;

const deepseek_core = @import("deepseek_core");
const Blas = deepseek_core.blas.Blas;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🚀 DeepZig V3 CUDA GPU Benchmark\n", .{});
    print("=====================================\n\n", .{});

    // Initialize BLAS with hardware detection
    var blas = try Blas.init(allocator);
    defer blas.deinit();

    print("🎮 BLAS Backend: {}\n", .{blas.backend});
    print("🎮 Expected Performance: {d:.1} GFLOPS\n\n", .{blas.performance_info.peak_gflops});

    // Test different matrix sizes
    const test_configs = [_]struct { size: u32, iterations: u32 }{
        .{ .size = 512, .iterations = 20 },
        .{ .size = 1024, .iterations = 10 },
        .{ .size = 2048, .iterations = 5 },
        .{ .size = 4096, .iterations = 2 },
        .{ .size = 8192, .iterations = 1 },
    };

    var max_gflops: f64 = 0;
    var optimal_size: u32 = 512;

    for (test_configs) |config| {
        print("📊 Testing {}x{} matrices ({} iterations)...\n", .{ config.size, config.size, config.iterations });

        const gflops = try benchmarkMatrixMultiplication(allocator, &blas, config.size, config.iterations);

        print("   Performance: {d:.1} GFLOPS\n", .{gflops});

        if (gflops > max_gflops) {
            max_gflops = gflops;
            optimal_size = config.size;
        }

        // Check if we're using GPU acceleration effectively
        if (blas.backend == .cuda) {
            if (gflops > 2000.0) {
                print("   ✅ EXCELLENT GPU performance!\n", .{});
            } else if (gflops > 1000.0) {
                print("   ✅ GOOD GPU performance\n", .{});
            } else {
                print("   ⚠️ GPU performance lower than expected\n", .{});
            }
        } else {
            if (gflops > 800.0) {
                print("   ✅ EXCELLENT CPU performance!\n", .{});
            } else if (gflops > 400.0) {
                print("   ✅ GOOD CPU performance\n", .{});
            } else {
                print("   ⚠️ CPU performance lower than expected\n", .{});
            }
        }

        print("\n", .{});

        // Small delay between tests
        std.time.sleep(500_000_000); // 500ms
    }

    print("🎯 Benchmark Summary:\n", .{});
    print("===================\n", .{});
    print("   Peak Performance: {d:.1} GFLOPS\n", .{max_gflops});
    print("   Optimal Matrix Size: {}x{}\n", .{ optimal_size, optimal_size });
    print("   Hardware Backend: {}\n", .{blas.backend});

    // Compare to hardware expectations
    const expected_gflops = blas.performance_info.peak_gflops;
    const efficiency = (max_gflops / expected_gflops) * 100.0;

    print("   Hardware Efficiency: {d:.1}%\n", .{efficiency});

    if (efficiency > 80.0) {
        print("   🎉 OUTSTANDING performance! Hardware fully utilized.\n", .{});
    } else if (efficiency > 60.0) {
        print("   ✅ GOOD performance! Hardware well utilized.\n", .{});
    } else if (efficiency > 40.0) {
        print("   ⚠️ MODERATE performance. Consider optimization.\n", .{});
    } else {
        print("   ❌ POOR performance. Hardware underutilized.\n", .{});
    }

    // GPU-specific analysis
    if (blas.backend == .cuda) {
        print("\n🎮 GPU Analysis:\n", .{});
        if (max_gflops > 3000.0) {
            print("   🚀 RTX 2070 SUPER performing excellently!\n", .{});
            print("   💡 Ready for RTX 5080 upgrade (estimated 10,000+ GFLOPS)\n", .{});
        } else if (max_gflops > 1500.0) {
            print("   ✅ Good GPU utilization\n", .{});
        } else {
            print("   ⚠️ GPU may not be fully utilized\n", .{});
            print("   💡 Check CUDA driver and GPU memory bandwidth\n", .{});
        }
    }

    print("\n🏁 Benchmark completed successfully!\n", .{});
}

fn benchmarkMatrixMultiplication(
    allocator: std.mem.Allocator,
    blas: *const Blas,
    size: u32,
    iterations: u32,
) !f64 {
    const matrix_size = size * size;

    // Allocate matrices
    const a = try allocator.alloc(f32, matrix_size);
    defer allocator.free(a);

    const b = try allocator.alloc(f32, matrix_size);
    defer allocator.free(b);

    const c = try allocator.alloc(f32, matrix_size);
    defer allocator.free(c);

    // Initialize with simple pattern for consistent results
    for (0..matrix_size) |i| {
        a[i] = @as(f32, @floatFromInt(i % 100)) / 100.0;
        b[i] = @as(f32, @floatFromInt((i * 2) % 100)) / 100.0;
        c[i] = 0.0;
    }

    // Warmup run
    blas.sgemm(
        .row_major,
        .no_trans,
        .no_trans,
        .{ .m = size, .n = size, .k = size },
        1.0,
        a,
        b,
        0.0,
        c,
    );

    // Actual benchmark
    var timer = try Timer.start();
    const start_time = timer.lap();

    for (0..iterations) |_| {
        blas.sgemm(
            .row_major,
            .no_trans,
            .no_trans,
            .{ .m = size, .n = size, .k = size },
            1.0,
            a,
            b,
            0.0,
            c,
        );
    }

    const end_time = timer.read();
    const elapsed_ns = end_time - start_time;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

    // Calculate GFLOPS: 2 * M * N * K operations per matrix multiplication
    const operations = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(iterations));
    const gflops = operations / elapsed_s / 1_000_000_000.0;

    return gflops;
}
