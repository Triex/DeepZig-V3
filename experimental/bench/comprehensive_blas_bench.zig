// Comprehensive BLAS benchmark for AMD Ryzen 9 3900X
// Tests multiple matrix sizes to find optimal performance

const std = @import("std");
const print = std.debug.print;
const Timer = std.time.Timer;
const Random = std.Random;

const deepseek_core = @import("deepseek_core");
const Blas = deepseek_core.blas.Blas;

fn benchmarkMatrixSize(allocator: std.mem.Allocator, size: u32, iterations: u32) !f64 {
    const blas = try Blas.global(allocator);

    // Create test matrices
    const matrix_a = try allocator.alloc(f32, size * size);
    const matrix_b = try allocator.alloc(f32, size * size);
    const matrix_c = try allocator.alloc(f32, size * size);
    defer allocator.free(matrix_a);
    defer allocator.free(matrix_b);
    defer allocator.free(matrix_c);

    // Fill with random data
    var prng = Random.DefaultPrng.init(42);
    const random = prng.random();
    for (matrix_a) |*val| val.* = random.float(f32);
    for (matrix_b) |*val| val.* = random.float(f32);

    // Warmup
    @memset(matrix_c, 0.0);
    blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });

    // Benchmark
    @memset(matrix_c, 0.0);
    var timer = try Timer.start();

    for (0..iterations) |_| {
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
    }

    const elapsed_ns = timer.read();

    // Calculate GFLOPS
    const ops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(iterations));
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
    const gflops = ops / elapsed_s / 1e9;

    return gflops;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🚀 Comprehensive BLAS Benchmark for AMD Ryzen 9 3900X\n", .{});
    print("======================================================\n\n", .{});

    // Test different matrix sizes to find optimal performance
    const test_configs = [_]struct { size: u32, iterations: u32 }{
        .{ .size = 512, .iterations = 20 },   // Small matrices
        .{ .size = 1024, .iterations = 10 },  // Medium matrices
        .{ .size = 2048, .iterations = 5 },   // Large matrices
        .{ .size = 4096, .iterations = 2 },   // Very large matrices
        .{ .size = 8192, .iterations = 1 },   // Extreme matrices (if memory allows)
    };

    var max_gflops: f64 = 0;
    var optimal_size: u32 = 0;

    for (test_configs) |config| {
        print("📊 Testing {}x{} matrices ({} iterations)...\n", .{ config.size, config.size, config.iterations });

        const gflops = benchmarkMatrixSize(allocator, config.size, config.iterations) catch |err| {
            print("❌ Failed to benchmark {}x{}: {}\n", .{ config.size, config.size, err });
            continue;
        };

        print("   Performance: {d:.1} GFLOPS\n", .{gflops});

        if (gflops > max_gflops) {
            max_gflops = gflops;
            optimal_size = config.size;
        }

        print("\n", .{});

        // Small delay between tests
        std.time.sleep(500_000_000); // 500ms
    }

    print("🎯 Performance Summary:\n", .{});
    print("======================\n", .{});
    print("   Peak Performance: {d:.1} GFLOPS\n", .{max_gflops});
    print("   Optimal Matrix Size: {}x{}\n", .{ optimal_size, optimal_size });

    // Compare to expectations
    const expected_gflops: f64 = 1200.0; // Conservative estimate for Ryzen 9 3900X
    const efficiency = (max_gflops / expected_gflops) * 100.0;

    print("   Efficiency: {d:.1}% of expected peak\n", .{efficiency});

    if (max_gflops >= 800.0) {
        print("   ✅ Excellent performance!\n", .{});
    } else if (max_gflops >= 400.0) {
        print("   ⚠️  Good performance, but room for improvement\n", .{});
    } else {
        print("   ❌ Performance below expectations\n", .{});
    }

    print("\n🔧 Optimization Tips:\n", .{});
    if (max_gflops < 800.0) {
        print("   • Consider installing Intel MKL for 20-30% better performance\n", .{});
        print("   • Verify CPU governor is set to 'performance'\n", .{});
        print("   • Check that all 24 threads are being utilized\n", .{});
        print("   • Monitor with: htop during benchmark execution\n", .{});
    }
    print("   • For production: Use matrices >= {}x{} for best performance\n", .{ optimal_size, optimal_size });
}
