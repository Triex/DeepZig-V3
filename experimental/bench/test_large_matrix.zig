// Test large matrix performance to verify 993 GFLOPS restoration
const std = @import("std");
const print = std.debug.print;
const Timer = std.time.Timer;

const deepseek_core = @import("deepseek_core");
const Blas = deepseek_core.blas.Blas;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🚀 Large Matrix Performance Test\n", .{});
    print("=================================\n\n", .{});

    var blas = try Blas.init(allocator);
    defer blas.deinit();

    print("✅ Backend: {}\n", .{blas.backend});
    print("✅ Expected: {d:.1} GFLOPS\n\n", .{blas.performance_info.peak_gflops});

    // Test the same sizes that achieved 993 GFLOPS
    const test_configs = [_]struct { size: u32, iterations: u32 }{
        .{ .size = 2048, .iterations = 3 },
        .{ .size = 4096, .iterations = 2 },
        .{ .size = 8192, .iterations = 1 },
    };

    var max_gflops: f64 = 0;

    for (test_configs) |config| {
        print("📊 Testing {}x{} matrices ({} iterations)...\n", .{ config.size, config.size, config.iterations });

        const matrix_a = try allocator.alloc(f32, config.size * config.size);
        const matrix_b = try allocator.alloc(f32, config.size * config.size);
        const matrix_c = try allocator.alloc(f32, config.size * config.size);
        defer allocator.free(matrix_a);
        defer allocator.free(matrix_b);
        defer allocator.free(matrix_c);

        // Fill with data
        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();
        for (matrix_a) |*val| val.* = random.float(f32);
        for (matrix_b) |*val| val.* = random.float(f32);

        // Warmup
        @memset(matrix_c, 0.0);
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = config.size, .n = config.size, .k = config.size });

        // Benchmark
        @memset(matrix_c, 0.0);
        var timer = try Timer.start();

        for (0..config.iterations) |_| {
            blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = config.size, .n = config.size, .k = config.size });
        }

        const elapsed_ns = timer.read();
        const ops = 2.0 * @as(f64, @floatFromInt(config.size)) * @as(f64, @floatFromInt(config.size)) * @as(f64, @floatFromInt(config.size)) * @as(f64, @floatFromInt(config.iterations));
        const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        const gflops = ops / elapsed_s / 1e9;

        print("   Performance: {d:.1} GFLOPS\n", .{gflops});

        if (gflops > max_gflops) {
            max_gflops = gflops;
        }

        // Small delay
        std.time.sleep(500_000_000); // 500ms
    }

    print("\n🎯 Results:\n", .{});
    print("===========\n", .{});
    print("   Peak Performance: {d:.1} GFLOPS\n", .{max_gflops});

    if (max_gflops >= 900.0) {
        print("   ✅ EXCELLENT! Back to high performance levels\n", .{});
    } else if (max_gflops >= 400.0) {
        print("   ⚠️  GOOD but not at original 993 GFLOPS level\n", .{});
        print("   This may be due to Zig 0.15.0-dev regression (~20%)\n", .{});
    } else {
        print("   ❌ Still below expectations\n", .{});
    }

    print("\n💡 Next Steps:\n", .{});
    if (max_gflops < 900.0) {
        print("   • Original performance was 993 GFLOPS\n", .{});
        print("   • Current performance: {d:.1} GFLOPS\n", .{max_gflops});
        print("   • Gap may be due to Zig compiler regression\n", .{});
        print("   • Consider GPU acceleration for 3000-5000 GFLOPS\n", .{});
    }
}
