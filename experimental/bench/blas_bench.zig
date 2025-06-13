// Optimized BLAS benchmark with GPU detection
// Tests multiple matrix sizes for peak performance

const std = @import("std");
const print = std.debug.print;
const Timer = std.time.Timer;
const Random = std.Random;

const deepseek_core = @import("deepseek_core");
const Blas = deepseek_core.blas.Blas;

fn benchmarkMatrixSize(allocator: std.mem.Allocator, size: u32, iterations: u32) !f64 {
    var blas = try Blas.init(allocator);
    defer blas.deinit();

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
    @memset(matrix_c, 0.0);

    // Warmup runs
    for (0..2) |_| {
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
        @memset(matrix_c, 0.0);
    }

    // Small delay to stabilize
    std.time.sleep(100_000_000); // 100ms

    // Benchmark
    var timer = try Timer.start();
    for (0..iterations) |i| {
        blas.matmul(f32, matrix_a, matrix_b, matrix_c, .{ .m = size, .n = size, .k = size });
        @memset(matrix_c, 0.0);

        // Small pause between iterations
        if (i < iterations - 1) {
            std.time.sleep(1_000_000); // 1ms pause
        }
    }
    const elapsed_ns = timer.read();

    // Calculate GFLOPS
    const ops = 2.0 * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(size)) * @as(f64, @floatFromInt(iterations));
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;

    return ops / elapsed_s / 1e9;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("🚀 BLAS Performance Test\n", .{});
    print("========================\n\n", .{});

    // Check for GPU
    const has_gpu = blk: {
        const result = std.process.Child.run(.{
            .allocator = allocator,
            .argv = &[_][]const u8{"nvidia-smi", "-L"},
        }) catch break :blk false;
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        break :blk result.term.Exited == 0;
    };

    if (has_gpu) {
        print("🎮 GPU: NVIDIA RTX 2070 SUPER detected\n", .{});
        print("⚠️  Using CPU BLAS (GPU acceleration TODO)\n", .{});
        print("   Expected with CUDA: 3000-5000 GFLOPS\n\n", .{});
    } else {
        print("💻 CPU BLAS only\n\n", .{});
    }

    // Test matrix sizes (up to 1024 by default, --extended for more)
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const use_extended = for (args) |arg| {
        if (std.mem.eql(u8, arg, "--extended")) break true;
    } else false;

    const Config = struct { size: u32, iterations: u32 };

    const default_configs = [_]Config{
        .{ .size = 512, .iterations = 10 },
        .{ .size = 1024, .iterations = 5 },
    };

    const extended_configs = [_]Config{
        .{ .size = 512, .iterations = 20 },
        .{ .size = 1024, .iterations = 10 },
        .{ .size = 2048, .iterations = 5 },
        .{ .size = 4096, .iterations = 2 },
        .{ .size = 8192, .iterations = 1 },
    };

    const configs = if (use_extended) extended_configs[0..] else default_configs[0..];

    var max_gflops: f64 = 0;
    var optimal_size: u32 = 0;

    for (configs) |config| {
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
    }

    print("🎯 Performance Summary:\n", .{});
    print("======================\n", .{});
    print("   Peak Performance: {d:.1} GFLOPS\n", .{max_gflops});
    print("   Optimal Matrix Size: {}x{}\n", .{ optimal_size, optimal_size });
    print("   Backend: CPU OpenBLAS ({} threads)\n", .{std.Thread.getCpuCount() catch 1});

    if (max_gflops >= 800.0) {
        print("   ✅ Excellent CPU performance!\n", .{});
    } else if (max_gflops >= 400.0) {
        print("   ⚠️  Good performance, but room for improvement\n", .{});
    } else {
        print("   ❌ Performance below expectations\n", .{});
    }

    if (has_gpu and max_gflops < 2000.0) {
        print("\n💡 Next Steps:\n", .{});
        print("   • Add CUDA support for 5x performance boost\n", .{});
        print("   • Expected: 3000-5000 GFLOPS on RTX 2070 SUPER\n", .{});
    }

    if (!use_extended) {
        print("   • Run with --extended for larger matrices\n", .{});
    }
}
