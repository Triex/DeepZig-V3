// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable
    const exe = b.addExecutable(.{
        .name = "deepseek-v3-zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply release mode optimizations
    configureOptimizations(exe, optimize);

    // BLAS library configuration based on target platform
    configureBlas(exe, target);

    // Create the main module for DeepSeek core
    const deepseek_core = b.addModule("deepseek_core", .{
        .root_source_file = b.path("src/core/root.zig"),
    });

    // Create backend modules
    const cpu_backend = b.addModule("cpu_backend", .{
        .root_source_file = b.path("src/backends/cpu/root.zig"),
    });

    const cuda_backend = b.addModule("cuda_backend", .{
        .root_source_file = b.path("src/backends/cuda/root.zig"),
    });

    const metal_backend = b.addModule("metal_backend", .{
        .root_source_file = b.path("src/backends/metal/root.zig"),
    });

    // Add backend dependencies to the core module
    deepseek_core.addImport("cpu_backend", cpu_backend);
    deepseek_core.addImport("cuda_backend", cuda_backend);
    deepseek_core.addImport("metal_backend", metal_backend);

    // Backend modules also need deepseek_core for shared types
    cpu_backend.addImport("deepseek_core", deepseek_core);
    cuda_backend.addImport("deepseek_core", deepseek_core);
    metal_backend.addImport("deepseek_core", deepseek_core);

    // Add module dependencies to main executable
    exe.root_module.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("cpu_backend", cpu_backend);
    exe.root_module.addImport("cuda_backend", cuda_backend);
    exe.root_module.addImport("metal_backend", metal_backend);

    // Add module dependencies
    const web_layer = b.addModule("web_layer", .{
        .root_source_file = b.path("src/web/root.zig"),
    });
    web_layer.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("web_layer", web_layer);

    // Add Metal framework for macOS
    if (target.result.os.tag == .macos) {
        exe.linkFramework("Metal");
        exe.linkFramework("Foundation");
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Benchmarks
    const benchmark_exe = b.addExecutable(.{
        .name = "deepseek-v3-benchmark",
        .root_source_file = b.path("bench/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to benchmarks
    configureOptimizations(benchmark_exe, optimize);

    // Add the same modules to benchmark
    benchmark_exe.root_module.addImport("deepseek_core", deepseek_core);

    const cpu_backend_bench = b.addModule("cpu_backend", .{
        .root_source_file = b.path("src/backends/cpu/root.zig"),
    });
    cpu_backend_bench.addImport("deepseek_core", deepseek_core);
    benchmark_exe.root_module.addImport("cpu_backend", cpu_backend_bench);

    // Configure BLAS for benchmarks too
    configureBlas(benchmark_exe, target);

    // Add Metal framework for benchmarks on macOS
    if (target.result.os.tag == .macos) {
        benchmark_exe.linkFramework("Metal");
        benchmark_exe.linkFramework("Foundation");
    }

    b.installArtifact(benchmark_exe);

    const benchmark_run_cmd = b.addRunArtifact(benchmark_exe);
    benchmark_run_cmd.step.dependOn(b.getInstallStep());

    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&benchmark_run_cmd.step);

    // BLAS benchmarks specifically
    const blas_bench_exe = b.addExecutable(.{
        .name = "blas-benchmark",
        .root_source_file = b.path("bench/blas_bench.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to BLAS benchmark
    configureOptimizations(blas_bench_exe, optimize);

    blas_bench_exe.root_module.addImport("deepseek_core", deepseek_core);
    configureBlas(blas_bench_exe, target);

    const blas_bench_run = b.addRunArtifact(blas_bench_exe);
    const blas_bench_step = b.step("bench-blas", "Run BLAS performance benchmarks");
    blas_bench_step.dependOn(&blas_bench_run.step);

    // Ceate a validation test executable
    const validation_exe = b.addExecutable(.{
        .name = "validation-test",
        .root_source_file = b.path("src/test_validation.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to validation
    configureOptimizations(validation_exe, optimize);

    // Add core modules
    validation_exe.root_module.addImport("deepseek_core", deepseek_core);

    // Link BLAS for validation
    validation_exe.linkLibC();
    if (builtin.os.tag == .macos) {
        validation_exe.linkFramework("Accelerate");
    } else if (builtin.os.tag == .linux) {
        validation_exe.linkSystemLibrary("openblas");
    }

    b.installArtifact(validation_exe);

    // Main validate run step (now using enhanced validation)
    const validation_run_cmd = b.addRunArtifact(validation_exe);
    validation_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        validation_run_cmd.addArgs(args);
    }

    const validation_run_step = b.step("validate", "Run comprehensive validation suite");
    validation_run_step.dependOn(&validation_run_cmd.step);

    // Add dedicated MoE test executable
    const moe_test_exe = b.addExecutable(.{
        .name = "moe-test",
        .root_source_file = b.path("src/test_moe.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to MoE tests
    configureOptimizations(moe_test_exe, optimize);

    // Add core modules
    moe_test_exe.root_module.addImport("deepseek_core", deepseek_core);

    // Link BLAS for MoE tests
    configureBlas(moe_test_exe, target);

    b.installArtifact(moe_test_exe);

    // Main MoE test step
    const moe_test_run_cmd = b.addRunArtifact(moe_test_exe);
    moe_test_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        moe_test_run_cmd.addArgs(args);
    }

    const moe_test_step = b.step("test-moe", "Run Mixture of Experts (MoE) test suite");
    moe_test_step.dependOn(&moe_test_run_cmd.step);

    // Create training module
    const training_module = b.addModule("training", .{
        .root_source_file = b.path("src/training/root.zig"),
    });
    // Training module depends on core
    training_module.addImport("deepseek_core", deepseek_core);

    // Add native Zig training executable
    const train_exe = b.addExecutable(.{
        .name = "train-model",
        .root_source_file = b.path("src/train.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to training
    configureOptimizations(train_exe, optimize);

    // Add modules to train
    train_exe.root_module.addImport("deepseek_core", deepseek_core);
    train_exe.root_module.addImport("training", training_module);

    // Link BLAS for training
    configureBlas(train_exe, target);

    b.installArtifact(train_exe);

    // Main training step
    const train_run_cmd = b.addRunArtifact(train_exe);
    train_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        train_run_cmd.addArgs(args);
    }

    const train_step = b.step("train-model", "Run native Zig training for the medium model");
    train_step.dependOn(&train_run_cmd.step);

    // Add model size specific training commands
    addTrainingCommands(b, train_exe);

    // Add diagnostic executable for performance debugging
    const diagnostic_exe = b.addExecutable(.{
        .name = "diagnostic",
        .root_source_file = b.path("bench/diagnostic.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to diagnostic
    configureOptimizations(diagnostic_exe, optimize);

    // Add core modules to diagnostic
    diagnostic_exe.root_module.addImport("deepseek_core", deepseek_core);

    // Link BLAS for diagnostic
    configureBlas(diagnostic_exe, target);

    b.installArtifact(diagnostic_exe);

    // Main diagnostic step
    const diagnostic_run_cmd = b.addRunArtifact(diagnostic_exe);
    diagnostic_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        diagnostic_run_cmd.addArgs(args);
    }

    const diagnostic_step = b.step("diagnostic", "Run performance diagnostic to identify issues");
    diagnostic_step.dependOn(&diagnostic_run_cmd.step);

    // Add large matrix test executable
    const large_matrix_exe = b.addExecutable(.{
        .name = "test-large-matrix",
        .root_source_file = b.path("bench/test_large_matrix.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to large matrix test
    configureOptimizations(large_matrix_exe, optimize);

    // Add core modules to large matrix test
    large_matrix_exe.root_module.addImport("deepseek_core", deepseek_core);

    // Link BLAS for large matrix test
    configureBlas(large_matrix_exe, target);

    b.installArtifact(large_matrix_exe);

    // Main large matrix test step
    const large_matrix_run_cmd = b.addRunArtifact(large_matrix_exe);
    large_matrix_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        large_matrix_run_cmd.addArgs(args);
    }

    const large_matrix_step = b.step("test-large", "Run large matrix performance test");
    large_matrix_step.dependOn(&large_matrix_run_cmd.step);

    // Add CUDA GPU benchmark executable
    const cuda_bench_exe = b.addExecutable(.{
        .name = "cuda-gpu-benchmark",
        .root_source_file = b.path("bench/cuda_bench.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to CUDA benchmark
    configureOptimizations(cuda_bench_exe, optimize);

    // Add core modules to CUDA benchmark
    cuda_bench_exe.root_module.addImport("deepseek_core", deepseek_core);

    // Link BLAS for CUDA benchmark
    configureBlas(cuda_bench_exe, target);

    b.installArtifact(cuda_bench_exe);

    // Main CUDA benchmark step
    const cuda_bench_run_cmd = b.addRunArtifact(cuda_bench_exe);
    cuda_bench_run_cmd.step.dependOn(b.getInstallStep());

    const cuda_bench_step = b.step("bench-cuda", "Run CUDA GPU benchmark");
    cuda_bench_step.dependOn(&cuda_bench_run_cmd.step);
}

/// Configure optimizations for maximum performance
fn configureOptimizations(step: *std.Build.Step.Compile, optimize: std.builtin.OptimizeMode) void {
    switch (optimize) {
        .ReleaseFast, .ReleaseSmall => {
            // Maximum performance optimizations
            step.root_module.addCMacro("NDEBUG", "1");
            // Disable LTO for now to avoid build issues
            // step.want_lto = true;

            // Enable advanced vectorization and optimizations
            step.root_module.addCMacro("ZIG_FAST_MATH", "1");
            step.root_module.addCMacro("ZIG_RELEASE_MODE", "1");
        },
        .ReleaseSafe => {
            // Balanced optimizations with safety
            step.root_module.addCMacro("ZIG_RELEASE_SAFE", "1");
            // step.want_lto = true;
        },
        .Debug => {
            // Debug mode - enable debugging features
            step.root_module.addCMacro("ZIG_DEBUG_MODE", "1");
        },
    }
}

/// Add training commands for different model sizes
fn addTrainingCommands(b: *std.Build, train_exe: *std.Build.Step.Compile) void {
    // Test model - ultra fast for testing (< 1 minute)
    const train_test_cmd = b.addRunArtifact(train_exe);
    train_test_cmd.step.dependOn(b.getInstallStep());
    train_test_cmd.addArgs(&[_][]const u8{ "--model-size", "test", "--max-samples", "2000", "--epochs", "4" });

    const train_test_step = b.step("train-test", "🧪 Train test model (1M params, REAL learning, fast results)");
    train_test_step.dependOn(&train_test_cmd.step);

    // Small model - quick validation (few minutes)
    const train_small_cmd = b.addRunArtifact(train_exe);
    train_small_cmd.step.dependOn(b.getInstallStep());
    train_small_cmd.addArgs(&[_][]const u8{ "--model-size", "small", "--max-samples", "1000", "--epochs", "3" });

    const train_small_step = b.step("train-small", "🏃 Train small model (~5M params, ~5min)");
    train_small_step.dependOn(&train_small_cmd.step);

    // Conversational model - designed for chat and tool calling (recommended)
    const train_conversational_cmd = b.addRunArtifact(train_exe);
    train_conversational_cmd.step.dependOn(b.getInstallStep());
    train_conversational_cmd.addArgs(&[_][]const u8{ "--model-size", "conversational", "--max-samples", "20000", "--epochs", "8" });

    const train_conversational_step = b.step("train-conversational", "💬 Train conversational AI model (~60M params, chat + tool calling)");
    train_conversational_step.dependOn(&train_conversational_cmd.step);

    // Medium model - balanced training
    const train_medium_cmd = b.addRunArtifact(train_exe);
    train_medium_cmd.step.dependOn(b.getInstallStep());
    train_medium_cmd.addArgs(&[_][]const u8{ "--model-size", "medium", "--max-samples", "10000", "--epochs", "3" });

    const train_medium_step = b.step("train-medium", "⚡ Train medium model (~50M params, ~25min)");
    train_medium_step.dependOn(&train_medium_cmd.step);

    // Large model - production quality (longer training)
    const train_large_cmd = b.addRunArtifact(train_exe);
    train_large_cmd.step.dependOn(b.getInstallStep());
    train_large_cmd.addArgs(&[_][]const u8{ "--model-size", "large", "--max-samples", "20000", "--epochs", "5" });

    const train_large_step = b.step("train-large", "🚀 Train large model (~125M params, hours)");
    train_large_step.dependOn(&train_large_cmd.step);

    // Add LoRA variants for parameter-efficient training
    const train_small_lora_cmd = b.addRunArtifact(train_exe);
    train_small_lora_cmd.step.dependOn(b.getInstallStep());
    train_small_lora_cmd.addArgs(&[_][]const u8{ "--model-size", "small", "--use-lora", "--lora-rank", "16", "--epochs", "5" });

    const train_small_lora_step = b.step("train-small-lora", "🔧 Train small model with LoRA (parameter efficient)");
    train_small_lora_step.dependOn(&train_small_lora_cmd.step);

    const train_medium_lora_cmd = b.addRunArtifact(train_exe);
    train_medium_lora_cmd.step.dependOn(b.getInstallStep());
    train_medium_lora_cmd.addArgs(&[_][]const u8{ "--model-size", "medium", "--use-lora", "--lora-rank", "32", "--epochs", "5" });

    const train_medium_lora_step = b.step("train-medium-lora", "🔧 Train medium model with LoRA (parameter efficient)");
    train_medium_lora_step.dependOn(&train_medium_lora_cmd.step);
}

/// Configure BLAS linking (CUDA cuBLAS prioritized, then OpenBLAS, Accelerate, Intel MKL, etc.)
fn configureBlas(exe: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    switch (target.result.os.tag) {
        .macos => {
            // Use Accelerate framework on macOS
            exe.linkFramework("Accelerate");
        },
        .linux, .windows => {
            exe.linkLibC();

            // CUDA Detection and Linking (auto-detect CUDA availability)
            const cuda_available = detectCuda();
            if (cuda_available) {
                std.log.info("🎮 CUDA detected - enabling GPU acceleration!", .{});

                // Link CUDA libraries for GPU acceleration
                exe.linkSystemLibrary("cuda");      // CUDA Driver API (for cuInit, cuDeviceGetCount, etc.)
                exe.linkSystemLibrary("cudart");    // CUDA Runtime API
                exe.linkSystemLibrary("cublas");    // cuBLAS for matrix operations
                exe.linkSystemLibrary("cublasLt");  // cuBLAS LT (lightweight)

                // Add CUDA include paths
                exe.addIncludePath(.{ .cwd_relative = "/usr/include" });
                exe.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

                // Enable CUDA compilation
                exe.root_module.addCMacro("CUDA_ENABLED", "1");
                exe.root_module.addCMacro("GPU_ACCELERATION", "1");

                std.log.info("✅ CUDA libraries linked successfully", .{});
            } else {
                std.log.info("⚠️ CUDA not available - using CPU-only mode", .{});
                exe.root_module.addCMacro("CUDA_COMPILATION_DISABLED", "1");
                exe.root_module.addCMacro("CPU_ONLY_MODE", "1");
            }

            // Always add OpenBLAS as fallback for CPU operations
            exe.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu/openblas-pthread" });
            exe.addIncludePath(.{ .cwd_relative = "/usr/include/x86_64-linux-gnu/openblas-pthread" });
            exe.linkSystemLibrary("openblas");
        },
        else => {
            // Use CBLAS fallback for other platforms
            exe.linkSystemLibrary("cblas");
            exe.linkLibC();
        },
    }
}

/// Detect if CUDA is available on the system
fn detectCuda() bool {
    // Check for CUDA compiler
    const nvcc_result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{ "which", "nvcc" },
    }) catch return false;
    defer std.heap.page_allocator.free(nvcc_result.stdout);
    defer std.heap.page_allocator.free(nvcc_result.stderr);

    if (nvcc_result.term.Exited != 0) {
        return false;
    }

    // Check for cuBLAS library
    const cublas_check = std.fs.cwd().access("/usr/lib/x86_64-linux-gnu/libcublas.so", .{}) catch return false;
    _ = cublas_check;

    // Check for CUDA runtime library
    const cudart_check = std.fs.cwd().access("/usr/lib/x86_64-linux-gnu/libcudart.so", .{}) catch return false;
    _ = cudart_check;

    return true;
}
