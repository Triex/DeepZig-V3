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

    // Add module dependencies
    const deepseek_core = b.addModule("deepseek_core", .{
        .root_source_file = b.path("src/core/root.zig"),
    });
    exe.root_module.addImport("deepseek_core", deepseek_core);

    const web_layer = b.addModule("web_layer", .{
        .root_source_file = b.path("src/web/root.zig"),
    });
    web_layer.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("web_layer", web_layer);

    const cpu_backend = b.addModule("cpu_backend", .{
        .root_source_file = b.path("src/backends/cpu/root.zig"),
    });
    cpu_backend.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("cpu_backend", cpu_backend);

    const metal_backend = b.addModule("metal_backend", .{
        .root_source_file = b.path("src/backends/metal/root.zig"),
    });
    metal_backend.addImport("deepseek_core", deepseek_core);
    exe.root_module.addImport("metal_backend", metal_backend);

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
    const blas_bench_step = b.step("bench-blas", "Run BLAS-specific benchmarks");
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
    const train_medium_exe = b.addExecutable(.{
        .name = "train-medium",
        .root_source_file = b.path("src/train_medium.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply optimizations to training
    configureOptimizations(train_medium_exe, optimize);

    // Add modules to train_medium
    train_medium_exe.root_module.addImport("deepseek_core", deepseek_core);
    train_medium_exe.root_module.addImport("training", training_module);

    // Link BLAS for training
    configureBlas(train_medium_exe, target);

    b.installArtifact(train_medium_exe);

    // Main training step
    const train_medium_run_cmd = b.addRunArtifact(train_medium_exe);
    train_medium_run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        train_medium_run_cmd.addArgs(args);
    }

    const train_medium_step = b.step("train-medium", "Run native Zig training for the medium model");
    train_medium_step.dependOn(&train_medium_run_cmd.step);
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

/// Configure BLAS linking for the given compile step based on target platform
fn configureBlas(step: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    const target_os = target.result.os.tag;

    // Link libc for all targets since BLAS requires it
    step.linkLibC();

    switch (target_os) {
        .macos => {
            // Use Apple's Accelerate framework
            step.linkFramework("Accelerate");
            step.root_module.addCMacro("HAVE_ACCELERATE", "1");
        },
        .linux => {
            // Use OpenBLAS on Linux
            step.linkSystemLibrary("openblas");
            step.root_module.addCMacro("HAVE_OPENBLAS", "1");
        },
        .windows => {
            // Use OpenBLAS on Windows (if available)
            step.linkSystemLibrary("openblas");
            step.root_module.addCMacro("HAVE_OPENBLAS", "1");
        },
        else => {
            // Fallback to naive implementation
            step.root_module.addCMacro("HAVE_NAIVE_BLAS", "1");
        },
    }
}
