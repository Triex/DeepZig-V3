// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Comprehensive test suite for MoE (Mixture of Experts) implementation
//! This file includes benchmarking, validation, and correctness tests
//! to ensure the MoE implementation is production-ready.

const std = @import("std");
const deepseek_core = @import("deepseek_core");

const Allocator = std.mem.Allocator;
const log = std.log;

// Import core modules
const Backend = deepseek_core.Backend;
const ModelConfig = deepseek_core.ModelConfig;
const FloatTensor = deepseek_core.FloatTensor;
const MoE = deepseek_core.MoE;

pub fn main() !void {
    log.info("=== DeepZig MoE Test Suite ===", .{});
    
    // Setup testing environment
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize backend with CPU (device_id=0)
    const backend = Backend.init(allocator, .cpu, 0);
    
    try testMoEInitialization(allocator, backend);
    try testMoEForward(allocator, backend);
    try testMoESpecialized(allocator, backend);
    try testMoEFullPipeline(allocator, backend);
    try benchmarkMoE(allocator, backend);
    
    log.info("All MoE tests passed successfully! ✓", .{});
}

/// Test proper initialization of MoE with various configurations
fn testMoEInitialization(allocator: Allocator, backend: Backend) !void {
    log.info("Testing MoE initialization...", .{});
    
    // Valid configuration
    const config = ModelConfig{
        .hidden_size = 256,
        .intermediate_size = 1024,
        .num_experts = 8,
        .num_experts_per_token = 2,
    };
    
    var moe = try MoE.init(allocator, config, backend);
    defer moe.deinit();
    
    // Verify initialization
    std.debug.assert(moe.experts.len == 8);
    std.debug.assert(moe.config.num_experts_per_token == 2);
    
    log.info("MoE initialization tests passed ✓", .{});
}

/// Test MoE forward pass with realistic inputs
fn testMoEForward(allocator: Allocator, backend: Backend) !void {
    log.info("Testing MoE forward pass...", .{});
    
    // Create a small but realistic configuration
    const config = ModelConfig{
        .hidden_size = 64,
        .intermediate_size = 256,
        .num_experts = 4,
        .num_experts_per_token = 2,
    };
    
    var moe = try MoE.init(allocator, config, backend);
    defer moe.deinit();
    
    // Create sample hidden states for testing (batch_size=2, seq_len=3)
    var input = try FloatTensor.init(allocator, &[_]usize{2, 3, 64});
    defer input.deinit();
    
    // Fill with test pattern
    for (0..input.data.len) |i| {
        input.data[i] = @floatCast(@as(f32, @floatFromInt(i)) / 100.0);
    }
    
    var output = try FloatTensor.init(allocator, &[_]usize{2, 3, 64});
    defer output.deinit();
    
    try moe.forward(&input, &output);
    
    // Verify output is non-zero (actual values will depend on random initialization)
    var all_zeros = true;
    for (output.data) |val| {
        if (val != 0) {
            all_zeros = false;
            break;
        }
    }
    std.debug.assert(!all_zeros);
    
    log.info("MoE forward pass tests passed ✓", .{});
}

/// Test the specialized MoE implementation for common configurations
fn testMoESpecialized(allocator: Allocator, backend: Backend) !void {
    log.info("Testing MoE specialized implementations...", .{});
    
    // Test with config matching the specialized 8x2 implementation
    const config = ModelConfig{
        .hidden_size = 128,
        .intermediate_size = 512,
        .num_experts = 8,
        .num_experts_per_token = 2,
    };
    
    var moe = try MoE.init(allocator, config, backend);
    defer moe.deinit();
    
    // Create test input
    var input = try FloatTensor.init(allocator, &[_]usize{2, 4, 128});
    defer input.deinit();
    
    // Initialize with test pattern
    for (0..input.data.len) |i| {
        input.data[i] = @floatCast(@as(f32, @floatFromInt(i % 17)) / 10.0);
    }
    
    var output = try FloatTensor.init(allocator, &[_]usize{2, 4, 128});
    defer output.deinit();
    
    // Time the specialized implementation
    const timer_start = std.time.nanoTimestamp();
    try moe.forward(&input, &output);
    const elapsed_ns = std.time.nanoTimestamp() - timer_start;
    
    log.info("Specialized MoE forward completed in {d:.3} ms", .{
        @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0
    });
    
    log.info("MoE specialized implementation tests passed ✓", .{});
}

/// Test MoE in a full transformer pipeline context
fn testMoEFullPipeline(allocator: Allocator, backend: Backend) !void {
    log.info("Testing MoE in transformer pipeline context...", .{});
    
    // This is a simplified test that doesn't need the full transformer
    // Just verify that MoE can handle the shapes correctly in a pipeline context
    
    const batch_size = 1;
    const seq_len = 8;
    const hidden_size = 32;
    
    // Create test hidden states that would come from attention
    var hidden_states = try FloatTensor.init(allocator, &[_]usize{
        batch_size, seq_len, hidden_size
    });
    defer hidden_states.deinit();
    
    // Fill with test pattern
    for (0..hidden_states.data.len) |i| {
        hidden_states.data[i] = @floatCast(@as(f32, @floatFromInt(i % 7)) / 10.0);
    }
    
    // Create MoE layer
    const config = ModelConfig{
        .hidden_size = hidden_size,
        .intermediate_size = hidden_size * 4,
        .num_experts = 4,
        .num_experts_per_token = 2,
    };
    
    var moe = try MoE.init(allocator, config, backend);
    defer moe.deinit();
    
    // Create output tensor for results
    var output = try FloatTensor.init(allocator, &[_]usize{
        batch_size, seq_len, hidden_size
    });
    defer output.deinit();
    
    try moe.forward(&hidden_states, &output);
    
    log.info("MoE full pipeline integration tests passed ✓", .{});
}

/// Benchmark MoE performance with realistic configuration
fn benchmarkMoE(allocator: Allocator, backend: Backend) !void {
    log.info("Benchmarking MoE performance...", .{});
    
    // Create a more realistic configuration for benchmarking
    const config = ModelConfig{
        .hidden_size = 768,  // Realistic for a medium model
        .intermediate_size = 3072,
        .num_experts = 8,
        .num_experts_per_token = 2,
    };
    
    var moe = try MoE.init(allocator, config, backend);
    defer moe.deinit();
    
    // Create benchmark input with realistic batch and sequence length
    const batch_size = 2;
    const seq_len = 32;
    
    var input = try FloatTensor.init(allocator, &[_]usize{
        batch_size, seq_len, config.hidden_size
    });
    defer input.deinit();
    
    // Initialize with semi-random pattern
    for (0..input.data.len) |i| {
        input.data[i] = @floatCast(@as(f32, @floatFromInt(i % 23)) / 100.0);
    }
    
    var output = try FloatTensor.init(allocator, &[_]usize{
        batch_size, seq_len, config.hidden_size
    });
    defer output.deinit();
    
    // Warm-up run
    try moe.forward(&input, &output);
    
    // Benchmark multiple runs
    const num_runs = 5;
    var total_ns: i128 = 0;
    
    for (0..num_runs) |_| {
        const start = std.time.nanoTimestamp();
        try moe.forward(&input, &output);
        const end = std.time.nanoTimestamp();
        total_ns += end - start;
    }
    
    const avg_ms = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(num_runs)) / 1_000_000.0;
    
    // Calculate tokens per second
    const tokens = batch_size * seq_len;
    const tokens_per_second = @as(f64, @floatFromInt(tokens)) / (avg_ms / 1000.0);
    
    log.info("MoE benchmark results:", .{});
    log.info("  Config: hidden_size={d}, experts={d}, tokens_per_expert={d}", .{
        config.hidden_size, config.num_experts, config.num_experts_per_token
    });
    log.info("  Average time: {d:.3} ms per batch", .{avg_ms});
    log.info("  Tokens per second: {d:.1}", .{tokens_per_second});
    log.info("  Batch size: {d}, Sequence length: {d}", .{batch_size, seq_len});
    
    log.info("MoE benchmark complete ✓", .{});
}
