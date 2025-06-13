// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Optimizer implementation for DeepZig V3
//! Optimized for AMD Ryzen 9 3900X (24 cores) + NVIDIA RTX 2070 SUPER
//!
//! Features:
//! - Multi-threaded AdamW optimizer with SIMD acceleration
//! - CUDA Tensor Core support for mixed precision
//! - AVX2-accelerated vector operations
//! - Memory-efficient gradient accumulation
//! - Zero-copy GPU-CPU transfers where possible

const std = @import("std");
const math = std.math;
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const backend_detection = @import("backend_detection.zig");

// Use the main core BLAS implementation instead of redundant cuda_backend
const deepseek_core = @import("deepseek_core");
const Blas = deepseek_core.blas.Blas;

/// Optimizer configuration optimized for high-end hardware
pub const OptimizerConfig = struct {
    learning_rate: f32 = 1e-4,
    weight_decay: f32 = 0.01,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    use_mixed_precision: bool = true,
    gradient_clip_norm: f32 = 1.0,

    // Hardware-specific optimizations
    use_cuda: bool = false,
    num_worker_threads: u32 = 20,
    use_avx2_simd: bool = true,
    parameter_partitioning: bool = true,
    gpu_memory_fraction: f32 = 0.75, // Use 75% of GPU memory
};

/// Thread worker for parallel gradient updates
const OptimizerWorker = struct {
    thread: ?std.Thread = null,
    start_idx: usize,
    end_idx: usize,
    params: []f32,
    gradients: []f32,
    momentum: []f32,
    velocity: []f32,
    config: OptimizerConfig,
    step_count: u64,
    learning_rate: f32,
    finished: std.atomic.Value(bool),
    thread_running: std.atomic.Value(bool),

    fn workerThread(self: *OptimizerWorker) void {
        defer self.thread_running.store(false, .seq_cst);
        self.updateParametersRange();
        self.finished.store(true, .seq_cst);
    }

    fn updateParametersRange(self: *OptimizerWorker) void {
        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;
        const epsilon = self.config.epsilon;
        const weight_decay = self.config.weight_decay;

        // Bias correction
        const bias_correction1 = 1.0 - math.pow(f32, beta1, @floatFromInt(self.step_count));
        const bias_correction2 = 1.0 - math.pow(f32, beta2, @floatFromInt(self.step_count));
        const corrected_lr = self.learning_rate * math.sqrt(bias_correction2) / bias_correction1;

        // Process parameters in chunks for better cache locality
        const chunk_size = 16; // Process 16 parameters at a time for SIMD
        var i = self.start_idx;

        while (i < self.end_idx) {
            const end_chunk = @min(i + chunk_size, self.end_idx);

            if (self.config.use_avx2_simd and builtin.cpu.arch == .x86_64) {
                self.updateParametersAVX2(i, end_chunk, corrected_lr, beta1, beta2, epsilon, weight_decay);
            } else {
                self.updateParametersScalar(i, end_chunk, corrected_lr, beta1, beta2, epsilon, weight_decay);
            }

            i = end_chunk;
        }
    }

    fn updateParametersAVX2(self: *OptimizerWorker, start: usize, end: usize,
                           lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) void {
        // AVX2 SIMD implementation for AMD processors
        // Process 8 f32 values at once
        var i = start;
        const simd_end = start + ((end - start) / 8) * 8;

        const beta1_vec: @Vector(8, f32) = @splat(beta1);
        const beta2_vec: @Vector(8, f32) = @splat(beta2);
        const one_minus_beta1: @Vector(8, f32) = @splat(1.0 - beta1);
        const one_minus_beta2: @Vector(8, f32) = @splat(1.0 - beta2);
        const lr_vec: @Vector(8, f32) = @splat(lr);
        const epsilon_vec: @Vector(8, f32) = @splat(epsilon);
        const weight_decay_vec: @Vector(8, f32) = @splat(weight_decay);

        while (i < simd_end) {
            // Load 8 values at once
            const grad_vec: @Vector(8, f32) = self.gradients[i..i+8][0..8].*;
            const param_vec: @Vector(8, f32) = self.params[i..i+8][0..8].*;
            var momentum_vec: @Vector(8, f32) = self.momentum[i..i+8][0..8].*;
            var velocity_vec: @Vector(8, f32) = self.velocity[i..i+8][0..8].*;

            // Add weight decay to gradients
            const grad_with_decay = grad_vec + weight_decay_vec * param_vec;

            // Update momentum and velocity
            momentum_vec = beta1_vec * momentum_vec + one_minus_beta1 * grad_with_decay;
            velocity_vec = beta2_vec * velocity_vec + one_minus_beta2 * grad_with_decay * grad_with_decay;

            // Compute parameter update
            const sqrt_velocity = @sqrt(velocity_vec + epsilon_vec);
            const param_update = lr_vec * momentum_vec / sqrt_velocity;
            const new_params = param_vec - param_update;

            // Store results
            const new_params_array: [8]f32 = new_params;
            const momentum_array: [8]f32 = momentum_vec;
            const velocity_array: [8]f32 = velocity_vec;
            @memcpy(self.params[i..i+8], &new_params_array);
            @memcpy(self.momentum[i..i+8], &momentum_array);
            @memcpy(self.velocity[i..i+8], &velocity_array);

            i += 8;
        }

        // Handle remaining elements
        self.updateParametersScalar(i, end, lr, beta1, beta2, epsilon, weight_decay);
    }

    fn updateParametersScalar(self: *OptimizerWorker, start: usize, end: usize,
                             lr: f32, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32) void {
        for (start..end) |i| {
            const grad = self.gradients[i] + weight_decay * self.params[i];

            // Update momentum and velocity
            self.momentum[i] = beta1 * self.momentum[i] + (1.0 - beta1) * grad;
            self.velocity[i] = beta2 * self.velocity[i] + (1.0 - beta2) * grad * grad;

            // Update parameter
            const update = lr * self.momentum[i] / (math.sqrt(self.velocity[i]) + epsilon);
            self.params[i] -= update;
        }
    }
};

/// High-performance AdamW optimizer
pub const Optimizer = struct {
    allocator: Allocator,
    config: OptimizerConfig,

    // Optimizer state
    step_count: u64 = 0,
    param_count: usize = 0,

    // CPU storage
    params: ?[]f32 = null,
    gradients: ?[]f32 = null,
    momentum: ?[]f32 = null,
    velocity: ?[]f32 = null,

    // Mixed precision support
    loss_scale: f32 = 32768.0,
    fp16_gradients: ?[]f16 = null,

    // Multi-threading with improved lifecycle management
    workers: ?[]OptimizerWorker = null,
    threads_active: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    is_initialized: bool = false,

    pub fn init(allocator: Allocator, config: OptimizerConfig) !Optimizer {
        const optimizer = Optimizer{
            .allocator = allocator,
            .config = config,
        };

        return optimizer;
    }

    pub fn deinit(self: *Optimizer) void {
        // Prevent double-free
        if (!self.is_initialized) {
            return;
        }

        // Safely cleanup threads first with timeout
        self.cleanupThreads();

        // Defensive memory cleanup with null checks
        if (self.params) |params| {
            self.allocator.free(params);
            self.params = null;
        }
        if (self.gradients) |gradients| {
            self.allocator.free(gradients);
            self.gradients = null;
        }
        if (self.momentum) |momentum| {
            self.allocator.free(momentum);
            self.momentum = null;
        }
        if (self.velocity) |velocity| {
            self.allocator.free(velocity);
            self.velocity = null;
        }
        if (self.fp16_gradients) |fp16_gradients| {
            self.allocator.free(fp16_gradients);
            self.fp16_gradients = null;
        }

        if (self.workers) |workers| {
            self.allocator.free(workers);
            self.workers = null;
        }

        // Mark as not initialized
        self.is_initialized = false;
    }

    fn cleanupThreads(self: *Optimizer) void {
        if (self.workers) |workers| {
            // Wait for any active threads to complete with timeout
            if (self.threads_active.load(.seq_cst)) {
                for (workers) |*worker| {
                    // Wait for worker to finish with timeout
                    var timeout_count: u32 = 0;
                    while (worker.thread_running.load(.seq_cst) and timeout_count < 10000) { // 1 second timeout
                        std.time.sleep(100_000); // 100 microseconds
                        timeout_count += 1;
                    }

                    // Force thread to stop if timeout
                    if (timeout_count >= 10000) {
                        worker.thread_running.store(false, .seq_cst);
                        worker.finished.store(true, .seq_cst);
                    }

                    // Join thread if it was spawned
                    if (worker.thread) |thread| {
                        thread.join();
                        worker.thread = null;
                    }
                }
                self.threads_active.store(false, .seq_cst);
            }

            // Clear all worker state to prevent use-after-free
            for (workers) |*worker| {
                worker.thread = null;
                worker.thread_running.store(false, .seq_cst);
                worker.finished.store(true, .seq_cst);
            }
        }
    }

    /// Initialize optimizer state for given parameter count
    pub fn initializeState(self: *Optimizer, param_count: usize) !void {
        // Prevent double initialization
        if (self.is_initialized) {
            return error.AlreadyInitialized;
        }

        self.param_count = param_count;

        // Allocate CPU memory
        self.params = try self.allocator.alloc(f32, param_count);
        self.gradients = try self.allocator.alloc(f32, param_count);
        self.momentum = try self.allocator.alloc(f32, param_count);
        self.velocity = try self.allocator.alloc(f32, param_count);

        // Initialize to zero
        @memset(self.params.?, 0.0);
        @memset(self.gradients.?, 0.0);
        @memset(self.momentum.?, 0.0);
        @memset(self.velocity.?, 0.0);

        // Mixed precision buffer
        if (self.config.use_mixed_precision) {
            self.fp16_gradients = try self.allocator.alloc(f16, param_count);
            @memset(self.fp16_gradients.?, 0.0);
        }

        // Initialize worker threads
        try self.initializeWorkers();

        // Mark as initialized
        self.is_initialized = true;
    }

    fn initializeWorkers(self: *Optimizer) !void {
        const num_workers = self.config.num_worker_threads;
        self.workers = try self.allocator.alloc(OptimizerWorker, num_workers);

        const params_per_worker = self.param_count / num_workers;

        for (self.workers.?, 0..) |*worker, i| {
            const start_idx = i * params_per_worker;
            const end_idx = if (i == num_workers - 1) self.param_count else (i + 1) * params_per_worker;

            worker.* = OptimizerWorker{
                .thread = null, // Explicitly initialize to null
                .start_idx = start_idx,
                .end_idx = end_idx,
                .params = self.params.?,
                .gradients = self.gradients.?,
                .momentum = self.momentum.?,
                .velocity = self.velocity.?,
                .config = self.config,
                .step_count = 0,
                .learning_rate = 0,
                .finished = std.atomic.Value(bool).init(false),
                .thread_running = std.atomic.Value(bool).init(false),
            };
        }
    }

    /// Zero gradients efficiently
    pub fn zeroGrad(self: *Optimizer) !void {
        if (self.gradients == null) return;

        const gradients = self.gradients.?;

        // Zero gradients on CPU using SIMD
        if (self.config.use_avx2_simd and builtin.cpu.arch == .x86_64) {
            self.zeroGradientsSIMD(gradients);
        } else {
            @memset(gradients, 0.0);
        }

        if (self.fp16_gradients) |fp16_grads| {
            @memset(fp16_grads, 0.0);
        }
    }

    fn zeroGradientsSIMD(self: *Optimizer, gradients: []f32) void {
        _ = self;
        const simd_width = 8; // AVX2 can handle 8 f32 values
        const simd_end = (gradients.len / simd_width) * simd_width;

        var i: usize = 0;
        while (i < simd_end) {
            const zero_vec: @Vector(8, f32) = @splat(0.0);
            const zero_array: [8]f32 = zero_vec;
            @memcpy(gradients[i..i+8], &zero_array);
            i += 8;
        }

        // Handle remaining elements
        @memset(gradients[i..], 0.0);
    }

    /// High-performance optimization step
    pub fn step(self: *Optimizer, learning_rate: f32) !void {
        if (self.params == null or self.momentum == null or self.velocity == null) {
            return error.OptimizerNotInitialized;
        }

        self.step_count += 1;

        try self.stepCPU(learning_rate);
    }

    fn stepCPU(self: *Optimizer, learning_rate: f32) !void {
        if (self.workers == null) return error.WorkersNotInitialized;

        // Ensure any previous threads are cleaned up
        self.cleanupThreads();

        // Mark threads as active
        self.threads_active.store(true, .seq_cst);

        // Launch all worker threads
        for (self.workers.?) |*worker| {
            worker.step_count = self.step_count;
            worker.learning_rate = learning_rate;
            worker.finished.store(false, .seq_cst);
            worker.thread_running.store(true, .seq_cst);
            worker.thread = try std.Thread.spawn(.{}, OptimizerWorker.workerThread, .{worker});
        }

        // Wait for all workers to complete
        for (self.workers.?) |*worker| {
            while (!worker.finished.load(.seq_cst)) {
                std.time.sleep(1000); // Sleep 1 microsecond
            }
            if (worker.thread) |thread| {
                thread.join();
                worker.thread = null;
            }
        }

        // Mark threads as inactive
        self.threads_active.store(false, .seq_cst);
    }

    /// SIMD-accelerated gradient clipping
    pub fn clipGradients(self: *Optimizer, max_norm: f32) !void {
        if (self.gradients == null) return;

        const gradients = self.gradients.?;

        // Compute gradient norm using SIMD
        var norm_sq: f32 = 0.0;
        if (self.config.use_avx2_simd and builtin.cpu.arch == .x86_64) {
            norm_sq = self.computeGradientNormSIMD(gradients);
        } else {
            for (gradients) |grad| {
                norm_sq += grad * grad;
            }
        }

        const norm = math.sqrt(norm_sq);
        if (norm > max_norm) {
            const scale = max_norm / norm;
            self.scaleGradients(gradients, scale);
        }
    }

    fn computeGradientNormSIMD(self: *Optimizer, gradients: []f32) f32 {
        _ = self;
        var norm_sq: f32 = 0.0;
        const simd_width = 8;
        const simd_end = (gradients.len / simd_width) * simd_width;

        var accumulator: @Vector(8, f32) = @splat(0.0);
        var i: usize = 0;

        while (i < simd_end) {
            const grad_vec: @Vector(8, f32) = gradients[i..i+8][0..8].*;
            accumulator += grad_vec * grad_vec;
            i += 8;
        }

        // Sum the accumulator
        for (0..8) |j| {
            norm_sq += accumulator[j];
        }

        // Handle remaining elements
        for (i..gradients.len) |j| {
            norm_sq += gradients[j] * gradients[j];
        }

        return norm_sq;
    }

    fn scaleGradients(self: *Optimizer, gradients: []f32, scale: f32) void {
        if (self.config.use_avx2_simd and builtin.cpu.arch == .x86_64) {
            const simd_width = 8;
            const simd_end = (gradients.len / simd_width) * simd_width;
            const scale_vec: @Vector(8, f32) = @splat(scale);

            var i: usize = 0;
                         while (i < simd_end) {
                 const grad_vec: @Vector(8, f32) = gradients[i..i+8][0..8].*;
                 const scaled_vec = grad_vec * scale_vec;
                 const scaled_array: [8]f32 = scaled_vec;
                 @memcpy(gradients[i..i+8], &scaled_array);
                 i += 8;
             }

            // Handle remaining elements
            for (i..gradients.len) |j| {
                gradients[j] *= scale;
            }
        } else {
            for (gradients) |*grad| {
                grad.* *= scale;
            }
        }
    }

    /// Update loss scale for mixed precision training
    pub fn updateLossScale(self: *Optimizer, has_inf_or_nan: bool) void {
        if (has_inf_or_nan) {
            self.loss_scale = @max(self.loss_scale * 0.5, 1.0);
        } else {
            // Increase loss scale every 2000 successful steps
            if (self.step_count % 2000 == 0) {
                self.loss_scale = @min(self.loss_scale * 2.0, 65536.0);
            }
        }
    }

    /// Get optimizer statistics
    pub fn getStats(self: *Optimizer) OptimizerStats {
        return OptimizerStats{
            .step_count = self.step_count,
            .loss_scale = self.loss_scale,
            .param_count = self.param_count,
            .using_cuda = false,
            .num_workers = self.config.num_worker_threads,
        };
    }
};

pub const OptimizerStats = struct {
    step_count: u64,
    loss_scale: f32,
    param_count: usize,
    using_cuda: bool,
    num_workers: u32,
};
