// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Optimizer implementations for DeepZig model training
//! Includes standard optimizers like Adam with weight decay

const std = @import("std");
const deepseek_core = @import("deepseek_core");
const tensor = deepseek_core.tensor;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const log = std.log;

/// Adam optimizer configuration parameters
pub const AdamConfig = struct {
    learning_rate: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    weight_decay: f32 = 0.0,
};

/// Adam optimizer implementation with weight decay
/// Based on the paper "Adam: A Method for Stochastic Optimization"
pub const Adam = struct {
    allocator: Allocator,
    params: ArrayList(*tensor.Tensor(.f32)),
    exp_avg: ArrayList(*tensor.Tensor(.f32)),    // First moment estimates (m)
    exp_avg_sq: ArrayList(*tensor.Tensor(.f32)), // Second moment estimates (v)
    step_count: usize,
    config: AdamConfig,

    /// Initialize Adam optimizer with given parameters
    pub fn init(allocator: Allocator, params: ArrayList(*tensor.Tensor(.f32)), config: AdamConfig) !Adam {
        var exp_avg = ArrayList(*tensor.Tensor(.f32)).init(allocator);
        errdefer {
            for (exp_avg.items) |item| {
                item.deinit();
            }
            exp_avg.deinit();
        }

        var exp_avg_sq = ArrayList(*tensor.Tensor(.f32)).init(allocator);
        errdefer {
            for (exp_avg_sq.items) |item| {
                item.deinit();
            }
            exp_avg_sq.deinit();
        }

        // Initialize momentum buffers
        for (params.items) |param| {
            const avg = try tensor.zeros(allocator, param.shape(), .f32);
            try exp_avg.append(avg);

            const avg_sq = try tensor.zeros(allocator, param.shape(), .f32);
            try exp_avg_sq.append(avg_sq);
        }

        return Adam{
            .allocator = allocator,
            .params = params,
            .exp_avg = exp_avg,
            .exp_avg_sq = exp_avg_sq,
            .step_count = 0,
            .config = config,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *Adam) void {
        for (self.exp_avg.items) |item| {
            item.deinit();
        }
        self.exp_avg.deinit();

        for (self.exp_avg_sq.items) |item| {
            item.deinit();
        }
        self.exp_avg_sq.deinit();

        // Note: We don't own the params, so we don't deinit them
        self.params.deinit();
    }

    /// Perform a single optimization step 
    pub fn step(self: *Adam) !void {
        self.step_count += 1;

        const lr = self.config.learning_rate;
        const beta1 = self.config.beta1;
        const beta2 = self.config.beta2;
        const epsilon = self.config.epsilon;
        const weight_decay = self.config.weight_decay;

        // Bias correction terms
        const bias_correction1 = 1.0 - std.math.pow(f32, beta1, @floatFromInt(self.step_count));
        const bias_correction2 = 1.0 - std.math.pow(f32, beta2, @floatFromInt(self.step_count));
        const corrected_lr = lr * @sqrt(bias_correction2) / bias_correction1;

        for (self.params.items, self.exp_avg.items, self.exp_avg_sq.items) |param, exp_avg, exp_avg_sq| {
            if (param.grad == null) {
                // Skip parameters that don't have gradients
                continue;
            }

            // Get param grad
            const grad = param.grad.?;
            const grad_data = try grad.asType(f32);
            defer grad_data.deinit();

            // Update exp_avg: m = beta1 * m + (1 - beta1) * grad
            try exp_avg.mul_(beta1);
            {
                var scaled_grad = try grad_data.clone();
                defer scaled_grad.deinit();
                try scaled_grad.mul_(1.0 - beta1);
                try exp_avg.add_(scaled_grad);
            }

            // Update exp_avg_sq: v = beta2 * v + (1 - beta2) * grad^2
            try exp_avg_sq.mul_(beta2);
            {
                var squared_grad = try grad_data.clone();
                defer squared_grad.deinit();
                try squared_grad.mul_(grad_data); // Element-wise square
                try squared_grad.mul_(1.0 - beta2);
                try exp_avg_sq.add_(squared_grad);
            }

            // Update parameters
            var update = try exp_avg.clone();
            defer update.deinit();

            // Divide by sqrt(v) + epsilon
            {
                var denom = try exp_avg_sq.clone();
                defer denom.deinit();
                try denom.sqrt_();
                try denom.add_(epsilon);
                try update.div_(denom);
            }

            try update.mul_(corrected_lr);

            // Apply weight decay if configured
            if (weight_decay > 0.0) {
                var decay_term = try param.clone();
                defer decay_term.deinit();
                try decay_term.mul_(weight_decay * lr);
                try update.add_(decay_term);
            }

            // Update parameter: param -= update
            try param.sub_(update);
        }
    }

    /// Zero out parameter gradients
    pub fn zeroGrad(self: *Adam) !void {
        for (self.params.items) |param| {
            if (param.grad != null) {
                try param.grad.?.zero_();
            }
        }
    }
};

/// A simple StochasticGradientDescent optimizer
pub const SGD = struct {
    allocator: Allocator,
    params: ArrayList(*tensor.Tensor(.f32)),
    config: struct {
        learning_rate: f32 = 0.01,
        momentum: f32 = 0.0,
        weight_decay: f32 = 0.0,
    },
    momentum_buffers: ?ArrayList(*tensor.Tensor(.f32)),

    /// Initialize SGD optimizer with given parameters
    pub fn init(allocator: Allocator, params: ArrayList(*tensor.Tensor(.f32)), config: struct {
        learning_rate: f32 = 0.01,
        momentum: f32 = 0.0,
        weight_decay: f32 = 0.0,
    }) !SGD {
        var momentum_buffers: ?ArrayList(*tensor.Tensor) = null;
        
        // Initialize momentum buffers if momentum > 0
        if (config.momentum > 0.0) {
            var buffers = ArrayList(*tensor.Tensor(.f32)).init(allocator);
            errdefer {
                for (buffers.items) |item| {
                    item.deinit();
                }
                buffers.deinit();
            }

            for (params.items) |param| {
                const buffer = try tensor.zeros(allocator, param.shape(), .f32);
                try buffers.append(buffer);
            }
            
            momentum_buffers = buffers;
        }

        return SGD{
            .allocator = allocator,
            .params = params,
            .config = config,
            .momentum_buffers = momentum_buffers,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *SGD) void {
        if (self.momentum_buffers) |*buffers| {
            for (buffers.items) |item| {
                item.deinit();
            }
            buffers.deinit();
        }
        
        // Note: We don't own the params, so we don't deinit them
        self.params.deinit();
    }

    /// Perform a single optimization step
    pub fn step(self: *SGD) !void {
        const lr = self.config.learning_rate;
        const momentum = self.config.momentum;
        const weight_decay = self.config.weight_decay;
        
        for (self.params.items, 0..) |param, i| {
            if (param.grad == null) {
                // Skip parameters that don't have gradients
                continue;
            }
            
            const grad = param.grad.?;
            
            // Apply weight decay
            if (weight_decay > 0.0) {
                var decay_term = try param.clone();
                defer decay_term.deinit();
                try decay_term.mul_(weight_decay);
                try grad.add_(decay_term);
            }
            
            // Apply momentum if configured
            if (momentum > 0.0 and self.momentum_buffers != null) {
                const buf = self.momentum_buffers.?.items[i];
                
                // buf = momentum * buf + grad
                try buf.mul_(momentum);
                try buf.add_(grad);
                
                // param -= lr * buf
                var update = try buf.clone();
                defer update.deinit();
                try update.mul_(lr);
                try param.sub_(update);
            } else {
                // Standard SGD update: param -= lr * grad
                var update = try grad.clone();
                defer update.deinit();
                try update.mul_(lr);
                try param.sub_(update);
            }
        }
    }
    
    /// Zero out parameter gradients
    pub fn zeroGrad(self: *SGD) !void {
        for (self.params.items) |param| {
            if (param.grad != null) {
                try param.grad.?.zero_();
            }
        }
    }
};

/// Factory function to create an optimizer by name
pub fn createOptimizer(allocator: Allocator, name: []const u8, params: ArrayList(*tensor.Tensor(.f32)), config: anytype) !anyerror {
    if (std.mem.eql(u8, name, "adam")) {
        return try Adam.init(allocator, params, .{
            .learning_rate = config.learning_rate,
            .weight_decay = config.weight_decay,
            .beta1 = if (@hasField(@TypeOf(config), "beta1")) config.beta1 else 0.9,
            .beta2 = if (@hasField(@TypeOf(config), "beta2")) config.beta2 else 0.999,
            .epsilon = if (@hasField(@TypeOf(config), "epsilon")) config.epsilon else 1e-8,
        });
    } else if (std.mem.eql(u8, name, "sgd")) {
        return try SGD.init(allocator, params, .{
            .learning_rate = config.learning_rate,
            .weight_decay = config.weight_decay,
            .momentum = if (@hasField(@TypeOf(config), "momentum")) config.momentum else 0.0,
        });
    } else {
        return error.UnsupportedOptimizer;
    }
}
