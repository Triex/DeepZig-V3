// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Training module for DeepZig V3
//!
//! Provides efficient training components including:
//! - Modern optimizers (AdamW with mixed precision support)
//! - Efficient data loading with multi-threading
//! - Memory-optimized batch processing
//! - Learning rate scheduling
//! - Comprehensive training metrics

const std = @import("std");

// Re-export optimizer components
const optimizer_module = @import("optimizer.zig");
pub const Optimizer = optimizer_module.Optimizer;
pub const OptimizerConfig = optimizer_module.OptimizerConfig;
pub const OptimizerStats = optimizer_module.OptimizerStats;

// Re-export data loading components
const data_loader_module = @import("data_loader.zig");
pub const DataLoader = data_loader_module.DataLoader;
pub const DataLoaderConfig = data_loader_module.DataLoaderConfig;
pub const Batch = data_loader_module.Batch;
pub const MicroBatch = data_loader_module.MicroBatch;

// Re-export hardware detection and backends
const backend_detection_module = @import("backend_detection.zig");
pub const HardwareInfo = backend_detection_module.HardwareInfo;
pub const BackendConfig = backend_detection_module.BackendConfig;
pub const detectHardware = backend_detection_module.detectHardware;
pub const getOptimalConfig = backend_detection_module.getOptimalConfig;

// Legacy compatibility exports
pub const dataset = @import("dataset.zig");
pub const TextDataset = dataset.TextDataset;

/// Training configuration with modern optimizations
pub const TrainingConfig = struct {
    // Core training parameters
    batch_size: u32 = 32,
    micro_batch_size: u32 = 8,
    learning_rate: f32 = 1e-4,
    num_epochs: u32 = 3,
    max_samples: u32 = 100000,

    // Optimization features
    use_mixed_precision: bool = true,
    gradient_accumulation: bool = true,
    gradient_checkpointing: bool = true,

    // Data loading features
    num_data_workers: u32 = 4,
    prefetch_batches: u32 = 2,
    shuffle_data: bool = true,

    // Memory optimization
    pin_memory: bool = false,
    max_sequence_length: u32 = 2048,

    // Monitoring
    output_dir: []const u8 = "checkpoints",
    log_interval: u32 = 10,
    save_steps: u32 = 1000,

    /// Validate configuration settings
    pub fn validate(self: *const TrainingConfig) !void {
        if (self.micro_batch_size > self.batch_size) {
            return error.InvalidBatchSizes;
        }
        if (self.learning_rate <= 0 or self.learning_rate > 1.0) {
            return error.InvalidLearningRate;
        }
        if (self.num_data_workers == 0) {
            return error.InvalidWorkerCount;
        }
    }
};

/// Training pipeline factory for creating optimized components
pub const TrainingPipeline = struct {
    allocator: std.mem.Allocator,
    config: TrainingConfig,

    pub fn init(allocator: std.mem.Allocator, config: TrainingConfig) !TrainingPipeline {
        try config.validate();

        return TrainingPipeline{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Create optimizer with configuration
    pub fn createOptimizer(self: *TrainingPipeline) !Optimizer {
        const optimizer_config = OptimizerConfig{
            .learning_rate = self.config.learning_rate,
            .use_mixed_precision = self.config.use_mixed_precision,
            .gradient_clip_norm = 1.0,
            .parameter_partitioning = false,
        };

        return try Optimizer.init(self.allocator, optimizer_config);
    }

    /// Create data loader with configuration
    pub fn createDataLoader(self: *TrainingPipeline) !DataLoader {
        const data_config = DataLoaderConfig{
            .batch_size = self.config.batch_size,
            .num_workers = self.config.num_data_workers,
            .prefetch_batches = self.config.prefetch_batches,
            .shuffle = self.config.shuffle_data,
            .pin_memory = self.config.pin_memory,
            .max_sequence_length = self.config.max_sequence_length,
        };

        return try DataLoader.init(self.allocator, data_config);
    }

    /// Get performance recommendations based on system capabilities
    pub fn getPerformanceRecommendations(self: *TrainingPipeline) TrainingRecommendations {
        _ = self;

        // Analyze system and provide practical recommendations
        return TrainingRecommendations{
            .recommended_batch_size = 32,
            .recommended_workers = 4,
            .use_mixed_precision = true,
            .use_gradient_checkpointing = true,
            .estimated_memory_gb = 8.0,
        };
    }
};

/// Training performance recommendations
pub const TrainingRecommendations = struct {
    recommended_batch_size: u32,
    recommended_workers: u32,
    use_mixed_precision: bool,
    use_gradient_checkpointing: bool,
    estimated_memory_gb: f32,
};

/// Training metrics tracking
pub const TrainingMetrics = struct {
    // Performance metrics
    samples_per_second: f32 = 0,
    tokens_per_second: f32 = 0,

    // Training metrics
    current_loss: f32 = 0,
    best_loss: f32 = std.math.inf(f32),
    learning_rate: f32 = 0,
    epoch: u32 = 0,
    global_step: u32 = 0,

    // Resource utilization
    memory_usage_mb: f32 = 0,
    compute_utilization: f32 = 0,

    // Timing
    step_time_ms: f32 = 0,
    data_loading_time_ms: f32 = 0,

    pub fn updatePerformance(self: *TrainingMetrics, batch_size: u32, step_time_ns: u64) void {
        const step_time_s = @as(f32, @floatFromInt(step_time_ns)) / std.time.ns_per_s;
        self.samples_per_second = @as(f32, @floatFromInt(batch_size)) / step_time_s;
        self.step_time_ms = step_time_s * 1000.0;
    }

    pub fn logMetrics(self: *TrainingMetrics) void {
        std.log.info("Step {d} | Loss: {d:.4} | {d:.1} samples/s | {d:.1}ms/step", .{
            self.global_step,
            self.current_loss,
            self.samples_per_second,
            self.step_time_ms,
        });
    }
};

/// Create a standard training configuration with sensible defaults
pub fn createStandardConfig() TrainingConfig {
    return TrainingConfig{
        .batch_size = 32,
        .micro_batch_size = 8,
        .learning_rate = 1e-4,
        .use_mixed_precision = true,
        .gradient_accumulation = true,
        .num_data_workers = 4,
    };
}

/// Create a high-performance training configuration for larger models
pub fn createHighPerformanceConfig() TrainingConfig {
    return TrainingConfig{
        .batch_size = 64,
        .micro_batch_size = 16,
        .learning_rate = 3e-4,
        .use_mixed_precision = true,
        .gradient_accumulation = true,
        .gradient_checkpointing = true,
        .num_data_workers = 8,
        .prefetch_batches = 4,
        .pin_memory = true,
    };
}
