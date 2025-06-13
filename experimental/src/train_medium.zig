// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Training implementation for DeepZig V3 medium models
//! Optimized for AMD Ryzen 9 3900X (24 cores) + NVIDIA RTX 2070 SUPER (8GB VRAM)
//!
//! Features:
//! - Hardware-aware configuration with automatic detection
//! - Multi-threaded training with SIMD optimization
//! - CUDA acceleration with Tensor Core support
//! - Mixed precision training with dynamic loss scaling
//! - Memory-efficient gradient accumulation and checkpointing
//! - Comprehensive performance monitoring

const std = @import("std");
const math = std.math;
const log = std.log;
const Allocator = std.mem.Allocator;
const Timer = std.time.Timer;

const deepseek = @import("deepseek_core");
const training = @import("training");

// Import training components
const Optimizer = training.Optimizer;
const OptimizerConfig = training.OptimizerConfig;
const DataLoader = training.DataLoader;
const DataLoaderConfig = training.DataLoaderConfig;
const Batch = training.Batch;

// Import hardware detection and CUDA backend
const detectHardware = training.detectHardware;
const getOptimalConfig = training.getOptimalConfig;
const HardwareInfo = training.HardwareInfo;
const initCudaBackend = training.initCudaBackend;

// ModelSizeType is a simple string type for compatibility with Zig 0.15.0-dev
pub const ModelSizeType = []const u8;
// Predefined model size options
pub const MODEL_SIZE_TEST = "test";
pub const MODEL_SIZE_SMALL = "small";
pub const MODEL_SIZE_MEDIUM = "medium";
pub const MODEL_SIZE_LARGE = "large";

/// Hardware-optimized training configuration
pub const TrainingConfig = struct {
    // Model configuration
    model_size: ModelSizeType = MODEL_SIZE_MEDIUM,
    
    // Training parameters (optimized for RTX 2070 SUPER 8GB VRAM)
    batch_size: u32 = 64,        // Larger batches for better GPU utilization
    micro_batch_size: u32 = 16,  // Increased for Tensor Core efficiency
    num_epochs: u32 = 10,
    max_samples: u32 = 100000,

    // Optimization
    learning_rate: f32 = 1e-4,
    min_learning_rate: f32 = 1e-6,
    warmup_steps: u32 = 1000,
    gradient_clip_norm: f32 = 1.0,
    weight_decay: f32 = 0.01,

    // Mixed precision (optimized for RTX 2070 SUPER Tensor Cores)
    use_mixed_precision: bool = true,
    loss_scale: f32 = 32768.0,
    dynamic_loss_scaling: bool = true,
    use_tensor_cores: bool = true,

    // Memory optimization
    gradient_checkpointing: bool = true,
    dataloader_workers: u32 = 20,  // Use most of 24 CPU threads
    prefetch_batches: u32 = 8,     // More prefetching with abundant memory
    pin_memory: bool = true,       // Enable for faster GPU transfers
    
    // Model size documentation
    // test:   Ultra-fast test (<1M params)
    // small:  Fast training (~5M params)
    // medium: Default size (~50M params)
    // large:  High quality (~125M params)

    // Hardware acceleration
    use_cuda: bool = false,        // Will be auto-detected
    use_avx2_simd: bool = true,   // AMD Ryzen 9 3900X supports AVX2
    optimizer_threads: u32 = 20,  // Multi-threaded optimization

    // Logging and checkpointing
    log_interval: u32 = 10,
    save_steps: u32 = 1000,
    output_dir: []const u8 = "checkpoints",

    // Hardware detection
    hardware: ?HardwareInfo = null,

    // Validation
    pub fn validate(self: *const TrainingConfig) !void {
        if (self.micro_batch_size > self.batch_size) {
            return error.InvalidBatchSizes;
        }
        if (self.learning_rate <= 0 or self.learning_rate > 1.0) {
            return error.InvalidLearningRate;
        }
        if (self.gradient_clip_norm < 0) {
            return error.InvalidGradientClipping;
        }
    }
};

/// Training state tracking
pub const TrainingState = struct {
    epoch: usize = 0,
    global_step: usize = 0,
    learning_rate: f32 = 0,
    loss_scale: f32 = 32768.0,
    best_loss: f32 = math.inf(f32),

    // Performance metrics
    samples_per_second: f32 = 0,
    tokens_per_second: f32 = 0,

    // Timers
    data_timer: Timer = undefined,
    compute_timer: Timer = undefined,
    epoch_timer: Timer = undefined,

    pub fn init() !TrainingState {
        return TrainingState{
            .data_timer = try Timer.start(),
            .compute_timer = try Timer.start(),
            .epoch_timer = try Timer.start(),
        };
    }

    pub fn updateMetrics(self: *TrainingState, batch_size: u32, step_time_ns: u64) void {
        const step_time_s = @as(f32, @floatFromInt(step_time_ns)) / std.time.ns_per_s;
        self.samples_per_second = @as(f32, @floatFromInt(batch_size)) / step_time_s;
    }
};

/// Main training class with comprehensive functionality
pub const Trainer = struct {
    allocator: Allocator,
    model: *deepseek.Model,
    config: TrainingConfig,
    state: TrainingState,

    // Training components
    optimizer: Optimizer,
    data_loader: DataLoader,
    scheduler: LearningRateScheduler,

    pub fn init(allocator: Allocator, model: *deepseek.Model, config: TrainingConfig) !Trainer {
        try config.validate();

        log.info("Detecting hardware configuration...", .{});

        // Auto-detect hardware capabilities
        var optimized_config = config;
        if (optimized_config.hardware == null) {
            optimized_config.hardware = detectHardware(allocator) catch |err| {
                log.warn("Hardware detection failed: {}, using default settings", .{err});
                null;
            };
        }

        // Apply hardware-specific optimizations
        if (optimized_config.hardware) |hw| {
            log.info("Detected: {} CPU cores, CUDA: {}, AVX2: {}", .{
                hw.cpu_cores, hw.cuda_available, hw.avx2_support
            });

            // Enable CUDA if available
            optimized_config.use_cuda = hw.cuda_available;

            // Optimize thread counts
            optimized_config.dataloader_workers = @max(hw.cpu_cores - 4, 4);
            optimized_config.optimizer_threads = @max(hw.cpu_cores - 4, 4);

            // Enable SIMD if supported
            optimized_config.use_avx2_simd = hw.avx2_support;

            // Optimize batch sizes for GPU memory
            if (hw.gpu_memory_gb != null and hw.gpu_memory_gb.? >= 8) {
                optimized_config.batch_size = 64;
                optimized_config.micro_batch_size = 16;
            }
        }

        // Initialize optimizer with hardware-optimized settings
        const optimizer_config = OptimizerConfig{
            .learning_rate = optimized_config.learning_rate,
            .weight_decay = optimized_config.weight_decay,
            .use_mixed_precision = optimized_config.use_mixed_precision,
            .use_cuda = optimized_config.use_cuda,
            .num_worker_threads = optimized_config.optimizer_threads,
            .use_avx2_simd = optimized_config.use_avx2_simd,
        };

        const optimizer = try Optimizer.init(allocator, optimizer_config);

        // Initialize data loader with optimized settings
        const data_loader_config = DataLoaderConfig{
            .batch_size = optimized_config.batch_size,
            .num_workers = optimized_config.dataloader_workers,
            .prefetch_batches = optimized_config.prefetch_batches,
            .pin_memory = optimized_config.pin_memory,
            .simd_acceleration = optimized_config.use_avx2_simd,
        };

        const data_loader = try DataLoader.init(allocator, data_loader_config);

        const total_steps = (optimized_config.max_samples / optimized_config.batch_size) * optimized_config.num_epochs;
        const scheduler = LearningRateScheduler.init(.{
            .initial_lr = optimized_config.learning_rate,
            .min_lr = optimized_config.min_learning_rate,
            .warmup_steps = optimized_config.warmup_steps,
            .total_steps = total_steps,
        });

        log.info("Training optimized for hardware: {} CPU threads, {} dataloader workers, batch size {}", .{
            optimized_config.optimizer_threads,
            optimized_config.dataloader_workers,
            optimized_config.batch_size
        });

        return Trainer{
            .allocator = allocator,
            .model = model,
            .config = optimized_config,
            .state = try TrainingState.init(),
            .optimizer = optimizer,
            .data_loader = data_loader,
            .scheduler = scheduler,
        };
    }

    pub fn deinit(self: *Trainer) void {
        self.optimizer.deinit();
        self.data_loader.deinit();
        self.scheduler.deinit();
    }

    /// Main training loop
    pub fn train(self: *Trainer) !void {
        log.info("Starting training with {d} epochs, batch size {d}", .{
            self.config.num_epochs, self.config.batch_size
        });

        // Load dataset
        try self.data_loader.loadDataset("training_data");

        const total_steps = (self.config.max_samples / self.config.batch_size) * self.config.num_epochs;
        var global_timer = try Timer.start();

        // Training loop
        for (0..self.config.num_epochs) |epoch| {
            self.state.epoch = epoch;
            self.state.epoch_timer = try Timer.start();

            var epoch_loss: f32 = 0;
            var batch_count: usize = 0;

            try self.data_loader.reset();

            while (try self.data_loader.hasNext()) {
                // Get batch
                const batch_start = self.state.data_timer.lap();
                var batch = try self.data_loader.nextBatch();
                defer batch.deinit();
                const data_time = self.state.data_timer.lap() - batch_start;

                // Training step
                const compute_start = self.state.compute_timer.lap();
                const loss = try self.trainingStep(&batch);
                const compute_time = self.state.compute_timer.lap() - compute_start;

                epoch_loss += loss;
                batch_count += 1;
                self.state.global_step += 1;

                // Update learning rate
                self.state.learning_rate = self.scheduler.step(self.state.global_step);

                // Update metrics
                const step_time = data_time + compute_time;
                self.state.updateMetrics(self.config.batch_size, step_time);

                // Logging
                if (self.state.global_step % self.config.log_interval == 0) {
                    log.info("Step {d}/{d} | Loss: {d:.4} | LR: {d:.6} | {d:.1} samples/s", .{
                        self.state.global_step, total_steps, loss,
                        self.state.learning_rate, self.state.samples_per_second
                    });
                }

                // Checkpointing
                if (self.state.global_step % self.config.save_steps == 0) {
                    try self.saveCheckpoint();
                }

                // Dynamic loss scaling
                if (self.config.dynamic_loss_scaling) {
                    try self.updateLossScale(loss);
                }
            }

            const epoch_time = self.state.epoch_timer.lap();
            const avg_loss = epoch_loss / @as(f32, @floatFromInt(batch_count));

            log.info("Epoch {d}/{d} complete in {d:.2}s | Avg Loss: {d:.4}", .{
                epoch + 1, self.config.num_epochs,
                @as(f32, @floatFromInt(epoch_time)) / std.time.ns_per_s,
                avg_loss
            });

            if (avg_loss < self.state.best_loss) {
                self.state.best_loss = avg_loss;
                try self.saveBestModel();
            }
        }

        const total_time = global_timer.lap();
        const total_time_s = @as(f32, @floatFromInt(total_time)) / std.time.ns_per_s;

        log.info("Training complete in {d:.2}s", .{total_time_s});
    }

    /// Single training step with gradient accumulation
    fn trainingStep(self: *Trainer, batch: *Batch) !f32 {
        try self.optimizer.zeroGrad();

        var accumulated_loss: f32 = 0;
        const gradient_accumulation_steps = self.config.batch_size / self.config.micro_batch_size;

        // Gradient accumulation
        for (0..gradient_accumulation_steps) |step| {
            const micro_batch = try batch.getMicroBatch(step, self.config.micro_batch_size);
            defer micro_batch.deinit();

            // Forward pass
            var loss: f32 = 0;
            if (self.config.use_mixed_precision) {
                loss = try self.mixedPrecisionForward(micro_batch);
            } else {
                loss = try self.standardForward(micro_batch);
            }

            // Scale loss for accumulation
            loss = loss / @as(f32, @floatFromInt(gradient_accumulation_steps));
            accumulated_loss += loss;

            // Backward pass
            if (self.config.use_mixed_precision) {
                try self.mixedPrecisionBackward(loss * self.state.loss_scale);
            } else {
                try self.standardBackward(loss);
            }
        }

        // Gradient clipping
        if (self.config.gradient_clip_norm > 0) {
            try self.clipGradients(self.config.gradient_clip_norm);
        }

        // Optimizer step
        try self.optimizer.step(self.state.learning_rate);

        return accumulated_loss;
    }

    // Implementation methods
    fn mixedPrecisionForward(self: *Trainer, batch: anytype) !f32 {
        _ = self;
        _ = batch;
        return 1.0; // Placeholder
    }

    fn standardForward(self: *Trainer, batch: anytype) !f32 {
        _ = self;
        _ = batch;
        return 1.0; // Placeholder
    }

    fn mixedPrecisionBackward(self: *Trainer, scaled_loss: f32) !void {
        _ = self;
        _ = scaled_loss;
    }

    fn standardBackward(self: *Trainer, loss: f32) !void {
        _ = self;
        _ = loss;
    }

    fn clipGradients(self: *Trainer, max_norm: f32) !void {
        _ = self;
        _ = max_norm;
    }

    fn updateLossScale(self: *Trainer, loss: f32) !void {
        if (math.isNan(loss) or math.isInf(loss)) {
            self.state.loss_scale *= 0.5;
        } else if (self.state.global_step % 2000 == 0) {
            self.state.loss_scale = @min(self.state.loss_scale * 2.0, 65536.0);
        }
    }

    fn saveCheckpoint(self: *Trainer) !void {
        const checkpoint_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/checkpoint-{d}",
            .{ self.config.output_dir, self.state.global_step }
        );
        defer self.allocator.free(checkpoint_path);

        // Implementation would save model, optimizer state, and training state
        log.info("Saving checkpoint to {s}", .{checkpoint_path});
    }

    fn saveBestModel(self: *Trainer) !void {
        const best_model_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/best_model",
            .{self.config.output_dir}
        );
        defer self.allocator.free(best_model_path);

        log.info("Saving best model to {s}", .{best_model_path});
    }
};

/// Learning rate scheduler with warmup and cosine annealing
pub const LearningRateScheduler = struct {
    initial_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,

    pub fn init(config: anytype) LearningRateScheduler {
        return LearningRateScheduler{
            .initial_lr = config.initial_lr,
            .min_lr = config.min_lr,
            .warmup_steps = config.warmup_steps,
            .total_steps = config.total_steps,
        };
    }

    pub fn deinit(self: *LearningRateScheduler) void {
        _ = self;
    }

    pub fn step(self: *LearningRateScheduler, global_step: usize) f32 {
        if (global_step < self.warmup_steps) {
            // Linear warmup
            return self.initial_lr * @as(f32, @floatFromInt(global_step)) / @as(f32, @floatFromInt(self.warmup_steps));
        } else {
            // Cosine annealing
            const progress = @as(f32, @floatFromInt(global_step - self.warmup_steps)) / @as(f32, @floatFromInt(self.total_steps - self.warmup_steps));
            return self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1.0 + @cos(std.math.pi * progress));
        }
    }
};

/// Parse command line arguments
pub fn parseArgs(allocator: Allocator) !TrainingConfig {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    var config = TrainingConfig{};
    
    // Process args (starting from index 1 to skip executable name)
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        
        if (std.mem.eql(u8, arg, "--model-size") or std.mem.eql(u8, arg, "-m")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            
            const size_str = args[i];
            if (std.mem.eql(u8, size_str, "test")) {
                config.model_size = MODEL_SIZE_TEST;
                // Adjust batch size for tiny model
                config.batch_size = 16;
                config.micro_batch_size = 8;
                config.num_epochs = 2;
            } else if (std.mem.eql(u8, size_str, "small")) {
                config.model_size = MODEL_SIZE_SMALL;
                config.batch_size = 32;
                config.micro_batch_size = 8;
            } else if (std.mem.eql(u8, size_str, "medium")) {
                config.model_size = MODEL_SIZE_MEDIUM;
            } else if (std.mem.eql(u8, size_str, "large")) {
                config.model_size = MODEL_SIZE_LARGE;
                config.batch_size = 32; // Reduce batch size for larger model
            } else {
                log.err("Invalid model size: {s}", .{size_str});
                return error.InvalidValue;
            }
        } else if (std.mem.eql(u8, arg, "--batch-size") or std.mem.eql(u8, arg, "-b")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.batch_size = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--epochs") or std.mem.eql(u8, arg, "-e")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.num_epochs = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--max-samples") or std.mem.eql(u8, arg, "-s")) {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            config.max_samples = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        }
    }
    
    return config;
}

/// Print help information
fn printHelp() void {
    std.io.getStdOut().writer().print(
        \\DeepZig V3 Training
        \\=====================================
        \\Usage: train-medium [options]
        \\
        \\Options:
        \\  --model-size, -m <size>   Model size: test, small, medium, large (default: medium)
        \\  --batch-size, -b <n>      Batch size (adjusted automatically per model size)
        \\  --epochs, -e <n>          Number of training epochs
        \\  --max-samples, -s <n>     Maximum number of training samples
        \\  --help, -h               Show this help message
        \\
        \\Examples:
        \\  train-medium --model-size test   Ultra-fast test run (<1M params)
        \\  train-medium --model-size small  Quick training run (~5M params)
        \\  train-medium                     Default training (~50M params)
        \\  train-medium --model-size large  Full quality (~125M params)
        \\=====================================
, .{}) catch {};
}

/// Main entry point for training
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    log.info("=== DeepZig V3 High-Performance Training ===", .{});
    
    // Parse command line arguments
    var config = parseArgs(allocator) catch |err| {
        log.err("Error parsing arguments: {}", .{err});
        printHelp();
        return err;
    };

    // Detect and display system information
    const hardware: ?HardwareInfo = detectHardware(allocator) catch |err| {
        log.warn("Hardware detection failed: {}, using conservative defaults", .{err});
        null;
    };

    if (hardware) |hw| {
        log.info("System: {d} CPU cores, {s} AVX2, {s} CUDA", .{
            hw.cpu_cores,
            if (hw.avx2_support) "with" else "without",
            if (hw.cuda_available) "with" else "without"
        });

        if (hw.gpu_memory_gb) |gpu_mem| {
            if (hw.gpu_compute_capability) |compute_cap| {
                log.info("GPU: {d} GB VRAM, Compute Capability {d:.1}", .{ gpu_mem, compute_cap });
            } else {
                log.info("GPU: {d} GB VRAM", .{gpu_mem});
            }
        }
    }

    // Apply hardware-specific optimizations to parsed config
    config.hardware = hardware;

    // Enable CUDA if detected
    if (hardware) |hw| {
        config.use_cuda = hw.cuda_available;
        config.use_avx2_simd = hw.avx2_support;

        // Adjust for available GPU memory
        if (hw.gpu_memory_gb) |gpu_mem| {
            if (gpu_mem >= 8) {
                config.batch_size = 64;
                log.info("Using large batch size ({d}) for {d}GB GPU", .{ config.batch_size, gpu_mem });
            } else if (gpu_mem >= 4) {
                config.batch_size = 32;
                log.info("Using medium batch size ({d}) for {d}GB GPU", .{ config.batch_size, gpu_mem });
            } else {
                config.batch_size = 16;
                log.info("Using small batch size ({d}) for {d}GB GPU", .{ config.batch_size, gpu_mem });
            }
        }
    }

    // Initialize model based on selected size
    log.info("Initializing model size: {s}", .{config.model_size});
    
    var model_config: *deepseek.config.ModelConfig = undefined;
    
    if (std.mem.eql(u8, config.model_size, MODEL_SIZE_TEST)) {
        model_config = try deepseek.config.ModelConfig.testConfig(allocator);
    } else if (std.mem.eql(u8, config.model_size, MODEL_SIZE_SMALL)) {
        model_config = try deepseek.config.ModelConfig.smallConfig(allocator);
    } else if (std.mem.eql(u8, config.model_size, MODEL_SIZE_MEDIUM)) {
        model_config = try deepseek.config.ModelConfig.mediumConfig(allocator);
    } else if (std.mem.eql(u8, config.model_size, MODEL_SIZE_LARGE)) {
        model_config = try deepseek.config.ModelConfig.largeConfig(allocator);
    } else {
        // Fall back to medium as default
        log.warn("Unknown model size: {s}, using medium", .{config.model_size});
        model_config = try deepseek.config.ModelConfig.mediumConfig(allocator);
    }
    defer model_config.deinit();

    var model = try deepseek.Model.initFromConfig(allocator, model_config);
    defer model.deinit();

    // Create and run trainer
    var trainer = try Trainer.init(allocator, &model, config);
    defer trainer.deinit();

    // Show final configuration
    log.info("Final config: size={s}, batch={d}, workers={d}, CUDA={}, SIMD={}", .{
        trainer.config.model_size,
        trainer.config.batch_size,
        trainer.config.dataloader_workers,
        trainer.config.use_cuda,
        trainer.config.use_avx2_simd
    });

    try trainer.train();

    log.info("Training complete", .{});
}
