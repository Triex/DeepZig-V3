// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! DeepZig Medium Model Training Implementation
//! Pure Zig implementation for training the medium-sized model

const std = @import("std");
const builtin = @import("builtin");
const fs = std.fs;
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const ArrayList = std.ArrayList;
const Timer = std.time.Timer;
const log = std.log;

// Core components - imported as modules defined in build.zig
const deepseek = @import("deepseek_core");
const tensor = @import("deepseek_core").tensor;
const config = @import("deepseek_core").config;
const tokenizer_mod = @import("deepseek_core").tokenizer;
const Tokenizer = tokenizer_mod.Tokenizer;
const moe = @import("deepseek_core").moe;

// Training components - imported from training module
const training = @import("training");
const optimizer = training.optimizer;
const Adam = training.Adam;
const TextDataset = training.TextDataset;

/// Training configuration parameters
pub const TrainingConfig = struct {
    batch_size: usize = 8,
    num_epochs: usize = 3, 
    learning_rate: f32 = 5e-5,
    weight_decay: f32 = 0.01,
    warmup_steps: usize = 10,
    save_steps: usize = 50,
    max_samples: usize = 1000,
    output_dir: []const u8 = "deepzig-medium-demo",
};

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = general_purpose_allocator.deinit();
    const gpa = general_purpose_allocator.allocator();

    var arena = ArenaAllocator.init(gpa);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const training_config = TrainingConfig{};
    try trainMediumModel(arena_allocator, training_config);
}

/// Main training function for the medium model
fn trainMediumModel(allocator: Allocator, training_config: TrainingConfig) !void {
    log.info("=== DeepZig Medium Model Training ===", .{});
    
    // 1. Load and prepare dataset
    log.info("Loading dataset...", .{});
    var data_loader = try TextDataset.init(allocator, "wikitext", training_config.max_samples);
    defer data_loader.deinit();
    
    log.info("Using {d} samples from dataset", .{data_loader.sample_count});
    
    // 2. Create and train tokenizer
    log.info("Creating and training tokenizer...", .{});
    var tokenizer = try Tokenizer.trainFromDataset(allocator, &data_loader, .{
        .vocab_size = 32000,
        .model_type = .bpe,
    });
    defer tokenizer.deinit();
    
    const tokenizer_path = try std.fmt.allocPrint(
        allocator, 
        "{s}/tokenizer.json", 
        .{training_config.output_dir}
    );
    defer allocator.free(tokenizer_path);
    
    try fs.cwd().makePath(training_config.output_dir);
    try tokenizer.saveToFile(tokenizer_path);
    log.info("✅ Saved tokenizer to {s}", .{tokenizer_path});
    
    // 3. Tokenize dataset
    log.info("Tokenizing dataset...", .{});
    try data_loader.tokenize(&tokenizer);
    
    // 4. Initialize model with medium config
    log.info("Initializing model...", .{});
    const medium_config = try config.ModelConfig.mediumConfig(allocator);
    defer medium_config.deinit();
    
    var model = try deepseek.Model.initFromConfig(allocator, medium_config);
    defer model.deinit();
    
    // 5. Initialize optimizer
    log.info("Setting up optimizer...", .{});
    var optimizer_instance = try Adam.init(allocator, model.parameters(), .{
        .learning_rate = training_config.learning_rate,
        .weight_decay = training_config.weight_decay,
    });
    defer optimizer_instance.deinit();
    
    // 6. Training loop
    log.info("\nTraining the model...", .{});
    const epochs = training_config.num_epochs;
    const steps_per_epoch = data_loader.sample_count / training_config.batch_size;
    const total_steps = epochs * steps_per_epoch;

    var timer = try Timer.start();
    var global_step: usize = 0;
    
    for (0..epochs) |epoch| {
        var epoch_loss: f32 = 0;
        _ = timer.lap(); // Clear timer for this epoch
        
        for (0..steps_per_epoch) |_| {
            // Get batch
            const batch = try data_loader.getBatch(allocator, training_config.batch_size);
            defer batch.deinit();
            
            // Forward pass
            var outputs = try model.forward(allocator, batch.input_ids, batch.attention_mask);
            defer outputs.deinit();
            
            // Calculate loss
            var loss = try calculateLoss(allocator, outputs.logits, batch.labels);
            defer loss.deinit();
            epoch_loss += loss.value();
            
            // Backward pass
            try loss.backward();
            
            // Optimizer step
            try optimizer_instance.step();
            try optimizer_instance.zeroGrad();
            
            global_step += 1;
            
            // Report progress
            if (global_step % 5 == 0 or global_step == 1) {
                const progress = @as(f32, @floatFromInt(global_step)) / @as(f32, @floatFromInt(total_steps));
                log.info("Step {d}/{d} ({d:.1}%) - Loss: {d:.4}", .{
                    global_step, 
                    total_steps, 
                    progress * 100, 
                    loss.value()
                });
            }
            
            // Save checkpoint
            if (global_step % training_config.save_steps == 0) {
                const checkpoint_dir = try std.fmt.allocPrint(
                    allocator,
                    "{s}/checkpoint-{d}", 
                    .{training_config.output_dir, global_step}
                );
                defer allocator.free(checkpoint_dir);
                
                try saveCheckpoint(allocator, model, checkpoint_dir);
                log.info("Saved checkpoint to {s}", .{checkpoint_dir});
            }
        }
        
        const epoch_time_ns = timer.lap();
        const epoch_time_s = @as(f32, @floatFromInt(epoch_time_ns)) / std.time.ns_per_s;
        
        log.info("Epoch {d}/{d} completed in {d:.2}s - Avg loss: {d:.4}", .{
            epoch + 1,
            epochs,
            epoch_time_s,
            epoch_loss / @as(f32, @floatFromInt(steps_per_epoch))
        });
    }
    
    // 7. Save final model
    log.info("\nSaving model weights...", .{});
    try saveCheckpoint(allocator, model, training_config.output_dir);
    log.info("✅ Model training complete", .{});
}

/// Calculate cross-entropy loss between logits and labels
fn calculateLoss(allocator: Allocator, logits: *tensor.Tensor, labels: *tensor.Tensor) !*tensor.Tensor {
    return tensor.crossEntropyLoss(allocator, logits, labels, .{
        .reduction = .mean,
        .ignore_index = -100,
    });
}

/// Save model checkpoint to specified directory
fn saveCheckpoint(allocator: Allocator, model: deepseek.Model, dir_path: []const u8) !void {
    try fs.cwd().makePath(dir_path);
    
    // Save model config
    const config_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir_path});
    defer allocator.free(config_path);
    try model.config.saveToFile(config_path);
    
    // Save model weights
    const weights_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{dir_path});
    defer allocator.free(weights_path);
    try model.saveWeights(weights_path);
}
