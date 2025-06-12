// DeepSeek V3 Core Module
// This module contains the fundamental components for LLM inference

const std = @import("std");

pub const Attention = @import("attention.zig").Attention;
pub const Backend = @import("backend.zig").Backend;
pub const blas = @import("blas.zig");
pub const Config = @import("config.zig").Config;
pub const math = @import("math/root.zig");
pub const memory = @import("memory.zig");
pub const Model = @import("model.zig").Model;
pub const ModelConfig = @import("config.zig").ModelConfig;
pub const MoE = @import("moe.zig").MoE;
pub const Shape = @import("tensor.zig").Shape;
pub const tensor = @import("tensor.zig");
pub const FloatTensor = tensor.FloatTensor;
pub const DoubleTensor = tensor.DoubleTensor;
pub const IntTensor = tensor.IntTensor;
pub const ByteTensor = tensor.ByteTensor;
pub const createMatrix = tensor.createMatrix;
pub const createVector = tensor.createVector;
pub const benchmarkTensorOps = tensor.benchmarkTensorOps;
pub const TensorDType = @import("tensor.zig").TensorDType;
pub const TensorShape = @import("tensor.zig").TensorShape;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const Transformer = @import("transformer.zig").Transformer;

// Generation pipeline
pub const Generation = struct {
    model: *Model,
    tokenizer: *Tokenizer,

    const Self = @This();

    pub fn init(model: *Model, tokenizer: *Tokenizer) Self {
        return Self{
            .model = model,
            .tokenizer = tokenizer,
        };
    }

    /// Generate text using greedy decoding
    pub fn generate(
        self: *Self,
        prompt: []const u8,
        max_new_tokens: u32,
        temperature: f32,
        top_k: u32,
    ) ![]u8 {
        std.log.info("🎯 Generating text for prompt: '{s}'", .{prompt});

        // Tokenize prompt
        const input_tokens = try self.tokenizer.encodeWithSpecialTokens(prompt, true, false);
        defer self.tokenizer.allocator.free(input_tokens);

        std.log.debug("📝 Prompt tokens: {any}", .{input_tokens});

        // TODO: Run model inference
        // For now, return a placeholder response
        _ = temperature;
        _ = top_k;
        _ = max_new_tokens;

        const response = "This is a placeholder response from DeepZig V3! Model inference pipeline coming soon.";
        return try self.tokenizer.allocator.dupe(u8, response);
    }

    /// Simple greedy decoding (deterministic)
    pub fn generateGreedy(self: *Self, prompt: []const u8, max_new_tokens: u32) ![]u8 {
        return try self.generate(prompt, max_new_tokens, 0.0, 1);
    }
};

// Core tensor and math components
// Tensor type aliases for convenience
// Helper functions
// Other core components (may need implementation)
// Math utilities
// Memory management
// Configuration
// Error types
pub const CoreError = error{
    InvalidTensorShape,
    UnsupportedOperation,
    ModelLoadError,
    TokenizerError,
    BackendError,
    OutOfMemory,
    InvalidConfiguration,
    GenerationError,
};

// Version information
pub const version = struct {
    pub const major = 0;
    pub const minor = 1;
    pub const patch = 0;
    pub const string = "0.1.0";
};

// Core test suite
test "core module" {
    const testing = std.testing;

    // Basic smoke tests
    try testing.expect(version.major == 0);
    try testing.expect(version.minor == 1);
}

// Utility functions
pub fn init() void {
    // TODO: Initialize any global state if needed
    std.log.info("DeepSeek V3 Core initialized (v{s})", .{version.string});
}

pub fn deinit() void {
    // TODO: Cleanup any global state
    std.log.info("DeepSeek V3 Core deinitialized");
}

// Tests for new components
test "ModelConfig validation" {
    const config = ModelConfig.defaultDeepSeekV3();
    try config.validate();
}

test "Generation pipeline init" {
    const testing = std.testing;

    // We can't easily test the full pipeline without a real model,
    // but we can test the basic structure
    const backend_instance = Backend.init(.cpu);

    var model = try Model.loadDefault(testing.allocator, backend_instance);
    defer model.deinit();

    var tokenizer = try Tokenizer.init(testing.allocator, 1000);
    defer tokenizer.deinit();

    var generator = Generation.init(&model, &tokenizer);
    const result = try generator.generateGreedy("Hello", 10);
    defer testing.allocator.free(result);

    try testing.expect(result.len > 0);
}
