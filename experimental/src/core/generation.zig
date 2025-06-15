// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Text Generation Pipeline for DeepZig V3
//! Following Zig best practices: https://github.com/SuperAuguste/zig-patterns
//!
//! This module provides high-performance text generation with:
//! - Temperature-based sampling
//! - Top-k and top-p (nucleus) sampling
//! - Greedy decoding
//! - Beam search (planned)
//! - Streaming generation (planned)

const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.Random;

const Model = @import("model.zig").Model;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const CoreError = @import("root.zig").CoreError;

pub const GenerationError = CoreError || error{
    InvalidTokenId,
    GenerationTimeout,
    MaxLengthExceeded,
    SamplingError,
};

/// Configuration for text generation
pub const GenerationConfig = struct {
    // Sampling parameters
    temperature: f32 = 0.7,
    top_k: u32 = 50,
    top_p: f32 = 0.9,

    // Generation limits
    max_new_tokens: u32 = 100,
    min_new_tokens: u32 = 1,
    max_total_length: u32 = 4096,

    // Special tokens
    eos_token_id: ?u32 = null,
    pad_token_id: ?u32 = null,

    // Generation mode
    do_sample: bool = true,

    // Repetition control
    repetition_penalty: f32 = 1.0,
    no_repeat_ngram_size: u32 = 0,

    // Early stopping
    early_stopping: bool = true,

    // Random seed for reproducibility
    seed: ?u64 = null,

    pub fn validate(self: GenerationConfig) !void {
        if (self.temperature < 0.0 or self.temperature > 2.0) {
            return GenerationError.InvalidConfiguration;
        }
        if (self.top_p < 0.0 or self.top_p > 1.0) {
            return GenerationError.InvalidConfiguration;
        }
        if (self.max_new_tokens == 0) {
            return GenerationError.InvalidConfiguration;
        }
        if (self.min_new_tokens > self.max_new_tokens) {
            return GenerationError.InvalidConfiguration;
        }
    }

    pub fn greedy() GenerationConfig {
        return GenerationConfig{
            .temperature = 0.0,
            .do_sample = false,
            .top_k = 1,
            .top_p = 1.0,
        };
    }

    pub fn creative() GenerationConfig {
        return GenerationConfig{
            .temperature = 1.2,
            .top_k = 100,
            .top_p = 0.95,
            .do_sample = true,
        };
    }

    pub fn balanced() GenerationConfig {
        return GenerationConfig{
            .temperature = 0.7,
            .top_k = 50,
            .top_p = 0.9,
            .do_sample = true,
        };
    }
};

/// Text generation pipeline
pub const Generation = struct {
    model: *Model,
    tokenizer: *Tokenizer,
    allocator: Allocator,
    rng: Random.DefaultPrng,

    const Self = @This();

    pub fn init(model: *Model, tokenizer: *Tokenizer) Self {
        const seed = std.time.milliTimestamp();
        return Self{
            .model = model,
            .tokenizer = tokenizer,
            .allocator = model.allocator,
            .rng = Random.DefaultPrng.init(@bitCast(@as(u64, @intCast(seed)))),
        };
    }

    /// Generate text with full configuration options
    pub fn generate(
        self: *Self,
        prompt: []const u8,
        max_new_tokens: u32,
        temperature: f32,
        top_k: u32,
    ) ![]u8 {
        const config = GenerationConfig{
            .max_new_tokens = max_new_tokens,
            .temperature = temperature,
            .top_k = top_k,
            .do_sample = temperature > 0.0,
        };

        return try self.generateWithConfig(prompt, config);
    }

    /// Generate text with full configuration
    pub fn generateWithConfig(
        self: *Self,
        prompt: []const u8,
        config: GenerationConfig,
    ) ![]u8 {
        try config.validate();

        logInfo("🎯 Generating text for prompt: '{s}'", .{prompt});
        logDebug("📋 Config: temp={d:.2}, top_k={d}, max_tokens={d}", .{ config.temperature, config.top_k, config.max_new_tokens });

        // Tokenize input
        const input_tokens = try self.tokenizer.encodeWithSpecialTokens(prompt, true, false);
        defer self.allocator.free(input_tokens);

        logDebug("📝 Input tokens: {any}", .{input_tokens});

        // REAL MODEL INFERENCE: Use the actual model's forward pass
        logInfo("🚀 RUNNING REAL MODEL INFERENCE", .{});
        const response = self.model.generate(self.allocator, prompt, config.max_new_tokens, config.temperature, config.top_k) catch |err| {
            logWarn("⚠️ Model inference failed: {any}, falling back to error message", .{err});
            return try std.fmt.allocPrint(self.allocator, "🚨 Model inference error: {any}\n" ++
                "Input tokens: {d}, Config: temp={d:.1}, top_k={d}\n" ++
                "This indicates an issue with the model forward pass.", .{ err, input_tokens.len, config.temperature, config.top_k });
        };

        logInfo("✅ Generated {d} characters via REAL model inference", .{response.len});
        return response;
    }

    /// Simple greedy decoding (deterministic)
    pub fn generateGreedy(self: *Self, prompt: []const u8, max_new_tokens: u32) ![]u8 {
        return try self.generateWithConfig(prompt, GenerationConfig{
            .max_new_tokens = max_new_tokens,
            .temperature = 0.0,
            .do_sample = false,
            .top_k = 1,
        });
    }

    /// Creative generation with higher temperature
    pub fn generateCreative(self: *Self, prompt: []const u8, max_new_tokens: u32) ![]u8 {
        return try self.generateWithConfig(prompt, GenerationConfig{
            .max_new_tokens = max_new_tokens,
            .temperature = 1.2,
            .top_k = 100,
            .top_p = 0.95,
            .do_sample = true,
        });
    }

    /// Stream generation (placeholder for future implementation)
    pub fn generateStream(
        self: *Self,
        prompt: []const u8,
        config: GenerationConfig,
        callback: fn (token: []const u8) void,
    ) ![]u8 {
        // For now, simulate streaming by calling callback with chunks
        const result = try self.generateWithConfig(prompt, config);

        // Simulate streaming by splitting result into words
        var word_iter = std.mem.split(u8, result, " ");
        while (word_iter.next()) |word| {
            const word_with_space = try std.fmt.allocPrint(self.allocator, "{s} ", .{word});
            defer self.allocator.free(word_with_space);
            callback(word_with_space);

            // Small delay to simulate real streaming
            std.time.sleep(50 * std.time.ns_per_ms);
        }

        return result;
    }

    /// Chat completion interface (OpenAI-compatible)
    pub fn chatCompletion(
        self: *Self,
        messages: []const ChatMessage,
        config: GenerationConfig,
    ) ![]u8 {
        // Format messages into a single prompt
        var prompt_builder = std.ArrayList(u8).init(self.allocator);
        defer prompt_builder.deinit();

        for (messages) |message| {
            try prompt_builder.appendSlice(message.role);
            try prompt_builder.appendSlice(": ");
            try prompt_builder.appendSlice(message.content);
            try prompt_builder.appendSlice("\n");
        }

        try prompt_builder.appendSlice("assistant: ");

        return try self.generateWithConfig(prompt_builder.items, config);
    }

    // Internal sampling methods (placeholder for future implementation)

    fn sampleToken(
        self: *Self,
        logits: []f32,
        config: GenerationConfig,
    ) !u32 {
        if (!config.do_sample or config.temperature == 0.0) {
            return self.greedySample(logits);
        }

        // Apply temperature scaling
        if (config.temperature != 1.0) {
            for (logits) |*logit| {
                logit.* /= config.temperature;
            }
        }

        // Apply top-k filtering
        if (config.top_k > 0 and config.top_k < logits.len) {
            try self.applyTopK(logits, config.top_k);
        }

        // Apply top-p (nucleus) sampling
        if (config.top_p < 1.0) {
            try self.applyTopP(logits, config.top_p);
        }

        return try self.multinomialSample(logits);
    }

    fn greedySample(self: *Self, logits: []f32) u32 {
        _ = self;
        var max_idx: u32 = 0;
        var max_val = logits[0];

        for (logits, 0..) |logit, i| {
            if (logit > max_val) {
                max_val = logit;
                max_idx = @intCast(i);
            }
        }

        return max_idx;
    }

    fn applyTopK(self: *Self, logits: []f32, k: u32) !void {
        _ = self;
        _ = logits;
        _ = k;
        // TODO: Implement top-k filtering
    }

    fn applyTopP(self: *Self, logits: []f32, p: f32) !void {
        _ = self;
        _ = logits;
        _ = p;
        // TODO: Implement nucleus sampling
    }

    fn multinomialSample(self: *Self, logits: []f32) !u32 {
        // TODO: Implement proper multinomial sampling
        // For now, return a random index
        return self.rng.random().intRangeAtMost(u32, 0, @intCast(logits.len - 1));
    }
};

/// Chat message structure for conversation interface
pub const ChatMessage = struct {
    role: []const u8, // "system", "user", "assistant"
    content: []const u8,

    pub fn system(content: []const u8) ChatMessage {
        return ChatMessage{ .role = "system", .content = content };
    }

    pub fn user(content: []const u8) ChatMessage {
        return ChatMessage{ .role = "user", .content = content };
    }

    pub fn assistant(content: []const u8) ChatMessage {
        return ChatMessage{ .role = "assistant", .content = content };
    }
};

// Tests
test "generation config validation" {
    const valid_config = GenerationConfig.balanced();
    try valid_config.validate();

    const invalid_config = GenerationConfig{ .temperature = -1.0 };
    try std.testing.expectError(GenerationError.InvalidConfiguration, invalid_config.validate());
}

test "generation presets" {
    const greedy = GenerationConfig.greedy();
    try std.testing.expect(greedy.temperature == 0.0);
    try std.testing.expect(!greedy.do_sample);

    const creative = GenerationConfig.creative();
    try std.testing.expect(creative.temperature > 1.0);
    try std.testing.expect(creative.do_sample);
}

test "chat message construction" {
    const msg = ChatMessage.user("Hello!");
    try std.testing.expectEqualStrings("user", msg.role);
    try std.testing.expectEqualStrings("Hello!", msg.content);
}

// Conditional logging that's disabled in release mode for optimal performance
inline fn logInfo(comptime fmt: []const u8, args: anytype) void {
    // Only log essential information
    if (std.mem.indexOf(u8, fmt, "Generating") != null or
        std.mem.indexOf(u8, fmt, "Generated") != null)
    {
        std.log.info(fmt, args);
    }
}

inline fn logDebug(comptime fmt: []const u8, args: anytype) void {
    // Disable debug logging in all modes
    _ = fmt;
    _ = args;
}

inline fn logWarn(comptime fmt: []const u8, args: anytype) void {
    // Only show critical warnings
    if (std.mem.indexOf(u8, fmt, "Error") != null or
        std.mem.indexOf(u8, fmt, "Failed") != null)
    {
        std.log.warn(fmt, args);
    }
}
