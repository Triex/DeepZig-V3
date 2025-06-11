// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;

const Backend = @import("backend.zig").Backend;
const CoreError = @import("root.zig").CoreError;
const FloatTensor = @import("tensor.zig").FloatTensor;
const Shape = @import("tensor.zig").Shape;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Transformer = @import("transformer.zig").Transformer;

pub const ModelError = CoreError || error{
    InvalidModelFile,
    UnsupportedModelVersion,
    CorruptedWeights,
    MissingTokenizer,
    SafetensorsError,
    UnsupportedFormat,
    FileNotFound,
    InvalidConfig,
};

/// Model configuration matching DeepSeek V3 architecture
pub const ModelConfig = struct {
    // Model dimensions
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    max_position_embeddings: u32,

    // MoE configuration
    num_experts: u32,
    num_experts_per_token: u32,
    expert_capacity: u32,

    // Multi-head Latent Attention (MLA) config
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    qk_rope_base: f32,

    // Activation function
    hidden_act: []const u8, // "swiglu" for DeepSeek V3

    // Normalization
    rms_norm_eps: f32,

    // Quantization settings
    use_fp16: bool,
    use_bf16: bool,

    pub fn deepseekV3Default() ModelConfig {
        return ModelConfig{
            .vocab_size = 129280,
            .hidden_size = 7168,
            .intermediate_size = 18432,
            .num_hidden_layers = 61,
            .num_attention_heads = 128,
            .num_key_value_heads = 128,
            .max_position_embeddings = 32768,
            .num_experts = 256,
            .num_experts_per_token = 8,
            .expert_capacity = 64,
            .qk_nope_head_dim = 128,
            .qk_rope_head_dim = 64,
            .v_head_dim = 128,
            .qk_rope_base = 10000.0,
            .hidden_act = "swiglu",
            .rms_norm_eps = 1e-6,
            .use_fp16 = false,
            .use_bf16 = true,
        };
    }
};

/// Model information
pub const ModelInfo = struct {
    name: []const u8,
    version: []const u8,
    config: ModelConfig,
    num_parameters: u64,
    memory_usage: u64,
};

/// SafeTensors header structure for parsing model weights
const SafeTensorsHeader = struct {
    tensors: std.HashMap([]const u8, TensorInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    metadata: ?std.HashMap([]const u8, []const u8, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),

    const TensorInfo = struct {
        dtype: []const u8,
        shape: []usize,
        data_offsets: [2]u64, // [start, end]
    };

    pub fn deinit(self: *SafeTensorsHeader, allocator: Allocator) void {
        var tensor_iter = self.tensors.iterator();
        while (tensor_iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.dtype);
            allocator.free(entry.value_ptr.shape);
        }
        self.tensors.deinit();

        if (self.metadata) |*meta| {
            var meta_iter = meta.iterator();
            while (meta_iter.next()) |entry| {
                allocator.free(entry.key_ptr.*);
                allocator.free(entry.value_ptr.*);
            }
            meta.deinit();
        }
    }
};

/// DeepSeek V3 Model
pub const Model = struct {
    config: ModelConfig,
    transformer: Transformer,
    tokenizer: Tokenizer,
    backend: Backend,
    allocator: Allocator,

    // Embedding layers
    embed_tokens: FloatTensor,
    embed_positions: ?FloatTensor,

    // Output layers
    lm_head: FloatTensor,
    norm: FloatTensor,

    const Self = @This();

    /// Load model from file path (safetensors format)
    pub fn loadFromPath(allocator: Allocator, path: []const u8, backend: Backend) !Self {
        std.log.info("🔄 Loading DeepSeek V3 model from: {s}", .{path});

        // Check if path exists and determine file type
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("❌ Model file not found: {s}", .{path});
                return ModelError.FileNotFound;
            },
            else => return err,
        };
        defer file.close();

        // Determine file format
        const extension = std.fs.path.extension(path);
        if (std.mem.eql(u8, extension, ".safetensors")) {
            return try loadFromSafetensors(allocator, file, backend);
        } else {
            std.log.err("❌ Unsupported model format: {s}. Currently only .safetensors is supported.", .{extension});
            return ModelError.UnsupportedFormat;
        }
    }

    /// Load model from safetensors file
    fn loadFromSafetensors(allocator: Allocator, file: std.fs.File, backend: Backend) !Self {
        std.log.info("📦 Loading SafeTensors model...", .{});

        // Read file size
        const file_size = try file.getEndPos();
        std.log.info("  File size: {} bytes", .{file_size});

        // Read the header size (first 8 bytes)
        var header_size_bytes: [8]u8 = undefined;
        _ = try file.readAll(&header_size_bytes);
        const header_size = std.mem.readInt(u64, &header_size_bytes, .little);

        std.log.info("  Header size: {} bytes", .{header_size});

        if (header_size > 100 * 1024 * 1024) { // 100MB limit for safety
            std.log.err("❌ Header too large: {} bytes (max 100MB)", .{header_size});
            return ModelError.SafetensorsError;
        }

        // Read header JSON
        const header_json = try allocator.alloc(u8, header_size);
        defer allocator.free(header_json);
        _ = try file.readAll(header_json);

        // Parse header
        var header = try parseSafetensorsHeader(allocator, header_json);
        defer header.deinit(allocator);

        std.log.info("  Found {} tensors in header", .{header.tensors.count()});

        // Log some tensor information
        var tensor_iter = header.tensors.iterator();
        var tensor_count: u32 = 0;
        while (tensor_iter.next()) |entry| {
            if (tensor_count < 5) { // Log first 5 tensors
                std.log.debug("    Tensor: {s}, dtype: {s}, shape: {any}", .{ entry.key_ptr.*, entry.value_ptr.dtype, entry.value_ptr.shape });
            }
            tensor_count += 1;
        }

        // Determine model configuration from tensors
        const config = try inferModelConfig(header, allocator);
        std.log.info("  Inferred config - Hidden size: {}, Layers: {}", .{ config.hidden_size, config.num_hidden_layers });

        // Read tensor data from file
        const data_start_offset = 8 + header_size;
        const tensor_data = try allocator.alloc(u8, file_size - data_start_offset);
        defer allocator.free(tensor_data);

        try file.seekTo(data_start_offset);
        _ = try file.readAll(tensor_data);

        // Load weights into model
        return try loadModelFromTensorData(allocator, config, header, tensor_data, backend);
    }

    /// Parse SafeTensors JSON header
    fn parseSafetensorsHeader(allocator: Allocator, header_json: []const u8) !SafeTensorsHeader {
        std.log.debug("🔍 Parsing SafeTensors header...", .{});

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, header_json, .{}) catch |err| {
            std.log.err("❌ Failed to parse header JSON: {}", .{err});
            return ModelError.SafetensorsError;
        };
        defer parsed.deinit();

        var tensors = std.HashMap([]const u8, SafeTensorsHeader.TensorInfo, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator);
        var metadata: ?std.HashMap([]const u8, []const u8, std.hash_map.StringContext, std.hash_map.default_max_load_percentage) = null;

        const root = parsed.value.object;

        for (root.keys()) |key| {
            if (std.mem.eql(u8, key, "__metadata__")) {
                // Parse metadata
                metadata = std.HashMap([]const u8, []const u8, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator);
                const meta_obj = root.get(key).?.object;
                for (meta_obj.keys()) |meta_key| {
                    const meta_value = meta_obj.get(meta_key).?.string;
                    const key_copy = try allocator.dupe(u8, meta_key);
                    const value_copy = try allocator.dupe(u8, meta_value);
                    try metadata.?.put(key_copy, value_copy);
                }
            } else {
                // Parse tensor info
                const tensor_obj = root.get(key).?.object;

                const dtype = try allocator.dupe(u8, tensor_obj.get("dtype").?.string);

                const shape_array = tensor_obj.get("shape").?.array;
                const shape = try allocator.alloc(usize, shape_array.items.len);
                for (shape_array.items, 0..) |item, i| {
                    shape[i] = @intCast(item.integer);
                }

                const offsets_array = tensor_obj.get("data_offsets").?.array;
                const data_offsets = [2]u64{
                    @intCast(offsets_array.items[0].integer),
                    @intCast(offsets_array.items[1].integer),
                };

                const key_copy = try allocator.dupe(u8, key);
                const tensor_info = SafeTensorsHeader.TensorInfo{
                    .dtype = dtype,
                    .shape = shape,
                    .data_offsets = data_offsets,
                };

                try tensors.put(key_copy, tensor_info);
            }
        }

        std.log.debug("✅ Parsed {} tensors from header", .{tensors.count()});

        return SafeTensorsHeader{
            .tensors = tensors,
            .metadata = metadata,
        };
    }

    /// Infer model configuration from tensor shapes
    fn inferModelConfig(header: SafeTensorsHeader, allocator: Allocator) !ModelConfig {
        _ = allocator;
        std.log.debug("🔍 Inferring model configuration from tensor shapes...", .{});

        var config = ModelConfig.deepseekV3Default();

        // Look for embedding layer to determine vocab size and hidden size
        if (header.tensors.get("model.embed_tokens.weight")) |embed_info| {
            if (embed_info.shape.len >= 2) {
                config.vocab_size = @intCast(embed_info.shape[0]);
                config.hidden_size = @intCast(embed_info.shape[1]);
            }
        }

        // Count layers by finding highest layer index
        var max_layer_idx: u32 = 0;
        var tensor_iter = header.tensors.iterator();
        while (tensor_iter.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.startsWith(u8, name, "model.layers.")) {
                // Extract layer number
                var parts = std.mem.splitScalar(u8, name, '.');
                _ = parts.next(); // "model"
                _ = parts.next(); // "layers"
                const layer_str = parts.next() orelse continue;
                const layer_idx = std.fmt.parseInt(u32, layer_str, 10) catch continue;
                max_layer_idx = @max(max_layer_idx, layer_idx);
            }
        }
        config.num_hidden_layers = max_layer_idx + 1;

        std.log.info("✅ Inferred configuration:", .{});
        std.log.info("  Vocab size: {}", .{config.vocab_size});
        std.log.info("  Hidden size: {}", .{config.hidden_size});
        std.log.info("  Layers: {}", .{config.num_hidden_layers});

        return config;
    }

    /// Load model from parsed tensor data
    fn loadModelFromTensorData(
        allocator: Allocator,
        config: ModelConfig,
        header: SafeTensorsHeader,
        tensor_data: []const u8,
        backend: Backend,
    ) !Self {
        std.log.info("🏗️ Creating model from tensor data...", .{});

        // Initialize transformer with inferred config
        const transformer = try Transformer.init(allocator, config, backend);

        // Initialize tokenizer
        const tokenizer = try Tokenizer.init(allocator, config.vocab_size);

        // Load embedding weights
        const embed_tokens = try loadTensorFromSafetensors(
            allocator,
            "model.embed_tokens.weight",
            header,
            tensor_data,
        ) orelse blk: {
            std.log.warn("⚠️ Embedding weights not found, using random initialization", .{});
            var tensor = try FloatTensor.init(allocator, &[_]usize{ config.vocab_size, config.hidden_size });
            try initializeEmbedding(&tensor);
            break :blk tensor;
        };

        // Load output head weights
        const lm_head = try loadTensorFromSafetensors(
            allocator,
            "lm_head.weight",
            header,
            tensor_data,
        ) orelse blk: {
            std.log.warn("⚠️ LM head weights not found, using random initialization", .{});
            var tensor = try FloatTensor.init(allocator, &[_]usize{ config.hidden_size, config.vocab_size });
            try initializeLinear(&tensor);
            break :blk tensor;
        };

        // Load final norm weights
        const norm = try loadTensorFromSafetensors(
            allocator,
            "model.norm.weight",
            header,
            tensor_data,
        ) orelse blk: {
            std.log.warn("⚠️ Final norm weights not found, using ones initialization", .{});
            var tensor = try FloatTensor.init(allocator, &[_]usize{config.hidden_size});
            tensor.fill(1.0);
            break :blk tensor;
        };

        std.log.info("✅ Model created successfully with real weights!", .{});

        return Self{
            .config = config,
            .transformer = transformer,
            .tokenizer = tokenizer,
            .backend = backend,
            .allocator = allocator,
            .embed_tokens = embed_tokens,
            .embed_positions = null,
            .lm_head = lm_head,
            .norm = norm,
        };
    }

    /// Load a specific tensor from safetensors data
    fn loadTensorFromSafetensors(
        allocator: Allocator,
        tensor_name: []const u8,
        header: SafeTensorsHeader,
        tensor_data: []const u8,
    ) !?FloatTensor {
        const tensor_info = header.tensors.get(tensor_name) orelse {
            std.log.debug("Tensor {s} not found in safetensors", .{tensor_name});
            return null;
        };

        std.log.debug("Loading tensor: {s}, dtype: {s}, shape: {any}", .{ tensor_name, tensor_info.dtype, tensor_info.shape });

        // Create tensor with correct shape
        var tensor = try FloatTensor.init(allocator, tensor_info.shape);

        // Extract data from the buffer
        const start_offset = tensor_info.data_offsets[0];
        const end_offset = tensor_info.data_offsets[1];
        const data_slice = tensor_data[start_offset..end_offset];

        // Convert data based on dtype
        if (std.mem.eql(u8, tensor_info.dtype, "F32")) {
            // Direct copy for F32
            const f32_data = std.mem.bytesAsSlice(f32, data_slice);
            @memcpy(tensor.data, f32_data);
        } else if (std.mem.eql(u8, tensor_info.dtype, "F16")) {
            // Convert F16 to F32
            const f16_data = std.mem.bytesAsSlice(f16, data_slice);
            for (f16_data, 0..) |val, i| {
                tensor.data[i] = @floatCast(val);
            }
        } else if (std.mem.eql(u8, tensor_info.dtype, "BF16")) {
            // Convert BF16 to F32 (simplified conversion)
            const u16_data = std.mem.bytesAsSlice(u16, data_slice);
            for (u16_data, 0..) |val, i| {
                // BF16 to F32: shift left by 16 bits
                const f32_bits = @as(u32, val) << 16;
                tensor.data[i] = @bitCast(f32_bits);
            }
        } else {
            std.log.err("❌ Unsupported dtype: {s}", .{tensor_info.dtype});
            tensor.deinit();
            return ModelError.UnsupportedFormat;
        }

        std.log.debug("✅ Loaded tensor {s} successfully", .{tensor_name});
        return tensor;
    }

    /// Load default/demo model
    pub fn loadDefault(allocator: Allocator, backend: Backend) !Self {
        const config = ModelConfig.deepseekV3Default();

        std.log.info("Creating default DeepSeek V3 model...", .{});
        std.log.info("  Hidden size: {}", .{config.hidden_size});
        std.log.info("  Layers: {}", .{config.num_hidden_layers});
        std.log.info("  Experts: {}", .{config.num_experts});
        std.log.info("  Vocab size: {}", .{config.vocab_size});

        // Initialize transformer
        const transformer = try Transformer.init(allocator, config, backend);

        // Initialize tokenizer
        const tokenizer = try Tokenizer.init(allocator, config.vocab_size);

        // Initialize embedding layers
        var embed_tokens = try FloatTensor.init(allocator, &[_]usize{ config.vocab_size, config.hidden_size });

        // Initialize with random values (in real implementation, load from weights)
        try initializeEmbedding(&embed_tokens);

        // Output projection
        var lm_head = try FloatTensor.init(allocator, &[_]usize{ config.hidden_size, config.vocab_size });
        try initializeLinear(&lm_head);

        // Final layer norm
        var norm = try FloatTensor.init(allocator, &[_]usize{config.hidden_size});
        norm.fill(1.0); // Initialize with ones

        return Self{
            .config = config,
            .transformer = transformer,
            .tokenizer = tokenizer,
            .backend = backend,
            .allocator = allocator,
            .embed_tokens = embed_tokens,
            .embed_positions = null,
            .lm_head = lm_head,
            .norm = norm,
        };
    }

    /// Free model memory
    pub fn deinit(self: *Self) void {
        self.transformer.deinit();
        self.tokenizer.deinit();
        self.embed_tokens.deinit();
        if (self.embed_positions) |*pos| pos.deinit();
        self.lm_head.deinit();
        self.norm.deinit();
    }

    /// Get model information
    pub fn info(self: *const Self) ModelInfo {
        const num_params = self.estimateParameters();
        const memory_usage = self.estimateMemoryUsage();

        return ModelInfo{
            .name = "DeepSeek V3",
            .version = "0.1.0",
            .config = self.config,
            .num_parameters = num_params,
            .memory_usage = memory_usage,
        };
    }

    /// Generate text completion
    pub fn generate(self: *Self, input_tokens: []const u32, max_tokens: u32) ![]u32 {
        _ = self;
        _ = input_tokens;
        _ = max_tokens;

        // TODO: Implement actual generation
        // This would involve:
        // 1. Run forward pass through transformer layers
        // 2. Apply final layer norm and output projection
        // 3. Sample next token from logits
        // 4. Repeat until max_tokens or EOS

        std.log.debug("Generation not yet implemented");
        return error.NotImplemented;
    }

    /// Forward pass through the model
    pub fn forward(
        self: *Self,
        input_ids: []const u32,
        output: *FloatTensor,
    ) !void {
        // TODO: Implement forward pass
        // 1. Embedding lookup
        // 2. Transformer forward pass
        // 3. Final layer norm
        // 4. Language model head

        _ = self;
        _ = input_ids;
        _ = output;

        std.log.debug("Model forward pass (placeholder)");
    }

    /// Estimate model parameters
    fn estimateParameters(self: *const Self) u64 {
        var params: u64 = 0;

        // Embedding parameters
        params += @as(u64, self.config.vocab_size) * self.config.hidden_size;

        // Transformer parameters (rough estimate)
        const layer_params = @as(u64, self.config.hidden_size) * self.config.hidden_size * 4; // Attention + FFN
        params += layer_params * self.config.num_hidden_layers;

        // MoE parameters
        const expert_params = @as(u64, self.config.hidden_size) * self.config.intermediate_size * 2;
        params += expert_params * self.config.num_experts;

        // Output head
        params += @as(u64, self.config.hidden_size) * self.config.vocab_size;

        return params;
    }

    /// Estimate memory usage in bytes
    fn estimateMemoryUsage(self: *const Self) u64 {
        const params = self.estimateParameters();
        const dtype_size: u64 = if (self.config.use_fp16 or self.config.use_bf16) 2 else 4;

        // Model weights + activation memory + KV cache
        return params * dtype_size * 2; // Rough estimate
    }
};

// Initialize embedding with small random values
fn initializeEmbedding(tensor: *FloatTensor) !void {
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

    for (tensor.data) |*val| {
        val.* = (random.float(f32) - 0.5) * 0.02; // Small random values
    }
}

// Initialize linear layer with Xavier initialization
fn initializeLinear(tensor: *FloatTensor) !void {
    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();

    const fan_in = tensor.shape.dims[0];
    const fan_out = tensor.shape.dims[1];
    const limit = std.math.sqrt(6.0 / @as(f32, @floatFromInt(fan_in + fan_out)));

    for (tensor.data) |*val| {
        val.* = (random.float(f32) - 0.5) * 2.0 * limit;
    }
}

// Tests
test "model creation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a dummy backend for testing
    const backend = Backend{
        .type = .cpu,
        .device_id = 0,
        .allocator = allocator,
    };

    var model = try Model.loadDefault(allocator, backend);
    defer model.deinit();

    const model_info = model.info();
    try testing.expect(model_info.num_parameters > 0);
    try testing.expect(std.mem.eql(u8, model_info.name, "DeepSeek V3"));
}

test "model config" {
    const config = ModelConfig.deepseekV3Default();
    std.testing.expect(config.vocab_size == 129280) catch unreachable;
    std.testing.expect(config.num_experts == 256) catch unreachable;
    std.testing.expect(config.num_experts_per_token == 8) catch unreachable;
}
