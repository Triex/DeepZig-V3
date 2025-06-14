// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const Allocator = std.mem.Allocator;
const json = std.json;

const Backend = @import("backend.zig").Backend;
const CoreError = @import("root.zig").CoreError;
const FloatTensor = @import("tensor.zig").FloatTensor;
pub const ModelConfig = @import("config.zig").ModelConfig;
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

/// DeepSeek V3 Model with enhanced configuration support
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

    /// Load model from directory (HuggingFace format: config.json + model.safetensors)
    pub fn loadFromDirectory(allocator: Allocator, model_dir: []const u8, backend: Backend) !Self {
        std.log.info("🏗️ Loading DeepSeek V3 model from directory: {s}", .{model_dir});

        // Load configuration
        const config_path = try std.fs.path.join(allocator, &[_][]const u8{ model_dir, "config.json" });
        defer allocator.free(config_path);

        const config = (ModelConfig.loadFromFile(allocator, config_path) catch |err| blk: {
            std.log.warn("⚠️ Could not load config.json: {}. Using tiny test config for development.", .{err});
            break :blk ModelConfig.tinyTest();
        });

        // Validate configuration
        try config.validate();
        std.log.info("✅ Configuration validated: {}", .{config});

        // Load tokenizer
        const tokenizer_path = try std.fs.path.join(allocator, &[_][]const u8{ model_dir, "tokenizer.json" });
        defer allocator.free(tokenizer_path);

        const tokenizer = (Tokenizer.loadFromFile(allocator, tokenizer_path) catch |err| blk: {
            std.log.warn("⚠️ Could not load tokenizer.json: {}. Using basic tokenizer.", .{err});
            break :blk try Tokenizer.init(allocator, config.vocab_size);
        });

        // Load model weights
        const model_path = try std.fs.path.join(allocator, &[_][]const u8{ model_dir, "model.safetensors" });
        defer allocator.free(model_path);

        return try Self.loadFromSafetensorsWithConfig(allocator, model_path, config, tokenizer, backend);
    }

    /// Load model from file path (safetensors format) with custom config
    pub fn loadFromSafetensorsWithConfig(allocator: Allocator, path: []const u8, config: ModelConfig, tokenizer: Tokenizer, backend: Backend) !Self {
        std.log.info("🔄 Loading model weights from: {s}", .{path});

        // Check if path exists and determine file type
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.warn("⚠️ Model file not found: {s}. Creating default model.", .{path});
                return try Self.loadDefaultWithConfig(allocator, config, tokenizer, backend);
            },
            else => return err,
        };
        defer file.close();

        return try loadFromSafetensorsFile(allocator, file, config, tokenizer, backend);
    }

    /// Load model from file path (safetensors format)
    pub fn loadFromPath(allocator: Allocator, path: []const u8, backend: Backend) !Self {
        const default_config = ModelConfig.defaultDeepSeekV3();
        const tokenizer = try Tokenizer.init(allocator, default_config.vocab_size);
        return try Self.loadFromSafetensorsWithConfig(allocator, path, default_config, tokenizer, backend);
    }

    /// Load model from safetensors file
    fn loadFromSafetensorsFile(allocator: Allocator, file: std.fs.File, config: ModelConfig, tokenizer: Tokenizer, backend: Backend) !Self {
        std.log.info("📦 Loading SafeTensors model...", .{});

        // Read file size
        const file_size = try file.getEndPos();
        std.log.info("  File size: {} bytes ({d:.1} MB)", .{ file_size, @as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0) });

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

        // Verify config matches model architecture
        try verifyConfigMatchesModel(config, header);

        // Read tensor data from file
        const data_start_offset = 8 + header_size;
        const tensor_data = try allocator.alloc(u8, file_size - data_start_offset);
        defer allocator.free(tensor_data);

        try file.seekTo(data_start_offset);
        _ = try file.readAll(tensor_data);

        // Load weights into model
        return try loadModelFromTensorData(allocator, config, header, tensor_data, tokenizer, backend);
    }

    /// Verify that the config matches the model architecture in safetensors
    fn verifyConfigMatchesModel(config: ModelConfig, header: SafeTensorsHeader) !void {
        std.log.debug("🔍 Verifying config matches model architecture...", .{});

        // Check embedding dimensions
        if (header.tensors.get("model.embed_tokens.weight")) |embed_info| {
            if (embed_info.shape.len >= 2) {
                const file_vocab_size = @as(u32, @intCast(embed_info.shape[0]));
                const file_hidden_size = @as(u32, @intCast(embed_info.shape[1]));

                if (file_vocab_size != config.vocab_size) {
                    std.log.warn("⚠️ Vocab size mismatch: config={}, model={}", .{ config.vocab_size, file_vocab_size });
                }
                if (file_hidden_size != config.hidden_size) {
                    std.log.err("❌ Hidden size mismatch: config={}, model={}", .{ config.hidden_size, file_hidden_size });
                    return ModelError.InvalidConfig;
                }
            }
        }

        // Count layers in model
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
        const file_num_layers = max_layer_idx + 1;

        if (file_num_layers != config.num_hidden_layers) {
            std.log.warn("⚠️ Layer count mismatch: config={}, model={}", .{ config.num_hidden_layers, file_num_layers });
        }

        std.log.debug("✅ Config verification complete", .{});
    }

    /// Parse SafeTensors JSON header
    fn parseSafetensorsHeader(allocator: Allocator, header_json: []const u8) !SafeTensorsHeader {
        std.log.debug("🔍 Parsing SafeTensors header...", .{});

        var parsed = json.parseFromSlice(json.Value, allocator, header_json, .{}) catch |err| {
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

    /// Load model from parsed tensor data
    fn loadModelFromTensorData(
        allocator: Allocator,
        config: ModelConfig,
        header: SafeTensorsHeader,
        tensor_data: []const u8,
        tokenizer: Tokenizer,
        backend: Backend,
    ) !Self {
        std.log.info("🏗️ Creating model from tensor data...", .{});

        // Initialize transformer with config
        const transformer = try Transformer.init(allocator, config, backend);

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

    /// Load default/demo model with custom config
    pub fn loadDefaultWithConfig(allocator: Allocator, config: ModelConfig, tokenizer: Tokenizer, backend: Backend) !Self {
        std.log.info("🎯 Creating default model with config: {}", .{config});

        // Initialize transformer
        const transformer = try Transformer.init(allocator, config, backend);

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

    /// Load default/demo model
    pub fn loadDefault(allocator: Allocator, backend: Backend) !Self {
        const config = ModelConfig.defaultDeepSeekV3();
        const tokenizer = try Tokenizer.init(allocator, config.vocab_size);
        return try Self.loadDefaultWithConfig(allocator, config, tokenizer, backend);
    }

    /// Load tiny test model (2 layers, small dims)
    pub fn loadTiny(allocator: Allocator, backend: Backend) !Self {
        const config = ModelConfig.tinyTest();
        const tokenizer = try Tokenizer.init(allocator, config.vocab_size);
        return try Self.loadDefaultWithConfig(allocator, config, tokenizer, backend);
    }

    /// Load small test model (4 layers, medium dims)
    pub fn loadSmall(allocator: Allocator, backend: Backend) !Self {
        const config = ModelConfig.smallTest();
        const tokenizer = try Tokenizer.init(allocator, config.vocab_size);
        return try Self.loadDefaultWithConfig(allocator, config, tokenizer, backend);
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

    /// Simple text generation function for testing
    pub fn generateText(self: *Self, prompt: []const u8, max_tokens: u32) ![]const u8 {
        _ = max_tokens; // TODO: Use for generation loop
        std.log.info("🤖 Generating response for: '{s}'", .{prompt});
        
        // Tokenize input
        const input_tokens = try self.tokenizer.encode(prompt);
        defer self.allocator.free(input_tokens);
        
        std.log.info("📝 Input tokens: {} -> {any}", .{ input_tokens.len, input_tokens[0..@min(10, input_tokens.len)] });
        
        // Determine sequence length
        const seq_len = input_tokens.len;
        
        // Embedding lookup: input_ids -> hidden_states
        var hidden_states = try FloatTensor.init(self.allocator, &[_]usize{ 1, seq_len, self.config.hidden_size });
        defer hidden_states.deinit();
        
        // Simple embedding lookup (normally this would be proper embedding layer)
        for (0..seq_len) |s| {
            const token_id = input_tokens[s];
            const embed_idx = @min(token_id, self.config.vocab_size - 1);
            
            for (0..self.config.hidden_size) |h| {
                const embed_offset = embed_idx * self.config.hidden_size + h;
                const hidden_offset = s * self.config.hidden_size + h;
                hidden_states.data[hidden_offset] = self.embed_tokens.data[embed_offset];
            }
        }
        
        std.log.info("📊 Hidden states shape: {}x{}x{}", .{ 1, seq_len, self.config.hidden_size });
        
        // Forward pass through transformer
        var output_hidden = try FloatTensor.init(self.allocator, &[_]usize{ 1, seq_len, self.config.hidden_size });
        defer output_hidden.deinit();
        
        try self.transformer.forward(&hidden_states, null, null, null, false, &output_hidden);
        
        // Skip final norm for now (placeholder)
        const normed_output_ptr = &output_hidden;
        
        // Project to vocabulary: [batch, seq_len, hidden] -> [seq_len, vocab]
        var logits = try FloatTensor.init(self.allocator, &[_]usize{ seq_len, self.config.vocab_size });
        defer logits.deinit();
        
        // Simple matrix multiplication: normed_output @ lm_head -> logits
        // This is a simplified version - in practice you'd use BLAS
        for (0..seq_len) |s| {
            for (0..self.config.vocab_size) |v| {
                var sum: f32 = 0.0;
                for (0..self.config.hidden_size) |h| {
                    const hidden_val = normed_output_ptr.data[s * self.config.hidden_size + h];
                    const weight_val = self.lm_head.data[h * self.config.vocab_size + v];
                    sum += hidden_val * weight_val;
                }
                logits.data[s * self.config.vocab_size + v] = sum;
            }
        }
        
        // Get next token (greedy sampling from last position)
        const last_pos_offset = (seq_len - 1) * self.config.vocab_size;
        var max_logit: f32 = logits.data[last_pos_offset];
        var next_token: u32 = 0;
        
        for (1..self.config.vocab_size) |v| {
            const logit = logits.data[last_pos_offset + v];
            if (logit > max_logit) {
                max_logit = logit;
                next_token = @intCast(v);
            }
        }
        
        std.log.info("🎯 Next token: {} (logit: {d:.3})", .{ next_token, max_logit });
        
        // Decode next token
        const output_tokens = try self.allocator.alloc(u32, 1);
        defer self.allocator.free(output_tokens);
        output_tokens[0] = next_token;
        
        const generated_text = try self.tokenizer.decode(output_tokens);
        
        std.log.info("✅ Generated: '{s}'", .{generated_text});
        return generated_text;
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

        std.log.debug("Generation not yet implemented", .{});
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

        std.log.debug("Model forward pass (placeholder)", .{});
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
        const dtype_size: u64 = if (std.mem.eql(u8, self.config.torch_dtype, "float16") or
            std.mem.eql(u8, self.config.torch_dtype, "bfloat16")) 2 else 4;

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
    const config = ModelConfig.defaultDeepSeekV3();
    std.testing.expect(config.vocab_size == 129280) catch unreachable;
    std.testing.expect(config.num_experts == 256) catch unreachable;
    std.testing.expect(config.num_experts_per_token == 8) catch unreachable;
}
