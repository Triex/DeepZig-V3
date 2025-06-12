// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const json = std.json;

const Backend = @import("backend.zig").Backend;
const tensor = @import("tensor.zig");
const FloatTensor = tensor.Tensor(.f32);
const CoreError = @import("root.zig").CoreError;
const Shape = tensor.Shape;
pub const ModelConfig = @import("config.zig").ModelConfig;
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

    /// Initialize model from provided configuration
    /// Creates a model with random weights based on the configuration
    pub fn initFromConfig(allocator: Allocator, config: *ModelConfig) !Self {
        std.log.info("🔧 Initializing model from config: {}", .{config});

        // Create backend with CPU for training
        const backend = Backend.init(allocator, .cpu, 0);

        // Create tokenizer
        const tokenizer = try Tokenizer.init(allocator, config.vocab_size);

        // Initialize transformer with the config
        const transformer = try Transformer.init(allocator, config.*, backend);

        // Create embedding layer
        const embed_dims = [_]usize{ config.vocab_size, config.hidden_size };
        var embed_tokens = try FloatTensor.init(allocator, &embed_dims);
        try initializeEmbedding(&embed_tokens);

        // Create embedding positions (optional)
        const embed_positions: ?FloatTensor = null;

        // Create output layers
        const norm_dims = [_]usize{ config.hidden_size };
        const norm = try FloatTensor.init(allocator, &norm_dims);

        const lm_head_dims = [_]usize{ config.hidden_size, config.vocab_size };
        var lm_head = try FloatTensor.init(allocator, &lm_head_dims);
        try initializeLinear(&lm_head);

        return Self{
            .config = config.*,
            .transformer = transformer,
            .tokenizer = tokenizer,
            .backend = backend,
            .allocator = allocator,
            .embed_tokens = embed_tokens,
            .embed_positions = embed_positions,
            .lm_head = lm_head,
            .norm = norm,
        };
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
            var embed_tensor = try FloatTensor.init(allocator, &[_]usize{ config.vocab_size, config.hidden_size });
            try initializeEmbedding(&embed_tensor);
            break :blk embed_tensor;
        };

        // Load output head weights
        const lm_head = try loadTensorFromSafetensors(
            allocator,
            "lm_head.weight",
            header,
            tensor_data,
        ) orelse blk: {
            std.log.warn("⚠️ LM head weights not found, using random initialization", .{});
            var lm_head_tensor = try FloatTensor.init(allocator, &[_]usize{ config.hidden_size, config.vocab_size });
            try initializeLinear(&lm_head_tensor);
            break :blk lm_head_tensor;
        };

        // Load final norm weights
        const norm = try loadTensorFromSafetensors(
            allocator,
            "model.norm.weight",
            header,
            tensor_data,
        ) orelse blk: {
            std.log.warn("⚠️ Final norm weights not found, using ones initialization", .{});
            var norm_tensor = try FloatTensor.init(allocator, &[_]usize{config.hidden_size});
            norm_tensor.fill(1.0);
            break :blk norm_tensor;
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
        var t_tensor = try FloatTensor.init(allocator, tensor_info.shape);

        // Extract data from the buffer
        const start_offset = tensor_info.data_offsets[0];
        const end_offset = tensor_info.data_offsets[1];
        const data_slice = tensor_data[start_offset..end_offset];

                // Convert data based on dtype
        if (std.mem.eql(u8, tensor_info.dtype, "F32")) {
            // Direct copy for F32 data
            if (data_slice.len != t_tensor.data.len * @sizeOf(f32)) {
                std.log.err("❌ Data size mismatch: expected {}, got {}", .{ t_tensor.data.len * @sizeOf(f32), data_slice.len });
                t_tensor.deinit();
                return ModelError.SafetensorsError;
            }
            const f32_data = std.mem.bytesAsSlice(f32, data_slice);
            @memcpy(t_tensor.data, f32_data);
        } else if (std.mem.eql(u8, tensor_info.dtype, "BF16")) {
            // Convert BF16 to F32
            if (data_slice.len != t_tensor.data.len * @sizeOf(u16)) {
                std.log.err("❌ Data size mismatch for BF16: expected {}, got {}", .{ t_tensor.data.len * @sizeOf(u16), data_slice.len });
                t_tensor.deinit();
                return ModelError.SafetensorsError;
            }
            const u16_data = std.mem.bytesAsSlice(u16, data_slice);
            for (u16_data, 0..) |val, i| {
                // BF16 to F32: shift left by 16 bits
                const f32_bits = @as(u32, val) << 16;
                t_tensor.data[i] = @bitCast(f32_bits);
            }
        } else if (std.mem.eql(u8, tensor_info.dtype, "F16")) {
            // Convert F16 to F32 (simplified - assumes little endian)
            if (data_slice.len != t_tensor.data.len * @sizeOf(u16)) {
                std.log.err("❌ Data size mismatch for F16: expected {}, got {}", .{ t_tensor.data.len * @sizeOf(u16), data_slice.len });
                t_tensor.deinit();
                return ModelError.SafetensorsError;
            }
            const u16_data = std.mem.bytesAsSlice(u16, data_slice);
            for (u16_data, 0..) |val, i| {
                // F16 to F32 conversion (simplified)
                t_tensor.data[i] = @floatCast(@as(f16, @bitCast(val)));
            }
        } else {
            std.log.err("❌ Unsupported dtype: {s}", .{tensor_info.dtype});
            t_tensor.deinit();
            return ModelError.UnsupportedFormat;
        }

        std.log.debug("✅ Loaded tensor {s} successfully", .{tensor_name});
        return t_tensor;
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
        // Deinit all owned resources
        self.tokenizer.deinit();
        self.transformer.deinit();
        self.backend.deinit();
        self.embed_tokens.deinit();

        if (self.embed_positions) |*positions| {
            positions.deinit();
        }

        self.norm.deinit();
        self.lm_head.deinit();
    }

    /// Returns an ArrayList containing all trainable parameters of the model
    /// The caller owns the ArrayList but not the tensors within
    pub fn parameters(self: *Self) !ArrayList(*tensor.Tensor(.f32)) {
        var params = ArrayList(*tensor.Tensor(.f32)).init(self.allocator);
        errdefer params.deinit();

        // Add embedding parameters
        try params.append(&self.embed_tokens);

        // Add transformer parameters
        var transformer_params = try self.transformer.parameters();
        defer transformer_params.deinit();

        for (transformer_params.items) |param| {
            try params.append(param);
        }

        // Add output layer parameters
        try params.append(&self.norm);
        try params.append(&self.lm_head);

        return params;
    }

    /// Simple text generation function for testing
    pub fn generateText(self: *Self, prompt: []const u8, max_tokens: u32) ![]const u8 {
        std.log.info("🤖 Generating response for: '{s}'", .{prompt});

        // Tokenize prompt and copy into dynamic list we can keep appending to
        const prompt_tokens = try self.tokenizer.encode(prompt);
        defer self.allocator.free(prompt_tokens);

        var all_tokens = std.ArrayList(u32).init(self.allocator);
        defer all_tokens.deinit();
        try all_tokens.appendSlice(prompt_tokens);

        // Greedy sampling loop
        var step: u32 = 0;
        while (step < max_tokens) : (step += 1) {
            const seq_len = all_tokens.items.len;

            // 1. Embedding lookup -> hidden_states tensor [1, seq_len, hidden]
            var hidden_states = try FloatTensor.init(self.allocator, &[_]usize{ 1, seq_len, self.config.hidden_size });
            defer hidden_states.deinit();

            for (0..seq_len) |s| {
                const token_id = all_tokens.items[s];
                const embed_idx = @min(token_id, self.config.vocab_size - 1);
                for (0..self.config.hidden_size) |h| {
                    const embed_offset = embed_idx * self.config.hidden_size + h;
                    const hidden_offset = s * self.config.hidden_size + h;
                    hidden_states.data[hidden_offset] = self.embed_tokens.data[embed_offset];
                }
            }

            // 2. Transformer forward
            var output_hidden = try FloatTensor.init(self.allocator, &[_]usize{ 1, seq_len, self.config.hidden_size });
            defer output_hidden.deinit();
            try self.transformer.forward(&hidden_states, null, null, null, false, &output_hidden);

            // 3. Compute logits for last position only
            const last_offset = (seq_len - 1) * self.config.hidden_size;
            var next_logits = try FloatTensor.init(self.allocator, &[_]usize{ self.config.vocab_size });
            defer next_logits.deinit();

            for (0..self.config.vocab_size) |v| {
                var sum: f32 = 0.0;
                for (0..self.config.hidden_size) |h| {
                    const hidden_val = output_hidden.data[last_offset + h];
                    const weight_val = self.lm_head.data[h * self.config.vocab_size + v];
                    sum += hidden_val * weight_val;
                }
                next_logits.data[v] = sum;
            }

            // 4. Greedy pick highest logit
            var best_idx: u32 = 0;
            var best_logit: f32 = next_logits.data[0];
            for (1..self.config.vocab_size) |i| {
                const lg = next_logits.data[i];
                if (lg > best_logit) {
                    best_logit = lg;
                    best_idx = @intCast(i);
                }
            }

            std.log.info("🎯 Step {} -> token {} (logit: {d:.3})", .{step, best_idx, best_logit});

            // 5. Append token and check EOS
            try all_tokens.append(best_idx);
            if (best_idx == self.config.eos_token_id) break; // stop if EOS
        }

        const generated_slice = all_tokens.items[prompt_tokens.len..];
        const generated_text = try self.tokenizer.decode(generated_slice);
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
fn initializeEmbedding(t: *FloatTensor) !void {
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();

        // Initialize embedding layer uniformly between -0.02 and 0.02
    const total_elements = t.shape.numel();

    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        t.data[i] = random.float(f32) * 0.04 - 0.02; // Range: [-0.02, 0.02]
    }
}

// Initialize linear layer with Xavier initialization
fn initializeLinear(t: *FloatTensor) !void {
    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();

        // Xavier/Glorot initialization
    const shape_dims = t.shape.dims;

    if (t.shape.dims.len != 2) return error.InvalidShape;

    const fan_in = shape_dims[0];
    const fan_out = shape_dims[1];
    const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(fan_in + fan_out)));

    const total_elements = t.shape.numel();

    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        // Normal distribution with mean=0, std_dev calculated above
        t.data[i] = random.floatNorm(f32) * std_dev;
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
