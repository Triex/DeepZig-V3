const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;

/// Global configuration for DeepSeek V3
pub const Config = struct {
    log_level: std.log.Level = .info,
    enable_telemetry: bool = false,
    cache_dir: ?[]const u8 = null,

    pub fn loadFromEnv() Config {
        // TODO: Load configuration from environment variables
        return Config{};
    }
};

/// Model configuration matching DeepSeek V3 architecture
/// Based on HuggingFace transformers config.json format
pub const ModelConfig = struct {
    // Core architecture
    architectures: []const []const u8 = &[_][]const u8{"DeepseekV3ForCausalLM"},
    model_type: []const u8 = "deepseek_v3",

    // Model dimensions
    vocab_size: u32 = 129280,
    hidden_size: u32 = 7168,
    intermediate_size: u32 = 18432,
    num_hidden_layers: u32 = 61,
    num_attention_heads: u32 = 128,
    num_key_value_heads: u32 = 128,
    max_position_embeddings: u32 = 32768,

    // Multi-Head Latent Attention (MLA) - DeepSeek V3's key innovation
    qk_nope_head_dim: u32 = 128,
    qk_rope_head_dim: u32 = 64,
    v_head_dim: u32 = 128,
    qk_rope_base: f32 = 10000.0,

    // Mixture of Experts (MoE) configuration
    num_experts: u32 = 256,
    num_experts_per_token: u32 = 8,
    expert_capacity: ?u32 = null,
    moe_layer_freq: u32 = 1, // Apply MoE every N layers
    first_k_dense_replace: u32 = 1, // First K layers are dense (not MoE)
    moe_intermediate_size: u32 = 2048,

    // Activation and normalization
    hidden_act: []const u8 = "swiglu",
    rms_norm_eps: f32 = 1e-6,

    // Training configuration
    bos_token_id: u32 = 100000,
    eos_token_id: u32 = 100001,
    pad_token_id: ?u32 = null,
    tie_word_embeddings: bool = false,

    // Precision and optimization
    torch_dtype: []const u8 = "bfloat16",
    use_cache: bool = true,
    attention_dropout: f32 = 0.0,
    hidden_dropout: f32 = 0.0,

    // Rope configuration
    rope_theta: f32 = 10000.0,
    rope_scaling: ?RopeScaling = null,

    // Generation defaults
    max_new_tokens: u32 = 4096,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    top_k: u32 = 50,
    repetition_penalty: f32 = 1.1,

    pub const RopeScaling = struct {
        type: []const u8, // "linear" or "dynamic"
        factor: f32,
    };

    /// Load configuration from HuggingFace config.json
    pub fn loadFromFile(allocator: Allocator, config_path: []const u8) !ModelConfig {
        std.log.info("🔧 Loading model config from: {s}", .{config_path});

        const file = std.fs.cwd().openFile(config_path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.warn("⚠️ Config file not found, using default DeepSeek V3 config");
                return defaultDeepSeekV3();
            },
            else => return err,
        };
        defer file.close();

        const contents = try file.readToEndAlloc(allocator, 1024 * 1024); // 1MB limit
        defer allocator.free(contents);

        return try parseFromJson(allocator, contents);
    }

    /// Parse configuration from JSON string
    pub fn parseFromJson(allocator: Allocator, json_str: []const u8) !ModelConfig {
        var parsed = json.parseFromSlice(json.Value, allocator, json_str, .{}) catch |err| {
            std.log.err("❌ Failed to parse config JSON: {}", .{err});
            std.log.info("📋 Using default DeepSeek V3 configuration");
            return defaultDeepSeekV3();
        };
        defer parsed.deinit();

        var config = defaultDeepSeekV3();
        const root = parsed.value.object;

        // Parse core dimensions
        if (root.get("vocab_size")) |val| config.vocab_size = @intCast(val.integer);
        if (root.get("hidden_size")) |val| config.hidden_size = @intCast(val.integer);
        if (root.get("intermediate_size")) |val| config.intermediate_size = @intCast(val.integer);
        if (root.get("num_hidden_layers")) |val| config.num_hidden_layers = @intCast(val.integer);
        if (root.get("num_attention_heads")) |val| config.num_attention_heads = @intCast(val.integer);
        if (root.get("num_key_value_heads")) |val| config.num_key_value_heads = @intCast(val.integer);
        if (root.get("max_position_embeddings")) |val| config.max_position_embeddings = @intCast(val.integer);

        // Parse MLA dimensions
        if (root.get("qk_nope_head_dim")) |val| config.qk_nope_head_dim = @intCast(val.integer);
        if (root.get("qk_rope_head_dim")) |val| config.qk_rope_head_dim = @intCast(val.integer);
        if (root.get("v_head_dim")) |val| config.v_head_dim = @intCast(val.integer);
        if (root.get("qk_rope_base")) |val| config.qk_rope_base = @floatCast(val.float);

        // Parse MoE configuration
        if (root.get("num_experts")) |val| config.num_experts = @intCast(val.integer);
        if (root.get("num_experts_per_token")) |val| config.num_experts_per_token = @intCast(val.integer);
        if (root.get("moe_layer_freq")) |val| config.moe_layer_freq = @intCast(val.integer);
        if (root.get("first_k_dense_replace")) |val| config.first_k_dense_replace = @intCast(val.integer);
        if (root.get("moe_intermediate_size")) |val| config.moe_intermediate_size = @intCast(val.integer);

        // Parse training parameters
        if (root.get("bos_token_id")) |val| config.bos_token_id = @intCast(val.integer);
        if (root.get("eos_token_id")) |val| config.eos_token_id = @intCast(val.integer);
        if (root.get("rms_norm_eps")) |val| config.rms_norm_eps = @floatCast(val.float);

        // Parse generation parameters
        if (root.get("temperature")) |val| config.temperature = @floatCast(val.float);
        if (root.get("top_p")) |val| config.top_p = @floatCast(val.float);
        if (root.get("top_k")) |val| config.top_k = @intCast(val.integer);

        std.log.info("✅ Loaded config: {} layers, {} heads, {} hidden", .{ config.num_hidden_layers, config.num_attention_heads, config.hidden_size });

        return config;
    }

    /// Get default DeepSeek V3 configuration (full model)
    pub fn defaultDeepSeekV3() ModelConfig {
        return ModelConfig{};
    }

    /// Get tiny test configuration for development
    pub fn tinyTest() ModelConfig {
        return ModelConfig{
            .vocab_size = 1000,
            .hidden_size = 256,
            .intermediate_size = 512,
            .num_hidden_layers = 2,
            .num_attention_heads = 4,
            .num_key_value_heads = 4,
            .max_position_embeddings = 512,
            .qk_nope_head_dim = 32,
            .qk_rope_head_dim = 32,
            .v_head_dim = 64,
            .num_experts = 4,
            .num_experts_per_token = 2,
            .moe_layer_freq = 2, // Only second layer has MoE
            .first_k_dense_replace = 1,
            .moe_intermediate_size = 256,
        };
    }

    /// Get small test configuration for testing
    pub fn smallTest() ModelConfig {
        return ModelConfig{
            .vocab_size = 10000,
            .hidden_size = 512,
            .intermediate_size = 1024,
            .num_hidden_layers = 4,
            .num_attention_heads = 8,
            .num_key_value_heads = 8,
            .max_position_embeddings = 2048,
            .qk_nope_head_dim = 64,
            .qk_rope_head_dim = 32,
            .v_head_dim = 64,
            .num_experts = 8,
            .num_experts_per_token = 2,
            .moe_layer_freq = 2,
            .first_k_dense_replace = 1,
            .moe_intermediate_size = 512,
        };
    }

    /// Validate configuration for consistency
    pub fn validate(self: *const ModelConfig) !void {
        if (self.hidden_size % self.num_attention_heads != 0) {
            std.log.err("❌ Hidden size {} not divisible by num heads {}", .{ self.hidden_size, self.num_attention_heads });
            return error.InvalidConfiguration;
        }

        if (self.qk_nope_head_dim + self.qk_rope_head_dim != self.hidden_size / self.num_attention_heads) {
            std.log.err("❌ MLA head dimensions inconsistent with hidden_size/num_heads", .{});
            return error.InvalidConfiguration;
        }

        if (self.num_experts_per_token > self.num_experts) {
            std.log.err("❌ Experts per token {} > total experts {}", .{ self.num_experts_per_token, self.num_experts });
            return error.InvalidConfiguration;
        }

        std.log.debug("✅ Configuration validation passed", .{});
    }

    /// Estimate memory usage for this configuration
    pub fn estimateMemoryUsage(self: *const ModelConfig) u64 {
        const bytes_per_param = 2; // BF16

        // Embedding layers
        const embed_params = @as(u64, self.vocab_size) * self.hidden_size;

        // Transformer layers
        const attention_params_per_layer =
            self.hidden_size * (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads + // Q proj
            self.hidden_size * (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_key_value_heads + // K proj
            self.hidden_size * self.v_head_dim * self.num_key_value_heads + // V proj
            self.hidden_size * self.hidden_size; // O proj

        const mlp_params_per_layer = if (self.num_experts > 1)
            self.hidden_size * self.moe_intermediate_size * 3 * self.num_experts // Gate, Up, Down for each expert
        else
            self.hidden_size * self.intermediate_size * 3; // Dense SwiGLU

        const layer_norm_params_per_layer = self.hidden_size * 2; // Attention + MLP norms

        const params_per_layer = attention_params_per_layer + mlp_params_per_layer + layer_norm_params_per_layer;
        const total_layer_params = params_per_layer * self.num_hidden_layers;

        // Output head
        const lm_head_params = self.hidden_size * self.vocab_size;

        const total_params = embed_params + total_layer_params + lm_head_params;
        return total_params * bytes_per_param;
    }

    /// Format configuration for logging
    pub fn format(self: *const ModelConfig, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("ModelConfig{{ layers: {}, heads: {}, hidden: {}, experts: {}, vocab: {} }}", .{ self.num_hidden_layers, self.num_attention_heads, self.hidden_size, self.num_experts, self.vocab_size });
    }
};

// Tests
test "ModelConfig validation" {
    var config = ModelConfig.defaultDeepSeekV3();
    try config.validate();
}

test "ModelConfig memory estimation" {
    const config = ModelConfig.defaultDeepSeekV3();
    const memory = config.estimateMemoryUsage();
    try std.testing.expect(memory > 0);
}
