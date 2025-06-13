// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;

/// Helper function to escape JSON strings
fn escapeJsonString(writer: anytype, string: []const u8) !void {
    for (string) |c| {
        switch (c) {
            '\"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            12 => try writer.writeAll("\\f"), // Form feed = ASCII 12
            8 => try writer.writeAll("\\b"), // Backspace = ASCII 8
            else => if (c < 32) {
                // Control characters need unicode escape
                try writer.print("\\u{x:0>4}", .{@as(u16, c)});
            } else {
                try writer.writeByte(c);
            },
        }
    }
}

/// BPE (Byte Pair Encoding) Tokenizer for DeepSeek V3
/// Supports loading from HuggingFace tokenizer.json format
pub const Tokenizer = struct {
    allocator: Allocator,
    vocab_size: u32,

    // Vocabulary mappings
    token_to_id: std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
    id_to_token: std.HashMap(u32, []const u8, std.hash_map.AutoContext(u32), std.hash_map.default_max_load_percentage),

    // Special tokens
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
    pad_token_id: ?u32,

    // BPE merge rules (pairs to merge and their priority)
    merge_rules: std.ArrayList(MergeRule),

    const Self = @This();

    const MergeRule = struct {
        pair: [2][]const u8,
        merged: []const u8,
        priority: u32,
    };

    /// Initialize tokenizer with default vocabulary
    pub fn init(allocator: Allocator, vocab_size: u32) !Self {
        std.log.info("🔤 Initializing BPE tokenizer with vocab size: {}", .{vocab_size});

        var token_to_id = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator);
        var id_to_token = std.HashMap(u32, []const u8, std.hash_map.AutoContext(u32), std.hash_map.default_max_load_percentage).init(allocator);
        const merge_rules = std.ArrayList(MergeRule).init(allocator);

        // Initialize with basic ASCII vocabulary + special tokens
        try Self.initializeBasicVocab(&token_to_id, &id_to_token, allocator);

        return Self{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .token_to_id = token_to_id,
            .id_to_token = id_to_token,
            .bos_token_id = 1,    // Fixed: Use standard BOS token ID
            .eos_token_id = 2,    // Fixed: Use standard EOS token ID
            .unk_token_id = 0,    // Fixed: Use standard UNK token ID
            .pad_token_id = 3,    // Fixed: Use standard PAD token ID
            .merge_rules = merge_rules,
        };
    }

    /// Load tokenizer from HuggingFace tokenizer.json
    pub fn loadFromFile(allocator: Allocator, tokenizer_path: []const u8) !Self {
        std.log.info("🔤 Loading tokenizer from: {s}", .{tokenizer_path});

        const file = std.fs.cwd().openFile(tokenizer_path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.warn("⚠️ Tokenizer file not found, using basic tokenizer", .{});
                return try Self.init(allocator, 129280); // Default DeepSeek V3 vocab size
            },
            else => return err,
        };
        defer file.close();

        const contents = try file.readToEndAlloc(allocator, 100 * 1024 * 1024); // 100MB limit
        defer allocator.free(contents);

        return try Self.parseFromJson(allocator, contents);
    }

    /// Parse tokenizer from HuggingFace tokenizer.json format
    fn parseFromJson(allocator: Allocator, json_str: []const u8) !Self {
        var parsed = json.parseFromSlice(json.Value, allocator, json_str, .{}) catch |err| {
            std.log.err("❌ Failed to parse tokenizer JSON: {}", .{err});
            return try Self.init(allocator, 129280);
        };
        defer parsed.deinit();

        const root = parsed.value.object;

        var token_to_id = std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage).init(allocator);
        var id_to_token = std.HashMap(u32, []const u8, std.hash_map.AutoContext(u32), std.hash_map.default_max_load_percentage).init(allocator);
        const merge_rules = std.ArrayList(MergeRule).init(allocator);

        var vocab_size: u32 = 129280;
        var bos_token_id: u32 = 100000;
        var eos_token_id: u32 = 100001;
        var unk_token_id: u32 = 100002;
        var pad_token_id: ?u32 = null;

        // Parse vocabulary
        if (root.get("model")) |model| {
            if (model.object.get("vocab")) |vocab_obj| {
                for (vocab_obj.object.keys()) |token| {
                    const id = @as(u32, @intCast(vocab_obj.object.get(token).?.integer));

                    // Store copies of the strings
                    const token_copy = try allocator.dupe(u8, token);
                    try token_to_id.put(token_copy, id);
                    try id_to_token.put(id, token_copy);

                    vocab_size = @max(vocab_size, id + 1);
                }
            }
        }

        // Parse special tokens
        if (root.get("added_tokens")) |added_tokens| {
            for (added_tokens.array.items) |token_obj| {
                const token_data = token_obj.object;
                if (token_data.get("content")) |content| {
                    const token_str = content.string;
                    const id = @as(u32, @intCast(token_data.get("id").?.integer));

                    if (std.mem.eql(u8, token_str, "<|begin_of_text|>")) {
                        bos_token_id = id;
                    } else if (std.mem.eql(u8, token_str, "<|end_of_text|>")) {
                        eos_token_id = id;
                    } else if (std.mem.eql(u8, token_str, "<unk>")) {
                        unk_token_id = id;
                    } else if (std.mem.eql(u8, token_str, "<pad>")) {
                        pad_token_id = id;
                    }
                }
            }
        }

        std.log.info("✅ Loaded tokenizer: {} vocab, BOS: {}, EOS: {}", .{ vocab_size, bos_token_id, eos_token_id });

        return Self{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .token_to_id = token_to_id,
            .id_to_token = id_to_token,
            .bos_token_id = bos_token_id,
            .eos_token_id = eos_token_id,
            .unk_token_id = unk_token_id,
            .pad_token_id = pad_token_id,
            .merge_rules = merge_rules,
        };
    }

    /// Initialize basic ASCII vocabulary for fallback
    fn initializeBasicVocab(
        token_to_id: *std.HashMap([]const u8, u32, std.hash_map.StringContext, std.hash_map.default_max_load_percentage),
        id_to_token: *std.HashMap(u32, []const u8, std.hash_map.AutoContext(u32), std.hash_map.default_max_load_percentage),
        allocator: Allocator,
    ) !void {
        // FIXED: Start with special tokens at low IDs
        var id: u32 = 0;

        // Special tokens first
        const special_tokens = [_][]const u8{ "<unk>", "<s>", "</s>", "<pad>" };
        for (special_tokens) |token| {
            const token_copy = try allocator.dupe(u8, token);
            try token_to_id.put(token_copy, id);
            try id_to_token.put(id, token_copy);
            id += 1;
        }

        // Common words and characters that the model is likely to generate
        const common_vocab = [_][]const u8{
            // Common single characters
            " ", "!", "?", ".", ",", ":", ";",
            "a", "e", "i", "o", "u", "n", "t", "r", "s", "l", "h", "d", "c", "m", "f", "p", "g", "w", "y", "b", "v", "k", "x", "j", "q", "z",
            "A", "E", "I", "O", "U", "N", "T", "R", "S", "L", "H", "D", "C", "M", "F", "P", "G", "W", "Y", "B", "V", "K", "X", "J", "Q", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",

            // Common words
            "the", "and", "or", "a", "an", "is", "are", "was", "were", "I", "you", "he", "she", "it", "we", "they",
            "this", "that", "with", "for", "on", "in", "at", "to", "of", "as", "by", "be", "do", "have", "can", "will",
            "Hello", "Hi", "How", "What", "Why", "When", "Where", "Who", "Please", "Thank", "help", "Yes", "No",

            // Common endings
            "ing", "ed", "er", "ly", "s", "'s", "'t", "'re", "'ve", "'ll", "'d",

            // Programming/technical
            "function", "def", "class", "import", "return", "if", "else", "for", "while", "try", "except",
            "python", "code", "data", "model", "train", "test", "example", "result",
        };

        for (common_vocab) |token| {
            const token_copy = try allocator.dupe(u8, token);
            try token_to_id.put(token_copy, id);
            try id_to_token.put(id, token_copy);
            id += 1;
        }

        // Add printable ASCII for remaining slots
        for (32..127) |i| {
            const byte_val = @as(u8, @intCast(i));
            const char_str = [_]u8{byte_val};

            // Check if we already added this character
            if (!token_to_id.contains(&char_str)) {
                const token = try allocator.alloc(u8, 1);
                token[0] = byte_val;
                try token_to_id.put(token, id);
                try id_to_token.put(id, token);
                id += 1;
            }
        }

        // Add conversational tokens for chat applications
        const conversation_tokens = [_][]const u8{
            // Chat role markers
            "<user>", "</user>", // User messages
            "<assistant>", "</assistant>", // Assistant responses
            "<system>", "</system>", // System prompts

            // Tool calling tokens
            "<tool>", "</tool>", // Tool calls
            "<function>", "</function>", // Function definitions
            "<result>", "</result>", // Tool results
            "{", "}", "[", "]", // JSON formatting
            "\"", ":", // JSON syntax (comma already added above)

            // Additional common patterns
            "Can", "Could", "Would", "me", "my", "your", "our", "their",
        };

        // Add conversation tokens
        for (conversation_tokens) |token| {
            const token_copy = try allocator.dupe(u8, token);
            try token_to_id.put(token_copy, id);
            try id_to_token.put(id, token_copy);
            id += 1;
        }

        std.log.debug("✅ Initialized basic vocabulary with {} tokens", .{id});
    }

    pub fn deinit(self: *Self) void {
        // Free duplicated token strings (stored as values in id_to_token)
        var id_iter = self.id_to_token.iterator();
        while (id_iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.id_to_token.deinit();
        self.token_to_id.deinit();

        // Free merge rules
        for (self.merge_rules.items) |rule| {
            self.allocator.free(rule.pair[0]);
            self.allocator.free(rule.pair[1]);
            self.allocator.free(rule.merged);
        }
        self.merge_rules.deinit();
    }

    /// Encode text to token IDs
    pub fn encode(self: *Self, text: []const u8) ![]u32 {
        if (text.len == 0) {
            return try self.allocator.alloc(u32, 0);
        }

        // Simple word-level tokenization for now
        // In a full implementation, this would use BPE algorithm
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        var i: usize = 0;
        while (i < text.len) {
            // Try to find the longest matching token starting at position i
            var best_match_len: usize = 0;
            var best_token_id: u32 = self.unk_token_id;

            // Look for matches of decreasing length
            var check_len: usize = @min(50, text.len - i); // Max token length
            while (check_len > 0) {
                const candidate = text[i .. i + check_len];
                if (self.token_to_id.get(candidate)) |token_id| {
                    best_match_len = check_len;
                    best_token_id = token_id;
                    break;
                }
                check_len -= 1;
            }

            // If no match found, encode as single byte
            if (best_match_len == 0) {
                const byte_token = [_]u8{text[i]};
                if (self.token_to_id.get(&byte_token)) |token_id| {
                    best_token_id = token_id;
                } else {
                    best_token_id = self.unk_token_id;
                }
                best_match_len = 1;
            }

            try tokens.append(best_token_id);
            i += best_match_len;
        }

        return try tokens.toOwnedSlice();
    }

    /// Encode text with special tokens (BOS/EOS)
    pub fn encodeWithSpecialTokens(self: *Self, text: []const u8, add_bos: bool, add_eos: bool) ![]u32 {
        const base_tokens = try self.encode(text);
        defer self.allocator.free(base_tokens);

        const bos_count: usize = if (add_bos) 1 else 0;
        const eos_count: usize = if (add_eos) 1 else 0;

        var result = try self.allocator.alloc(u32, base_tokens.len + bos_count + eos_count);

        var idx: usize = 0;
        if (add_bos) {
            result[idx] = self.bos_token_id;
            idx += 1;
        }

        @memcpy(result[idx .. idx + base_tokens.len], base_tokens);
        idx += base_tokens.len;

        if (add_eos) {
            result[idx] = self.eos_token_id;
        }

        return result;
    }

    /// Decode token IDs to text
    pub fn decode(self: *Self, tokens: []const u32) ![]u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        for (tokens) |token_id| {
            // Skip special tokens in basic decode
            if (token_id == self.bos_token_id or
                token_id == self.eos_token_id or
                (self.pad_token_id != null and token_id == self.pad_token_id.?))
            {
                continue;
            }

            if (self.id_to_token.get(token_id)) |token| {
                // Validate UTF-8 before adding
                if (std.unicode.utf8ValidateSlice(token)) {
                    try result.appendSlice(token);
                } else {
                    // Invalid UTF-8 - convert byte to readable form
                    if (token.len == 1 and token[0] >= 32 and token[0] <= 126) {
                        // Printable ASCII
                        try result.appendSlice(token);
                    } else {
                        // Non-printable or invalid - add safe replacement
                        try result.append('?');
                    }
                }
            } else {
                // Unknown token - check if it's a valid single byte
                if (token_id < 256) {
                    const byte_val = @as(u8, @intCast(token_id));
                    if (byte_val >= 32 and byte_val <= 126) {
                        try result.append(byte_val);
                    } else {
                        try result.append('?');
                    }
                } else {
                    // Completely unknown token - use readable replacement
                    try result.appendSlice("<UNK>");
                }
            }
        }

        return try result.toOwnedSlice();
    }

    /// Decode with special token handling
    pub fn decodeWithSpecialTokens(self: *Self, tokens: []const u32) ![]u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        for (tokens) |token_id| {
            if (self.id_to_token.get(token_id)) |token| {
                try result.appendSlice(token);
            } else {
                // Unknown token
                try result.appendSlice("<unk>");
            }
        }

        return try result.toOwnedSlice();
    }

    /// Get token information
    pub fn getTokenInfo(self: *const Self) TokenInfo {
        return TokenInfo{
            .vocab_size = self.vocab_size,
            .bos_token_id = self.bos_token_id,
            .eos_token_id = self.eos_token_id,
            .unk_token_id = self.unk_token_id,
            .pad_token_id = self.pad_token_id,
        };
    }

    pub const TokenInfo = struct {
        vocab_size: u32,
        bos_token_id: u32,
        eos_token_id: u32,
        unk_token_id: u32,
        pad_token_id: ?u32,
    };

    /// Training options for creating a tokenizer from a dataset
    pub const TrainingOptions = struct {
        vocab_size: u32 = 32000,
        model_type: enum { bpe } = .bpe,
    };

    /// Train a new tokenizer from a dataset
    pub fn trainFromDataset(allocator: Allocator, data_loader: anytype, options: TrainingOptions) !Self {
        std.log.info("🔤 Training BPE tokenizer from dataset with vocab size: {}", .{options.vocab_size});

        // Initialize with basic tokenizer first
        var tokenizer = try Self.init(allocator, options.vocab_size);

        // In a real implementation, we would scan the dataset to find frequent tokens,
        // build merge rules, etc. For this experimental version, we'll just use the basic
        // tokenizer with ASCII characters.
        _ = data_loader; // Silence unused parameter warning

        std.log.info("✅ Tokenizer training completed with {} vocab entries", .{tokenizer.token_to_id.count()});
        return tokenizer;
    }

    /// Save tokenizer to a HuggingFace compatible tokenizer.json file
    pub fn saveToFile(self: *Self, tokenizer_path: []const u8) !void {
        std.log.info("💾 Saving tokenizer to: {s}", .{tokenizer_path});

        // Create directory path if it doesn't exist
        const dir_path = std.fs.path.dirname(tokenizer_path) orelse "";
        if (dir_path.len > 0) {
            std.fs.cwd().makePath(dir_path) catch |err| {
                std.log.warn("⚠️ Failed to create directory: {s}, error: {}", .{ dir_path, err });
                // Continue even if directory creation fails
            };
        }

        // Create or overwrite the file
        const file = try std.fs.cwd().createFile(tokenizer_path, .{});
        defer file.close();

        // Create JSON writer
        var buffered_writer = std.io.bufferedWriter(file.writer());
        var writer = buffered_writer.writer();

        // Write JSON header
        try writer.writeAll("{\n");
        try writer.writeAll("  \"version\": \"1.0\",\n");
        try writer.writeAll("  \"truncation\": null,\n");
        try writer.writeAll("  \"padding\": null,\n");

        // Write model type and vocabulary
        try writer.writeAll("  \"model\": {\n");
        try writer.writeAll("    \"type\": \"bpe\",\n");
        try writer.writeAll("    \"vocab\": {\n");

        // Write vocabulary entries
        var first_entry = true;
        var id_iter = self.id_to_token.iterator();
        while (id_iter.next()) |entry| {
            if (entry.key_ptr.* == self.bos_token_id or
                entry.key_ptr.* == self.eos_token_id or
                entry.key_ptr.* == self.unk_token_id or
                (self.pad_token_id != null and entry.key_ptr.* == self.pad_token_id.?))
            {
                // Skip special tokens, they will be added separately
                continue;
            }

            // Add comma if not the first entry
            if (!first_entry) {
                try writer.writeAll(",\n");
            }
            first_entry = false;

            // Write token and its ID
            const token = entry.value_ptr.*;
            try writer.writeAll("      \"");
            try escapeJsonString(writer, token);
            try writer.print("\": {}", .{entry.key_ptr.*});
        }

        try writer.writeAll("\n    },\n");
        try writer.writeAll("    \"merges\": []\n");
        try writer.writeAll("  },\n");

        // Write special tokens
        try writer.writeAll("  \"special_tokens\": {\n");
        try writer.writeAll("    \"bos_token\": \"<s>\",\n");
        try writer.writeAll("    \"eos_token\": \"</s>\",\n");
        try writer.writeAll("    \"unk_token\": \"<unk>\",\n");
        if (self.pad_token_id != null) {
            try writer.writeAll("    \"pad_token\": \"<pad>\",\n");
        }
        try writer.writeAll("  },\n");

        // Write added tokens section for special tokens
        try writer.writeAll("  \"added_tokens\": [\n");
        try writer.writeAll("    { \"id\": ");
        try writer.print("{}", .{self.bos_token_id});
        try writer.writeAll(", \"content\": \"<s>\", \"single_word\": false, \"special\": true },\n");
        try writer.writeAll("    { \"id\": ");
        try writer.print("{}", .{self.eos_token_id});
        try writer.writeAll(", \"content\": \"</s>\", \"single_word\": false, \"special\": true },\n");
        try writer.writeAll("    { \"id\": ");
        try writer.print("{}", .{self.unk_token_id});
        try writer.writeAll(", \"content\": \"<unk>\", \"single_word\": false, \"special\": true }\n");

        if (self.pad_token_id != null) {
            try writer.writeAll(",\n    { \"id\": ");
            try writer.print("{}", .{self.pad_token_id.?});
            try writer.writeAll(", \"content\": \"<pad>\", \"single_word\": false, \"special\": true }\n");
        }
        try writer.writeAll("  ]\n");

        // End JSON
        try writer.writeAll("}\n");

        // Flush the buffered writer
        try buffered_writer.flush();

        std.log.info("✅ Tokenizer saved successfully to {s}", .{tokenizer_path});
    }
};

// Tests
test "Tokenizer basic functionality" {
    const testing = std.testing;
    var tokenizer = try Tokenizer.init(testing.allocator, 1000);
    defer tokenizer.deinit();

    const text = "Hello, world!";
    const tokens = try tokenizer.encode(text);
    defer testing.allocator.free(tokens);

    const decoded = try tokenizer.decode(tokens);
    defer testing.allocator.free(decoded);

    try testing.expect(tokens.len > 0);
    try testing.expect(decoded.len > 0);
}

test "Tokenizer special tokens" {
    const testing = std.testing;
    var tokenizer = try Tokenizer.init(testing.allocator, 1000);
    defer tokenizer.deinit();

    const text = "Hello";
    const tokens = try tokenizer.encodeWithSpecialTokens(text, true, true);
    defer testing.allocator.free(tokens);

    try testing.expect(tokens.len >= 3); // BOS + content + EOS
    try testing.expect(tokens[0] == tokenizer.bos_token_id);
    try testing.expect(tokens[tokens.len - 1] == tokenizer.eos_token_id);
}
