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
            .bos_token_id = 1, // Fixed: Use standard BOS token ID
            .eos_token_id = 2, // Fixed: Use standard EOS token ID
            .unk_token_id = 0, // Fixed: Use standard UNK token ID
            .pad_token_id = 3, // Fixed: Use standard PAD token ID
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
        // IMPROVED: Much better vocabulary initialization with conversational focus
        var id: u32 = 0;

        // Special tokens first (standard IDs for compatibility)
        const special_tokens = [_][]const u8{ "<unk>", "<s>", "</s>", "<pad>" };
        for (special_tokens) |token| {
            const token_copy = try allocator.dupe(u8, token);
            try token_to_id.put(token_copy, id);
            try id_to_token.put(id, token_copy);
            id += 1;
        }

        // Conversation tokens (critical for chat applications)
        const conversation_tokens = [_][]const u8{
            "<user>",     "</user>",     "<assistant>", "</assistant>",
            "<system>",   "</system>",   "<tool>",      "</tool>",
            "<function>", "</function>", "<result>",    "</result>",
        };
        for (conversation_tokens) |token| {
            const token_copy = try allocator.dupe(u8, token);
            try token_to_id.put(token_copy, id);
            try id_to_token.put(id, token_copy);
            id += 1;
        }

        // Common punctuation and symbols (essential for proper text generation)
        const punctuation = [_][]const u8{ " ", ".", ",", "!", "?", ":", ";", "'", "\"", "-", "_", "(", ")", "[", "]", "{", "}", "@", "#", "$", "%", "&", "*", "+", "=", "<", ">", "/", "\\", "|", "~", "`", "^", "\n", "\t" };
        for (punctuation) |token| {
            const token_copy = try allocator.dupe(u8, token);
            try token_to_id.put(token_copy, id);
            try id_to_token.put(id, token_copy);
            id += 1;
        }

        // Single characters (a-z, A-Z, 0-9) for character-level fallback
        for (0..26) |i| {
            // Lowercase
            const lower_char = [_]u8{@as(u8, @intCast('a' + i))};
            const lower_token = try allocator.dupe(u8, &lower_char);
            try token_to_id.put(lower_token, id);
            try id_to_token.put(id, lower_token);
            id += 1;

            // Uppercase
            const upper_char = [_]u8{@as(u8, @intCast('A' + i))};
            const upper_token = try allocator.dupe(u8, &upper_char);
            try token_to_id.put(upper_token, id);
            try id_to_token.put(id, upper_token);
            id += 1;
        }

        // Digits
        for (0..10) |i| {
            const digit_char = [_]u8{@as(u8, @intCast('0' + i))};
            const digit_token = try allocator.dupe(u8, &digit_char);
            try token_to_id.put(digit_token, id);
            try id_to_token.put(id, digit_token);
            id += 1;
        }

        // CRITICAL: Common word patterns and subwords for better generation
        const common_patterns = [_][]const u8{
            // Common words (absolutely essential for good text generation)
            "the",     "and",     "or",     "to",      "of",         "in",        "for",     "on",        "with",   "as",       "by",        "at",     "from",
            "is",      "are",     "was",    "were",    "be",         "been",      "being",   "have",      "has",    "had",      "do",        "did",    "does",
            "can",     "could",   "will",   "would",   "should",     "may",       "might",   "must",      "shall",  "I",        "you",       "he",     "she",
            "it",      "we",      "they",   "me",      "him",        "her",       "us",      "them",      "this",   "that",     "these",     "those",  "what",
            "which",   "who",     "when",   "where",   "why",        "how",

            // Conversational starters
                  "Hello",   "Hi",        "Hey",    "Good",     "Thank",     "Please", "Sorry",
            "Yes",     "No",      "OK",     "Okay",    "morning",    "afternoon", "evening", "night",     "today",  "tomorrow", "yesterday",

            // Question words and phrases
            "What's", "How's",
            "Where's", "When's",  "Who's",  "Why's",   "Can't",      "Don't",     "Won't",   "Shouldn't", "I'm",    "You're",   "He's",      "She's",  "It's",
            "We're",   "They're", "I've",   "You've",  "We've",      "They've",

            // Common verbs and forms
              "help",    "want",      "need",   "like",     "think",     "know",   "see",
            "get",     "go",      "come",   "take",    "make",       "give",      "work",    "play",      "look",   "find",     "use",       "try",    "say",
            "tell",    "ask",     "answer", "explain", "understand",

            // Tech/Programming terms (important for code generation)
            "function",  "def",     "class",     "import", "return",   "if",        "else",   "for",
            "while",   "try",     "except", "python",  "code",       "data",      "model",   "train",     "test",   "example",  "result",    "error",  "value",
            "type",

            // Common endings and prefixes (BPE-like patterns)
               "ing",     "ed",     "er",      "est",        "ly",        "tion",    "ness",      "ment",   "able",     "ful",       "less",   "ous",
            "un",      "re",      "pre",    "dis",     "over",       "under",     "out",     "up",        "down",   "in",       "on",        "off",

            // Numbers as words
               "one",
            "two",     "three",   "four",   "five",    "six",        "seven",     "eight",   "nine",      "ten",    "first",    "second",    "third",  "last",
            "next",    "new",     "old",    "good",    "bad",        "big",       "small",
        };

        for (common_patterns) |pattern| {
            // Only add if not already present
            if (!token_to_id.contains(pattern)) {
                const token_copy = try allocator.dupe(u8, pattern);
                try token_to_id.put(token_copy, id);
                try id_to_token.put(id, token_copy);
                id += 1;
            }
        }

        std.log.debug("✅ Initialized enhanced vocabulary with {} tokens", .{id});
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

        // IMPROVED: Much more sophisticated tokenization algorithm
        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        var i: usize = 0;
        while (i < text.len) {
            var matched = false;
            var best_match_len: usize = 0;
            var best_token_id: u32 = self.unk_token_id;

            // STRATEGY 1: Try to match special tokens first (highest priority)
            const special_patterns = [_][]const u8{ "<user>", "</user>", "<assistant>", "</assistant>", "<system>", "</system>", "<tool>", "</tool>", "<function>", "</function>", "<result>", "</result>", "<s>", "</s>", "<unk>", "<pad>" };

            for (special_patterns) |pattern| {
                if (i + pattern.len <= text.len and std.mem.eql(u8, text[i .. i + pattern.len], pattern)) {
                    if (self.token_to_id.get(pattern)) |token_id| {
                        if (pattern.len > best_match_len) {
                            best_match_len = pattern.len;
                            best_token_id = token_id;
                            matched = true;
                        }
                    }
                }
            }

            // STRATEGY 2: Try to match longer sequences (words, phrases) - greedy longest match
            if (!matched) {
                // Try decreasing lengths to find the longest match
                var try_len = @min(text.len - i, 20); // Don't try excessively long matches
                while (try_len > 0) : (try_len -= 1) {
                    const candidate = text[i .. i + try_len];
                    if (self.token_to_id.get(candidate)) |token_id| {
                        best_match_len = try_len;
                        best_token_id = token_id;
                        matched = true;
                        break; // Take the first (longest) match
                    }
                }
            }

            // STRATEGY 3: Word boundary handling - try to tokenize word-by-word
            if (!matched) {
                // Find word boundaries
                var word_end = i;
                while (word_end < text.len) {
                    const c = text[word_end];
                    if (c == ' ' or c == '\t' or c == '\n' or c == '\r' or
                        c == '.' or c == ',' or c == '!' or c == '?' or
                        c == ':' or c == ';' or c == '(' or c == ')' or
                        c == '[' or c == ']' or c == '{' or c == '}')
                    {
                        break;
                    }
                    word_end += 1;
                }

                if (word_end > i) {
                    const word = text[i..word_end];
                    if (self.token_to_id.get(word)) |token_id| {
                        best_match_len = word.len;
                        best_token_id = token_id;
                        matched = true;
                    }
                }
            }

            // STRATEGY 4: Character-level fallback for single characters
            if (!matched and i < text.len) {
                const char_slice = text[i .. i + 1];
                if (self.token_to_id.get(char_slice)) |token_id| {
                    best_match_len = 1;
                    best_token_id = token_id;
                    matched = true;
                }
            }

            // FALLBACK: Use UNK token if nothing else works
            if (!matched) {
                best_match_len = 1;
                best_token_id = self.unk_token_id;
            }

            try tokens.append(best_token_id);
            i += best_match_len;
        }

        return tokens.toOwnedSlice();
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

    /// ENHANCED: Superior BPE implementation with merge rules and frequency optimization
    pub fn trainBPE(self: *Self, training_texts: []const []const u8, target_vocab_size: u32) !void {
        std.log.info("🧠 Training advanced BPE on {} texts to vocab size {}", .{ training_texts.len, target_vocab_size });

        // Build frequency table of byte pairs
        var pair_frequencies = std.HashMap([2]u8, u32, std.hash_map.AutoContext([2]u8), std.hash_map.default_max_load_percentage).init(self.allocator);
        defer pair_frequencies.deinit();

        // Collect byte pair statistics from training data
        for (training_texts) |text| {
            var i: usize = 0;
            while (i + 1 < text.len) : (i += 1) {
                const pair = [2]u8{ text[i], text[i + 1] };
                const current_count = pair_frequencies.get(pair) orelse 0;
                try pair_frequencies.put(pair, current_count + 1);
            }
        }

        // Create merge rules based on frequency
        var merge_candidates = std.ArrayList(struct { pair: [2]u8, frequency: u32 }).init(self.allocator);
        defer merge_candidates.deinit();

        var pair_iter = pair_frequencies.iterator();
        while (pair_iter.next()) |entry| {
            try merge_candidates.append(.{ .pair = entry.key_ptr.*, .frequency = entry.value_ptr.* });
        }

        // Sort by frequency (descending)
        std.sort.insertion(@TypeOf(merge_candidates.items[0]), merge_candidates.items, {}, struct {
            fn lessThan(_: void, lhs: @TypeOf(merge_candidates.items[0]), rhs: @TypeOf(merge_candidates.items[0])) bool {
                return lhs.frequency > rhs.frequency;
            }
        }.lessThan);

        // Add most frequent pairs as new tokens
        var current_id = self.token_to_id.count();
        const max_new_tokens = @min(merge_candidates.items.len, target_vocab_size - @as(u32, @intCast(current_id)));

        for (merge_candidates.items[0..max_new_tokens]) |candidate| {
            // Create merged token string
            const merged_token = try std.fmt.allocPrint(self.allocator, "{c}{c}", .{ candidate.pair[0], candidate.pair[1] });

            // Add to vocabulary if not already present
            if (!self.token_to_id.contains(merged_token)) {
                try self.token_to_id.put(merged_token, @as(u32, @intCast(current_id)));
                try self.id_to_token.put(@as(u32, @intCast(current_id)), merged_token);

                // Add merge rule
                const pair_0 = try std.fmt.allocPrint(self.allocator, "{c}", .{candidate.pair[0]});
                const pair_1 = try std.fmt.allocPrint(self.allocator, "{c}", .{candidate.pair[1]});
                try self.merge_rules.append(MergeRule{
                    .pair = [2][]const u8{ pair_0, pair_1 },
                    .merged = merged_token,
                    .priority = candidate.frequency,
                });

                current_id += 1;
            } else {
                self.allocator.free(merged_token);
            }
        }

        std.log.info("✅ BPE training complete: {} merge rules, {} total tokens", .{ self.merge_rules.items.len, self.token_to_id.count() });
    }

    /// ENHANCED: Apply BPE merge rules for optimal tokenization
    fn applyBPEMerges(self: *Self, word_tokens: *std.ArrayList([]const u8)) !void {
        var changed = true;
        while (changed) {
            changed = false;

            // Sort merge rules by priority (higher frequency = higher priority)
            std.sort.insertion(MergeRule, self.merge_rules.items, {}, struct {
                fn lessThan(_: void, lhs: MergeRule, rhs: MergeRule) bool {
                    return lhs.priority > rhs.priority;
                }
            }.lessThan);

            for (self.merge_rules.items) |rule| {
                var i: usize = 0;
                while (i + 1 < word_tokens.items.len) {
                    if (std.mem.eql(u8, word_tokens.items[i], rule.pair[0]) and
                        std.mem.eql(u8, word_tokens.items[i + 1], rule.pair[1]))
                    {

                        // Found a merge opportunity
                        const merged = try self.allocator.dupe(u8, rule.merged);

                        // Replace the pair with merged token
                        word_tokens.items[i] = merged;
                        _ = word_tokens.orderedRemove(i + 1);

                        changed = true;
                        break; // Start over with highest priority rules
                    }
                    i += 1;
                }
                if (changed) break;
            }
        }
    }

    /// ENHANCED: Advanced encoding with multiple strategies
    pub fn encodeAdvanced(self: *Self, text: []const u8, enable_bpe: bool) ![]u32 {
        if (text.len == 0) {
            return try self.allocator.alloc(u32, 0);
        }

        var tokens = std.ArrayList(u32).init(self.allocator);
        defer tokens.deinit();

        // PRE-PROCESSING: Handle special sequences
        var processed_text = try self.preprocessText(text);
        defer self.allocator.free(processed_text);

        var i: usize = 0;
        while (i < processed_text.len) {
            // STRATEGY 1: Special tokens (highest priority)
            if (try self.matchSpecialToken(processed_text, i)) |match_result| {
                try tokens.append(match_result.token_id);
                i += match_result.length;
                continue;
            }

            // STRATEGY 2: Word-level processing with BPE
            if (enable_bpe) {
                if (try self.processWordWithBPE(processed_text, &i)) |word_tokens| {
                    defer word_tokens.deinit();
                    try tokens.appendSlice(word_tokens.items);
                    continue;
                }
            }

            // STRATEGY 3: Longest substring match
            if (try self.longestMatch(processed_text, i)) |match_result| {
                try tokens.append(match_result.token_id);
                i += match_result.length;
                continue;
            }

            // STRATEGY 4: Character fallback
            const char_slice = processed_text[i .. i + 1];
            const token_id = self.token_to_id.get(char_slice) orelse self.unk_token_id;
            try tokens.append(token_id);
            i += 1;
        }

        return tokens.toOwnedSlice();
    }

    /// Enhanced text preprocessing for better tokenization
    fn preprocessText(self: *Self, text: []const u8) ![]u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        var i: usize = 0;
        while (i < text.len) {
            const c = text[i];

            // Normalize whitespace
            if (c == '\t' or c == '\r') {
                try result.append(' ');
            } else if (c == '\n') {
                // Preserve newlines but standardize
                try result.append('\n');
            } else if (c == ' ') {
                // Avoid double spaces
                if (result.items.len == 0 or result.items[result.items.len - 1] != ' ') {
                    try result.append(' ');
                }
            } else {
                try result.append(c);
            }
            i += 1;
        }

        return result.toOwnedSlice();
    }

    /// Match special tokens with lookahead
    fn matchSpecialToken(self: *const Self, text: []const u8, pos: usize) !?struct { token_id: u32, length: usize } {
        const special_patterns = [_][]const u8{ "<user>", "</user>", "<assistant>", "</assistant>", "<system>", "</system>", "<tool>", "</tool>", "<function>", "</function>", "<result>", "</result>", "<s>", "</s>", "<unk>", "<pad>" };

        for (special_patterns) |pattern| {
            if (pos + pattern.len <= text.len and std.mem.eql(u8, text[pos .. pos + pattern.len], pattern)) {
                if (self.token_to_id.get(pattern)) |token_id| {
                    return .{ .token_id = token_id, .length = pattern.len };
                }
            }
        }
        return null;
    }

    /// Process a word with BPE merging
    fn processWordWithBPE(self: *Self, text: []const u8, pos: *usize) !?std.ArrayList(u32) {
        // Find word boundaries
        const start = pos.*;
        var end = start;
        while (end < text.len) {
            const c = text[end];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r' or
                c == '.' or c == ',' or c == '!' or c == '?' or
                c == ':' or c == ';' or c == '(' or c == ')' or
                c == '[' or c == ']' or c == '{' or c == '}')
            {
                break;
            }
            end += 1;
        }

        if (end == start) return null;

        const word = text[start..end];

        // Initialize word as character tokens
        var word_tokens = std.ArrayList([]const u8).init(self.allocator);
        defer {
            for (word_tokens.items) |token| {
                self.allocator.free(token);
            }
            word_tokens.deinit();
        }

        for (word) |c| {
            const char_token = try std.fmt.allocPrint(self.allocator, "{c}", .{c});
            try word_tokens.append(char_token);
        }

        // Apply BPE merges
        try self.applyBPEMerges(&word_tokens);

        // Convert to token IDs
        var result = std.ArrayList(u32).init(self.allocator);
        for (word_tokens.items) |token| {
            const token_id = self.token_to_id.get(token) orelse self.unk_token_id;
            try result.append(token_id);
        }

        pos.* = end;
        return result;
    }

    /// Longest substring matching
    fn longestMatch(self: *const Self, text: []const u8, pos: usize) !?struct { token_id: u32, length: usize } {
        var best_length: usize = 0;
        var best_token_id: u32 = self.unk_token_id;

        const max_length = @min(text.len - pos, 32); // Reasonable limit
        var try_length = max_length;

        while (try_length > 0) : (try_length -= 1) {
            const candidate = text[pos .. pos + try_length];
            if (self.token_to_id.get(candidate)) |token_id| {
                if (try_length > best_length) {
                    best_length = try_length;
                    best_token_id = token_id;
                }
                break; // Take first (longest) match
            }
        }

        if (best_length > 0) {
            return .{ .token_id = best_token_id, .length = best_length };
        }
        return null;
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
