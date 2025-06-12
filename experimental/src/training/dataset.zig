// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Dataset handling for DeepZig model training
//! Provides utilities for loading, processing and batching text datasets

const std = @import("std");
const deepseek_core = @import("deepseek_core");
const tokenizer_mod = deepseek_core.tokenizer;
const tensor = deepseek_core.tensor;
const Tokenizer = tokenizer_mod.Tokenizer;

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const fs = std.fs;
const log = std.log;
const uri = std.Uri;

// In Zig 0.15.0-dev, random number generation is in std.Random instead of std.rand

pub const DatasetError = error{
    InvalidDataset,
    DatasetTooLarge,
    TokenizationFailed,
    NetworkError,
    FileNotFound,
};

/// Text sample with optional metadata
pub const TextSample = struct {
    text: []const u8,
    metadata: StringHashMap([]const u8),
    
    pub fn init(allocator: Allocator, text: []const u8) !TextSample {
        return TextSample{
            .text = text,
            .metadata = StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *TextSample) void {
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            self.metadata.allocator.free(entry.key_ptr.*);
            self.metadata.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
};

/// Tokenized batch for model training
pub const Batch = struct {
    allocator: Allocator,
    input_ids: *tensor.Tensor,
    attention_mask: *tensor.Tensor,
    labels: *tensor.Tensor,
    
    pub fn deinit(self: *Batch) void {
        self.input_ids.deinit();
        self.attention_mask.deinit();
        self.labels.deinit();
    }
};

/// Represents a tokenized sample for model training
pub const TokenizedSample = struct {
    input_ids: ArrayList(i32),
    attention_mask: ArrayList(bool),
};

/// Text dataset loader and processor for language model training
pub const TextDataset = struct {
    allocator: Allocator,
    samples: ArrayList(TextSample),
    tokenized_samples: ?ArrayList(TokenizedSample),
    dataset_name: []const u8,
    sample_count: usize,
    max_length: usize,
    prng: std.Random.DefaultPrng,
    
    /// Initialize dataset loader with specified parameters
    pub fn init(allocator: Allocator, dataset_name: []const u8, max_samples: usize) !TextDataset {
        var samples = ArrayList(TextSample).init(allocator);
        
        // For simplicity in this implementation, we'll use a dummy dataset loader
        // In a real implementation, this would download or load files from disk
        try loadDataset(allocator, &samples, dataset_name, max_samples);
        
        // In Zig 0.15.0-dev, we directly store the PRNG from std.Random
        const seed: u64 = @bitCast(std.time.timestamp());
        const prng = std.Random.DefaultPrng.init(seed);
        
        return TextDataset{
            .allocator = allocator,
            .samples = samples,
            .tokenized_samples = null,
            .dataset_name = try allocator.dupe(u8, dataset_name),
            .sample_count = samples.items.len,
            .max_length = 512,  // Default sequence length
            .prng = prng,
        };
    }
    
    /// Clean up resources
    pub fn deinit(self: *TextDataset) void {
        // Free tokenized samples if they exist
        if (self.tokenized_samples) |*tokenized| {
            for (tokenized.items) |*item| {
                item.input_ids.deinit();
                item.attention_mask.deinit();
            }
            tokenized.deinit();
        }
        
        // Free text samples
        for (self.samples.items) |*sample| {
            sample.deinit();
        }
        self.samples.deinit();
        
        // Free dataset name
        self.allocator.free(self.dataset_name);
    }
    
    /// Tokenize all samples in the dataset
    pub fn tokenize(self: *TextDataset, tokenizer: *Tokenizer) !void {
        if (self.tokenized_samples != null) {
            // Already tokenized
            return;
        }
        
        var tokenized = ArrayList(TokenizedSample).init(self.allocator);
        errdefer {
            for (tokenized.items) |*item| {
                item.input_ids.deinit();
                item.attention_mask.deinit();
            }
            tokenized.deinit();
        }
        
        for (self.samples.items) |sample| {
            // Tokenize the sample
            const token_ids = try tokenizer.encode(sample.text);
            
            // Convert u32 token ids to i32 in an ArrayList
            var input_ids = try ArrayList(i32).initCapacity(self.allocator, token_ids.len);
            for (token_ids) |id| {
                try input_ids.append(@intCast(id));
            }
            
            // Create attention masks (all true)
            var masks = try ArrayList(bool).initCapacity(self.allocator, token_ids.len);
            for (token_ids) |_| {
                try masks.append(true);
            }
            
            // Store tokenized sample
            try tokenized.append(.{
                .input_ids = input_ids,
                .attention_mask = masks,
            });
        }
        
        self.tokenized_samples = tokenized;
    }
    
    /// Prepare a batch of samples for training
    pub fn getBatch(self: *TextDataset, allocator: Allocator, batch_size: usize) !Batch {
        if (self.tokenized_samples == null) {
            return error.DatasetNotTokenized;
        }
        
        const samples = self.tokenized_samples.?;
        if (samples.items.len == 0) {
            return error.EmptyDataset;
        }
        
        // Create tensors for the batch
        const max_seq_len = self.max_length;
        
        var input_ids = try tensor.zeros(allocator, &[_]usize{ batch_size, max_seq_len }, .i32);
        errdefer input_ids.deinit();
        
        var attention_mask = try tensor.zeros(allocator, &[_]usize{ batch_size, max_seq_len }, .bool);
        errdefer attention_mask.deinit();
        
        var labels = try tensor.full(allocator, &[_]usize{ batch_size, max_seq_len }, -100, .i32);
        errdefer labels.deinit();
        
        // Fill batch with random samples
        var i: usize = 0;
        while (i < batch_size) : (i += 1) {
            // Select a random sample
            const sample_idx = self.prng.random().uintLessThan(usize, samples.items.len);
            const sample = samples.items[sample_idx];
            
            // Determine sequence length
            const seq_length = @min(sample.input_ids.items.len, max_seq_len);
            
            // Update tensors with sample data
            var j: usize = 0;
            while (j < seq_length) : (j += 1) {
                try input_ids.setItem(&[_]usize{ i, j }, sample.input_ids.items[j]);
                try attention_mask.setItem(&[_]usize{ i, j }, sample.attention_mask.items[j]);
                try labels.setItem(&[_]usize{ i, j }, sample.input_ids.items[j]);
            }
        }
        
        return Batch{
            .allocator = allocator,
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .labels = labels,
        };
    }
    
    /// Shuffle the dataset samples
    pub fn shuffle(self: *TextDataset) void {
        // Shuffle raw samples
        self.prng.random().shuffle(TextSample, self.samples.items);
        
        // Shuffle tokenized samples if they exist
        if (self.tokenized_samples) |*tokenized| {
            // Get the correct type for shuffling
            const ItemType = @TypeOf(tokenized.items[0]);
            self.prng.random().shuffle(ItemType, tokenized.items);
        }
    }
};

/// Load dataset from sources (local or remote)
fn loadDataset(allocator: Allocator, samples: *ArrayList(TextSample), dataset_name: []const u8, max_samples: usize) !void {
    // This is a simplified implementation for demonstration
    // In a real app, we would handle downloading/caching datasets
    
    if (std.mem.eql(u8, dataset_name, "wikitext")) {
        try loadWikitextSample(allocator, samples, max_samples);
    } else if (std.mem.eql(u8, dataset_name, "custom")) {
        try loadLocalDataset(allocator, samples, "data/custom", max_samples);
    } else {
        log.err("Unknown dataset: {s}", .{dataset_name});
        return error.InvalidDataset;
    }
}

/// Load a sample of the Wikitext dataset
fn loadWikitextSample(allocator: Allocator, samples: *ArrayList(TextSample), max_samples: usize) !void {
    // Simulated dataset for demonstration
    // In production, this would download/read the actual dataset
    
    const sample_texts = [_][]const u8{
        "The quick brown fox jumps over the lazy dog.",
        "In natural language processing, tokenization is the process of breaking down text into individual words or tokens.",
        "The Transformer model architecture revolutionized machine learning for sequential data.",
        "DeepZig is a Zig implementation of state-of-the-art language models.",
        "Mixture of Experts (MoE) allows for conditional computation in neural networks.",
        "Zig is a general-purpose programming language designed for robustness and optimal interoperability.",
        "Large language models have billions of parameters and require significant computational resources to train.",
        "The attention mechanism allows models to focus on relevant parts of the input sequence.",
        "Transfer learning enables models to leverage knowledge from pre-training for downstream tasks.",
        "Tokenizers convert text into numerical representations that neural networks can process.",
    };
    
    // Add samples up to max_samples
    const count = @min(sample_texts.len, max_samples);
    for (0..count) |i| {
        const sample = try TextSample.init(allocator, try allocator.dupe(u8, sample_texts[i]));
        try samples.append(sample);
    }
    
    // Repeat samples to reach max_samples if needed
    var i: usize = count;
    while (i < max_samples) : (i += 1) {
        const text = sample_texts[i % sample_texts.len];
        const sample = try TextSample.init(allocator, try allocator.dupe(u8, text));
        try samples.append(sample);
    }
}

/// Load dataset from local files
fn loadLocalDataset(allocator: Allocator, samples: *ArrayList(TextSample), dir_path: []const u8, max_samples: usize) !void {
    var dir = try fs.cwd().openDir(dir_path, .{ .iterate = true });
    defer dir.close();
    
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    
    var count: usize = 0;
    while (try walker.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".txt")) continue;
        
        const file_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, entry.path });
        defer allocator.free(file_path);
        
        const max_file_size = 10 * 1024 * 1024; // 10MB limit
        const file_content = try fs.cwd().readFileAlloc(allocator, file_path, max_file_size);
        
        const sample = try TextSample.init(allocator, file_content);
        try samples.append(sample);
        
        count += 1;
        if (count >= max_samples) break;
    }
}
