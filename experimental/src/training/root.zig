// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Training module for DeepZig V3
//! Provides optimizers, datasets, and training utilities

const std = @import("std");

// Re-export training components
pub const optimizer = @import("optimizer.zig");
pub const Adam = optimizer.Adam;
pub const SGD = optimizer.SGD;
pub const createOptimizer = optimizer.createOptimizer;

pub const dataset = @import("dataset.zig");
pub const TextDataset = dataset.TextDataset;

// Link to core module
pub const deepseek_core = @import("deepseek_core");
