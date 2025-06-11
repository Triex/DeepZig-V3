// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

//! Test runner for Comprehensive Validation Framework
//! Demonstrates comprehensive validation across multiple testing levels

const std = @import("std");

const Backend = @import("core/backend.zig").Backend;
const validation = @import("validation.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 DeepSeek V3 Validation Test Runner", .{});
    std.log.info("=================================================================", .{});

    const backend = Backend{
        .type = .cpu,
        .device_id = 0,
        .allocator = allocator,
    };

    // Test different validation levels
    const validation_levels = [_]validation.ValidationLevel{
        .Basic,
        .Comprehensive,
        .Numerical,
        .Architectural,
    };

    for (validation_levels) |level| {
        std.log.info("", .{});
        std.log.info("🧪 Testing Validation Level: {s}", .{@tagName(level)});
        std.log.info("-----------------------------------------------------------------", .{});

        var validator = validation.Validator.init(allocator, backend, level);

        const results = try validator.runEnhancedValidation();
        defer {
            // Clean up each result's internal ArrayLists
            for (results) |*result| {
                result.deinit();
            }
            allocator.free(results);
        }

        // Calculate overall metrics
        var total_passed: u32 = 0;
        var total_score: f32 = 0.0;

        for (results) |result| {
            if (result.passed) total_passed += 1;
            total_score += result.metrics.getOverallScore();
        }

        const avg_score = total_score / @as(f32, @floatFromInt(results.len));
        const pass_rate = @as(f32, @floatFromInt(total_passed)) / @as(f32, @floatFromInt(results.len));

        std.log.info("📈 Level {s} Results: {d:.1}% pass rate, {d:.3} avg score", .{ @tagName(level), pass_rate * 100, avg_score });
    }

    std.log.info("", .{});
    std.log.info("✅ Validation testing completed!", .{});
    std.log.info("=================================================================", .{});
}
