// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2025 TriexDev

const std = @import("std");
const print = std.debug.print;
const Allocator = std.mem.Allocator;

const cpu_backend = @import("cpu_backend");
const deepseek_core = @import("deepseek_core");
const metal_backend = @import("metal_backend");
const web_layer = @import("web_layer");

const Config = struct {
    port: u16 = 8080,
    host: []const u8 = "127.0.0.1",
    model_path: ?[]const u8 = null,
    backend: Backend = .cpu,
    max_concurrent_requests: u32 = 100,
    max_sequence_length: u32 = 32768,
    model_size: ModelSize = .tiny, // default to tiny for fast startup
    allocator: ?Allocator = null, // Store allocator for cleanup

    const Backend = enum { cpu, metal, cuda, webgpu };
    const ModelSize = enum { tiny, small, full };
    
    /// Free any memory allocated for this config
    pub fn deinit(self: *Config) void {
        if (self.allocator) |alloc| {
            // Free host if it's not the default value
            if (!std.mem.eql(u8, self.host, "127.0.0.1")) {
                alloc.free(self.host);
            }
            
            // Free model path if it exists
            if (self.model_path) |path| {
                alloc.free(path);
                self.model_path = null;
            }
            
            self.allocator = null;
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    var config = try parseArgs(allocator);
    defer config.deinit();

    // Initialize the selected backend
    var backend = try initBackend(allocator, config.backend);
    defer backend.deinit();

    // Load the model
    var model: deepseek_core.Model = switch (config.model_size) {
        .tiny => try deepseek_core.Model.loadTiny(allocator, backend),
        .small => try deepseek_core.Model.loadSmall(allocator, backend),
        .full => if (config.model_path) |path| blk: {
            const stat = std.fs.cwd().statFile(path) catch |err| {
                std.log.err("❌ Unable to stat model path {s}: {}", .{path, err});
                return err;
            };
            if (stat.kind == .directory) {
                break :blk try deepseek_core.Model.loadFromDirectory(allocator, path, backend);
            } else {
                break :blk try deepseek_core.Model.loadFromPath(allocator, path, backend);
            }
        } else
            try deepseek_core.Model.loadDefault(allocator, backend),
    };
    defer model.deinit();

    print("🚀 DeepZig V3 Server Starting...\n", .{});
    print("   Backend: {s}\n", .{@tagName(config.backend)});
    print("   Host: {s}:{d}\n", .{ config.host, config.port });
    print("   Model: {s}\n", .{model.info().name});
    print("   Max Context: {} tokens\n", .{config.max_sequence_length});

    // Test generation
    print("\n🧪 Testing text generation...\n", .{});
    const test_prompt = "Hello, world!";
    const generated = try model.generateText(test_prompt, 10);
    defer allocator.free(generated);
    print("🎉 Generated text: '{s}'\n", .{generated});

    print("\n✅ Model loaded successfully!\n", .{});
    print("💡 Run with --no-server to skip web server\n", .{});
    
    // Skip server for now - just test the model
    return;

    // Start the web server (unreachable for now)
    // var server = try web_layer.Server.init(allocator, .{
    //     .host = config.host,
    //     .port = config.port,
    //     .model = model,
    //     .max_concurrent_requests = config.max_concurrent_requests,
    // });
    // defer server.deinit();
    // 
    // print("✅ Server ready! Send requests to http://{s}:{d}\n", .{ config.host, config.port });
    // print("   Endpoints:\n", .{});
    // print("   - POST /v1/chat/completions (OpenAI compatible)\n", .{});
    // print("   - POST /v1/completions\n", .{});
    // print("   - GET  /v1/models\n", .{});
    // print("   - GET  /health\n", .{});
    // print("   - WebSocket /ws (streaming)\n", .{});
    // try server.listen();
}

fn parseArgs(allocator: Allocator) !Config {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var config = Config{ .allocator = allocator };

    // Environment variable overrides
    if ((std.process.hasEnvVar(allocator, "DZ_TINY") catch false))
        config.model_size = .tiny;
    if ((std.process.hasEnvVar(allocator, "DZ_SMALL") catch false))
        config.model_size = .small;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            config.port = try std.fmt.parseInt(u16, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--host") and i + 1 < args.len) {
            // Duplicate the host string so it remains valid after args are freed
            config.host = try allocator.dupe(u8, args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, arg, "--model") and i + 1 < args.len) {
            // Duplicate the model path slice because args will be freed before use
            config.model_path = try allocator.dupe(u8, args[i + 1]);
            config.model_size = .full;
            i += 1;
        } else if (std.mem.eql(u8, arg, "--backend") and i + 1 < args.len) {
            const backend_str = args[i + 1];
            config.backend = std.meta.stringToEnum(Config.Backend, backend_str) orelse {
                print("Unknown backend: {s}\n", .{backend_str});
                print("Available backends: cpu, metal, cuda, webgpu\n", .{});
                std.process.exit(1);
            };
            i += 1;
        } else if (std.mem.eql(u8, arg, "--tiny")) {
            config.model_size = .tiny;
        } else if (std.mem.eql(u8, arg, "--small")) {
            config.model_size = .small;
        } else if (std.mem.eql(u8, arg, "--full")) {
            config.model_size = .full;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--no-server")) {
            return config;
        }
    }

    return config;
}

fn initBackend(allocator: Allocator, backend_type: Config.Backend) !deepseek_core.Backend {
    return switch (backend_type) {
        .cpu => cpu_backend.init(allocator),
        .metal => metal_backend.init(allocator),
        .cuda => {
            print("CUDA backend not yet implemented, falling back to CPU\n", .{});
            return cpu_backend.init(allocator);
        },
        .webgpu => {
            print("WebGPU backend not yet implemented, falling back to CPU\n", .{});
            return cpu_backend.init(allocator);
        },
    };
}

fn printHelp() void {
    print("DeepZig V3 - High-Performance LLM Inference\n\n", .{});
    print("Usage: deepzig-v3 [OPTIONS]\n\n", .{});
    print("Options:\n", .{});
    print("  --port <PORT>        Port to listen on (default: 8080)\n", .{});
    print("  --host <HOST>        Host to bind to (default: 127.0.0.1)\n", .{});
    print("  --model <PATH>       Path to model weights (loads in FULL size)\n", .{});
    print("  --tiny               Use tiny test model (default)\n", .{});
    print("  --small              Use small test model\n", .{});
    print("  --full               Use full DeepZig model\n", .{});
    print("  --backend <BACKEND>  Backend to use: cpu, metal, cuda, webgpu (default: cpu)\n", .{});
    print("  --help, -h           Show this help message\n", .{});
    print("  --no-server          Skip starting the web server\n", .{});
    print("\n", .{});
    print("Examples:\n", .{});
    print("  deepzig-v3 --port 3000 --backend metal\n", .{});
    print("  deepzig-v3 --model ./models/deepseek-v3.bin --backend cuda\n", .{});
}
