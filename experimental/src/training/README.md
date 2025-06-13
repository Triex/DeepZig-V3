# DeepZig Native Training Implementation

This directory contains a pure Zig 0.15.0-dev implementation of the training pipeline for DeepZig models. The implementation is designed to be efficient, memory-safe, and take full advantage of Zig's features.

## Overview

The training implementation is divided into several components:

- **train.zig**: Main training entry point and implementation for medium-sized model
- **optimizer.zig**: Implementations of Adam and SGD optimizers for parameter updates
- **dataset.zig**: Dataset handling, batching, and tokenization utilities

## Usage

To train the model using the native Zig implementation:

```bash
cd /path/to/DeepZig
zig build train-model
```

This will compile and run the training pipeline with default parameters.

## Configuration

Training configuration options include:

- Batch size
- Number of epochs
- Learning rate
- Weight decay
- Output directory
- Maximum number of samples

These can be modified directly in the `TrainingConfig` struct in `train.zig`.

## Benefits of Zig Training

Training in Zig offers several advantages over the Python implementation:

1. **Memory efficiency**: Precise control over memory allocation and deallocation
2. **Performance**: Direct BLAS integration and efficient tensor operations
3. **Safety**: Compile-time checks and explicit error handling
4. **Consistency**: Same language for both training and inference
5. **No serialization overhead**: No need to convert between formats

## Technical Details

The implementation uses:

- Arena allocators for temporary allocations during training
- Vectorized operations where possible (using SIMD)
- Explicit memory management with proper cleanup
- Integration with BLAS libraries for matrix operations
- Custom dataset handling optimized for language model training

## Current Limitations

This is an experimental implementation and has some limitations:

- Limited dataset handling compared to HuggingFace datasets
- Fewer optimizer options than PyTorch
- Basic tokenization compared to specialized tokenizers
- Limited parallelism compared to PyTorch's distributed training

These limitations will be addressed in future updates.
