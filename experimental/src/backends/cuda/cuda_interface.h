// CUDA Interface for DeepZig V3
// Provides C-compatible interface for CUDA/cuBLAS operations
// Compatible with Zig @cImport

#ifndef DEEPZIG_CUDA_INTERFACE_H
#define DEEPZIG_CUDA_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// Error codes
typedef enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INIT_FAILED = 1,
    CUDA_ERROR_DEVICE_NOT_FOUND = 2,
    CUDA_ERROR_OUT_OF_MEMORY = 3,
    CUDA_ERROR_INVALID_DEVICE = 4,
    CUDA_ERROR_INVALID_VALUE = 5,
    CUDA_ERROR_LAUNCH_FAILED = 6,
    CUDA_ERROR_CUBLAS_FAILED = 7,
    CUDA_ERROR_UNKNOWN = 99
} CudaError;

// GPU device information
typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    int major_compute_capability;
    int minor_compute_capability;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_blocks_per_grid;
    bool supports_tensor_cores;
} CudaDeviceInfo;

// Matrix layout for BLAS operations
typedef enum {
    CUDA_LAYOUT_ROW_MAJOR = 0,
    CUDA_LAYOUT_COL_MAJOR = 1
} CudaMatrixLayout;

// Transpose operations
typedef enum {
    CUDA_NO_TRANS = 0,
    CUDA_TRANS = 1,
    CUDA_CONJ_TRANS = 2
} CudaTranspose;

// Device Management
CudaError cuda_get_device_count(int* count);
CudaError cuda_get_device_info(int device_id, CudaDeviceInfo* info);
CudaError cuda_select_optimal_device(int* selected_device);

// Memory Management
CudaError cuda_malloc(void** ptr, size_t size);
CudaError cuda_free(void* ptr);
CudaError cuda_memcpy_h2d(void* dst, const void* src, size_t size);
CudaError cuda_memcpy_d2h(void* dst, const void* src, size_t size);

// Synchronization
CudaError cuda_device_synchronize(void);

// cuBLAS Operations
CudaError cuda_blas_create(void);
CudaError cuda_blas_destroy(void);

// Single-precision matrix multiplication: C = alpha * A * B + beta * C
CudaError cuda_blas_sgemm(
    CudaMatrixLayout layout,
    CudaTranspose transa,
    CudaTranspose transb,
    int m, int n, int k,
    float alpha,
    const float* a, int lda,
    const float* b, int ldb,
    float beta,
    float* c, int ldc
);

// Utility functions
const char* cuda_get_error_string(CudaError error);
bool cuda_is_available(void);

// Performance benchmarking
CudaError cuda_benchmark_sgemm(
    int size,
    int iterations,
    double* gflops_result
);

#ifdef __cplusplus
}
#endif

#endif // DEEPZIG_CUDA_INTERFACE_H
