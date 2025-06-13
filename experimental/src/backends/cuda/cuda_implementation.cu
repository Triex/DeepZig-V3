// CUDA Implementation for DeepZig V3
// High-performance GPU acceleration using CUDA and cuBLAS

#include "cuda_interface.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>

// Global cuBLAS handle
static cublasHandle_t g_cublas_handle = nullptr;
static bool g_cublas_initialized = false;

// Error conversion utilities
static CudaError convert_cuda_error(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return CUDA_SUCCESS;
        case cudaErrorInvalidDevice: return CUDA_ERROR_INVALID_DEVICE;
        case cudaErrorInvalidValue: return CUDA_ERROR_INVALID_VALUE;
        case cudaErrorMemoryAllocation: return CUDA_ERROR_OUT_OF_MEMORY;
        case cudaErrorLaunchFailure: return CUDA_ERROR_LAUNCH_FAILED;
        default: return CUDA_ERROR_UNKNOWN;
    }
}

static CudaError convert_cublas_error(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return CUDA_SUCCESS;
        case CUBLAS_STATUS_NOT_INITIALIZED: return CUDA_ERROR_INIT_FAILED;
        case CUBLAS_STATUS_ALLOC_FAILED: return CUDA_ERROR_OUT_OF_MEMORY;
        case CUBLAS_STATUS_INVALID_VALUE: return CUDA_ERROR_INVALID_VALUE;
        case CUBLAS_STATUS_ARCH_MISMATCH: return CUDA_ERROR_INVALID_DEVICE;
        case CUBLAS_STATUS_MAPPING_ERROR: return CUDA_ERROR_OUT_OF_MEMORY;
        case CUBLAS_STATUS_EXECUTION_FAILED: return CUDA_ERROR_LAUNCH_FAILED;
        default: return CUDA_ERROR_CUBLAS_FAILED;
    }
}

// Device Management
extern "C" CudaError cuda_get_device_count(int* count) {
    cudaError_t error = cudaGetDeviceCount(count);
    return convert_cuda_error(error);
}

extern "C" CudaError cuda_get_device_info(int device_id, CudaDeviceInfo* info) {
    if (!info) return CUDA_ERROR_INVALID_VALUE;

    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    if (error != cudaSuccess) {
        return convert_cuda_error(error);
    }

    info->device_id = device_id;
    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    info->total_memory = prop.totalGlobalMem;
    info->major_compute_capability = prop.major;
    info->minor_compute_capability = prop.minor;
    info->multiprocessor_count = prop.multiProcessorCount;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->max_blocks_per_grid = prop.maxGridSize[0];
    info->supports_tensor_cores = (prop.major >= 7);

    return CUDA_SUCCESS;
}

extern "C" CudaError cuda_select_optimal_device(int* selected_device) {
    if (!selected_device) return CUDA_ERROR_INVALID_VALUE;

    int device_count;
    CudaError error = cuda_get_device_count(&device_count);
    if (error != CUDA_SUCCESS || device_count == 0) {
        return CUDA_ERROR_DEVICE_NOT_FOUND;
    }

    int best_device = 0;
    float best_score = 0.0f;

    for (int i = 0; i < device_count; i++) {
        CudaDeviceInfo info;
        error = cuda_get_device_info(i, &info);
        if (error != CUDA_SUCCESS) continue;

        float compute_score = info.major_compute_capability * 10.0f + info.minor_compute_capability;
        float memory_score = static_cast<float>(info.total_memory) / (1024.0f * 1024.0f * 1024.0f);
        float sm_score = static_cast<float>(info.multiprocessor_count);

        float total_score = compute_score * 10.0f + memory_score + sm_score * 0.1f;

        if (total_score > best_score) {
            best_score = total_score;
            best_device = i;
        }
    }

    *selected_device = best_device;
    return CUDA_SUCCESS;
}

// Memory Management
extern "C" CudaError cuda_malloc(void** ptr, size_t size) {
    cudaError_t error = cudaMalloc(ptr, size);
    return convert_cuda_error(error);
}

extern "C" CudaError cuda_free(void* ptr) {
    cudaError_t error = cudaFree(ptr);
    return convert_cuda_error(error);
}

extern "C" CudaError cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    return convert_cuda_error(error);
}

extern "C" CudaError cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return convert_cuda_error(error);
}

extern "C" CudaError cuda_device_synchronize(void) {
    cudaError_t error = cudaDeviceSynchronize();
    return convert_cuda_error(error);
}

// cuBLAS Operations
extern "C" CudaError cuda_blas_create(void) {
    if (g_cublas_initialized) return CUDA_SUCCESS;

    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return convert_cublas_error(status);
    }

    status = cublasSetPointerMode(g_cublas_handle, CUBLAS_POINTER_MODE_HOST);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
        return convert_cublas_error(status);
    }

    g_cublas_initialized = true;
    return CUDA_SUCCESS;
}

extern "C" CudaError cuda_blas_destroy(void) {
    if (!g_cublas_initialized) return CUDA_SUCCESS;

    cublasStatus_t status = cublasDestroy(g_cublas_handle);
    g_cublas_handle = nullptr;
    g_cublas_initialized = false;

    return convert_cublas_error(status);
}

static cublasOperation_t convert_transpose(CudaTranspose trans) {
    switch (trans) {
        case CUDA_NO_TRANS: return CUBLAS_OP_N;
        case CUDA_TRANS: return CUBLAS_OP_T;
        case CUDA_CONJ_TRANS: return CUBLAS_OP_C;
        default: return CUBLAS_OP_N;
    }
}

extern "C" CudaError cuda_blas_sgemm(
    CudaMatrixLayout layout,
    CudaTranspose transa,
    CudaTranspose transb,
    int m, int n, int k,
    float alpha,
    const float* a, int lda,
    const float* b, int ldb,
    float beta,
    float* c, int ldc
) {
    if (!g_cublas_initialized) {
        CudaError error = cuda_blas_create();
        if (error != CUDA_SUCCESS) return error;
    }

    cublasOperation_t op_a = convert_transpose(transa);
    cublasOperation_t op_b = convert_transpose(transb);

    cublasStatus_t status;
    if (layout == CUDA_LAYOUT_ROW_MAJOR) {
        status = cublasSgemm(g_cublas_handle, op_b, op_a, n, m, k,
                           &alpha, b, ldb, a, lda, &beta, c, ldc);
    } else {
        status = cublasSgemm(g_cublas_handle, op_a, op_b, m, n, k,
                           &alpha, a, lda, b, ldb, &beta, c, ldc);
    }

    return convert_cublas_error(status);
}

extern "C" bool cuda_is_available(void) {
    int device_count;
    return (cuda_get_device_count(&device_count) == CUDA_SUCCESS && device_count > 0);
}

extern "C" const char* cuda_get_error_string(CudaError error) {
    switch (error) {
        case CUDA_SUCCESS: return "Success";
        case CUDA_ERROR_INIT_FAILED: return "Initialization failed";
        case CUDA_ERROR_DEVICE_NOT_FOUND: return "Device not found";
        case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case CUDA_ERROR_INVALID_DEVICE: return "Invalid device";
        case CUDA_ERROR_INVALID_VALUE: return "Invalid value";
        case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed";
        case CUDA_ERROR_CUBLAS_FAILED: return "cuBLAS operation failed";
        case CUDA_ERROR_UNKNOWN: return "Unknown error";
        default: return "Unrecognized error";
    }
}

extern "C" CudaError cuda_benchmark_sgemm(int size, int iterations, double* gflops_result) {
    if (!gflops_result) return CUDA_ERROR_INVALID_VALUE;

    if (!g_cublas_initialized) {
        CudaError error = cuda_blas_create();
        if (error != CUDA_SUCCESS) return error;
    }

    size_t matrix_size = size * size * sizeof(float);
    float *d_a, *d_b, *d_c;

    if (cuda_malloc((void**)&d_a, matrix_size) != CUDA_SUCCESS ||
        cuda_malloc((void**)&d_b, matrix_size) != CUDA_SUCCESS ||
        cuda_malloc((void**)&d_c, matrix_size) != CUDA_SUCCESS) {
        cuda_free(d_a); cuda_free(d_b); cuda_free(d_c);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    cudaMemset(d_a, 1, matrix_size);
    cudaMemset(d_b, 1, matrix_size);
    cudaMemset(d_c, 0, matrix_size);

    // Warmup
    cuda_blas_sgemm(CUDA_LAYOUT_ROW_MAJOR, CUDA_NO_TRANS, CUDA_NO_TRANS,
                    size, size, size, 1.0f, d_a, size, d_b, size, 0.0f, d_c, size);
    cuda_device_synchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cuda_blas_sgemm(CUDA_LAYOUT_ROW_MAJOR, CUDA_NO_TRANS, CUDA_NO_TRANS,
                        size, size, size, 1.0f, d_a, size, d_b, size, 0.0f, d_c, size);
    }
    cudaEventRecord(stop);
    cuda_device_synchronize();

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double ops = 2.0 * size * size * size * iterations;
    double elapsed_s = elapsed_ms / 1000.0;
    *gflops_result = ops / elapsed_s / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cuda_free(d_a);
    cuda_free(d_b);
    cuda_free(d_c);

    return CUDA_SUCCESS;
}
