#include <graphblas_gpu/kernels/spmv_csr.hpp>

namespace graphblas_gpu{
namespace kernels {

// thread -> row, arithmetic csr
template <typename T>
__device__ void spmv_csr(const size_t* row_offsets,
                        const int* col_indices,
                        const T* values,
                        const T* vector,
                        const T* mask,
                        bool mask_enabled,
                        T* output,
                        size_t num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    T sum = T(0);
    size_t row_start = row_offsets[row];
    size_t row_end = row_offsets[row + 1];
    for (size_t i = row_start; i < row_end; i++) {
        sum += values[i] * vector[col_indices[i]]; 
    }
    if (mask_enabled) { 
        output[row] = sum * T(mask[row] != T(0));
    } else {
        output[row] = sum;
    }
}

// thread -> row, or-and csr
template <typename T>
__device__ void spmv_csr_logical(const size_t* row_offsets,
                                 const int* col_indices,
                                 const T* values,
                                 const T* vector,
                                 const T* mask,
                                 bool mask_enabled,
                                 T* output,
                                 size_t num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    bool result = false;
    size_t row_start = row_offsets[row];
    size_t row_end = row_offsets[row + 1];
    for (size_t i = row_start; i < row_end; i++) {
        bool val = (values[i] != 0) && (vector[col_indices[i]] != 0);
        result = result || val;
        if (result) break; 
    }
    if (mask_enabled) {
        output[row] = static_cast<T>(result && (mask[row] != 0));
    } else {
        output[row] = static_cast<T>(result);
    }
}

// vector based csr, arithmetic
template <typename T, int THREADS_PER_ROW>
__device__ void spmv_csr_vector_arithmetic(const size_t* row_offsets,
                                           const int* col_indices,
                                           const T* values,
                                           const T* vector,
                                           const T* mask,
                                           bool mask_enabled,
                                           T* output,
                                           size_t num_rows) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;
    size_t start = row_offsets[row];
    size_t end = row_offsets[row + 1];

    T sum = T(0);
    for (size_t i = start + threadIdx.x; i < end; i += THREADS_PER_ROW) {
        sum += values[i] * vector[col_indices[i]];
    }

#pragma unroll
    for (int offset = THREADS_PER_ROW >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset, THREADS_PER_ROW);
    }

    if (threadIdx.x == 0) {
        if (mask_enabled) {
            output[row] = sum * T(mask[row] != T(0));
        } else {
            output[row] = sum;
        }
    }
}


// vector based csr or-and 
template <typename T, int THREADS_PER_ROW>
__device__ void spmv_csr_vector_logical(const size_t* row_offsets,
                                        const int* col_indices,
                                        const T* values,
                                        const T* vector,
                                        const T* mask,
                                        bool mask_enabled,
                                        T* output,
                                        size_t num_rows) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;
    size_t start = row_offsets[row];
    size_t end = row_offsets[row + 1];

    bool result = false;
    for (size_t i = start + threadIdx.x; i < end; i += THREADS_PER_ROW) {
        bool val = (values[i] != 0) && (vector[col_indices[i]] != 0);
        result = result || val;
    }

#pragma unroll
    for (int offset = THREADS_PER_ROW >> 1; offset > 0; offset >>= 1) {
        bool other = __shfl_down_sync(0xFFFFFFFF, result, offset, THREADS_PER_ROW);
        result = result || other;
    }

    if (threadIdx.x == 0) {
        if (mask_enabled) {
            output[row] = static_cast<T>(result && (mask[row] != 0));
        } else {
            output[row] = static_cast<T>(result);
        }
    }
}

} // namespace kernels
} // namespace graphblas_gpu