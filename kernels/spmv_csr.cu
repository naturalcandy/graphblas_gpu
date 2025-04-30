#include <graphblas_gpu/kernels/spmv_csr.hpp>

namespace graphblas_gpu{
namespace kernels {

// thread -> row arithmetic
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

// thread -> row, or-and
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

// row per warp, arithmetic
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
    const size_t start = row_offsets[row];
    const size_t end = row_offsets[row + 1];

    T partial = T(0);
    for (size_t p = start + threadIdx.x; p < end; p += THREADS_PER_ROW) {
        partial += values[p] * vector[col_indices[p]];
    }
#pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset, THREADS_PER_ROW);
    }

    if (threadIdx.x == 0) {
        output[row] = mask_enabled ? partial * T(mask[row] != T(0)) : partial;
    }
}


// row per warp or-and 
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
    const size_t start = row_offsets[row];
    const size_t end = row_offsets[row + 1];

    bool vote = false;
    for (size_t p = start + threadIdx.x; p < end; p += THREADS_PER_ROW) {
        vote |= (values[p] != T(0)) && (vector[col_indices[p]] != T(0));
    }
    unsigned int v = static_cast<unsigned int>(vote);
#pragma unroll
    for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
        v |= __shfl_down_sync(0xFFFFFFFF, v, offset, THREADS_PER_ROW);
    }

    if (threadIdx.x == 0) {
        bool row_out = static_cast<bool>(v);
        if (mask_enabled) {
            row_out = row_out && (mask[row] != T(0));
        }
        output[row] = static_cast<T>(row_out);
    }
}

} // namespace kernels
} // namespace graphblas_gpu