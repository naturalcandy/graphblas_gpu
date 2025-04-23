#include <graphblas_gpu/kernels/spmv_sellc.hpp>

namespace graphblas_gpu{
namespace kernels {

template <typename T>
__device__ void spmv_sell_c(const size_t* slice_offsets,
                            const size_t* col_indices,
                            const T* values,
                            size_t num_rows,
                            size_t c,
                            const T* vector,
                            T* output
                           ) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= num_rows) return;

    size_t slice_id = global_row / c;
    size_t local_row = global_row % c;

    size_t slice_start = slice_offsets[slice_id];
    size_t slice_start_idx = slice_start + local_row;

    size_t slice_width = slice_offsets[slice_id + 1] - slice_offsets[slice_id];
    size_t max_nnz_per_row = slice_width / c;

    T sum = T(0);
    for (size_t i = 0; i < max_nnz_per_row; ++i) {
        size_t idx = slice_start_idx + i * c;
        size_t col = col_indices[idx];
        T val = values[idx];

        if (val != T(0)) {
            sum += val * vector[col];
        }
    }

    output[global_row] = sum;
}

} // namespace kernels
} // namespace graphblas_gpu