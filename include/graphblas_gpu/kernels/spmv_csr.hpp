#ifndef GRAPHBLAS_GPU_OPERATIONS_HPP
#define GRAPHBLAS_GPU_OPERATIONS_HPP

namespace graphblas_gpu{
namespace kernels {

// Thread -> Row mapping
template <typename T>
__device__ void spmv_csr(const size_t* row_offsets,
                        const size_t* col_indices,
                        const T* values,
                        const T* vector,
                        T* output,
                        size_t num_rows);


// do warp level implementation also..

} // namespace kernels
} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_OPERATIONS_HPP
