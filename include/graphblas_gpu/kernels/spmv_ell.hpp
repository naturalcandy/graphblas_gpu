
namespace graphblas_gpu{
namespace kernels {

template <typename T>
__device__ void spmv_ell(const size_t* col_indices,
                            const T* values,
                            const T* vector,
                            T* output,
                            size_t num_rows,
                            size_t max_nnz_per_row);

} // namespace kernels
} // namespace graphblas_gpu