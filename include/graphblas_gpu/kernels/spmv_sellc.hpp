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
                            );
    
} // namespace kernels
} // namespace graphblas_gpu