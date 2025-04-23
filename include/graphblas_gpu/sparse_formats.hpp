#ifndef GRAPHBLAS_GPU_SPARSE_FORMATS_HPP
#define GRAPHBLAS_GPU_SPARSE_FORMATS_HPP

namespace graphblas_gpu {
    void csr_to_ell(
        const int* row_ptr,
        const int* col_idx,
        const float* values,
        int num_rows,
        int*& ell_col_indices,
        float*& ell_values,
        int& max_nnz_per_row
    );

    void ell_to_csr(
        const int* ell_col_indices,
        const float* ell_values,
        int num_rows,
        int max_nnz_per_row,
        int*& row_ptr,
        int*& col_idx,
        float*& values
    );
    
    void csr_to_sellc(
        const int* row_ptr,
        const int* col_idx,
        const float* values,
        int num_rows,
        int c,
        int& total_vals,
        int*& sell_col_idx,
        float*& sell_values,
        int*& slice_ptrs,
        int*& slice_lengths
    );

    void sellc_to_csr(
        const int* sell_col_idx,
        const float* sell_values,
        const int* slice_ptrs,
        const int* slice_lengths,
        int num_rows,
        int c,
        int*& row_ptr,
        int*& col_idx,
        float*& values
    );

} //namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_SPARSE_FORMATS_HPP