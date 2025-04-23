#include <iostream>
#include <graphblas_gpu/sparse_formats.hpp>
#include <algorithm>

namespace graphblas_gpu {
    void csr_to_ell(
        const int* row_ptr,
        const int* col_idx,
        const float* values,
        int num_rows,
        int*& ell_col_indices,
        float*& ell_values,
        int& max_nnz_per_row
    ) {
        max_nnz_per_row = 0;
        for (int i = 0; i < num_rows; i++) {
            int count = row_ptr[i + 1] - row_ptr[i];
            max_nnz_per_row = std::max(max_nnz_per_row, count);
        }

        ell_col_indices = new int[num_rows * max_nnz_per_row];
        ell_values = new float[num_rows * max_nnz_per_row];

        for (int i = 0; i < num_rows; i++) {
            int row_start = row_ptr[i];
            int row_end = row_ptr[i + 1];
            int k = 0;

            for (int j = row_start; j < row_end; j++, k++) {
                ell_col_indices[i * max_nnz_per_row + k] = col_idx[j];
                ell_values[i * max_nnz_per_row + k] = values[j];
            }

            for (; k < max_nnz_per_row; k++) {
                ell_col_indices[i * max_nnz_per_row + k] = -1;
                ell_values[i * max_nnz_per_row + k] = 0.0f;
            }
        }
    }

    int get_ell_nnz(const int* ell_col_indices, int num_rows, int max_nnz_per_row) {
        int nnz = 0;
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < max_nnz_per_row; ++j) {
                int idx = i * max_nnz_per_row + j;
                if (ell_col_indices[idx] != -1) {
                    ++nnz;
                }
            }
        }
        return nnz;
    }

    void ell_to_csr(
        const int* ell_col_indices,
        const float* ell_values,
        int num_rows,
        int max_nnz_per_row,
        int*& row_ptr,
        int*& col_idx,
        float*& values
    ) {
        int estimated_nnz = get_ell_nnz(ell_col_indices, num_rows, max_nnz_per_row);
        row_ptr = new int[num_rows + 1];
        col_idx = new int[estimated_nnz];
        values = new float[estimated_nnz];

        int nnz_counter = 0;
        row_ptr[0] = 0;

        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < max_nnz_per_row; j++) {
                int idx = i * max_nnz_per_row + j;
                int col = ell_col_indices[idx];
                float val = ell_values[idx];

                if (col == -1) continue;

                col_idx[nnz_counter] = col;
                values[nnz_counter] = val;
                nnz_counter++;
            }
            row_ptr[i + 1] = nnz_counter;
        }
    }

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
    ) {
        int num_slices = (num_rows + c - 1) / c;
        slice_ptrs = new int[num_slices + 1];
        slice_lengths = new int[num_slices];

        total_vals = 0;
        for (int slice = 0; slice < num_slices; slice++) {
            int max_nnz = 0;
            for (int i = 0; i < c; ++i) {
                int row = slice * c + i;
                if (row >= num_rows) break;
                int row_nnz = row_ptr[row + 1] - row_ptr[row];
                if (row_nnz > max_nnz) max_nnz = row_nnz;
            }
            slice_lengths[slice] = max_nnz;
            total_vals += max_nnz * c;
        }

        sell_col_idx = new int[total_vals];
        sell_values = new float[total_vals];
        slice_ptrs[0] = 0;
        for (int slice = 1; slice <= num_slices; slice++)
            slice_ptrs[slice] = slice_ptrs[slice - 1] + slice_lengths[slice - 1] * c;

        for (int slice = 0; slice < num_slices; slice++) {
            int slice_offset = slice_ptrs[slice];
            int max_cols = slice_lengths[slice];
            for (int j = 0; j < max_cols; ++j) {
                for (int i = 0; i < c; ++i) {
                    int row = slice * c + i;
                    int index = slice_offset + j * c + i;

                    if (row >= num_rows) {
                        sell_col_idx[index] = -1;
                        sell_values[index] = 0.0f;
                        continue;
                    }

                    int row_start = row_ptr[row];
                    int row_end = row_ptr[row + 1];
                    int row_len = row_end - row_start;
                    if (j < row_len) {
                        sell_col_idx[index] = col_idx[row_start + j];
                        sell_values[index] = values[row_start + j];
                    } else {
                        sell_col_idx[index] = -1;
                        sell_values[index] = 0.0f;
                    }
                }
            }
        }
    }

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
    ) {
        int estimated_nnz = 0;
        int num_slices = (num_rows + c - 1) / c;
        for (int slice = 0; slice < num_slices; ++slice) {
            estimated_nnz += slice_lengths[slice] * c;
        }

        row_ptr = new int[num_rows + 1];
        col_idx = new int[estimated_nnz];
        values = new float[estimated_nnz];

        int nnz = 0;
        row_ptr[0] = 0;

        for (int slice = 0; slice < num_slices; ++slice) {
            int slice_offset = slice_ptrs[slice];
            int max_cols = slice_lengths[slice];

            for (int i = 0; i < c; ++i) {
                int row = slice * c + i;
                if (row >= num_rows) break;

                for (int j = 0; j < max_cols; ++j) {
                    int idx = slice_offset + j * c + i;
                    int col = sell_col_idx[idx];
                    float val = sell_values[idx];

                    if (col != -1) {
                        col_idx[nnz] = col;
                        values[nnz] = val;
                        ++nnz;
                    }
                }

                row_ptr[row + 1] = nnz;
            }
        }
    }

} // namespace graphblas_gpu
