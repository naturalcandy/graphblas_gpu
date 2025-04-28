// graph_classifier.hpp
#ifndef GRAPHBLAS_GPU_GRAPH_CLASSIFIER_HPP
#define GRAPHBLAS_GPU_GRAPH_CLASSIFIER_HPP

#include <vector>
#include <cstddef>
#include <string>
#include <algorithm>

namespace graphblas_gpu {

class GraphClassifier {
public:
    /* Format classification,  we will implement later
    static std::string determineOptimalFormat(
        const std::vector<size_t>& row_offsets,
        const std::vector<int>& col_indices,
        size_t rows, size_t cols); */
    
    // maybe we could implement these conversion functions to be on device side 
    
    // CSR to ELL
    template <typename T>
    static void csr_to_ell(
        const std::vector<size_t>& row_offsets,
        const std::vector<int>& col_indices,
        const std::vector<T>& values,
        size_t num_rows,
        std::vector<int>& ell_col_indices,
        std::vector<T>& ell_values,
        size_t& max_nnz_per_row);
    
    // ELL to CSR 
    template <typename T>
    static void ell_to_csr(
        const std::vector<int>& ell_col_indices,
        const std::vector<T>& ell_values,
        size_t num_rows,
        size_t max_nnz_per_row,
        std::vector<size_t>& row_offsets,
        std::vector<int>& col_indices,
        std::vector<T>& values);
    
    // CSR to SELLC 
    template <typename T>
    static void csr_to_sellc(
        const std::vector<size_t>& row_offsets,
        const std::vector<int>& col_indices,
        const std::vector<T>& values,
        size_t num_rows,
        size_t slice_size,
        std::vector<size_t>& slice_ptrs,
        std::vector<size_t>& slice_lengths,
        std::vector<int>& sell_col_indices,
        std::vector<T>& sell_values);
    
    // SELLC to CSR 
    template <typename T>
    static void sellc_to_csr(
        const std::vector<size_t>& slice_ptrs,
        const std::vector<size_t>& slice_lengths,
        const std::vector<int>& sell_col_indices,
        const std::vector<T>& sell_values,
        size_t num_rows,
        size_t slice_size,
        std::vector<size_t>& row_offsets,
        std::vector<int>& col_indices,
        std::vector<T>& values);
    
    template <typename T>
    void dense_to_csr(const T* dense, size_t num_rows, size_t num_cols,
                      size_t*& row_offsets, size_t*& col_indices, T*& values);
                    
    template <typename T>
    void dense_to_ell(const T* dense, size_t num_rows, size_t num_cols,
                  size_t*& ell_col_indices, T*& ell_values, size_t& max_cols_per_row);

    template <typename T>
    void dense_to_sell_c(const T* dense, size_t num_rows, size_t num_cols, size_t slice_height,
                    size_t*& sell_col_indices, T*& sell_values,
                    size_t*& slice_ptrs, size_t*& slice_lengths);

    

private:
    static size_t countNonPadding(const std::vector<int>& col_indices);
};

// Helper function to count non-padding elements
inline size_t GraphClassifier::countNonPadding(const std::vector<int>& col_indices) {
    size_t count = 0;
    for (const auto& col : col_indices) {
        if (col != -1) {
            count++;
        }
    }
    return count;
}

// CSR to ELL conversion
template <typename T>
void GraphClassifier::csr_to_ell(
    const std::vector<size_t>& row_offsets,
    const std::vector<int>& col_indices,
    const std::vector<T>& values,
    size_t num_rows,
    std::vector<int>& ell_col_indices,
    std::vector<T>& ell_values,
    size_t& max_nnz_per_row) {

    // Find maximum non-zeros per row
    max_nnz_per_row = 0;
    for (size_t i = 0; i < num_rows; i++) {
        size_t count = row_offsets[i + 1] - row_offsets[i];
        max_nnz_per_row = std::max(max_nnz_per_row, count);
    }

    // Resize output vectors
    ell_col_indices.resize(num_rows * max_nnz_per_row);
    ell_values.resize(num_rows * max_nnz_per_row);

    // Convert CSR to ELL
    for (size_t i = 0; i < num_rows; i++) {
        size_t row_start = row_offsets[i];
        size_t row_end = row_offsets[i + 1];
        size_t k = 0;

        // Copy actual elements
        for (size_t j = row_start; j < row_end; j++, k++) {
            ell_col_indices[i * max_nnz_per_row + k] = col_indices[j];
            ell_values[i * max_nnz_per_row + k] = values[j];
        }

        // Pad with -1 for columns and 0 for values
        for (; k < max_nnz_per_row; k++) {
            ell_col_indices[i * max_nnz_per_row + k] = -1;  // Changed from static_cast<size_t>(-1)
            ell_values[i * max_nnz_per_row + k] = T(0);
        }
    }
}

// ELL to CSR conversion
template <typename T>
void GraphClassifier::ell_to_csr(
    const std::vector<int>& ell_col_indices,
    const std::vector<T>& ell_values,
    size_t num_rows,
    size_t max_nnz_per_row,
    std::vector<size_t>& row_offsets,
    std::vector<int>& col_indices,
    std::vector<T>& values) {
    
    // Count non-padding elements to determine CSR array sizes
    size_t nnz = 0;
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < max_nnz_per_row; j++) {
            size_t idx = i * max_nnz_per_row + j;
            if (idx < ell_col_indices.size() && ell_col_indices[idx] != -1) {  // Changed from static_cast<size_t>(-1)
                nnz++;
            }
        }
    }

    // Resize output vectors
    row_offsets.resize(num_rows + 1);
    col_indices.resize(nnz);
    values.resize(nnz);

    // Convert ELL to CSR
    size_t nnz_counter = 0;
    row_offsets[0] = 0;

    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < max_nnz_per_row; j++) {
            size_t idx = i * max_nnz_per_row + j;
            if (idx < ell_col_indices.size() && ell_col_indices[idx] != -1) {  // Changed from static_cast<size_t>(-1)
                col_indices[nnz_counter] = ell_col_indices[idx];
                values[nnz_counter] = ell_values[idx];
                nnz_counter++;
            }
        }
        row_offsets[i + 1] = nnz_counter;
    }
}

// CSR to SELLC conversion
template <typename T>
void GraphClassifier::csr_to_sellc(
    const std::vector<size_t>& row_offsets,
    const std::vector<int>&    col_indices,
    const std::vector<T>&      values,
    size_t                     num_rows,
    size_t                     slice_size,          // C
    std::vector<size_t>&       slice_ptrs,          // (numSlices+1)
    std::vector<size_t>&       slice_lengths,       // (numSlices)
    std::vector<int>&          sell_col_indices,    // length = total_vals
    std::vector<T>&            sell_values)         // length = total_vals
{
    /* ---------- 1. basic geometry ---------- */
    const size_t num_slices = (num_rows + slice_size - 1) / slice_size;
    slice_ptrs.resize   (num_slices + 1);
    slice_lengths.resize(num_slices);

    /* ---------- 2. determine max nnz/row in every slice ---------- */
    size_t total_vals = 0;
    for (size_t s = 0; s < num_slices; ++s)
    {
        size_t max_nnz = 0;
        for (size_t r = 0; r < slice_size; ++r)            // rows *inside* slice
        {
            const size_t row = s * slice_size + r;
            if (row >= num_rows) break;

            const size_t row_nnz = row_offsets[row + 1] - row_offsets[row];
            max_nnz = std::max(max_nnz, row_nnz);
        }
        slice_lengths[s] = max_nnz;            // k_max for this slice
        total_vals      += max_nnz * slice_size;
    }

    /* ---------- 3. prefix sum for slice_ptrs ---------- */
    slice_ptrs[0] = 0;
    for (size_t s = 1; s <= num_slices; ++s)
        slice_ptrs[s] = slice_ptrs[s - 1] + slice_lengths[s - 1] * slice_size;

    /* ---------- 4. allocate output buffers ---------- */
    sell_col_indices.assign(total_vals, -1);
    sell_values      .assign(total_vals, T(0));

    /* ---------- 5. fill buffers column-major inside each slice ---------- */
    for (size_t s = 0; s < num_slices; ++s)
    {
        const size_t slice_offset = slice_ptrs[s];
        const size_t k_max        = slice_lengths[s];

        for (size_t r = 0; r < slice_size; ++r)            // local row
        {
            const size_t global_row = s * slice_size + r;
            if (global_row >= num_rows) break;

            const size_t row_start = row_offsets[global_row];
            const size_t row_end   = row_offsets[global_row + 1];
            const size_t row_nnz   = row_end - row_start;

            /* copy real nnz first, then leave padding at -1 / 0 */
            for (size_t k = 0; k < row_nnz; ++k)
            {
                const size_t dst = slice_offset + k * slice_size + r; // k·C + r
                sell_col_indices[dst] = col_indices[row_start + k];
                sell_values      [dst] = values     [row_start + k];
            }
            /* remaining k positions already contain -1 / 0 from assign() */
        }
    }
}


// SELLC to CSR conversion
template <typename T>
void GraphClassifier::sellc_to_csr(
    const std::vector<size_t>& slice_ptrs,
    const std::vector<size_t>& slice_lengths,
    const std::vector<int>& sell_col_indices,
    const std::vector<T>& sell_values,
    size_t num_rows,
    size_t slice_size,
    std::vector<size_t>& row_offsets,
    std::vector<int>& col_indices,
    std::vector<T>& values) {
    
    // Count non-padding elements to determine CSR array sizes
    size_t num_slices = (num_rows + slice_size - 1) / slice_size;
    size_t nnz = 0;
    
    for (const auto& col : sell_col_indices) {
        if (col != -1) {  // Changed from static_cast<size_t>(-1)
            nnz++;
        }
    }
    
    // Resize CSR arrays
    row_offsets.resize(num_rows + 1);
    col_indices.resize(nnz);
    values.resize(nnz);
    
    // Convert SELL-C to CSR
    size_t nnz_counter = 0;
    row_offsets[0] = 0;
    
    for (size_t slice = 0; slice < num_slices; slice++) {
        size_t slice_offset = slice_ptrs[slice];
        size_t max_cols = slice_lengths[slice];
        
        for (size_t i = 0; i < slice_size; i++) {
            size_t row = slice * slice_size + i;
            if (row >= num_rows) break;
            
            size_t row_nnz = 0;
            for (size_t j = 0; j < max_cols; j++) {
                size_t idx = slice_offset + j * slice_size + i;
                if (sell_col_indices[idx] != -1) {  // Changed from static_cast<size_t>(-1)
                    col_indices[nnz_counter] = sell_col_indices[idx];
                    values[nnz_counter] = sell_values[idx];
                    nnz_counter++;
                    row_nnz++;
                }
            }
            
            row_offsets[row + 1] = row_offsets[row] + row_nnz;
        }
    }
}

template <typename T>
void GraphClassifier::dense_to_csr(const T* dense, size_t num_rows, size_t num_cols,
                  size_t*& row_offsets, size_t*& col_indices, T*& values) {
    // First pass: count non-zeros
    size_t nnz = 0;
    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (dense[i] != 0.0f) ++nnz;
    }

    row_offsets = new size_t[num_rows + 1];
    col_indices = new size_t[nnz];
    values = new float[nnz];

    size_t index = 0;
    row_offsets[0] = 0;
    for (size_t row = 0; row < num_rows; ++row) {
        for (size_t col = 0; col < num_cols; ++col) {
            float val = dense[row * num_cols + col];
            if (val != 0.0f) {
                col_indices[index] = col;
                values[index] = val;
                ++index;
            }
        }
        row_offsets[row + 1] = index;
    }
}

template <typename T>
void GraphClassifier::dense_to_ell(const T* dense, size_t num_rows, size_t num_cols,
    size_t*& ell_col_indices, T*& ell_values, size_t& max_cols_per_row) {
    // Determine max nnz per row
    max_cols_per_row = 0;
    for (size_t row = 0; row < num_rows; ++row) {
        int count = 0;
        for (size_t col = 0; col < num_cols; ++col) {
            if (dense[row * num_cols + col] != 0.0f) ++count;
        }
        if (count > max_cols_per_row) max_cols_per_row = count;
    }

    ell_col_indices = new size_t[num_rows * max_cols_per_row];
    ell_values = new float[num_rows * max_cols_per_row];

    for (size_t row = 0; row < num_rows; ++row) {
        int nz = 0;
        for (size_t col = 0; col < num_cols; ++col) {
            float val = dense[row * num_cols + col];
            if (val != 0.0f) {
                ell_col_indices[row + nz * num_rows] = col;
                ell_values[row + nz * num_rows] = val;
                ++nz;
            }
        }
        // Pad remaining
        for (int k = nz; k < max_cols_per_row; ++k) {
            ell_col_indices[row + k * num_rows] = -1;
            ell_values[row + k * num_rows] = 0.0f;
        }
    }
}

template <typename T>
void GraphClassifier::dense_to_sell_c(const T* dense, size_t num_rows, size_t num_cols, size_t slice_height,
    size_t*& sell_col_indices, T*& sell_values,
                    size_t*& slice_ptrs, size_t*& slice_lengths) {
    int num_slices = (num_rows + slice_height - 1) / slice_height;

    slice_ptrs = new size_t[num_slices + 1];
    slice_lengths = new size_t[num_slices];

    // First pass: find max nnz per row in each slice
    int total_nnz_entries = 0;
    for (int s = 0; s < num_slices; ++s) {
        int max_cols = 0;
        for (int i = 0; i < slice_height; ++i) {
            int row = s * slice_height + i;
            if (row >= (int)num_rows) break;
            int count = 0;
            for (size_t col = 0; col < num_cols; ++col) {
                if (dense[row * num_cols + col] != 0.0f) ++count;
            }
            if (count > max_cols) max_cols = count;
        }
        slice_lengths[s] = max_cols;
        total_nnz_entries += max_cols * slice_height;
    }

    sell_col_indices = new size_t[total_nnz_entries];
    sell_values = new float[total_nnz_entries];

    int offset = 0;
    for (int s = 0; s < num_slices; ++s) {
        slice_ptrs[s] = offset;
        int max_cols = slice_lengths[s];

        for (int j = 0; j < max_cols; ++j) {
            for (int i = 0; i < slice_height; ++i) {
                int row = s * slice_height + i;
                int idx = offset + j * slice_height + i;
                if (row >= (int)num_rows) {
                    sell_col_indices[idx] = -1;
                    sell_values[idx] = 0.0f;
                    continue;
                }

                // Find the j-th non-zero
                int found = 0;
                for (size_t col = 0; col < num_cols; ++col) {
                    float val = dense[row * num_cols + col];
                    if (val != 0.0f) {
                        if (found == j) {
                            sell_col_indices[idx] = col;
                            sell_values[idx] = val;
                            break;
                        }
                        ++found;
                    }
                }

                if (found <= j) {
                    sell_col_indices[idx] = -1;
                    sell_values[idx] = 0.0f;
                }
            }
        }

        offset += max_cols * slice_height;
    }

    slice_ptrs[num_slices] = offset;
}


} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_GRAPH_CLASSIFIER_HPP