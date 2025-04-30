// graph_classifier.hpp
#ifndef GRAPHBLAS_GPU_GRAPH_CLASSIFIER_HPP
#define GRAPHBLAS_GPU_GRAPH_CLASSIFIER_HPP

#include <vector>
#include <cstddef>
#include <string>
#include <cmath>
#include <algorithm>

namespace graphblas_gpu {

class GraphClassifier {
public:
    
    static std::string chooseFormat(const std::vector<size_t>& row_offsets,
                                    size_t rows);
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
    
    // CSR -> CSR (transpose)
    template <typename T>
    static void csr_transpose(
        size_t rows, size_t cols,
        const std::vector<size_t>& row_offsets,
        const std::vector<int>& col_indices,
        const std::vector<T>& values,
        std::vector<size_t>& row_offsets_T,
        std::vector<int>& col_indices_T,
        std::vector<T>& values_T);


private:
    static size_t countNonPadding(const std::vector<int>& col_indices);
};

inline size_t GraphClassifier::countNonPadding(const std::vector<int>& col_indices) {
    size_t count = 0;
    for (const auto& col : col_indices) {
        if (col != -1) {
            count++;
        }
    }
    return count;
}

inline std::string GraphClassifier::chooseFormat(const std::vector<size_t>& row_offsets,
                                                 size_t rows)
{
    const size_t nnz_total = row_offsets.back();
    const double mean = static_cast<double>(nnz_total) / rows;

    size_t max_nnz = 0;
    double var_acc = 0.0;

    for (size_t r = 0; r < rows; ++r) {
        const size_t len = row_offsets[r + 1] - row_offsets[r];
        max_nnz = std::max(max_nnz, len);
        const double diff = static_cast<double>(len) - mean;
        var_acc += diff * diff;
    }

    const double variance = var_acc / rows;
    const double cv = (mean == 0.0) ? 0.0 : (std::sqrt(variance) / mean);

    constexpr size_t ELL_ROW_LIMIT     = 32;   
    constexpr double CV_LOW_THRESHOLD  = 0.25; 
    constexpr double CV_HIGH_THRESHOLD = 0.75; 

    if (cv <= CV_LOW_THRESHOLD) {
        return (max_nnz <= ELL_ROW_LIMIT) ? "ELL" : "SELLC";
    }
    if (cv <= CV_HIGH_THRESHOLD) {
        return "SELLC";
    }
    return "CSR";
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

    max_nnz_per_row = 0;
    for (size_t i = 0; i < num_rows; i++) {
        size_t count = row_offsets[i + 1] - row_offsets[i];
        max_nnz_per_row = std::max(max_nnz_per_row, count);
    }

    ell_col_indices.resize(num_rows * max_nnz_per_row);
    ell_values.resize(num_rows * max_nnz_per_row);

    for (size_t i = 0; i < num_rows; i++) {
        size_t row_start = row_offsets[i];
        size_t row_end = row_offsets[i + 1];
        size_t k = 0;

        for (size_t j = row_start; j < row_end; j++, k++) {
            ell_col_indices[i * max_nnz_per_row + k] = col_indices[j];
            ell_values[i * max_nnz_per_row + k] = values[j];
        }

        for (; k < max_nnz_per_row; k++) {
            ell_col_indices[i * max_nnz_per_row + k] = -1;  
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
    size_t nnz = 0;
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < max_nnz_per_row; j++) {
            size_t idx = i * max_nnz_per_row + j;
            if (idx < ell_col_indices.size() && ell_col_indices[idx] != -1) {  
                nnz++;
            }
        }
    }
    row_offsets.resize(num_rows + 1);
    col_indices.resize(nnz);
    values.resize(nnz);
    size_t nnz_counter = 0;
    row_offsets[0] = 0;
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j < max_nnz_per_row; j++) {
            size_t idx = i * max_nnz_per_row + j;
            if (idx < ell_col_indices.size() && ell_col_indices[idx] != -1) {  
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
    const size_t num_slices = (num_rows + slice_size - 1) / slice_size;
    slice_ptrs.resize   (num_slices + 1);
    slice_lengths.resize(num_slices);

    size_t total_vals = 0;
    for (size_t s = 0; s < num_slices; ++s)
    {
        size_t max_nnz = 0;
        for (size_t r = 0; r < slice_size; ++r)            
        {
            const size_t row = s * slice_size + r;
            if (row >= num_rows) break;
            const size_t row_nnz = row_offsets[row + 1] - row_offsets[row];
            max_nnz = std::max(max_nnz, row_nnz);
        }
        slice_lengths[s] = max_nnz;            
        total_vals += max_nnz * slice_size;
    }

    slice_ptrs[0] = 0;
    for (size_t s = 1; s <= num_slices; ++s)
        slice_ptrs[s] = slice_ptrs[s - 1] + slice_lengths[s - 1] * slice_size;

    sell_col_indices.assign(total_vals, -1);
    sell_values.assign(total_vals, T(0));

    for (size_t s = 0; s < num_slices; ++s)
    {
        const size_t slice_offset = slice_ptrs[s];
        const size_t k_max = slice_lengths[s];
        for (size_t r = 0; r < slice_size; ++r)            
        {
            const size_t global_row = s * slice_size + r;
            if (global_row >= num_rows) break;

            const size_t row_start = row_offsets[global_row];
            const size_t row_end = row_offsets[global_row + 1];
            const size_t row_nnz = row_end - row_start;

            for (size_t k = 0; k < row_nnz; ++k)
            {
                const size_t dst = slice_offset + k * slice_size + r; 
                sell_col_indices[dst] = col_indices[row_start + k];
                sell_values[dst] = values[row_start + k];
            }
            
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
    
    size_t num_slices = (num_rows + slice_size - 1) / slice_size;
    size_t nnz = 0;
    for (const auto& col : sell_col_indices) {
        if (col != -1) {  
            nnz++;
        }
    }
    row_offsets.resize(num_rows + 1);
    col_indices.resize(nnz);
    values.resize(nnz);
    
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
                if (sell_col_indices[idx] != -1) {  
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
void GraphClassifier::csr_transpose(
        size_t rows, size_t cols,
        const std::vector<size_t>& row_offsets,
        const std::vector<int>& col_indices,
        const std::vector<T>& values,
        std::vector<size_t>& row_offsets_T,
        std::vector<int>& col_indices_T,
        std::vector<T>& values_T)
{
    const size_t nnz = col_indices.size();
    row_offsets_T.resize(cols + 1, 0);

    for (int c : col_indices) {
        if (c < 0 || static_cast<size_t>(c) >= cols)
            throw std::out_of_range("csr_transpose: column index out of range");
        ++row_offsets_T[c + 1];
    }

    for (size_t i = 0; i < cols; ++i)
        row_offsets_T[i + 1] += row_offsets_T[i];

    col_indices_T.resize(nnz);
    values_T.resize(nnz);
    std::vector<size_t> cursor(row_offsets_T.begin(), row_offsets_T.end() - 1);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t p = row_offsets[r]; p < row_offsets[r + 1]; ++p) {
            int c = col_indices[p];
            size_t dst = cursor[c]++;
            col_indices_T[dst] = static_cast<int>(r);  
            values_T[dst] = values[p];
        }
    }
}

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_GRAPH_CLASSIFIER_HPP