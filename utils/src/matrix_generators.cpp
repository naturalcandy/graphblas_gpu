// utils/src/matrix_generators.cpp
#include <graphblas_gpu_utils/matrix_generators.hpp>
#include <random>
#include <algorithm>
#include <cmath>
#include <set>

namespace graphblas_gpu {
namespace utils {

// Generate a uniform random sparse matrix
void generate_uniform_random_csr(size_t num_rows, size_t num_cols, float sparsity, 
                              std::vector<size_t>& row_offsets,
                              std::vector<int>& col_indices,
                              std::vector<float>& values,
                              unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);
    std::uniform_int_distribution<size_t> col_dist(0, num_cols - 1);
    std::bernoulli_distribution sparsity_dist(sparsity);
    
    row_offsets.clear();
    col_indices.clear();
    values.clear();
    
    size_t estimated_nnz = num_rows * num_cols * sparsity;
    col_indices.reserve(estimated_nnz);
    values.reserve(estimated_nnz);
    row_offsets.resize(num_rows + 1);
    size_t nnz = 0;
    row_offsets[0] = 0;
    
    for (size_t row = 0; row < num_rows; row++) {
        for (size_t col = 0; col < num_cols; col++) {
            if (sparsity_dist(gen)) {
                col_indices.push_back(col);
                values.push_back(val_dist(gen));
                nnz++;
            }
        }
        row_offsets[row + 1] = nnz;
    }
    
    // At least one element per row...
    for (size_t row = 0; row < num_rows; row++) {
        if (row_offsets[row] == row_offsets[row + 1]) {
            size_t col = col_dist(gen);
            
            col_indices.insert(col_indices.begin() + row_offsets[row], col);
            values.insert(values.begin() + row_offsets[row], val_dist(gen));
            for (size_t i = row + 1; i <= num_rows; i++) {
                row_offsets[i]++;
            }
        }
    }
}

// Generate a scale-free sparse matrix
void generate_powerlaw_csr(size_t num_rows, size_t num_cols, float sparsity,
                          std::vector<size_t>& row_offsets,
                          std::vector<int>& col_indices,
                          std::vector<float>& values,
                          unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);
    row_offsets.clear();
    col_indices.clear();
    values.clear();
    size_t estimated_nnz = num_rows * num_cols * sparsity;
    col_indices.reserve(estimated_nnz);
    values.reserve(estimated_nnz);
    row_offsets.resize(num_rows + 1);
    
    std::vector<float> column_weights(num_cols);
    for (size_t i = 0; i < num_cols; ++i) {
        column_weights[i] = 1.0f / std::pow(float(i + 1), 2.1f);
    }
    
    std::discrete_distribution<size_t> col_dist(column_weights.begin(), column_weights.end());
    
    size_t total_expected_edges = num_rows * num_cols * sparsity;
    size_t edges_per_row = total_expected_edges / num_rows;
    
    size_t nnz = 0;
    row_offsets[0] = 0;
    
    for (size_t row = 0; row < num_rows; row++) {
        std::set<size_t> row_cols; 
        
        while (row_cols.size() < edges_per_row && row_cols.size() < num_cols) {
            size_t col = col_dist(gen);
            row_cols.insert(col);
        }
        
        if (row_cols.empty() && num_cols > 0) {
            row_cols.insert(gen() % num_cols);
        }
        
        for (size_t col : row_cols) {
            col_indices.push_back(col);
            values.push_back(val_dist(gen));
            nnz++;
        }
        
        row_offsets[row + 1] = nnz;
    }
}

// Generate a diagonal-heavy sparse matrix 
void generate_diagonal_heavy_csr(size_t num_rows, size_t num_cols, float sparsity,
                              std::vector<size_t>& row_offsets,
                              std::vector<int>& col_indices,
                              std::vector<float>& values,
                              unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(0.1f, 10.0f);
    std::bernoulli_distribution diagonal_dist(0.9); 
    std::bernoulli_distribution band_dist(0.4);    
    std::bernoulli_distribution random_dist(0.05);  
    
    row_offsets.clear();
    col_indices.clear();
    values.clear();
    
    size_t bandwidth = 5;
    size_t estimated_nnz = num_rows * (2 * bandwidth + 1) * sparsity;
    col_indices.reserve(estimated_nnz);
    values.reserve(estimated_nnz);
    row_offsets.resize(num_rows + 1); 
    size_t nnz = 0;
    row_offsets[0] = 0;
    
    for (size_t row = 0; row < num_rows; row++) {
        if (row < num_cols && diagonal_dist(gen)) {
            col_indices.push_back(row);
            values.push_back(val_dist(gen) * 10.0f);
            nnz++;
        }     
        for (size_t k = 1; k <= bandwidth; k++) {
            if (row >= k && row - k < num_cols && band_dist(gen)) {
                col_indices.push_back(row - k);
                values.push_back(val_dist(gen));
                nnz++;
            }
            
            if (row + k < num_cols && band_dist(gen)) {
                col_indices.push_back(row + k);
                values.push_back(val_dist(gen));
                nnz++;
            }
        }
        
        for (size_t col = 0; col < num_cols; col++) {
            if (std::abs(static_cast<long>(row) - static_cast<long>(col)) > bandwidth && random_dist(gen)) {
                col_indices.push_back(col);
                values.push_back(val_dist(gen) * 0.5f);
                nnz++;
            }
        }
        
        size_t row_start = row_offsets[row];
        size_t row_size = nnz - row_start;
        if (row_size > 0) {
            std::vector<std::pair<size_t, float>> row_data;
            for (size_t i = 0; i < row_size; i++) {
                row_data.push_back({col_indices[row_start + i], values[row_start + i]});
            }
            
            std::sort(row_data.begin(), row_data.end());
            
            for (size_t i = 0; i < row_size; i++) {
                col_indices[row_start + i] = row_data[i].first;
                values[row_start + i] = row_data[i].second;
            }
        }
        if (row_offsets[row] == nnz && num_cols > 0) {
            size_t col = std::min(row, num_cols - 1);
            col_indices.push_back(col);
            values.push_back(val_dist(gen));
            nnz++;
        }
        
        row_offsets[row + 1] = nnz;
    }
}


    } // namespace utils
} // namespace graphblas_gpu