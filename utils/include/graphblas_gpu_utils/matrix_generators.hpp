#pragma once

#include <vector>
#include <cstddef>

namespace graphblas_gpu {
namespace utils {

// Generate a uniform random sparse matrix in CSR format
void generate_uniform_random_csr(
    size_t num_rows, 
    size_t num_cols, 
    float sparsity,
    std::vector<size_t>& row_offsets,
    std::vector<int>& col_indices,
    std::vector<float>& values,
    unsigned int seed = 42);

// Generate a power-law distribution sparse matrix in CSR format
void generate_powerlaw_csr(
    size_t num_rows,
    size_t num_cols, 
    float sparsity,
    std::vector<size_t>& row_offsets,
    std::vector<int>& col_indices,
    std::vector<float>& values,
    unsigned int seed = 42);

// Generate a diagonal-heavy sparse matrix in CSR format
void generate_diagonal_heavy_csr(
    size_t num_rows,
    size_t num_cols, 
    float sparsity,
    std::vector<size_t>& row_offsets,
    std::vector<int>& col_indices,
    std::vector<float>& values,
    unsigned int seed = 42);

} // namespace utils
} // namespace graphblas_gpu