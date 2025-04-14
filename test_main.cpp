#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/spmv_csr.hpp>
#include <iostream>
#include <cassert>


int test_mat_to_csr(){
    double m[] = {2, 0, -1, 3, 0, 0, 5, 0, 0, 4, 0, 0, 1, 0, -5, 6, 7, 0, 2, 0, 0, 0, 3, 0, 9};
    int num_rows = 5;
    int num_cols = 5;
    int* csr_row_offsets;
    int* csr_cols;
    double* csr_vals;
    int nnz = 0;

    int correct_row_offsets[] = {0, 3, 5, 7, 10, 12};
    int correct_cols[] = {0, 2, 3, 1, 4, 2, 4, 0, 1, 3, 2, 4};
    double correct_vals[] = {2, -1, 3, 5, 4, 1, -5, 6, 7, 2, 3, 9};
    graphblas_gpu::kernels::matrix_to_csr(m, num_rows, num_cols, &csr_row_offsets, &csr_cols, &csr_vals, &nnz, 512);
    // std::cout << "Row offsets: \n";
    for (int i = 0; i < num_rows + 1; ++i) {
        // std::cout << csr_row_offsets[i] << " ";
        if (csr_row_offsets[i] != correct_row_offsets[i]) {
            std::cout << "\nRow offsets do not match at index " << i << "\n";
            return 0;
        }
    }

    // std::cout << "\nCols: ";
    for (int i = 0; i < nnz; ++i) {
        // std::cout << csr_cols[i] << " ";
        if (csr_cols[i] != correct_cols[i]) {
            std::cout << "\nCols do not match at index " << i << "\n";
            return 0;
        }
    }

    // std::cout << "\nVals: ";
    for (int i = 0; i < nnz; ++i) {
        // std::cout << csr_vals[i] << " ";
        if (csr_vals[i] != correct_vals[i]) {
            std::cout << "\nVals do not match at index " << i << "\n";
            return 0;
        }
    }
    return 1;
}

int main() {
    using namespace graphblas_gpu;

    size_t rows = 2, cols = 3;
    std::vector<size_t> row_offsets = {0, 2, 3};
    std::vector<size_t> col_indices = {0, 2, 1};

    std::vector<double> values = {10.0, 20.0, 30.0};

    SparseMatrix<double> matrix(rows, cols, row_offsets, col_indices, values);

    size_t rows2 = 5, cols2 = 3;
    std::vector<size_t> row_offsets2 = {0, 2, 3};
    std::vector<size_t> col_indices2 = {0, 2, 1};

    std::vector<double> values2 = {1, 2, 3};
    SparseMatrix<double> matrix2(rows2, cols2, row_offsets2, col_indices2, values2);

    std::cout << "SparseMatrix Buffer ID: " << matrix.bufferId() << "\n";
    std::cout << "SparseMatrix Buffer ID 2: " << matrix2.bufferId() << "\n";

    std::cout << "Memory used (bytes): " << matrix.bytes() << "\n";
    std::cout << "Data type: " << matrix.dataTypeName() << "\n\n";

    const auto& ops = OpSequence::getInstance().getOps();
    for (const auto& op : ops) {
        std::cout << "Op Name: " << op.name << "\n";
        std::cout << "Buffer IDs: ";
        for (auto id : op.buffer_ids)
            std::cout << id << " ";
        std::cout << "\nOp Args:\n";
        for (const auto& arg : op.args) {
            std::cout << "  " << arg.first << ": " << arg.second << "\n";
        }
        std::cout << "------\n";
    }

    assert(test_mat_to_csr());
    
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
