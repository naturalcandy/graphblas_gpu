#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/op_sequence.hpp>
#include <iostream>

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

    return 0;
}
