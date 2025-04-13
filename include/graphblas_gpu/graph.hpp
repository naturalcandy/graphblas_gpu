#ifndef GRAPHBLAS_GPU_GRAPH_HPP
#define GRAPHBLAS_GPU_GRAPH_HPP

#include <vector>
#include <cstddef>
#include <graphblas_gpu/op_sequence.hpp>
#include <typeinfo>

namespace graphblas_gpu {

template <typename T>
class SparseMatrix {
public:
    using Value = T;

    // Initialization constructor
    SparseMatrix(size_t rows, size_t cols,
                 const std::vector<size_t>& row_offsets,
                 const std::vector<size_t>& col_indices,
                 const std::vector<Value>& values);
    
    // Staging op constructor
    SparseMatrix(size_t rows, size_t cols, size_t buffer_id);

    size_t numRows() const { return rows_; }
    size_t numCols() const { return cols_; }

    size_t bufferId() const;
    size_t bytes() const;
    const std::string& dataTypeName() const;

private:
    size_t rows_, cols_;
    std::vector<size_t> row_offsets_, col_indices_;
    std::vector<Value> values_;
    size_t buffer_id_;
    std::string datatype_name_;
};

// To do: add more user friendly datatype names.

template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols,
                              const std::vector<size_t>& row_offsets,
                              const std::vector<size_t>& col_indices,
                              const std::vector<Value>& values)
    : rows_(rows), cols_(cols),
      row_offsets_(row_offsets),
      col_indices_(col_indices),
      values_(values),
      datatype_name_(typeid(Value).name()),
      buffer_id_(OpSequence::getInstance().nextBufferId()) {

    OpSequence::getInstance().addOp({
        Op::Type::AllocGraph,
        "AllocGraph",
        {buffer_id_},
        {
            {"rows", std::to_string(rows_)},
            {"cols", std::to_string(cols_)},
            {"nnz", std::to_string(values_.size())},
            {"datatype", datatype_name_}
        }
    });
}

template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols, size_t buffer_id)
    : rows_(rows), cols_(cols),
      buffer_id_(buffer_id),
      datatype_name_(typeid(Value).name()) { }


template <typename T>
size_t SparseMatrix<T>::bufferId() const {
    return buffer_id_;
}

template <typename T>
size_t SparseMatrix<T>::bytes() const {
    return values_.size() * sizeof(Value);
}

template <typename T>
const std::string& SparseMatrix<T>::dataTypeName() const {
    return datatype_name_;
}

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_GRAPH_HPP
