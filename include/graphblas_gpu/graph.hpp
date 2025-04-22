#ifndef GRAPHBLAS_GPU_GRAPH_HPP
#define GRAPHBLAS_GPU_GRAPH_HPP

#include <vector>
#include <cstddef>
#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/data_type.hpp>
#include <string>

namespace graphblas_gpu {


// We shoudl support multiply compressed format and build
// runtime classification layer based on sparsity pattern of input
      

template <typename T>
class SparseMatrix {
public:
    using Value = T;

    // Initialization constructor
    SparseMatrix(size_t rows, size_t cols,
                 const std::vector<size_t>& row_offsets,
                 const std::vector<size_t>& col_indices,
                 const std::vector<Value>& values);
    
    // Staging op constructor  (later on add a data format field as input here)
    SparseMatrix(size_t rows, size_t cols, size_t buffer_id);

    size_t numRows() const { return rows_; }
    size_t numCols() const { return cols_; }
    size_t nnz() const { return values_.size(); }

    size_t bufferId() const;
    size_t bytes() const;
    const std::string& dataTypeName() const;
    DataType dataType() const;
    std::string format() const { return format_; }
    

private:
    size_t rows_, cols_;
    std::vector<size_t> row_offsets_, col_indices_;
    std::vector<Value> values_;
    size_t buffer_id_;
    DataType datatype_;
    std::string format_;
};

// CSR sparse matrix initialization
template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols,
                              const std::vector<size_t>& row_offsets,
                              const std::vector<size_t>& col_indices,
                              const std::vector<Value>& values)
    : rows_(rows), cols_(cols),
      row_offsets_(row_offsets),
      col_indices_(col_indices),
      values_(values),
      datatype_(TypeToDataType<T>::value()),
      buffer_id_(OpSequence::getInstance().nextBufferId()),
      format_("CSR") 
    {

    OpSequence::getInstance().addOp({
        Op::Type::AllocGraph,
        "AllocGraph",
        {buffer_id_},
        {
            {"rows", std::to_string(rows_)},
            {"cols", std::to_string(cols_)},
            {"nnz", std::to_string(values_.size())},
            {"datatype", datatype_.toString()},
            {"format", format_}
        }
    });
}

// Initialize sparse matrix that are products of intermediate computation.
template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols, size_t buffer_id)
    : rows_(rows), cols_(cols),
      buffer_id_(buffer_id),
      datatype_(TypeToDataType<T>::value()) { }


template <typename T>
size_t SparseMatrix<T>::bufferId() const {
    return buffer_id_;
}

template <typename T>
size_t SparseMatrix<T>::bytes() const {
    // we should case on the format of matrix here..
    // for now we will assume input is a CSR matrix
    if (format_ == "CSR") {
        return (rows_ + 1) * sizeof(size_t) +    // row_offsets
           values_.size() * sizeof(size_t) +  // col_indices
           values_.size() * datatype_.sizeInBytes(); // values
    }
    return 0;
}

template <typename T>
const std::string& SparseMatrix<T>::dataTypeName() const {
    static const std::string name = datatype_.toString();
    return name;
}

template <typename T>
DataType SparseMatrix<T>::dataType() const {
    return datatype_;
}

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_GRAPH_HPP