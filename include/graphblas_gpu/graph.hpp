#ifndef GRAPHBLAS_GPU_GRAPH_HPP
#define GRAPHBLAS_GPU_GRAPH_HPP

#include <vector>
#include <cstddef>
#include <variant>
#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/data_type.hpp>
#include <string>
#include <stdexcept>

namespace graphblas_gpu {


// We shoudl support multiply compressed format and build
// runtime classification layer based on sparsity pattern of input
      

template <typename T>
class SparseMatrix {
public:
    using Value = T;

    // CSR
    struct CSRData {
        std::vector<size_t> row_offsets;
        std::vector<int> col_indices;
        std::vector<Value> values;
    };

    // ELL 
    struct ELLData {
        size_t max_nnz_per_row;
        std::vector<int> col_indices;
        std::vector<Value> values;
    };

    // SELLC 
    struct SELLCData {
        size_t slice_size;
        std::vector<size_t> slice_ptrs;
        std::vector<size_t> slice_lengths;
        std::vector<int> col_indices;
        std::vector<Value> values;
    };

    using FormatData = std::variant<CSRData, ELLData, SELLCData>;
    using CSRFormat = CSRData;
    using ELLFormat = ELLData;
    using SELLCFormat = SELLCData;
    

    // CSR constructor
    SparseMatrix(size_t rows, size_t cols,
                 const std::vector<size_t>& row_offsets,
                 const std::vector<int>& col_indices,
                 const std::vector<Value>& values);
    
    // ELL initialization constructor
    SparseMatrix(size_t rows, size_t cols,
        size_t max_nnz_per_row,
        const std::vector<int>& ell_col_indices,
        const std::vector<Value>& ell_values);
    
    // SELLC initialization constructor
    SparseMatrix(size_t rows, size_t cols,
        size_t slice_size,
        const std::vector<size_t>& slice_ptrs,
        const std::vector<size_t>& slice_lengths,
        const std::vector<int>& sell_col_indices,
        const std::vector<Value>& sell_values);
    
    // Staging op constructor  (later on add a data format field as input here)
    SparseMatrix(size_t rows, size_t cols, size_t buffer_id);

    // Basic properties that work with all formats
    size_t numRows() const { return rows_; }
    size_t numCols() const { return cols_; }
    size_t bufferId() const { return buffer_id_; }
    const std::string& dataTypeName() const { return datatype_name_; }
    std::string format() const { return format_; }


    // Format Specific Properties
    size_t nnz() const;
    size_t bytes() const;
    DataType dataType() const { return datatype_; }

    const auto& get_format_data() const { return format_data_; }


private:
    size_t rows_, cols_;
    size_t buffer_id_;
    DataType datatype_;
    std::string format_;
    std::string datatype_name_;
    FormatData format_data_;

    size_t countNonPadding(const std::vector<int>& col_indices) const;
    std::string classifyOptimalFormat();
};

// CSR sparse matrix initialization
template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols,
                              const std::vector<size_t>& row_offsets,
                              const std::vector<int>& col_indices,
                              const std::vector<Value>& values)
    : rows_(rows), cols_(cols),
      buffer_id_(OpSequence::getInstance().nextBufferId()),
      datatype_(TypeToDataType<T>::value()),
      datatype_name_(datatype_.toString()),
      format_("CSR"),
      format_data_(CSRData{row_offsets, col_indices, values}) 
    {
    if (row_offsets.size() != rows + 1) {
        throw std::invalid_argument("CSR row_offsets size must be rows+1");
    }
    if (col_indices.size() != values.size()) {
        throw std::invalid_argument("CSR col_indices and values must be same size");
    }
    if (row_offsets.back() != col_indices.size()) {
        throw std::invalid_argument("CSR row_offsets last element must equal nnz");
    }
    const auto& csr_data = std::get<CSRData>(format_data_);
    OpSequence::getInstance().addOp({
        Op::Type::AllocGraph,
        "AllocGraph",
        {buffer_id_},
        {
            {"rows", std::to_string(rows_)},
            {"cols", std::to_string(cols_)},
            {"nnz", std::to_string(csr_data.values.size())},
            {"datatype", datatype_name_},
            {"format", format_}
        }
    });
}

// ELL sparse matrix initialization
template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols,
                            size_t max_nnz_per_row,
                            const std::vector<int>& ell_col_indices,
                            const std::vector<Value>& ell_values)
    : rows_(rows), cols_(cols),
      buffer_id_(OpSequence::getInstance().nextBufferId()),
      datatype_(TypeToDataType<T>::value()),
      datatype_name_(datatype_.toString()),
      format_("ELL"),
      format_data_(ELLData{max_nnz_per_row, ell_col_indices, ell_values})
{
    // Validate inputs
    if (ell_col_indices.size() != rows * max_nnz_per_row) {
        throw std::invalid_argument(
            "ELL column indices array size mismatch. Expected " + 
            std::to_string(rows * max_nnz_per_row) + " got " + 
            std::to_string(ell_col_indices.size()));
    }
    if (ell_values.size() != rows * max_nnz_per_row) {
        throw std::invalid_argument(
            "ELL values array size mismatch. Expected " + 
            std::to_string(rows * max_nnz_per_row) + " got " + 
            std::to_string(ell_values.size()));
    }
    
    size_t nnz_count = countNonPadding(ell_col_indices);
    
    OpSequence::getInstance().addOp({
        Op::Type::AllocGraph,
        "AllocGraph",
        {buffer_id_},
        {
            {"rows", std::to_string(rows_)},
            {"cols", std::to_string(cols_)},
            {"nnz", std::to_string(nnz_count)},
            {"datatype", datatype_name_},
            {"format", format_},
            {"max_nnz_per_row", std::to_string(max_nnz_per_row)}
        }
    });
}


// SELLC sparse matrix initialization
template <typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols,
                            size_t slice_size,
                            const std::vector<size_t>& slice_ptrs,
                            const std::vector<size_t>& slice_lengths,
                            const std::vector<int>& sell_col_indices,
                            const std::vector<Value>& sell_values)
    : rows_(rows), cols_(cols),
      buffer_id_(OpSequence::getInstance().nextBufferId()),
      datatype_(TypeToDataType<T>::value()),
      format_("SELLC"),
      datatype_name_(datatype_.toString()),
      format_data_(SELLCData{slice_size, slice_ptrs, slice_lengths, sell_col_indices, sell_values})
{
    // Validate input sizes
    size_t num_slices = (rows + slice_size - 1) / slice_size;
    if (slice_ptrs.size() != num_slices + 1) {
        throw std::invalid_argument(
            "SELLC slice_ptrs size mismatch. Expected " + 
            std::to_string(num_slices + 1) + " got " + 
            std::to_string(slice_ptrs.size()));
    }
    if (slice_lengths.size() != num_slices) {
        throw std::invalid_argument(
            "SELLC slice_lengths size mismatch. Expected " + 
            std::to_string(num_slices) + " got " + 
            std::to_string(slice_lengths.size()));
    }
    if (sell_col_indices.size() != slice_ptrs.back()) {
        throw std::invalid_argument(
            "SELLC col_indices size mismatch. Expected " + 
            std::to_string(slice_ptrs.back()) + " got " + 
            std::to_string(sell_col_indices.size()));
    }
    if (sell_values.size() != slice_ptrs.back()) {
        throw std::invalid_argument(
            "SELLC values size mismatch. Expected " + 
            std::to_string(slice_ptrs.back()) + " got " + 
            std::to_string(sell_values.size()));
    }
    
    size_t nnz_count = countNonPadding(sell_col_indices);
    
    OpSequence::getInstance().addOp({
        Op::Type::AllocGraph,
        "AllocGraph",
        {buffer_id_},
        {
            {"rows", std::to_string(rows_)},
            {"cols", std::to_string(cols_)},
            {"nnz", std::to_string(nnz_count)},
            {"datatype", datatype_name_},
            {"format", format_},
            {"slice_size", std::to_string(slice_size)},
            {"sellc_len", std::to_string(sell_col_indices.size())}
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
size_t SparseMatrix<T>::countNonPadding(const std::vector<int>& col_indices) const {
    size_t count = 0;
    for (const auto& col : col_indices) {
        if (col != -1) {
            count++;
        }
    }
    return count;
}

template <typename T>
size_t SparseMatrix<T>::nnz() const {
    if (format_ == "CSR") {
        const auto& data = std::get<CSRData>(format_data_);
        return data.values.size();
    } if (format_ == "ELL") {
        const auto& data = std::get<ELLData>(format_data_);
        return countNonPadding(data.col_indices);
    } else {
        const auto& data = std::get<SELLCData>(format_data_);
        return countNonPadding(data.col_indices);
    } 
}

template <typename T>
size_t SparseMatrix<T>::bytes() const {
    if (format_ == "CSR") {
        const auto& data = std::get<CSRData>(format_data_);
        return (rows_ + 1) * sizeof(size_t) +
               data.col_indices.size() * sizeof(int) +
               data.values.size() * datatype_.sizeInBytes();
    }
    else if (format_ == "ELL") {
        const auto& data = std::get<ELLData>(format_data_);
        return data.col_indices.size() * sizeof(int) +
               data.values.size() * datatype_.sizeInBytes();
    }
    else if (format_ == "SELLC") {
        const auto& data = std::get<SELLCData>(format_data_);
        return data.slice_ptrs.size() * sizeof(size_t) +
               data.slice_lengths.size() * sizeof(size_t) +
               data.col_indices.size() * sizeof(int) +
               data.values.size() * datatype_.sizeInBytes();
    }
    throw std::runtime_error("Unknown matrix format");
}

template <typename T>
std::string SparseMatrix<T>::classifyOptimalFormat() {
    if (format_ != "CSR") {
        return format_; // Only analyze CSR matrices
    }

    const auto& csr_data = std::get<CSRData>(format_data_);
    const std::vector<size_t>& row_offsets = csr_data.row_offsets;
    size_t n_rows = this->numRows();

    if (n_rows == 0) {
        return "CSR"; // Empty matrix, stay CSR
    }

    double sum = 0.0;
    for (size_t i = 0; i < n_rows; ++i) {
        sum += static_cast<double>(row_offsets[i+1] - row_offsets[i]);
    }
    double mean_nnz = sum / n_rows;

    double variance = 0.0;
    size_t max_row_nnz = 0;
    size_t empty_rows = 0;
    for (size_t i = 0; i < n_rows; ++i) {
        size_t row_nnz = row_offsets[i+1] - row_offsets[i];
        variance += (static_cast<double>(row_nnz) - mean_nnz) * (static_cast<double>(row_nnz) - mean_nnz);
        max_row_nnz = std::max(max_row_nnz, row_nnz);
        if (row_nnz == 0) {
            empty_rows++;
        }
    }
    variance /= n_rows;
    double normalized_variance = variance / mean_nnz;

    if (normalized_variance < 0.2) {
        if (max_row_nnz > 2 * mean_nnz || empty_rows > 0) {
            return "SELL-C"; // Small variance, but some heavy or empty rows
        } else {
            return "ELL"; // Uniform
        }
    } else if (normalized_variance <= 1.0) {
        return "SELL-C"; // Medium vvariance
    } else {
        return "CSR"; // High variance
    }
}

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_GRAPH_HPP