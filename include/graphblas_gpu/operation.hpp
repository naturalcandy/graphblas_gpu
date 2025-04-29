#ifndef GRAPHBLAS_GPU_OPERATIONS_HPP
#define GRAPHBLAS_GPU_OPERATIONS_HPP

#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/semiring.hpp>
#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/graph.hpp>
#include <string>

namespace graphblas_gpu {

template<typename T>
class Operations {
public:
    static Vector<T> add(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        size_t result_size = lhs.size();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseAdd,
            "EWiseAdd",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()}, 
                {"size", std::to_string(result_size)}
            }
        });

        return Vector<T>(result_size, buffer_id);
    }

    static Vector<T> sub(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        size_t result_size = lhs.size();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseSub,
            "EWiseSub",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()}, 
                {"size", std::to_string(result_size)}
            
            }
        });

        return Vector<T>(result_size, buffer_id);
    }

    static Vector<T> mul(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        size_t result_size = lhs.size();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseMul,
            "EWiseMul",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()}, 
                {"size", std::to_string(result_size)}
            
            }
        });

        return Vector<T>(result_size, buffer_id);
    }

    static Vector<T> div(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        size_t result_size = lhs.size();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseDiv,
            "EWiseDiv",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()},
                {"size", std::to_string(result_size)}
            }
        });

        return Vector<T>(result_size, buffer_id);
    }

    static Vector<T> lor(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t id = OpSequence::getInstance().nextBufferId();
        size_t n  = lhs.size();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseOr, "EWiseOr",
            {id, lhs.bufferId(), rhs.bufferId()},
            { {"datatype", lhs.dataType().toString()},
              {"size",     std::to_string(n)} }
        });
        return Vector<T>(n, id);
    }


    static void add(Vector<T>& lhs, const Vector<T>& rhs) {
        if (lhs.size() != rhs.size()) {
            throw std::invalid_argument("Vectors must have the same size for in-place addition");
        }

        OpSequence::getInstance().addOp({
            Op::Type::EWiseAddInPlace,
            "EWiseAddInPlace",
            {lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()}, 
                {"size", std::to_string(lhs.size())},
                {"in_place", "true"}
            }
        });
    }

    static void sub(Vector<T>& lhs, const Vector<T>& rhs) {
        if (lhs.size() != rhs.size()) {
            throw std::invalid_argument("Vectors must have the same size for in-place subtraction");
        }

        OpSequence::getInstance().addOp({
            Op::Type::EWiseSubInPlace,
            "EWiseSubInPlace",
            {lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()}, 
                {"size", std::to_string(lhs.size())},
                {"in_place", "true"}
            }
        });
    }

    static void mul(Vector<T>& lhs, const Vector<T>& rhs) {
        if (lhs.size() != rhs.size()) {
            throw std::invalid_argument("Vectors must have the same size for in-place multiplication");
        }

        OpSequence::getInstance().addOp({
            Op::Type::EWiseMulInPlace,
            "EWiseMulInPlace",
            {lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()}, 
                {"size", std::to_string(lhs.size())},
                {"in_place", "true"}
            }
        });
    }

    static void div(Vector<T>& lhs, const Vector<T>& rhs) {
        if (lhs.size() != rhs.size()) {
            throw std::invalid_argument("Vectors must have the same size for in-place division");
        }

        OpSequence::getInstance().addOp({
            Op::Type::EWiseDivInPlace,
            "EWiseDivInPlace",
            {lhs.bufferId(), rhs.bufferId()},
            { 
                {"datatype", lhs.dataType().toString()}, 
                {"size", std::to_string(lhs.size())},
                {"in_place", "true"}
            }
        });
    }

    static void lor(Vector<T>& dst, const Vector<T>& src) {
        OpSequence::getInstance().addOp({
            Op::Type::EWiseOrInPlace, "EWiseOrInPlace",
            {dst.bufferId(), src.bufferId()},
            { {"datatype", dst.dataType().toString()},
              {"size",     std::to_string(dst.size())},
              {"in_place", "true"} }
        });
    }


    // allocate new result vector
    static Vector<T> spmv(const SparseMatrix<T>& mat,
                          const Vector<T>& vec,
                          SemiringType semiring,
                          const Vector<T>* mask = nullptr) {

        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        std::vector<size_t> buffer_ids = {buffer_id, mat.bufferId(), vec.bufferId()};
        if (mask) buffer_ids.push_back(mask->bufferId());

        std::unordered_map<std::string, std::string> args = {
            {"semiring", std::to_string(static_cast<int>(semiring))},
            {"datatype", vec.dataType().toString()},
            {"mask", mask ? "true" : "false"},
            {"format", mat.format()},
            {"num_rows", std::to_string(mat.numRows())},
            {"nnz", std::to_string(mat.nnz())}
        };

        const auto& format_data = mat.get_format_data();

        if (mat.format() == "ELL") {
            const auto& ell_data = std::get<typename SparseMatrix<T>::ELLFormat>(format_data);
            args["max_nnz_per_row"] = std::to_string(ell_data.max_nnz_per_row);
        } 
        else if (mat.format() == "SELLC") {
            const auto& sellc_data = std::get<typename SparseMatrix<T>::SELLCFormat>(format_data);
            args["slice_size"] = std::to_string(sellc_data.slice_size);
            args["sellc_len"]  = std::to_string(sellc_data.col_indices.size());
        }

        OpSequence::getInstance().addOp({
            Op::Type::SpMV,
            "SpMV",
            buffer_ids,
            args
        });

        return Vector<T>(mat.numRows(), buffer_id);
    }


    // in-place
    static void spmv(const SparseMatrix<T>& mat,
                 const Vector<T>& vec,
                 Vector<T>& result,
                 SemiringType semiring,
                 const Vector<T>* mask = nullptr) {
        if (result.size() != mat.numRows()) {
            throw std::invalid_argument("Result vector must have same number of rows as matrix");
        }

        std::vector<size_t> buffer_ids = {result.bufferId(), mat.bufferId(), vec.bufferId()};
        if (mask) buffer_ids.push_back(mask->bufferId());

        std::unordered_map<std::string, std::string> args = {
            {"semiring", std::to_string(static_cast<int>(semiring))},
            {"datatype", vec.dataType().toString()},
            {"mask", mask ? "true" : "false"},
            {"format", mat.format()},
            {"num_rows", std::to_string(mat.numRows())},
            {"nnz", std::to_string(mat.nnz())},
            {"in_place", "true"}
        };

        const auto& format_data = mat.get_format_data();

        if (mat.format() == "ELL") {
            const auto& ell_data = std::get<typename SparseMatrix<T>::ELLFormat>(format_data);
            args["max_nnz_per_row"] = std::to_string(ell_data.max_nnz_per_row);
        } 
        else if (mat.format() == "SELLC") {
            const auto& sellc_data = std::get<typename SparseMatrix<T>::SELLCFormat>(format_data);
            args["slice_size"] = std::to_string(sellc_data.slice_size);
            args["sellc_len"] = std::to_string(sellc_data.col_indices.size());
        }

        OpSequence::getInstance().addOp({
            Op::Type::SpMV,
            "SpMV",
            buffer_ids,
            args
        });
    }


    static SparseMatrix<T> spmm(const SparseMatrix<T>& A,
                                const SparseMatrix<T>& B,
                                SemiringType semiring,
                                const SparseMatrix<T>* mask = nullptr) {

        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        std::vector<size_t> buffer_ids = {buffer_id, A.bufferId(), B.bufferId()};
        if (mask) buffer_ids.push_back(mask->bufferId());

        OpSequence::getInstance().addOp({
            Op::Type::SpMM,
            "SpMM",
            buffer_ids,
            {
                {"semiring", std::to_string(static_cast<int>(semiring))},
                {"datatype", A.dataType().toString()},
                {"mask", mask ? "true" : "false"}
            }
        });

        return SparseMatrix<T>(A.numRows(), B.numCols(), buffer_id);
    }
};

} // namespace graphblas_gpu

// Operator overloads
template<typename T>
graphblas_gpu::Vector<T> operator+(const graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    return graphblas_gpu::Operations<T>::add(lhs, rhs);
}

template<typename T>
graphblas_gpu::Vector<T> operator-(const graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    return graphblas_gpu::Operations<T>::sub(lhs, rhs);
}

template<typename T>
graphblas_gpu::Vector<T> operator*(const graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    return graphblas_gpu::Operations<T>::mul(lhs, rhs);
}

template<typename T>
graphblas_gpu::Vector<T> operator/(const graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    return graphblas_gpu::Operations<T>::div(lhs, rhs);
}

template<typename T>
graphblas_gpu::Vector<T>& operator+=(graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    graphblas_gpu::Operations<T>::add(lhs, rhs);
    return lhs;
}

template<typename T>
graphblas_gpu::Vector<T>& operator-=(graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    graphblas_gpu::Operations<T>::sub(lhs, rhs);
    return lhs;
}

template<typename T>
graphblas_gpu::Vector<T>& operator*=(graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    graphblas_gpu::Operations<T>::mul(lhs, rhs);
    return lhs;
}

template<typename T>
graphblas_gpu::Vector<T>& operator/=(graphblas_gpu::Vector<T>& lhs, const graphblas_gpu::Vector<T>& rhs) {
    graphblas_gpu::Operations<T>::div(lhs, rhs);
    return lhs;
}

template<typename T>
graphblas_gpu::Vector<T>  operator|(const graphblas_gpu::Vector<T>& a,
                                    const graphblas_gpu::Vector<T>& b)
{ return graphblas_gpu::Operations<T>::lor(a,b); }

template<typename T>
graphblas_gpu::Vector<T>& operator|=(graphblas_gpu::Vector<T>& a,
                                     const graphblas_gpu::Vector<T>& b)
{ graphblas_gpu::Operations<T>::lor(a,b); return a; }

#endif // GRAPHBLAS_GPU_OPERATIONS_HPP