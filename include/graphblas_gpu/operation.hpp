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
        OpSequence::getInstance().addOp({
            Op::Type::EWiseAdd,
            "EWiseAdd",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { {"datatype", lhs.dataType().toString()} }
        });

        return Vector<T>(lhs.size(), buffer_id);
    }

    static Vector<T> sub(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseSub,
            "EWiseSub",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { {"datatype", lhs.dataType().toString()} }
        });

        return Vector<T>(lhs.size(), buffer_id);
    }

    static Vector<T> mul(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseMul,
            "EWiseMul",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { {"datatype", lhs.dataType().toString()} }
        });

        return Vector<T>(lhs.size(), buffer_id);
    }

    static Vector<T> div(const Vector<T>& lhs, const Vector<T>& rhs) {
        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        OpSequence::getInstance().addOp({
            Op::Type::EWiseDiv,
            "EWiseDiv",
            {buffer_id, lhs.bufferId(), rhs.bufferId()},
            { {"datatype", lhs.dataType().toString()} }
        });

        return Vector<T>(lhs.size(), buffer_id);
    }

    static Vector<T> spmv(const SparseMatrix<T>& mat,
                          const Vector<T>& vec,
                          SemiringType semiring,
                          const Vector<T>* mask = nullptr) {

        size_t buffer_id = OpSequence::getInstance().nextBufferId();
        std::vector<size_t> buffer_ids = {buffer_id, mat.bufferId(), vec.bufferId()};
        if (mask) buffer_ids.push_back(mask->bufferId());

        OpSequence::getInstance().addOp({
            Op::Type::SpMV,
            "SpMV",
            buffer_ids,
            {
                {"semiring", std::to_string(static_cast<int>(semiring))},
                {"datatype", vec.dataType().toString()},
                {"mask", mask ? "true" : "false"}
            }
        });

        return Vector<T>(mat.numRows(), buffer_id);
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

#endif // GRAPHBLAS_GPU_OPERATIONS_HPP