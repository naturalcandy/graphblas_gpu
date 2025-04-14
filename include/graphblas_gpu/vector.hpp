#ifndef GRAPHBLAS_GPU_VECTOR_HPP
#define GRAPHBLAS_GPU_VECTOR_HPP

#include <vector>
#include <cstddef>
#include <string>
#include <graphblas_gpu/op_sequence.hpp>
#include <graphblas_gpu/data_type.hpp>

namespace graphblas_gpu {

template <typename T>
class Vector {
public:
    using Value = T;

    // Initialization constructor
    Vector(size_t size, const std::vector<Value>& values)
        : size_(size),
          values_(values),
          datatype_(TypeToDataType<T>::value()),
          buffer_id_(OpSequence::getInstance().nextBufferId()) {

        OpSequence::getInstance().addOp({
            Op::Type::AllocVector,
            "AllocVector",
            {buffer_id_},
            {
                {"size", std::to_string(size_)},
                {"datatype", datatype_.toString()}
            }
        });
    }

    // Staging op constructor
    Vector(size_t size, size_t buffer_id)
        : size_(size),
          buffer_id_(buffer_id),
          datatype_(TypeToDataType<T>::value()) {
        // Note: Intentionally empty. NO AllocVector op here.
    }

    size_t bufferId() const {
        return buffer_id_;
    }

    size_t bytes() const {
        return size_ * datatype_.sizeInBytes();
    }

    size_t size() const {
        return size_;
    }

    const std::string& dataTypeName() const {
        static const std::string name = datatype_.toString();
        return name;
    }

    DataType dataType() const {
        return datatype_;
    }

    const Value* data() const {
        return values_.data();
    }

private:
    size_t size_;
    std::vector<Value> values_; // can be empty if vector is output of staged op
    size_t buffer_id_;
    DataType datatype_;
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_VECTOR_HPP