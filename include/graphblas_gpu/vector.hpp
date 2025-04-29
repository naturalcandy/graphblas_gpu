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

    // Staged op constructor
    Vector(size_t size, size_t buffer_id)
        : size_(size),
          buffer_id_(buffer_id),
          datatype_(TypeToDataType<T>::value()) {
    }

    // Empty vector constructor
    Vector(size_t size)
        : size_(size),
        values_(),  
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

    static void copy(const Vector<T>& src, Vector<T>& dst) {
        if (src.size() != dst.size()) {
            throw std::invalid_argument("Source and destination vectors must have the same size");
        }
    
        OpSequence::getInstance().addOp({
            Op::Type::Copy,
            "Copy",
            {dst.bufferId(), src.bufferId()},
            {
                {"datatype", src.dataType().toString()},
                {"size", std::to_string(src.size())}
            }
        });
    }
    
private:
    size_t size_;
    // Can be empty if vector is output of staged op
    std::vector<Value> values_; 
    size_t buffer_id_;
    DataType datatype_;
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_VECTOR_HPP