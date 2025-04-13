#ifndef GRAPHBLAS_GPU_VECTOR_HPP
#define GRAPHBLAS_GPU_VECTOR_HPP

#include <vector>
#include <cstddef>
#include <string>
#include <graphblas_gpu/op_sequence.hpp>
#include <typeinfo>


namespace graphblas_gpu {

template <typename T>
struct TypeName { static constexpr const char* value = "unknown"; };

template <>
struct TypeName<double> { static constexpr const char* value = "double"; };
template <>
struct TypeName<float> { static constexpr const char* value = "float"; };
template <>
struct TypeName<int> { static constexpr const char* value = "int"; };

// We can add more types later if needed


template <typename T>
class Vector {
public:
    using Value = T;

    // Initialization constructor
    Vector(size_t size, const std::vector<Value>& values);

    // Staging op constructor
    Vector(size_t size, size_t buffer_id);

    size_t bufferId() const;
    size_t bytes() const;
    size_t size() const;
    const std::string& dataTypeName() const;
    const Value* data() const;

private:
    size_t size_;
    std::vector<Value> values_; // can be empty if vector is output of staged op
    size_t buffer_id_;
    std::string datatype_name_;
};

////////////////////////////////////////////////////////////////////////////////

template <typename T>
Vector<T>::Vector(size_t size, const std::vector<Value>& values)
    : size_(size),
      values_(values),
      datatype_name_(typeid(Value).name()),
      buffer_id_(OpSequence::getInstance().nextBufferId()) {

    OpSequence::getInstance().addOp({
        Op::Type::AllocVector,
        "AllocVector",
        {buffer_id_},
        {
            {"size", std::to_string(size_)},
            {"datatype", datatype_name_}
        }
    });
}

template <typename T>
Vector<T>::Vector(size_t size, size_t buffer_id)
    : size_(size),
      buffer_id_(buffer_id),
      datatype_name_(TypeName<Value>::value) {
    // Note: Intentionally empty. NO AllocVector op here.
}


template <typename T>
size_t Vector<T>::bufferId() const {
    return buffer_id_;
}

template <typename T>
size_t Vector<T>::bytes() const {
    return values_.size() * sizeof(Value);
}

template <typename T>
size_t Vector<T>::size() const {
    return size_;
}

template <typename T>
const std::string& Vector<T>::dataTypeName() const {
    return datatype_name_;
}

template <typename T>
const typename Vector<T>::Value* Vector<T>::data() const {
    return values_.data();
}

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_VECTOR_HPP
