#ifndef GRAPHBLAS_GPU_DATA_TYPE_HPP
#define GRAPHBLAS_GPU_DATA_TYPE_HPP

#include <string>
#include <stdexcept>
#include <unordered_map>
#include <cstdint> 

namespace graphblas_gpu {

enum class DataTypeEnum {
    Float,
    Double,
    Int32,
    Int64,
    Bool,
    Unknown
};

class DataType {
public:
    // Default is float
    DataType() : type_(DataTypeEnum::Float) {}
    
    DataType(DataTypeEnum type) : type_(type) {}
    
    // Get type as string
    std::string toString() const {
        switch (type_) {
            case DataTypeEnum::Float: return "float";
            case DataTypeEnum::Double: return "double";
            case DataTypeEnum::Int32: return "int32";
            case DataTypeEnum::Int64: return "int64";
            case DataTypeEnum::Bool: return "bool";
            default: return "unknown";
        }
    }
    
    // Get size in bytes
    size_t sizeInBytes() const {
        switch (type_) {
            case DataTypeEnum::Float: return sizeof(float);
            case DataTypeEnum::Double: return sizeof(double);
            case DataTypeEnum::Int32: return sizeof(int32_t);
            case DataTypeEnum::Int64: return sizeof(int64_t);
            case DataTypeEnum::Bool: return sizeof(bool);
            default: throw std::runtime_error("Unknown data type");
        }
    }
    
    // Equality operator
    bool operator==(const DataType& other) const {
        return type_ == other.type_;
    }
    
    // Get the enum value
    DataTypeEnum value() const {
        return type_;
    }
    
private:
    DataTypeEnum type_;
};

template <typename T>
struct TypeToDataType {
    static DataType value() { return DataType(DataTypeEnum::Unknown); }
};

template <>
struct TypeToDataType<float> {
    static DataType value() { return DataType(DataTypeEnum::Float); }
};

template <>
struct TypeToDataType<double> {
    static DataType value() { return DataType(DataTypeEnum::Double); }
};

template <>
struct TypeToDataType<int> {
    static DataType value() { return DataType(DataTypeEnum::Int32); }
};

template <>
struct TypeToDataType<long long> {
    static DataType value() { return DataType(DataTypeEnum::Int64); }
};

template <>
struct TypeToDataType<bool> {
    static DataType value() { return DataType(DataTypeEnum::Bool); }
};

inline DataType dataTypeFromString(const std::string& name) {
    if (name == "float") return DataType(DataTypeEnum::Float);
    if (name == "double") return DataType(DataTypeEnum::Double);
    if (name == "int32" || name == "int") return DataType(DataTypeEnum::Int32);
    if (name == "int64" || name == "long") return DataType(DataTypeEnum::Int64);
    if (name == "bool") return DataType(DataTypeEnum::Bool);
    
    // Float as default
    return DataType(DataTypeEnum::Float);
}

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_DATA_TYPE_HPP