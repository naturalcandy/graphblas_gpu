#ifndef GRAPHBLAS_GPU_SEMIRING_HPP
#define GRAPHBLAS_GPU_SEMIRING_HPP

namespace graphblas_gpu {

enum class SemiringType {
    Arithmetic,     // Standard arithmetic (+, *)
    LogicalOrAnd   // Logical OR and AND semiring
    
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_SEMIRING_HPP
