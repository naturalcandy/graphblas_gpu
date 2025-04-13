#ifndef GRAPHBLAS_GPU_SEMIRING_HPP
#define GRAPHBLAS_GPU_SEMIRING_HPP

namespace graphblas_gpu {

enum class SemiringType {
    Arithmetic,     // Standard arithmetic (+, *)
    LogicalOrAnd,   // Logical OR and AND semiring
    MinPlus         // Min-Plus semiring for shortest path
    
    // Probs should add more...
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_SEMIRING_HPP
