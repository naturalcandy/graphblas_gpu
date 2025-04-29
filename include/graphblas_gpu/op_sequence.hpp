#ifndef GRAPHBLAS_GPU_OP_SEQUENCE_HPP
#define GRAPHBLAS_GPU_OP_SEQUENCE_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <cstddef>

namespace graphblas_gpu {

struct Op {
    enum class Type {
        AllocGraph,
        AllocVector,
        SpMV,
        SpMM,
        EWiseAdd,
        EWiseSub,
        EWiseMul,
        EWiseDiv,
        EWiseOr, 
        EWiseAddInPlace,
        EWiseSubInPlace,
        EWiseMulInPlace,
        EWiseDivInPlace,
        EWiseOrInPlace,
        Copy
    };

    Type type;
    std::string name;
    std::vector<size_t> buffer_ids;
    std::unordered_map<std::string, std::string> args;
};

class OpSequence {
public:
    static OpSequence& getInstance();

    void addOp(const Op& op);
    const std::vector<Op>& getOps() const;
    void clear();

    size_t nextBufferId();

private:
    OpSequence() = default;

    size_t buffer_count_ = 0; 
    std::vector<Op> ops_;
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_OP_SEQUENCE_HPP
