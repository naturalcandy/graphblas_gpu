#ifndef GRAPHBLAS_GPU_OP_SEQUENCE_HPP
#define GRAPHBLAS_GPU_OP_SEQUENCE_HPP

#include <vector>
#include <string>
#include <unordered_map>

namespace graphblas_gpu {

struct Op {
    enum class Type {
        AllocGraph,
        AllocVector,
        SpMV,
        SpMM,
    };

    Type type;
    std::string name;               // Explicit operation name for readability
    std::vector<size_t> buffer_ids; // All associated buffer IDs clearly
    std::unordered_map<std::string, std::string> args;
};

class OpSequence {
public:
    static OpSequence& getInstance();

    void addOp(const Op& op);
    const std::vector<Op>& getOps() const;
    void clear();

private:
    OpSequence() = default;

    std::vector<Op> ops_;
};

} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_OP_SEQUENCE_HPP
