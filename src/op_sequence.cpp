#include <graphblas_gpu/op_sequence.hpp>

namespace graphblas_gpu {

OpSequence& OpSequence::getInstance() {
    static OpSequence instance;
    return instance;
}

void OpSequence::addOp(const Op& op) {
    ops_.push_back(op);
}

const std::vector<Op>& OpSequence::getOps() const {
    return ops_;
}

void OpSequence::clear() {
    ops_.clear();
}

} // namespace graphblas_gpu
