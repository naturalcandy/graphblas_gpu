#include <graphblas_gpu/termination_condition.hpp>

namespace graphblas_gpu {

TerminationCondition& TerminationCondition::getInstance() {
    static TerminationCondition instance;
    return instance;
}

TerminationCondition::TerminationCondition() 
    : is_active_(false), condition_code_("") {
}

void TerminationCondition::reset() {
    is_active_ = false;
    require_all_threads_ = false;        
    condition_code_.clear();
}

void TerminationCondition::setFixedIterations() {
    is_active_ = false;
    condition_code_ = "";
}

bool TerminationCondition::isActive() const {
    return is_active_;
}

std::string TerminationCondition::getConditionCode() const {
    return condition_code_;
}

} // namespace graphblas_gpu