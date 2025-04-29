#ifndef GRAPHBLAS_GPU_TERMINATION_CONDITION_HPP
#define GRAPHBLAS_GPU_TERMINATION_CONDITION_HPP

#include <string>
#include <sstream>
#include <memory>
#include <graphblas_gpu/vector.hpp>

namespace graphblas_gpu {

class TerminationCondition {
public:
    static TerminationCondition& getInstance();
    
    // Reset to default state
    void reset();
    
    // Set fixed iterations (default)
    void setFixedIterations();
    
    // Check if target node is reached in vector (for BFS)
    template<typename T>
    void setNodeReached(size_t target_node, const Vector<T>& frontier);
    
    // Check if frontier is unchanged (for convergence detection)
    template<typename T>
    void setFrontierUnchanged(const Vector<T>& current, const Vector<T>& previous);
    
    // setFixedIteration || setNodeReached
    template<typename T>
    void setBfsComplete(size_t target_node, const Vector<T>& frontier, 
                        const Vector<T>& previous);
    
    // Check if condition is active
    bool isActive() const;
    
    // Get condition code for kernel generator
    std::string getConditionCode() const;

    bool requireAllThreads() const;
    
private:
    TerminationCondition();
    
    bool is_active_;
    bool require_all_threads_ = false;
    std::string condition_code_;
};

template<typename T>
void TerminationCondition::setNodeReached(size_t target_node, const Vector<T>& frontier) {
    is_active_ = true;
    require_all_threads_ = false;  

    std::stringstream ss;
    ss << "bool should_terminate = false;\n"
       << frontier.dataTypeName() << "* frontier_vec = (" 
       << frontier.dataTypeName() << "*)(BUF" << frontier.bufferId() << ");\n"
       << "if (idx == " << target_node << " && frontier_vec[" << target_node << "] != 0) {\n"
       << "    should_terminate = true;\n"
       << "}\n";

    condition_code_ = ss.str();
}



template<typename T>
void TerminationCondition::setFrontierUnchanged(const Vector<T>& current, const Vector<T>& previous) {
    if (current.size() != previous.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    is_active_ = true;
    require_all_threads_ = true;   

    std::stringstream ss;
    ss << "bool should_terminate = true;\n"
       << current.dataTypeName() << "* current_vec = (" 
       << current.dataTypeName() << "*)(BUF" << current.bufferId() << ");\n"
       << previous.dataTypeName() << "* previous_vec = (" 
       << previous.dataTypeName() << "*)(BUF" << previous.bufferId() << ");\n"
       << "for (size_t i = idx; i < " << current.size() << "; i += grid_size) {\n"
       << "    if (current_vec[i] != previous_vec[i]) {\n"
       << "        should_terminate = false;\n"
       << "    }\n"
       << "}\n";

    condition_code_ = ss.str();
}



template<typename T>
void TerminationCondition::setBfsComplete(size_t target_node, const Vector<T>& frontier, 
                                          const Vector<T>& previous) {
    if (frontier.size() != previous.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }

    is_active_ = true;
    require_all_threads_ = false;

    std::stringstream ss;
    ss << "bool should_terminate = false;\n"
       << frontier.dataTypeName() << "* frontier_vec = (" 
       << frontier.dataTypeName() << "*)(BUF" << frontier.bufferId() << ");\n"
       << previous.dataTypeName() << "* previous_vec = (" 
       << previous.dataTypeName() << "*)(BUF" << previous.bufferId() << ");\n"
       << "if (idx == " << target_node << " && frontier_vec[" << target_node << "] != 0) {\n"
       << "    should_terminate = true;\n"
       << "}\n"
       << "for (size_t i = idx; i < " << frontier.size() << "; i += grid_size) {\n"
       << "    if (frontier_vec[i] != previous_vec[i]) {\n"
       << "        should_terminate = should_terminate || false;\n"
       << "    }\n"
       << "}\n";

    condition_code_ = ss.str();
}

inline bool TerminationCondition::requireAllThreads() const {
    return require_all_threads_;
}


} // namespace graphblas_gpu

#endif // GRAPHBLAS_GPU_TERMINATION_CONDITION_HPP