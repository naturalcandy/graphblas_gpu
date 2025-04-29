#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <graphblas_gpu/termination_condition.hpp>

#include <iostream>
#include <vector>
#include <cassert>

int main() {
    using namespace graphblas_gpu;

    const size_t N = 10;
    const size_t target_node = 5;

    {
        std::cout << "==== Test 1: setNodeReached ====" << std::endl;
        
        // Frontier vector, starts all zeros
        std::vector<float> host_frontier(N, 0.0f);
        Vector<float> frontier(N, host_frontier);

        // Increment vector
        std::vector<float> host_increment(N, 1.0f);
        Vector<float> increment(N, host_increment);

        TerminationCondition::getInstance().reset();
        TerminationCondition::getInstance().setNodeReached(target_node, frontier);

        frontier += increment; // Add increment every iteration

        auto& compiler = OpCompiler::getInstance();
        compiler.compile();

        compiler.copyHostToDevice(frontier);
        compiler.copyHostToDevice(increment);

        compiler.execute(std::nullopt); // Use termination

        std::vector<float> result(N);
        compiler.copyDeviceToHost(result, frontier);

        assert(result[target_node] != 0);
        std::cout << "Test PASSED: setNodeReached.\n" << std::endl;
        OpCompiler::getInstance().reset();
        OpSequence::getInstance().clear();
        TerminationCondition::getInstance().reset();

    }

    {
        std::cout << "==== Test 2: setFrontierUnchanged ====" << std::endl;

        // Two vectors that start unequal
        std::vector<float> host_v1(N, 5.0f);
        std::vector<float> host_v2(N, 2.0f);
        Vector<float> v1(N, host_v1);
        Vector<float> v2(N, host_v2);

        // Increment vectors
        std::vector<float> host_inc_v1(N, 1.0f);
        std::vector<float> host_inc_v2(N, 2.0f);
        Vector<float> inc_v1(N, host_inc_v1);
        Vector<float> inc_v2(N, host_inc_v2);

        TerminationCondition::getInstance().reset();
        TerminationCondition::getInstance().setFrontierUnchanged(v1, v2);

        v1 += inc_v1;
        v2 += inc_v2;

        auto& compiler = OpCompiler::getInstance();
        compiler.compile();

        compiler.copyHostToDevice(v1);
        compiler.copyHostToDevice(v2);
        compiler.copyHostToDevice(inc_v1);
        compiler.copyHostToDevice(inc_v2);

        compiler.execute();

        std::vector<float> result_v1(N);
        std::vector<float> result_v2(N);
        compiler.copyDeviceToHost(result_v1, v1);
        compiler.copyDeviceToHost(result_v2, v2);

        std::cout << "Vector 1:\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << result_v1[i] << " ";
    }
    std::cout << "\nVector 2:\n";
    for (size_t i = 0; i < N; ++i) {
        std::cout << result_v2[i] << " ";
    }
    std::cout << "\n";

        for (size_t i = 0; i < N; ++i) {
            assert(std::abs(result_v1[i] - result_v2[i]) < 1e-5f);
        }
        std::cout << "Test PASSED: setFrontierUnchanged.\n" << std::endl;
        OpCompiler::getInstance().reset();
        OpSequence::getInstance().clear();
        TerminationCondition::getInstance().reset();

    }

    {
        std::cout << "==== Test 3: setBfsComplete ====" << std::endl;

        // Reuse two vectors for BfsComplete
        std::vector<float> host_frontier(N, 0.0f);
        std::vector<float> host_previous(N, 0.0f);
        Vector<float> frontier(N, host_frontier);
        Vector<float> previous(N, host_previous);

        // Create an increment for frontier
        std::vector<float> host_increment(N, 0.0f);
        host_increment[target_node] = 1.0f; // Only target node will increment
        Vector<float> increment(N, host_increment);

        TerminationCondition::getInstance().reset();
        TerminationCondition::getInstance().setBfsComplete(target_node, frontier, previous);

        frontier += increment;

        auto& compiler = OpCompiler::getInstance();
        compiler.compile();

        compiler.copyHostToDevice(frontier);
        compiler.copyHostToDevice(previous);
        compiler.copyHostToDevice(increment);

        compiler.execute(std::nullopt);

        std::vector<float> result(N);
        compiler.copyDeviceToHost(result, frontier);

        assert(result[target_node] != 0);
        std::cout << "Test PASSED: setBfsComplete.\n" << std::endl;
        OpCompiler::getInstance().reset();
        OpSequence::getInstance().clear();
        TerminationCondition::getInstance().reset();

    }

    std::cout << "==== All Termination Tests Passed Successfully! ====" << std::endl;
    return 0;
}
