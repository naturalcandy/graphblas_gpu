#include <graphblas_gpu/vector.hpp>
#include <graphblas_gpu/graph.hpp>
#include <graphblas_gpu/operation.hpp>
#include <graphblas_gpu/op_compiler.hpp>
#include <graphblas_gpu/termination_condition.hpp>
#include <graphblas_gpu/graph_classifier.hpp>   //  ⟵  new include

#include <iostream>
#include <vector>
#include <cassert>

/* 
 * 0 → 1,2   1 → 3   2 → 3,4   3 → 5   4 → 5
 */
void build_csr(std::vector<size_t>& ro,
               std::vector<int>&    ci,
               std::vector<float>&  val)
{
    ro  = {0,2,3,5,6,7,7};
    ci  = {1,2, 3, 3,4, 5, 5};
    val.assign(ci.size(), 1.0f);
}

int main()
{
    using namespace graphblas_gpu;
    OpSequence::getInstance().clear();
    TerminationCondition::getInstance().reset();


    // tranpose our input graph
    std::vector<size_t> ro , roT;
    std::vector<int>    ci , ciT;
    std::vector<float>  val, valT;
    build_csr(ro, ci, val);

    const size_t N = 6;
    GraphClassifier::csr_transpose<float>(
        N, N, ro, ci, val,          //  A  (out-edges)
        roT, ciT, valT);            //  A transpose (in-edges)

    SparseMatrix<float> A_T(N, N, roT, ciT, valT);   // pull-based BFS matrix

    //  visited vector (source = 0)
    std::vector<float> h_vis(N, 0.0f);  h_vis[0] = 1.0f;
    Vector<float> visited(N, h_vis);     // V
    Vector<float> old    (N);            // placeholder

    // BFS termination predicates
    TerminationCondition::getInstance().setBfsComplete(/*target =*/5, visited, old);
    

    // Stage the BFS kernel
    Vector<float>::copy(visited, old);                               
    Vector<float> tmp = Operations<float>::spmv(A_T, visited,         
                               SemiringType::LogicalOrAnd);           
    visited |= tmp;                                                    

    // compile and run until our predicate fires
    auto& comp = OpCompiler::getInstance();
    comp.compile();

    comp.copyHostToDevice(A_T);
    comp.copyHostToDevice(visited);
    comp.copyHostToDevice(old);    
    comp.copyHostToDevice(tmp);     

    comp.execute();     

    // check
    std::vector<float> host(N);
    comp.copyDeviceToHost(host, visited);

    std::cout << "Visited: ";
    for (float x : host) std::cout << x << ' ';
    std::cout << '\n';

    assert(host[5] != 0.0f);        // target reached
    std::cout << "BFS test PASSED\n";
}
