#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <graphblas_gpu/graph_classifier.hpp>

using namespace graphblas_gpu;

bool test_csr_to_ell() {
    bool passed = true;
    std::cout << "Testing CSR to ELL conversion..." << std::endl;

    size_t num_rows = 5, num_cols = 5;
    std::vector<size_t> csr_row_offsets = {0, 3, 5, 7, 10, 12};
    std::vector<int> csr_cols = {0, 2, 3, 1, 4, 2, 4, 0, 1, 3, 2, 4};
    std::vector<float> csr_vals = {2, -1, 3, 5, 4, 1, -5, 6, 7, 2, 3, 9};

    std::vector<int> ell_cols;
    std::vector<float> ell_vals;
    size_t max_nnz_per_row;

    // Convert CSR to ELL
    GraphClassifier::csr_to_ell(csr_row_offsets, csr_cols, csr_vals, num_rows, 
                               ell_cols, ell_vals, max_nnz_per_row);

    std::vector<int> ell_cols_expected = {0, 2, 3, 1, 4, -1, 2, 4, -1, 0, 1, 3, 2, 4, -1};
    std::vector<float> ell_vals_expected = {2, -1, 3, 5, 4, 0, 1, -5, 0, 6, 7, 2, 3, 9, 0};

    // Max nonzero 
    if (max_nnz_per_row != 3) {
        std::cout << "Error: max_nnz_per_row = " << max_nnz_per_row 
                  << ", expected 3" << std::endl;
        passed = false;
    }

    // Output size
    if (ell_cols.size() != num_rows * max_nnz_per_row) {
        std::cout << "Error: ell_cols size = " << ell_cols.size() 
                  << ", expected " << num_rows * max_nnz_per_row << std::endl;
        passed = false;
    }

    // Column verification
    for (size_t i = 0; i < ell_cols_expected.size(); i++) {
        if (ell_cols[i] != ell_cols_expected[i]) {
            std::cout << "Error in ell_cols at index " << i << ": "
                      << ell_cols[i] << " != " << ell_cols_expected[i] << std::endl;
            passed = false;
        }
        if (ell_vals[i] != ell_vals_expected[i]) {
            std::cout << "Error in ell_vals at index " << i << ": "
                      << ell_vals[i] << " != " << ell_vals_expected[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "CSR to ELL test PASSED!" << std::endl;
    } else {
        std::cout << "CSR to ELL test FAILED!" << std::endl;
    }
    return passed;
}

bool test_ell_to_csr() {
    bool passed = true;
    std::cout << "Testing ELL to CSR conversion..." << std::endl;

    size_t num_rows = 5;
    size_t max_nnz_per_row = 3;

    std::vector<int> ell_cols = {0, 2, 3, 1, 4, -1, 2, 4, -1, 0, 1, 3, 2, 4, -1};
    std::vector<float> ell_vals = {2, -1, 3, 5, 4, 0, 1, -5, 0, 6, 7, 2, 3, 9, 0};

    std::vector<size_t> csr_row_offsets;
    std::vector<int> csr_cols;
    std::vector<float> csr_vals;

    // Convert ELL to CSR
    GraphClassifier::ell_to_csr(ell_cols, ell_vals, num_rows, max_nnz_per_row, 
                               csr_row_offsets, csr_cols, csr_vals);

    std::vector<size_t> csr_row_offsets_expected = {0, 3, 5, 7, 10, 12};
    std::vector<int> csr_cols_expected = {0, 2, 3, 1, 4, 2, 4, 0, 1, 3, 2, 4};
    std::vector<float> csr_vals_expected = {2, -1, 3, 5, 4, 1, -5, 6, 7, 2, 3, 9};

    // Row offsets
    if (csr_row_offsets.size() != num_rows + 1) {
        std::cout << "Error: csr_row_offsets size = " << csr_row_offsets.size() 
                  << ", expected " << num_rows + 1 << std::endl;
        passed = false;
    }
    for (size_t i = 0; i < csr_row_offsets_expected.size(); i++) {
        if (csr_row_offsets[i] != csr_row_offsets_expected[i]) {
            std::cout << "Error in csr_row_offsets at index " << i << ": "
                      << csr_row_offsets[i] << " != " << csr_row_offsets_expected[i] << std::endl;
            passed = false;
        }
    }

    // Column verification
    if (csr_cols.size() != csr_cols_expected.size()) {
        std::cout << "Error: csr_cols size = " << csr_cols.size() 
                  << ", expected " << csr_cols_expected.size() << std::endl;
        passed = false;
    }

    for (size_t i = 0; i < csr_cols_expected.size(); i++) {
        if (csr_cols[i] != csr_cols_expected[i]) {
            std::cout << "Error in csr_cols at index " << i << ": "
                      << csr_cols[i] << " != " << csr_cols_expected[i] << std::endl;
            passed = false;
        }
        if (csr_vals[i] != csr_vals_expected[i]) {
            std::cout << "Error in csr_vals at index " << i << ": "
                      << csr_vals[i] << " != " << csr_vals_expected[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "ELL to CSR test PASSED!" << std::endl;
    } else {
        std::cout << "ELL to CSR test FAILED!" << std::endl;
    }
    return passed;
}

bool test_round_trip_csr_ell() {
    bool passed = true;
    std::cout << "Testing CSR -> ELL -> CSR round-trip..." << std::endl;
    
    const size_t num_rows = 5;
    std::vector<size_t> row_offsets = {0, 3, 5, 7, 10, 12};
    std::vector<int> col_indices = {0, 2, 3, 1, 4, 2, 4, 0, 1, 3, 2, 4};
    std::vector<float> values = {2, -1, 3, 5, 4, 1, -5, 6, 7, 2, 3, 9};

    // Intermediate ELL format
    std::vector<int> ell_col_indices;
    std::vector<float> ell_values;
    size_t max_nnz_per_row;

    // Output CSR format
    std::vector<size_t> out_row_offsets;
    std::vector<int> out_col_indices;
    std::vector<float> out_values;

    // Convert CSR -> ELL -> CSR
    GraphClassifier::csr_to_ell(row_offsets, col_indices, values, num_rows, 
                               ell_col_indices, ell_values, max_nnz_per_row);
    
    GraphClassifier::ell_to_csr(ell_col_indices, ell_values, num_rows, max_nnz_per_row, 
                               out_row_offsets, out_col_indices, out_values);

    size_t original_nnz = row_offsets[num_rows];
    
    if (out_row_offsets[num_rows] != original_nnz) {
        std::cout << "Error: out_row_offsets[num_rows] = " << out_row_offsets[num_rows] 
                  << ", expected " << original_nnz << std::endl;
        passed = false;
    }

    for (size_t i = 0; i <= num_rows; ++i) {
        if (row_offsets[i] != out_row_offsets[i]) {
            std::cout << "Error in row_offsets at index " << i << ": "
                      << row_offsets[i] << " != " << out_row_offsets[i] << std::endl;
            passed = false;
        }
    }

    for (size_t i = 0; i < original_nnz; ++i) {
        if (col_indices[i] != out_col_indices[i]) {
            std::cout << "Error in col_indices at index " << i << ": "
                      << col_indices[i] << " != " << out_col_indices[i] << std::endl;
            passed = false;
        }
        if (values[i] != out_values[i]) {
            std::cout << "Error in values at index " << i << ": "
                      << values[i] << " != " << out_values[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "CSR <-> ELL round-trip test PASSED!" << std::endl;
    } else {
        std::cout << "CSR <-> ELL round-trip test FAILED!" << std::endl;
    }

    return passed;
}

bool test_csr_to_sellc() {
    bool passed = true;
    std::cout << "Testing CSR to SELLC conversion..." << std::endl;

    const size_t num_rows = 7;
    const size_t slice_size = 2;  // C = 2

    std::vector<size_t> row_offsets = {0, 5, 8, 10, 10, 10, 11, 12};
    std::vector<int> col_indices = {0, 1, 2, 5, 7, 0, 1, 2, 2, 7, 0, 6};
    std::vector<float> values = {5, 2, 4, 2, 5, 3, 7, 2, 7, 5, 8, 3};

    std::vector<size_t> slice_ptrs;
    std::vector<size_t> slice_lengths;
    std::vector<int> sell_col_indices;
    std::vector<float> sell_values;

    // Convert CSR to SELLC
    GraphClassifier::csr_to_sellc(row_offsets, col_indices, values, num_rows, slice_size,
                                 slice_ptrs, slice_lengths, sell_col_indices, sell_values);

    // Check number of slices
    size_t expected_num_slices = (num_rows + slice_size - 1) / slice_size;
    if (slice_ptrs.size() != expected_num_slices + 1) {
        std::cout << "Error: slice_ptrs size = " << slice_ptrs.size() 
                  << ", expected " << expected_num_slices + 1 << std::endl;
        passed = false;
    }

    size_t nnz_sellc = 0;
    for (const auto& col : sell_col_indices) {
        if (col != -1) nnz_sellc++;
    }

    size_t original_nnz = row_offsets[num_rows];
    if (nnz_sellc != original_nnz) {
        std::cout << "Error: nnz in SELLC = " << nnz_sellc 
                  << ", expected " << original_nnz << std::endl;
        passed = false;
    }

    if (passed) {
        std::cout << "CSR to SELLC test PASSED!" << std::endl;
    } else {
        std::cout << "CSR to SELLC test FAILED!" << std::endl;
    }
    return passed;
}

bool test_sellc_to_csr() {
    bool passed = true;
    std::cout << "Testing SELLC to CSR conversion..." << std::endl;

    const size_t num_rows = 5;
    const size_t slice_size = 2;
    const size_t num_slices = (num_rows + slice_size - 1) / slice_size;

    std::vector<size_t> slice_ptrs = {0, 6, 10, 12};
    std::vector<size_t> slice_lengths = {3, 2, 1};
    
    // First slice: rows 0,1 have 3 elements each
    // Second slice: rows 2,3 have 2 elements each
    // Third slice: row 4 has 1 element
    std::vector<int> sell_col_indices = {
        0, 1,   // row 0,1 element 1
        2, 2,   // row 0,1 element 2
        3, -1,  // row 0,1 element 3 (padding for row 1)
        0, 1,   // row 2,3 element 1
        2, -1,  // row 2,3 element 2 (padding for row 3)
        0, -1   // row 4 element 1 (padding for missing row)
    };
    
    std::vector<float> sell_values = {
        1.0f, 2.0f,  // row 0,1 element 1
        3.0f, 4.0f,  // row 0,1 element 2
        5.0f, 0.0f,  // row 0,1 element 3 (padding for row 1)
        6.0f, 7.0f,  // row 2,3 element 1
        8.0f, 0.0f,  // row 2,3 element 2 (padding for row 3)
        9.0f, 0.0f   // row 4 element 1 (padding for missing row)
    };

    // Output CSR format
    std::vector<size_t> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;

    GraphClassifier::sellc_to_csr(slice_ptrs, slice_lengths, sell_col_indices, sell_values,
                                 num_rows, slice_size, row_offsets, col_indices, values);

    std::vector<size_t> expected_row_offsets = {0, 3, 5, 7, 8, 9};
    std::vector<int> expected_col_indices = {0, 2, 3, 1, 2, 0, 2, 1, 0};
    std::vector<float> expected_values = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f, 8.0f, 7.0f, 9.0f};

    if (row_offsets != expected_row_offsets) {
        std::cout << "Error: row_offsets mismatch" << std::endl;
        for (size_t i = 0; i < row_offsets.size(); i++) {
            std::cout << row_offsets[i] << " vs expected " << expected_row_offsets[i] << std::endl;
        }
        passed = false;
    }

    if (col_indices != expected_col_indices) {
        std::cout << "Error: col_indices mismatch" << std::endl;
        for (size_t i = 0; i < col_indices.size(); i++) {
            std::cout << col_indices[i] << " vs expected " << expected_col_indices[i] << std::endl;
        }
        passed = false;
    }

    if (values != expected_values) {
        std::cout << "Error: values mismatch" << std::endl;
        for (size_t i = 0; i < values.size(); i++) {
            std::cout << values[i] << " vs expected " << expected_values[i] << std::endl;
        }
        passed = false;
    }

    if (passed) {
        std::cout << "SELLC to CSR test PASSED!" << std::endl;
    } else {
        std::cout << "SELLC to CSR test FAILED!" << std::endl;
    }
    return passed;
}

bool test_round_trip_csr_sellc() {
    bool passed = true;
    std::cout << "Testing CSR -> SELLC -> CSR round-trip..." << std::endl;
    
    const size_t num_rows = 7;
    const size_t slice_size = 2;

    std::vector<size_t> row_offsets = {0, 5, 8, 10, 10, 10, 11, 12};
    std::vector<int> col_indices = {0, 1, 2, 5, 7, 0, 1, 2, 2, 7, 0, 6};
    std::vector<float> values = {5, 2, 4, 2, 5, 3, 7, 2, 7, 5, 8, 3};

    // Intermediate SELLC format
    std::vector<size_t> slice_ptrs;
    std::vector<size_t> slice_lengths;
    std::vector<int> sell_col_indices;
    std::vector<float> sell_values;

    std::vector<size_t> out_row_offsets;
    std::vector<int> out_col_indices;
    std::vector<float> out_values;

    // Convert CSR -> SELLC -> CSR
    GraphClassifier::csr_to_sellc(row_offsets, col_indices, values, num_rows, slice_size,
                                 slice_ptrs, slice_lengths, sell_col_indices, sell_values);
    
    GraphClassifier::sellc_to_csr(slice_ptrs, slice_lengths, sell_col_indices, sell_values,
                                 num_rows, slice_size, out_row_offsets, out_col_indices, out_values);

    size_t original_nnz = row_offsets[num_rows];
    
    if (out_row_offsets[num_rows] != original_nnz) {
        std::cout << "Error: out_row_offsets[num_rows] = " << out_row_offsets[num_rows] 
                  << ", expected " << original_nnz << std::endl;
        passed = false;
    }

    if (row_offsets != out_row_offsets) {
        std::cout << "Error: row_offsets mismatch" << std::endl;
        for (size_t i = 0; i < row_offsets.size(); i++) {
            if (row_offsets[i] != out_row_offsets[i]) {
                std::cout << "  at index " << i << ": " << row_offsets[i] 
                          << " vs " << out_row_offsets[i] << std::endl;
            }
        }
        passed = false;
    }

    for (size_t row = 0; row < num_rows; ++row) {
        size_t start = row_offsets[row];
        size_t end = row_offsets[row + 1];
        size_t out_start = out_row_offsets[row];
        size_t out_end = out_row_offsets[row + 1];
        
        if (end - start != out_end - out_start) {
            std::cout << "Error: Row " << row << " has different number of elements: "
                      << (end - start) << " vs " << (out_end - out_start) << std::endl;
            passed = false;
            continue;
        }
        
        // Sort elements in both the original and output rows
        std::vector<std::pair<int, float>> orig_row;
        std::vector<std::pair<int, float>> out_row;
        
        for (size_t i = start; i < end; ++i) {
            orig_row.push_back({col_indices[i], values[i]});
        }
        
        for (size_t i = out_start; i < out_end; ++i) {
            out_row.push_back({out_col_indices[i], out_values[i]});
        }
        
        std::sort(orig_row.begin(), orig_row.end());
        std::sort(out_row.begin(), out_row.end());
        
        // Compare sorted elements
        for (size_t i = 0; i < orig_row.size(); ++i) {
            if (orig_row[i].first != out_row[i].first || 
                orig_row[i].second != out_row[i].second) {
                std::cout << "Error: Row " << row << ", element " << i << " mismatch: "
                          << "(" << orig_row[i].first << ", " << orig_row[i].second << ") vs "
                          << "(" << out_row[i].first << ", " << out_row[i].second << ")" << std::endl;
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "CSR <-> SELLC round-trip test PASSED!" << std::endl;
    } else {
        std::cout << "CSR <-> SELLC round-trip test FAILED!" << std::endl;
    }

    return passed;
}

int main() {
    bool all_passed = true;
    
    all_passed &= test_csr_to_ell();
    all_passed &= test_ell_to_csr();
    all_passed &= test_round_trip_csr_ell();
    all_passed &= test_csr_to_sellc();
    all_passed &= test_sellc_to_csr();
    all_passed &= test_round_trip_csr_sellc();
    
    if (all_passed) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << "Some tests failed!" << std::endl;
    }
    
    return all_passed ? 0 : 1;
}