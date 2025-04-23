// #include <iostream>
// #include <cassert>
// #include <algorithm>
// #include <graphblas_gpu/sparse_formats.hpp>
// #include <iomanip>

// bool test_csr_to_ell() {
//     bool passed = true;
//     std::cout << "Testing CSR to ELL conversion..." << std::endl;

//     int num_rows = 5, num_cols = 5;
//     int csr_row_offsets[] = {0, 3, 5, 7, 10, 12};
//     int csr_cols[] = {0, 2, 3, 1, 4, 2, 4, 0, 1, 3, 2, 4};
//     float csr_vals[] = {2, -1, 3, 5, 4, 1, -5, 6, 7, 2, 3, 9};

//     int* ell_cols = nullptr;
//     float* ell_vals = nullptr;
//     int max_nnz_per_row;

//     graphblas_gpu::csr_to_ell(csr_row_offsets, csr_cols, csr_vals, num_rows, ell_cols, ell_vals, max_nnz_per_row);

//     int ell_cols_expected[15] = {0, 2, 3, 1, 4, -1, 2, 4, -1, 0, 1, 3, 2, 4, -1};
//     float ell_vals_expected[15] = {2, -1, 3, 5, 4, 0, 1, -5, 0, 6, 7, 2, 3, 9, 0};

//     for (int i = 0; i < 15; i++) {
//         if (ell_cols[i] != ell_cols_expected[i]) {
//             std::cout << "Error in ell_cols at index " << i << ": "
//                       << ell_cols[i] << " != " << ell_cols_expected[i] << std::endl;
//             passed = false;
//         }
//         if (ell_vals[i] != ell_vals_expected[i]) {
//             std::cout << "Error in ell_vals at index " << i << ": "
//                       << ell_vals[i] << " != " << ell_vals_expected[i] << std::endl;
//             passed = false;
//         }
//     }

//     delete[] ell_cols;
//     delete[] ell_vals;

//     if (passed) {
//         std::cout << "CSR to ELL test PASSED!" << std::endl;
//     } else {
//         std::cout << "CSR to ELL test FAILED!" << std::endl;
//     }
//     return passed;
// }

// bool test_ell_to_csr() {
//     bool passed = true;
//     std::cout << "Testing ELL to CSR conversion..." << std::endl;

//     int num_rows = 5;
//     int max_nnz_per_row = 3;

//     int ell_cols[15] = {0, 2, 3, 1, 4, -1, 2, 4, -1, 0, 1, 3, 2, 4, -1};
//     float ell_vals[15] = {2, -1, 3, 5, 4, 0, 1, -5, 0, 6, 7, 2, 3, 9, 0};

//     int* csr_row_offsets = nullptr;
//     int* csr_cols = nullptr;
//     float* csr_vals = nullptr;

//     graphblas_gpu::ell_to_csr(ell_cols, ell_vals, num_rows, max_nnz_per_row, csr_row_offsets, csr_cols, csr_vals);

//     int csr_row_offsets_expected[] = {0, 3, 5, 7, 10, 12};
//     int csr_cols_expected[] = {0, 2, 3, 1, 4, 2, 4, 0, 1, 3, 2, 4};
//     float csr_vals_expected[] = {2, -1, 3, 5, 4, 1, -5, 6, 7, 2, 3, 9};

//     for (int i = 0; i < 6; i++) {
//         if (csr_row_offsets[i] != csr_row_offsets_expected[i]) {
//             std::cout << "Error in csr_row_offsets at index " << i << ": "
//                       << csr_row_offsets[i] << " != " << csr_row_offsets_expected[i] << std::endl;
//             passed = false;
//         }
//     }

//     for (int i = 0; i < 12; i++) {
//         if (csr_cols[i] != csr_cols_expected[i]) {
//             std::cout << "Error in csr_cols at index " << i << ": "
//                       << csr_cols[i] << " != " << csr_cols_expected[i] << std::endl;
//             passed = false;
//         }
//     }

//     for (int i = 0; i < 12; i++) {
//         if (csr_vals[i] != csr_vals_expected[i]) {
//             std::cout << "Error in csr_vals at index " << i << ": "
//                       << csr_vals[i] << " != " << csr_vals_expected[i] << std::endl;
//             passed = false;
//         }
//     }

//     delete[] csr_row_offsets;
//     delete[] csr_cols;
//     delete[] csr_vals;

//     if (passed) {
//         std::cout << "ELL to CSR test PASSED!" << std::endl;
//     } else {
//         std::cout << "ELL to CSR test FAILED!" << std::endl;
//     }
//     return passed;
// }

// bool test_round_trid_csr_ell() {
//     bool passed = true;
//     const int num_rows = 5;

//     int row_ptr[] = {0, 3, 5, 7, 10, 12};
//     int col_idx[] = {0, 2, 3, 1, 4, 2, 4, 0, 1, 3, 2, 4};
//     float values[] = {2, -1, 3, 5, 4, 1, -5, 6, 7, 2, 3, 9};

//     int* ell_col_idx = nullptr;
//     float* ell_values = nullptr;
//     int max_nnz_per_row;

//     graphblas_gpu::csr_to_ell(row_ptr, col_idx, values, num_rows, ell_col_idx, ell_values, max_nnz_per_row);

//     int* out_row_ptr = nullptr;
//     int* out_col_idx = nullptr;
//     float* out_values = nullptr;

//     graphblas_gpu::ell_to_csr(ell_col_idx, ell_values, num_rows, max_nnz_per_row, out_row_ptr, out_col_idx, out_values);

//     int original_nnz = row_ptr[num_rows];
//     if (out_row_ptr[num_rows] != original_nnz) {
//         std::cout << "Error: out_row_ptr[num_rows] != original_nnz" << std::endl;
//         passed = false;
//     }

//     for (int i = 0; i <= num_rows; ++i) {
//         if (row_ptr[i] != out_row_ptr[i]) {
//             std::cout << "Error in row_ptr at index " << i << ": "
//                       << row_ptr[i] << " != " << out_row_ptr[i] << std::endl;
//             passed = false;
//         }
//     }

//     for (int i = 0; i < original_nnz; ++i) {
//         if (col_idx[i] != out_col_idx[i]) {
//             std::cout << "Error in col_idx at index " << i << ": "
//                       << col_idx[i] << " != " << out_col_idx[i] << std::endl;
//             passed = false;
//         }
//         if (values[i] != out_values[i]) {
//             std::cout << "Error in values at index " << i << ": "
//                       << values[i] << " != " << out_values[i] << std::endl;
//             passed = false;
//         }
//     }

//     if (passed) {
//         std::cout << "CSR <-> ELL round-trip test PASSED!" << std::endl;
//     } else {
//         std::cout << "CSR <-> ELL round-trip test FAILED!" << std::endl;
//     }

//     delete[] ell_col_idx;
//     delete[] ell_values;
//     delete[] out_row_ptr;
//     delete[] out_col_idx;
//     delete[] out_values;

//     return passed;
// }

// bool test_round_trip_csr_sellc() {
//     std::cout << "Testing CSR -> SellC -> CSR conversion..." << std::endl;
//     bool passed = true;
//     const int num_rows = 7;

//     int row_ptr[] = {0, 5, 8, 10, 10, 10, 11, 12};
//     int col_idx[] = {0, 1, 2, 5, 7, 0, 1, 2, 2, 7, 0, 6};
//     float values[] = {5, 2, 4, 2, 5, 3, 7, 2, 7, 5, 8, 3};

//     const int c = 2;

//     int* sell_col_idx = nullptr;
//     float* sell_values = nullptr;
//     int* slice_ptrs = nullptr;
//     int* slice_lengths = nullptr;
//     int total_vals;

//     graphblas_gpu::csr_to_sellc(row_ptr, col_idx, values, num_rows, c, total_vals, sell_col_idx, sell_values, slice_ptrs, slice_lengths);

//     int* out_row_ptr = nullptr;
//     int* out_col_idx = nullptr;
//     float* out_values = nullptr;

//     graphblas_gpu::sellc_to_csr(sell_col_idx, sell_values, slice_ptrs, slice_lengths, num_rows, c, out_row_ptr, out_col_idx, out_values);

//     int original_nnz = row_ptr[num_rows];
//     assert(out_row_ptr[num_rows] == original_nnz);

//     for (int i = 0; i <= num_rows; ++i)
//         if (row_ptr[i] != out_row_ptr[i]) {
//             std::cout << "Error in row_ptr at index " << i << ": "
//                       << row_ptr[i] << " != " << out_row_ptr[i] << std::endl;
//         }

//     for (int i = 0; i < original_nnz; ++i) {
//         if (col_idx[i] != out_col_idx[i]) {
//             std::cout << "Error in col_idx at index " << i << ": "
//                       << col_idx[i] << " != " << out_col_idx[i] << std::endl;
//             passed = false;
//         }
//         if (values[i] != out_values[i]) {
//             std::cout << "Error in values at index " << i << ": "
//                       << values[i] << " != " << out_values[i] << std::endl;
//             passed = false;
//         }
//     }

//     if (passed) {
//         std::cout << "CSR -> Sellc Round-trip test PASSED!" << std::endl;
//     } else {
//         std::cout << "CSR -> Sellc Round-trip test FAILED!" << std::endl;
//     }

//     delete[] sell_col_idx;
//     delete[] sell_values;
//     delete[] slice_ptrs;
//     delete[] slice_lengths;
//     delete[] out_row_ptr;
//     delete[] out_col_idx;
//     delete[] out_values;

//     return passed;
// }

int main() {
//     assert(test_csr_to_ell());
//     assert(test_ell_to_csr());
//     test_round_trid_csr_ell();
//     test_round_trip_csr_sellc();
//     std::cout << "All tests passed!" << std::endl;
    return 0;
}
