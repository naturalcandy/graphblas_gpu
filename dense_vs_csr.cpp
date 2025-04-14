#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;
// Generate dense adjacency matrix with given sparsity
vector<vector<int>> generate_dense_matrix(int rows, int cols, double sparsity) {
    vector<vector<int>> matrix(rows, vector<int>(cols, 0));
    int total_elements = rows * cols;
    int nonzeros = total_elements * (1.0 - sparsity);

    // Create positions and shuffle
    vector<int> positions(total_elements);
    for (int i = 0; i < total_elements; ++i) positions[i] = i;

    random_device rd;
    mt19937 gen(rd());
    shuffle(positions.begin(), positions.end(), gen);

    // Set non-zero elements (e.g., all 1s for simplicity)
    for (int i = 0; i < nonzeros; ++i) {
        int pos = positions[i];
        int r = pos / cols;
        int c = pos % cols;
        matrix[r][c] = 1; // or random weights if desired
    }

    return matrix;
}

// Generate CSR representation directly
void generate_csr_matrix(int rows, int cols, double sparsity,
                         vector<int>& csr_values, vector<int>& csr_col_indices, vector<int>& csr_row_ptr) {
    int total_elements = rows * cols;
    int nonzeros = total_elements * (1.0 - sparsity);

    vector<int> positions(total_elements);
    for (int i = 0; i < total_elements; ++i) positions[i] = i;

    random_device rd;
    mt19937 gen(rd());
    shuffle(positions.begin(), positions.end(), gen);

    positions.resize(nonzeros);
    sort(positions.begin(), positions.end());

    csr_row_ptr.push_back(0);
    int current_row = 0;
    int count = 0;

    for (int pos : positions) {
        int r = pos / cols;
        int c = pos % cols;

        while (current_row < r) {
            csr_row_ptr.push_back(count);
            current_row++;
        }

        csr_values.push_back(1); // or random weights
        csr_col_indices.push_back(c);
        count++;
    }

    // Fill remaining row pointers
    while (csr_row_ptr.size() <= rows) {
        csr_row_ptr.push_back(count);
    }
}

// Helper function to print dense matrix
void print_dense_matrix(const vector<vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) cout << val << " ";
        cout << endl;
    }
}

// Helper function to print CSR matrix
void print_csr(const vector<int>& values, const vector<int>& col_indices, const vector<int>& row_ptr) {
    cout << "CSR Values: ";
    for (int v : values) cout << v << " ";
    cout << endl;

    cout << "CSR Column Indices: ";
    for (int idx : col_indices) cout << idx << " ";
    cout << endl;

    cout << "CSR Row Pointers: ";
    for (int ptr : row_ptr) cout << ptr << " ";
    cout << endl;
}

vector<int> dense_matvec(const vector<vector<int>>& matrix, const vector<int>& vec) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<int> result(rows, 0);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[i] += matrix[i][j] * vec[j];

    return result;
}

vector<int> csr_matvec(const vector<int>& values, const vector<int>& col_indices, const vector<int>& row_ptr, const vector<int>& vec) {
    int rows = row_ptr.size() - 1;
    vector<int> result(rows, 0);

    for (int i = 0; i < rows; ++i) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            result[i] += values[j] * vec[col_indices[j]];
        }
    }

    return result;
}

void run_test(int rows, int cols, double sparsity, int num_iterations) {
    long long total_dense_time = 0;
    long long total_csr_time = 0;

    for (int iter = 0; iter < num_iterations; ++iter) {
        auto dense_matrix = generate_dense_matrix(rows, cols, sparsity);
        vector<int> vec(cols, 1);

        vector<int> csr_values, csr_col_indices, csr_row_ptr;
        generate_csr_matrix(rows, cols, sparsity, csr_values, csr_col_indices, csr_row_ptr);

        auto start_dense = high_resolution_clock::now();
        auto result_dense = dense_matvec(dense_matrix, vec);
        auto end_dense = high_resolution_clock::now();
        total_dense_time += duration_cast<microseconds>(end_dense - start_dense).count();

        auto start_csr = high_resolution_clock::now();
        auto result_csr = csr_matvec(csr_values, csr_col_indices, csr_row_ptr, vec);
        auto end_csr = high_resolution_clock::now();
        total_csr_time += duration_cast<microseconds>(end_csr - start_csr).count();
    }

    double avg_dense_time = static_cast<double>(total_dense_time) / num_iterations;
    double avg_csr_time = static_cast<double>(total_csr_time) / num_iterations;

    cout << "Test Case: " << rows << "x" << cols << ", Sparsity: " << sparsity * 100 << "%" << endl;
    cout << "Average Dense Time (μs): " << avg_dense_time << endl;
    cout << "Average CSR Time (μs): " << avg_csr_time << endl;
    cout << "Speedup: " << avg_dense_time / avg_csr_time << "x" << endl << endl;
}

void test_non_acclerated_mat_vec_mul(){
    vector<pair<int, int>> sizes = {{100, 100}, {500, 500}, {1000, 1000}, {10000, 10000}};
    vector<double> sparsities = {0.7, 0.8, 0.85, 0.90, 0.95}; // 70% to 95% zeros

    for (auto [rows, cols] : sizes) {
        for (double sparsity : sparsities) {
            run_test(rows, cols, sparsity, 5);
        }
    }
}

int main() {

    test_non_acclerated_mat_vec_mul();

    return 0;
}

