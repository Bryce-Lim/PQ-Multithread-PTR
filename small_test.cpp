#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"

void print_vector(const std::vector<float>& vec, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void test_basic_functionality() {
    std::cout << "=== Test 1: Basic Functionality ===" << std::endl;
    
    // Test data: 2 points, 2 centroids, 3 dimensions
    std::vector<float> points = {
        1.0f, 2.0f, 3.0f,  // Point 0: [1, 2, 3]
        4.0f, 5.0f, 6.0f   // Point 1: [4, 5, 6]
    };
    
    std::vector<float> centroids = {
        1.0f, 0.0f, 0.0f,  // Centroid 0: [1, 0, 0]
        0.0f, 1.0f, 1.0f   // Centroid 1: [0, 1, 1]
    };
    
    const size_t point_count = 2;
    const size_t centroid_count = 2;
    const size_t dimension = 3;
    const size_t expected_distances = point_count * centroid_count;
    
    std::vector<float> distances(expected_distances);
    
    size_t result = compute(points.data(), point_count, centroids.data(), centroid_count, dimension, distances.data());
    
    std::cout << "Points:" << std::endl;
    std::cout << "  Point 0: [1.0, 2.0, 3.0]" << std::endl;
    std::cout << "  Point 1: [4.0, 5.0, 6.0]" << std::endl;
    
    std::cout << "Centroids:" << std::endl;
    std::cout << "  Centroid 0: [1.0, 0.0, 0.0]" << std::endl;
    std::cout << "  Centroid 1: [0.0, 1.0, 1.0]" << std::endl;
    
    std::cout << "Expected inner products:" << std::endl;
    std::cout << "  Point 0 · Centroid 0: 1*1 + 2*0 + 3*0 = 1.0" << std::endl;
    std::cout << "  Point 0 · Centroid 1: 1*0 + 2*1 + 3*1 = 5.0" << std::endl;
    std::cout << "  Point 1 · Centroid 0: 4*1 + 5*0 + 6*0 = 4.0" << std::endl;
    std::cout << "  Point 1 · Centroid 1: 4*0 + 5*1 + 6*1 = 11.0" << std::endl;
    
    std::cout << "Function returned: " << result << " distances" << std::endl;
    print_vector(distances, "Computed distances");
    
    // Verify results
    std::vector<float> expected = {1.0f, 5.0f, 4.0f, 11.0f};
    bool success = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(distances[i] - expected[i]) > 1e-5f) {
            success = false;
            break;
        }
    }
    
    std::cout << "Result: " << (success ? "PASS" : "FAIL") << std::endl << std::endl;
}

void test_edge_cases() {
    std::cout << "=== Test 2: Edge Cases ===" << std::endl;
    
    // Test with 1D vectors
    std::vector<float> points_1d = {2.0f, 3.0f, 4.0f};  // 3 points, 1 dimension each
    std::vector<float> centroids_1d = {5.0f, -1.0f};    // 2 centroids, 1 dimension each
    std::vector<float> distances_1d(6);  // 3 points × 2 centroids = 6 distances
    
    size_t result_1d = compute(points_1d.data(), 3, centroids_1d.data(), 2, 1, distances_1d.data());
    
    std::cout << "1D Test:" << std::endl;
    std::cout << "Points: [2.0], [3.0], [4.0]" << std::endl;
    std::cout << "Centroids: [5.0], [-1.0]" << std::endl;
    std::cout << "Expected: [10.0, -2.0, 15.0, -3.0, 20.0, -4.0]" << std::endl;
    print_vector(distances_1d, "Computed");
    std::cout << "Returned: " << result_1d << " distances" << std::endl;
    
    // Test with zero vectors
    std::vector<float> zero_points = {0.0f, 0.0f, 1.0f, 2.0f};  // 2 points, 2D
    std::vector<float> zero_centroids = {0.0f, 0.0f, 3.0f, 4.0f};  // 2 centroids, 2D
    std::vector<float> zero_distances(4);
    
    size_t result_zero = compute(zero_points.data(), 2, zero_centroids.data(), 2, 2, zero_distances.data());
    
    std::cout << "\nZero Vector Test:" << std::endl;
    std::cout << "Points: [0,0], [1,2]" << std::endl;
    std::cout << "Centroids: [0,0], [3,4]" << std::endl;
    std::cout << "Expected: [0.0, 0.0, 0.0, 11.0]" << std::endl;
    print_vector(zero_distances, "Computed");
    std::cout << "Returned: " << result_zero << " distances" << std::endl << std::endl;
}

void test_invalid_inputs() {
    std::cout << "=== Test 3: Invalid Inputs ===" << std::endl;
    
    std::vector<float> valid_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(4);
    
    // Test null pointers
    size_t result1 = compute(nullptr, 1, valid_data.data(), 1, 2, output.data());
    std::cout << "Null points: returned " << result1 << " (expected 0)" << std::endl;
    
    size_t result2 = compute(valid_data.data(), 1, nullptr, 1, 2, output.data());
    std::cout << "Null centroids: returned " << result2 << " (expected 0)" << std::endl;
    
    size_t result3 = compute(valid_data.data(), 1, valid_data.data(), 1, 2, nullptr);
    std::cout << "Null output: returned " << result3 << " (expected 0)" << std::endl;
    
    // Test zero counts
    size_t result4 = compute(valid_data.data(), 0, valid_data.data(), 1, 2, output.data());
    std::cout << "Zero point count: returned " << result4 << " (expected 0)" << std::endl;
    
    size_t result5 = compute(valid_data.data(), 1, valid_data.data(), 0, 2, output.data());
    std::cout << "Zero centroid count: returned " << result5 << " (expected 0)" << std::endl;
    
    size_t result6 = compute(valid_data.data(), 1, valid_data.data(), 1, 0, output.data());
    std::cout << "Zero dimension: returned " << result6 << " (expected 0)" << std::endl << std::endl;
}

int main() {
    std::cout << "Scalar Inner Product Calculator - Test Suite" << std::endl;
    std::cout << "=============================================" << std::endl << std::endl;
    
    test_basic_functionality();
    test_edge_cases();
    test_invalid_inputs();
    
    std::cout << "All tests completed!" << std::endl;
    
    return 0;
}
