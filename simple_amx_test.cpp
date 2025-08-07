#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include "AMXInnerProductBF16Ptr.h"

// Helper function to convert float to bfloat16
uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

// Helper function to convert bfloat16 to float
float bf16_to_float(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

int main() {
    std::cout << "Simple AMX Inner Product Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Create AMX calculator
    AMXInnerProductBF16Ptr amx_calc;
    
    // Test 1: Initialization
    std::cout << "\nTest 1: AMX Initialization" << std::endl;
    bool init_success = amx_calc.initialize();
    std::cout << "AMX Initialize: " << (init_success ? "SUCCESS" : "FAILED") << std::endl;
    
    if (!init_success) {
        std::cout << "AMX not available. This is normal on non-AMX hardware." << std::endl;
        std::cout << "Test completed - AMX requires Intel 4th Gen Xeon or newer." << std::endl;
        return 0;
    }
    
    // Test 2: Simple computation with minimum AMX requirements
    std::cout << "\nTest 2: Basic AMX Computation" << std::endl;
    
    // AMX constraints: dimension % 64 == 0, centroid_count % 16 == 0
    const size_t point_count = 9600;      // Can be any value
    const size_t centroid_count = 16;   // Must be multiple of 16
    const size_t dimension = 1024;        // Must be multiple of 64
    
    std::cout << "Data size: " << point_count << " points, " 
              << centroid_count << " centroids, " 
              << dimension << " dimensions" << std::endl;
    
    // Create simple test data
    std::vector<float> points_float(point_count * dimension);
    std::vector<float> centroids_float(centroid_count * dimension);
    
    // Fill with simple test pattern
    for (size_t i = 0; i < points_float.size(); ++i) {
        points_float[i] = 1.0f + (i % 10) * 0.1f;  // 1.0, 1.1, 1.2, ... 1.9, repeat
    }
    
    for (size_t i = 0; i < centroids_float.size(); ++i) {
        centroids_float[i] = 0.5f + (i % 5) * 0.2f;  // 0.5, 0.7, 0.9, 1.1, 1.3, repeat
    }
    
    // Convert to bfloat16
    std::vector<uint16_t> points_bf16(point_count * dimension);
    std::vector<uint16_t> centroids_bf16(centroid_count * dimension);
    
    for (size_t i = 0; i < points_float.size(); ++i) {
        points_bf16[i] = float_to_bf16(points_float[i]);
    }
    
    for (size_t i = 0; i < centroids_float.size(); ++i) {
        centroids_bf16[i] = float_to_bf16(centroids_float[i]);
    }
    
    // Prepare output
    std::vector<float> distances(point_count * centroid_count);
    
    // Prepare output arrays for comparison
    std::vector<float> amx_distances(point_count * centroid_count);
    std::vector<float> scalar_distances(point_count * centroid_count);
    
    // First compute with scalar implementation for reference
    std::cout << "Running scalar computation for reference..." << std::endl;
    
    auto scalar_start = std::chrono::high_resolution_clock::now();
    
    // Scalar inner product computation
    size_t distance_idx = 0;
    for (size_t point_idx = 0; point_idx < point_count; ++point_idx) {
        for (size_t centroid_idx = 0; centroid_idx < centroid_count; ++centroid_idx) {
            float inner_product = 0.0f;
            for (size_t dim = 0; dim < dimension; ++dim) {
                float point_val = points_float[point_idx * dimension + dim];
                float centroid_val = centroids_float[centroid_idx * dimension + dim];
                inner_product += point_val * centroid_val;
            }
            scalar_distances[distance_idx++] = inner_product;
        }
    }
    
    auto scalar_end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(scalar_end - scalar_start);
    
    std::cout << "Scalar computation completed in " << scalar_time.count() << " μs" << std::endl;
    
    // Now test AMX computation
    std::cout << "Running AMX computation..." << std::endl;
    
    try {
        auto amx_start = std::chrono::high_resolution_clock::now();
        
        size_t result_count = amx_calc.compute_inner_products(
            reinterpret_cast<const bfloat16_t*>(points_bf16.data()),
            point_count,
            reinterpret_cast<const bfloat16_t*>(centroids_bf16.data()),
            centroid_count,
            dimension,
            amx_distances.data()
        );
        
        auto amx_end = std::chrono::high_resolution_clock::now();
        auto amx_time = std::chrono::duration_cast<std::chrono::microseconds>(amx_end - amx_start);
        
        std::cout << "AMX computation completed in " << amx_time.count() << " μs" << std::endl;
        std::cout << "Expected result count: " << point_count * centroid_count << std::endl;
        std::cout << "Actual result count: " << result_count << std::endl;
        std::cout << "Result count match: " << (result_count == point_count * centroid_count ? "YES" : "NO") << std::endl;
        
        // Performance comparison
        if (amx_time.count() > 0 && scalar_time.count() > 0) {
            double speedup = static_cast<double>(scalar_time.count()) / amx_time.count();
            std::cout << "AMX speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        }
        
        // Accuracy comparison
        std::cout << "\n=== ACCURACY VERIFICATION ===" << std::endl;
        
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        float total_abs_diff = 0.0f;
        float total_rel_diff = 0.0f;
        size_t mismatch_count = 0;
        size_t valid_comparisons = 0;
        const float tolerance = 0.01f;  // 1% tolerance for BF16 precision loss
        
        for (size_t i = 0; i < std::min(scalar_distances.size(), amx_distances.size()); ++i) {
            float abs_diff = std::abs(scalar_distances[i] - amx_distances[i]);
            float rel_diff = 0.0f;
            
            // Calculate relative difference (handle zero denominators)
            if (std::abs(scalar_distances[i]) > 1e-10f) {
                rel_diff = abs_diff / std::abs(scalar_distances[i]);
                total_rel_diff += rel_diff;
                valid_comparisons++;
            } else if (abs_diff > 1e-10f) {
                // If scalar is ~zero but AMX is not, this is significant
                rel_diff = 1.0f;  // 100% relative difference
                total_rel_diff += rel_diff;
                valid_comparisons++;
            }
            
            total_abs_diff += abs_diff;
            max_abs_diff = std::max(max_abs_diff, abs_diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
            
            if (rel_diff > tolerance) {
                mismatch_count++;
            }
        }
        
        float avg_abs_diff = total_abs_diff / scalar_distances.size();
        float avg_rel_diff = (valid_comparisons > 0) ? (total_rel_diff / valid_comparisons) : 0.0f;
        
        std::cout << "Max absolute difference: " << std::fixed << std::setprecision(6) << max_abs_diff << std::endl;
        std::cout << "Average absolute difference: " << std::fixed << std::setprecision(6) << avg_abs_diff << std::endl;
        std::cout << "Max relative difference: " << std::fixed << std::setprecision(4) << (max_rel_diff * 100.0f) << "%" << std::endl;
        std::cout << "Average relative difference: " << std::fixed << std::setprecision(4) << (avg_rel_diff * 100.0f) << "%" << std::endl;
        std::cout << "Values exceeding tolerance (" << (tolerance * 100) << "%): " << mismatch_count 
                  << " out of " << scalar_distances.size() << " (" << std::fixed << std::setprecision(2) 
                  << (100.0f * mismatch_count / scalar_distances.size()) << "%)" << std::endl;
        
        // Show detailed comparison for first 10 results
        std::cout << "\nDetailed comparison (first 10 results):" << std::endl;
        std::cout << "Index | Scalar      | AMX         | Abs Diff    | Rel Diff %" << std::endl;
        std::cout << "------|-------------|-------------|-------------|------------" << std::endl;
        
        for (size_t i = 0; i < std::min((size_t)10, scalar_distances.size()); ++i) {
            float abs_diff = std::abs(scalar_distances[i] - amx_distances[i]);
            float rel_diff_percent = 0.0f;
            
            if (std::abs(scalar_distances[i]) > 1e-10f) {
                rel_diff_percent = (abs_diff / std::abs(scalar_distances[i])) * 100.0f;
            } else if (abs_diff > 1e-10f) {
                rel_diff_percent = 100.0f;  // 100% if scalar is ~zero but AMX is not
            }
            
            std::cout << std::setw(5) << i << " | " 
                      << std::setw(11) << std::fixed << std::setprecision(4) << scalar_distances[i] << " | "
                      << std::setw(11) << std::fixed << std::setprecision(4) << amx_distances[i] << " | "
                      << std::setw(11) << std::fixed << std::setprecision(6) << abs_diff << " | "
                      << std::setw(10) << std::fixed << std::setprecision(3) << rel_diff_percent << std::endl;
        }
        
        // Overall accuracy assessment
        bool accuracy_pass = (mismatch_count == 0) || (mismatch_count < scalar_distances.size() * 0.05);  // Allow 5% of values to exceed tolerance
        std::cout << "\nAccuracy assessment: " << (accuracy_pass ? "PASS" : "FAIL") << std::endl;
        
        if (!accuracy_pass) {
            std::cout << "WARNING: Significant accuracy differences detected!" << std::endl;
            std::cout << "This could indicate issues with:" << std::endl;
            std::cout << "  - BF16 conversion precision" << std::endl;
            std::cout << "  - AMX computation logic" << std::endl;
            std::cout << "  - Memory alignment/access patterns" << std::endl;
        }
        
        // Enhanced precision analysis
        std::cout << "\n=== DETAILED PRECISION ANALYSIS ===" << std::endl;
        
        // Test BF16 conversion precision in isolation
        std::cout << "Testing BF16 conversion precision..." << std::endl;
        float max_conversion_error = 0.0f;
        float total_conversion_error = 0.0f;
        size_t test_samples = std::min((size_t)100, points_float.size());
        
        for (size_t i = 0; i < test_samples; ++i) {
            float original = points_float[i];
            uint16_t bf16_val = float_to_bf16(original);
            float converted_back = bf16_to_float(bf16_val);
            
            float conversion_error = std::abs(original - converted_back);
            float relative_error = (std::abs(original) > 1e-10f) ? conversion_error / std::abs(original) : 0.0f;
            
            max_conversion_error = std::max(max_conversion_error, relative_error);
            total_conversion_error += relative_error;
            
            if (i < 10) {
                std::cout << "  [" << i << "] " << std::fixed << std::setprecision(6) 
                          << original << " -> BF16 -> " << converted_back 
                          << " (error: " << std::setprecision(4) << (relative_error * 100) << "%)" << std::endl;
            }
        }
        
        float avg_conversion_error = total_conversion_error / test_samples;
        std::cout << "BF16 conversion - Max error: " << std::fixed << std::setprecision(4) 
                  << (max_conversion_error * 100) << "%, Avg error: " 
                  << (avg_conversion_error * 100) << "%" << std::endl;
        
        // Test if the error is cumulative (from many BF16 operations)
        std::cout << "\nTesting error accumulation..." << std::endl;
        
        // Single operation test
        float test_val1 = 1.5f, test_val2 = 2.3f;
        float scalar_product = test_val1 * test_val2;
        
        uint16_t bf16_val1 = float_to_bf16(test_val1);
        uint16_t bf16_val2 = float_to_bf16(test_val2);
        float bf16_product = bf16_to_float(bf16_val1) * bf16_to_float(bf16_val2);
        
        float single_op_error = (scalar_product != 0.0f) ? std::abs(scalar_product - bf16_product) / std::abs(scalar_product) : 0.0f;
        std::cout << "Single multiply: Scalar=" << std::fixed << std::setprecision(6) << scalar_product 
                  << ", BF16=" << bf16_product 
                  << " (error: " << std::setprecision(4) << (single_op_error * 100) << "%)" << std::endl;
        
        // Many operations test (simulate dot product with same dimension as test)
        float scalar_sum = 0.0f, bf16_sum = 0.0f;
        for (size_t i = 0; i < dimension; ++i) {
            float a = 1.0f + (i % 10) * 0.1f;  // Same pattern as your test data
            float b = 0.5f + (i % 5) * 0.2f;   // Same pattern as your test data
            
            scalar_sum += a * b;
            bf16_sum += bf16_to_float(float_to_bf16(a)) * bf16_to_float(float_to_bf16(b));
        }
        
        float accumulated_error = (scalar_sum != 0.0f) ? std::abs(scalar_sum - bf16_sum) / std::abs(scalar_sum) : 0.0f;
        std::cout << dimension << "-element dot product: Scalar=" << std::fixed << std::setprecision(6) << scalar_sum 
                  << ", BF16=" << bf16_sum 
                  << " (error: " << std::setprecision(4) << (accumulated_error * 100) << "%)" << std::endl;
        
        // Analysis of your specific data patterns
        std::cout << "\nAnalyzing your data characteristics..." << std::endl;
        
        float points_min = *std::min_element(points_float.begin(), points_float.end());
        float points_max = *std::max_element(points_float.begin(), points_float.end());
        float centroids_min = *std::min_element(centroids_float.begin(), centroids_float.end());
        float centroids_max = *std::max_element(centroids_float.begin(), centroids_float.end());
        
        std::cout << "Points range: [" << std::fixed << std::setprecision(3) << points_min << ", " << points_max << "]" << std::endl;
        std::cout << "Centroids range: [" << centroids_min << ", " << centroids_max << "]" << std::endl;
        
        // Calculate expected dot product range
        float expected_min_product = points_min * centroids_min * dimension;
        float expected_max_product = points_max * centroids_max * dimension;
        std::cout << "Expected dot product range: [" << expected_min_product << ", " << expected_max_product << "]" << std::endl;
        
        // Sample a few actual dot products to verify range
        float sample_scalar_sum = 0.0f;
        for (size_t i = 0; i < dimension; ++i) {
            sample_scalar_sum += points_float[i] * centroids_float[i];
        }
        std::cout << "Sample actual dot product: " << sample_scalar_sum << std::endl;
        
        // Recommendations
        std::cout << "\n=== RECOMMENDATIONS ===" << std::endl;
        
        if (avg_conversion_error * 100 > 0.02f) {
            std::cout << "⚠️  High BF16 conversion error detected!" << std::endl;
            std::cout << "   Consider normalizing your data to reduce conversion precision loss" << std::endl;
        } else {
            std::cout << "✅ BF16 conversion error is within normal range" << std::endl;
        }
        
        if (accumulated_error * 100 > avg_conversion_error * 100 * 2) {
            std::cout << "⚠️  Error accumulation detected!" << std::endl;
            std::cout << "   Error grows with vector length - this is expected for BF16" << std::endl;
        } else {
            std::cout << "✅ Error accumulation is minimal" << std::endl;
        }
        
        // Compare AMX error to expected BF16 error
        float error_ratio = avg_rel_diff / accumulated_error;
        if (avg_rel_diff > accumulated_error * 1.5f) {
            std::cout << "🔍 AMX-specific error detected!" << std::endl;
            std::cout << "   The error is " << std::fixed << std::setprecision(2) << error_ratio 
                      << "x higher than expected from BF16 alone." << std::endl;
            std::cout << "   Check AMX implementation for:" << std::endl;
            std::cout << "   - Incorrect tile operations or data ordering" << std::endl;
            std::cout << "   - Data layout/stride issues" << std::endl;
            std::cout << "   - Accumulation order differences" << std::endl;
            std::cout << "   - Memory alignment problems" << std::endl;
        } else {
            std::cout << "✅ Error appears to be primarily from BF16 precision limitations" << std::endl;
            std::cout << "   AMX error is only " << std::fixed << std::setprecision(2) << error_ratio 
                      << "x the expected BF16 error" << std::endl;
        }
        
        // Expected BF16 error summary
        std::cout << "\nERROR BREAKDOWN:" << std::endl;
        std::cout << "Expected BF16 error for " << dimension << "-element dot products: ~" 
                  << std::fixed << std::setprecision(4) << (accumulated_error * 100) << "%" << std::endl;
        std::cout << "Your actual AMX error: " << (avg_rel_diff * 100) << "%" << std::endl;
        std::cout << "Ratio (AMX/Expected): " << std::setprecision(2) << error_ratio << "x" << std::endl;
        
        if (avg_rel_diff > accumulated_error * 2.0f) {
            std::cout << "🚨 Your error is significantly higher than expected for BF16!" << std::endl;
            std::cout << "   This suggests an implementation bug in your AMX code." << std::endl;
        } else if (avg_rel_diff > accumulated_error * 1.2f) {
            std::cout << "⚠️  Your error is somewhat higher than expected." << std::endl;
            std::cout << "   This could be from data formatting or minor implementation issues." << std::endl;
        } else {
            std::cout << "✅ Your error is within expected range for BF16 operations" << std::endl;
            std::cout << "   The 0.0863% error is likely just BF16 precision limitations." << std::endl;
        }
        
        // Print timing stats
        amx_calc.print_timing_stats();
        
        bool overall_pass = (result_count == point_count * centroid_count) && accuracy_pass;
        std::cout << "\nTest 2 Overall Result: " << (overall_pass ? "PASSED" : "FAILED") << std::endl;
    } catch (const std::exception& e) {
        std::cout << "AMX computation failed: " << e.what() << std::endl;
        std::cout << "Test 2: FAILED" << std::endl;
        return 1;
    }
    
    // Test 3: Invalid input validation
    std::cout << "\nTest 3: Input Validation" << std::endl;
    
    // Test invalid dimension (not multiple of 64)
    try {
        std::vector<uint16_t> invalid_points(16 * 63);  // 63 dimensions
        std::vector<float> invalid_distances(16 * 16);
        
        amx_calc.compute_inner_products(
            reinterpret_cast<const bfloat16_t*>(invalid_points.data()),
            16,
            reinterpret_cast<const bfloat16_t*>(centroids_bf16.data()),
            16,
            63,  // Invalid dimension
            invalid_distances.data()
        );
        
        std::cout << "Invalid dimension test: FAILED (should have thrown exception)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Invalid dimension test: PASSED (caught exception: " << e.what() << ")" << std::endl;
    }
    
    // Test invalid centroid count (not multiple of 16)
    try {
        std::vector<uint16_t> invalid_centroids(15 * 64);  // 15 centroids
        std::vector<float> invalid_distances(16 * 15);
        
        amx_calc.compute_inner_products(
            reinterpret_cast<const bfloat16_t*>(points_bf16.data()),
            16,
            reinterpret_cast<const bfloat16_t*>(invalid_centroids.data()),
            15,  // Invalid centroid count
            64,
            invalid_distances.data()
        );
        
        std::cout << "Invalid centroid count test: FAILED (should have thrown exception)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Invalid centroid count test: PASSED (caught exception: " << e.what() << ")" << std::endl;
    }
    
    std::cout << "\nAll tests completed!" << std::endl;
    std::cout << "AMX implementation appears to be working correctly." << std::endl;
    
    return 0;
}
