// verify_multithread.cpp - Simple verification test
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>

#include "AMXCommon.h"
#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"
#include "AMXInnerProductBF16PtrMTSimple.h"

uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

int main() {
    std::cout << "Multithreaded AMX Verification Test" << std::endl;
    std::cout << "===================================" << std::endl;

    // Small test that meets AMX constraints
    const size_t point_count = 1024;   // Divisible by 32
    const size_t centroid_count = 32;  // Divisible by 16  
    const size_t dimension = 1024;     // Divisible by 64

    std::cout << "Configuration: " << point_count << " points, " 
              << centroid_count << " centroids, " << dimension << " dims" << std::endl;

    // Generate simple test data
    std::vector<float> points_float(point_count * dimension);
    std::vector<float> centroids_float(centroid_count * dimension);

    // Fill with varied test pattern
    for (size_t i = 0; i < point_count; ++i) {
        for (size_t d = 0; d < dimension; ++d) {
            // Create more varied data patterns
            float base = 1.0f + (i % 10) * 0.1f;
            float variation = std::sin(static_cast<float>(d) * 0.01f) * 0.3f;
            points_float[i * dimension + d] = base + variation + (d % 7) * 0.05f;
        }
    }
    
    for (size_t i = 0; i < centroid_count; ++i) {
        for (size_t d = 0; d < dimension; ++d) {
            // Different pattern for centroids
            float base = 0.5f + (i % 5) * 0.2f;
            float variation = std::cos(static_cast<float>(d) * 0.02f) * 0.4f;
            centroids_float[i * dimension + d] = base + variation + (d % 11) * 0.03f;
        }
    }

    // Normalize all vectors to unit length
    std::cout << "Normalizing vectors..." << std::endl;
    
    // Normalize data points
    for (size_t i = 0; i < point_count; ++i) {
        float* vec = &points_float[i * dimension];
        float norm = 0.0f;
        
        // Calculate L2 norm
        for (size_t d = 0; d < dimension; ++d) {
            norm += vec[d] * vec[d];
        }
        norm = std::sqrt(norm);
        
        // Normalize to unit length (avoid division by zero)
        if (norm > 1e-10f) {
            for (size_t d = 0; d < dimension; ++d) {
                vec[d] /= norm;
            }
        }
    }
    
    // Normalize centroids
    for (size_t i = 0; i < centroid_count; ++i) {
        float* vec = &centroids_float[i * dimension];
        float norm = 0.0f;
        
        // Calculate L2 norm
        for (size_t d = 0; d < dimension; ++d) {
            norm += vec[d] * vec[d];
        }
        norm = std::sqrt(norm);
        
        // Normalize to unit length (avoid division by zero)
        if (norm > 1e-10f) {
            for (size_t d = 0; d < dimension; ++d) {
                vec[d] /= norm;
            }
        }
    }
    
    // Verify normalization (check a few vectors)
    float sample_norm = 0.0f;
    for (size_t d = 0; d < dimension; ++d) {
        sample_norm += points_float[d] * points_float[d];
    }
    std::cout << "Sample point norm: " << std::sqrt(sample_norm) << " (should be ~1.0)" << std::endl;
    
    sample_norm = 0.0f;
    for (size_t d = 0; d < dimension; ++d) {
        sample_norm += centroids_float[d] * centroids_float[d];
    }
    std::cout << "Sample centroid norm: " << std::sqrt(sample_norm) << " (should be ~1.0)" << std::endl;

    // Convert to bfloat16
    std::vector<uint16_t> points_bf16(point_count * dimension);
    std::vector<uint16_t> centroids_bf16(centroid_count * dimension);
    
    for (size_t i = 0; i < points_float.size(); ++i) {
        points_bf16[i] = float_to_bf16(points_float[i]);
    }
    
    for (size_t i = 0; i < centroids_float.size(); ++i) {
        centroids_bf16[i] = float_to_bf16(centroids_float[i]);
    }

    // Result arrays
    std::vector<float> scalar_results(point_count * centroid_count);
    std::vector<float> single_amx_results(point_count * centroid_count);
    std::vector<float> multi_amx_results(point_count * centroid_count);

    // 1. Scalar reference
    std::cout << "\n1. Computing scalar reference..." << std::endl;
    compute(points_float.data(), point_count, centroids_float.data(), 
            centroid_count, dimension, scalar_results.data());
    std::cout << "   Scalar complete" << std::endl;

    // 2. Single-threaded AMX
    std::cout << "\n2. Computing single-threaded AMX..." << std::endl;
    AMXInnerProductBF16Ptr single_amx;
    if (!single_amx.initialize()) {
        std::cout << "   AMX not available" << std::endl;
        return 0;
    }
    
    single_amx.compute_inner_products(
        reinterpret_cast<const bfloat16_t*>(points_bf16.data()), point_count,
        reinterpret_cast<const bfloat16_t*>(centroids_bf16.data()), centroid_count,
        dimension, single_amx_results.data());
    std::cout << "   Single AMX complete" << std::endl;

    // 3. Multi-threaded AMX
    std::cout << "\n3. Computing multi-threaded AMX..." << std::endl;
    AMXInnerProductBF16PtrMTSimple multi_amx(2); // Use 2 threads
    if (!multi_amx.initialize()) {
        std::cout << "   Multi AMX initialization failed" << std::endl;
        return 1;
    }
    
    multi_amx.compute_inner_products(
        reinterpret_cast<const bfloat16_t*>(points_bf16.data()), point_count,
        reinterpret_cast<const bfloat16_t*>(centroids_bf16.data()), centroid_count,
        dimension, multi_amx_results.data());
    std::cout << "   Multi AMX complete" << std::endl;

    // 4. Compare results
    std::cout << "\n4. Accuracy Analysis:" << std::endl;
    
    // Compare Single AMX vs Scalar
    float max_single_diff = 0.0f;
    for (size_t i = 0; i < scalar_results.size(); ++i) {
        float diff = std::abs(scalar_results[i] - single_amx_results[i]);
        max_single_diff = std::max(max_single_diff, diff);
    }
    std::cout << "   Single AMX vs Scalar max diff: " << std::scientific << std::setprecision(3) << max_single_diff << std::endl;
    
    // Compare Multi AMX vs Single AMX (should be nearly identical)
    float max_multi_diff = 0.0f;
    size_t mismatch_count = 0;
    for (size_t i = 0; i < single_amx_results.size(); ++i) {
        float diff = std::abs(single_amx_results[i] - multi_amx_results[i]);
        max_multi_diff = std::max(max_multi_diff, diff);
        if (diff > 1e-6f) {
            mismatch_count++;
        }
    }
    std::cout << "   Multi AMX vs Single AMX max diff: " << std::scientific << std::setprecision(3) << max_multi_diff << std::endl;
    std::cout << "   Mismatches > 1e-6: " << mismatch_count << " / " << single_amx_results.size() << std::endl;
    
    // Compare Multi AMX vs Scalar
    float max_multi_scalar_diff = 0.0f;
    for (size_t i = 0; i < scalar_results.size(); ++i) {
        float diff = std::abs(scalar_results[i] - multi_amx_results[i]);
        max_multi_scalar_diff = std::max(max_multi_scalar_diff, diff);
    }
    std::cout << "   Multi AMX vs Scalar max diff: " << std::scientific << std::setprecision(3) << max_multi_scalar_diff << std::endl;

    // Show sample values for debugging
    std::cout << "\n5. Sample Values (first 5 results):" << std::endl;
    std::cout << "Index | Scalar      | Single AMX  | Multi AMX   | S-M Diff" << std::endl;
    std::cout << "------|-------------|-------------|-------------|----------" << std::endl;
    for (size_t i = 0; i < 5; ++i) {
        float sm_diff = std::abs(single_amx_results[i] - multi_amx_results[i]);
        std::cout << std::setw(5) << i << " | " 
                  << std::setw(11) << std::fixed << std::setprecision(4) << scalar_results[i] << " | "
                  << std::setw(11) << single_amx_results[i] << " | "
                  << std::setw(11) << multi_amx_results[i] << " | "
                  << std::setw(9) << std::scientific << std::setprecision(2) << sm_diff << std::endl;
    }

    // 6. Assessment
    std::cout << "\n6. Assessment:" << std::endl;
    if (max_multi_diff < 1e-5f) {
        std::cout << "   ✅ SUCCESS: Multi-threaded results match single-threaded" << std::endl;
        std::cout << "   The multithreading implementation is working correctly!" << std::endl;
    } else if (max_multi_diff < 1e-3f) {
        std::cout << "   ⚠️  ACCEPTABLE: Small differences likely due to BF16 precision" << std::endl;
        std::cout << "   This level of error is expected with BF16 operations." << std::endl;
    } else {
        std::cout << "   ❌ FAILED: Large differences indicate implementation bug" << std::endl;
        
        // Find the worst mismatch for debugging
        size_t worst_idx = 0;
        float worst_diff = 0.0f;
        for (size_t i = 0; i < single_amx_results.size(); ++i) {
            float diff = std::abs(single_amx_results[i] - multi_amx_results[i]);
            if (diff > worst_diff) {
                worst_diff = diff;
                worst_idx = i;
            }
        }
        
        size_t point_idx = worst_idx / centroid_count;
        size_t centroid_idx = worst_idx % centroid_count;
        
        std::cout << "   Worst mismatch at index " << worst_idx 
                  << " (point " << point_idx << ", centroid " << centroid_idx << ")" << std::endl;
        std::cout << "   Single: " << single_amx_results[worst_idx] 
                  << ", Multi: " << multi_amx_results[worst_idx] 
                  << ", Diff: " << worst_diff << std::endl;
                  
        // Additional debugging info
        std::cout << "\n   Debugging suggestions:" << std::endl;
        std::cout << "   - Check for race conditions in memory access" << std::endl;
        std::cout << "   - Verify AMX tile state isolation between threads" << std::endl;
        std::cout << "   - Check data alignment and boundary conditions" << std::endl;
        std::cout << "   - Verify index calculations for global vs local addressing" << std::endl;
    }

    // 7. Performance comparison
    std::cout << "\n7. Performance Notes:" << std::endl;
    std::cout << "   With normalized vectors, dot products represent cosine similarity" << std::endl;
    std::cout << "   Expected range: [-1.0, 1.0] for normalized vectors" << std::endl;
    std::cout << "   This is more realistic for embedding similarity computations" << std::endl;
    
    // Show the actual range of computed values
    float min_val = *std::min_element(scalar_results.begin(), scalar_results.end());
    float max_val = *std::max_element(scalar_results.begin(), scalar_results.end());
    std::cout << "   Actual scalar result range: [" << std::fixed << std::setprecision(4) 
              << min_val << ", " << max_val << "]" << std::endl;

    return 0;
}
