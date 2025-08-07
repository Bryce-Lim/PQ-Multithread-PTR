#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>

typedef uint16_t bfloat16_t;

// Configuration constants
const int dim = 1024;             // Embedding dimension - must be multiple of 64 for AMX
const int max_elements = 960000;   // Maximum number of vectors to generate
const int num_centroids = 32;    // Number of centroids - must be multiple of 16 for AMX
const int rounds = 3;             // Number of test rounds for averaging

// Validate AMX constraints
static_assert(dim % 64 == 0, "Dimension must be multiple of 64 for AMX");
static_assert(num_centroids % 16 == 0, "Number of centroids must be multiple of 16 for AMX");

// Convert float32 to bfloat16
static bfloat16_t float_to_bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    // Round to nearest even and truncate to bfloat16
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

// Convert bfloat16 to float32 (unused but kept for completeness)
[[maybe_unused]] static float bfloat16_to_float(bfloat16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// Generate realistic synthetic embedding data
void generate_embeddings(std::vector<float>& data, size_t count, size_t dimension, std::mt19937& gen) {
    data.resize(count * dimension);
    
    // Use different distributions to make it more realistic
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
    std::exponential_distribution<float> exp_dist(1.0f);
    
    for (size_t i = 0; i < count; ++i) {
        float* vec = &data[i * dimension];
        
        // Mix different distribution types to simulate real embeddings
        for (size_t d = 0; d < dimension; ++d) {
            if (d % 4 == 0) {
                vec[d] = normal_dist(gen);
            } else if (d % 4 == 1) {
                vec[d] = uniform_dist(gen);
            } else if (d % 4 == 2) {
                vec[d] = exp_dist(gen) - 1.0f; // Center around 0
            } else {
                vec[d] = std::sin(static_cast<float>(i + d)) * 0.5f; // Some correlation
            }
        }
        
        // Normalize the vector to unit length (common for embeddings)
        float norm = 0.0f;
        for (size_t d = 0; d < dimension; ++d) {
            norm += vec[d] * vec[d];
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-10f) {
            for (size_t d = 0; d < dimension; ++d) {
                vec[d] /= norm;
            }
        }
    }
}

// Simplified accuracy analysis
static void accuracyAnalyzer(const std::vector<float>& scalar_results, 
                            const std::vector<float>& comparison_results, 
                            const std::string& comparison_name) {
    if (scalar_results.size() != comparison_results.size()) {
        std::cout << "ERROR: Result size mismatch - Scalar: " << scalar_results.size() 
                  << ", " << comparison_name << ": " << comparison_results.size() << std::endl;
        return;
    }

    float max_abs_diff = 0.0f;
    float total_abs_diff = 0.0f;
    size_t significant_errors = 0;
    const float tolerance = 0.001f; // Absolute tolerance

    for (size_t i = 0; i < scalar_results.size(); ++i) {
        float abs_diff = std::abs(scalar_results[i] - comparison_results[i]);
        
        total_abs_diff += abs_diff;
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        
        if (abs_diff > tolerance) {
            significant_errors++;
        }
    }

    float avg_abs_diff = total_abs_diff / scalar_results.size();

    std::cout << "\n=== " << comparison_name << " ACCURACY ANALYSIS ===" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Max absolute difference:     " << std::setw(12) << max_abs_diff << std::endl;
    std::cout << "Average absolute difference: " << std::setw(12) << avg_abs_diff << std::endl;
    std::cout << "Values exceeding tolerance:  " << std::setw(8) << significant_errors 
              << " / " << scalar_results.size() << " (" << std::fixed << std::setprecision(2)
              << (100.0f * significant_errors / scalar_results.size()) << "%)" << std::endl;
    std::cout << "Tolerance threshold:         " << std::setw(12) << std::setprecision(6) << tolerance << std::endl;

    // Show sample of worst absolute differences
    std::vector<std::pair<float, size_t>> errors;
    for (size_t i = 0; i < scalar_results.size(); ++i) {
        float abs_diff = std::abs(scalar_results[i] - comparison_results[i]);
        errors.push_back({abs_diff, i});
    }
    
    std::partial_sort(errors.begin(), errors.begin() + std::min((size_t)10, errors.size()), 
                      errors.end(), std::greater<std::pair<float, size_t>>());

    std::cout << "\nWorst 10 absolute differences:" << std::endl;
    std::cout << "Index    | Scalar       | " << comparison_name << std::setw(12-comparison_name.length()) << " | Abs Diff" << std::endl;
    std::cout << "---------|--------------|--------------|----------" << std::endl;
    
    for (size_t i = 0; i < std::min((size_t)10, errors.size()); ++i) {
        size_t idx = errors[i].second;
        std::cout << std::setw(8) << idx << " | "
                  << std::setw(12) << std::fixed << std::setprecision(6) << scalar_results[idx] << " | "
                  << std::setw(12) << comparison_results[idx] << " | "
                  << std::setw(10) << std::setprecision(6) << errors[i].first << std::endl;
    }

    // Assessment
    if (avg_abs_diff > 0.001f) {
        std::cout << "\n⚠️  HIGH ACCURACY DIFFERENCE DETECTED!" << std::endl;
        if (comparison_name == "AMX") {
            std::cout << "This suggests potential issues with:" << std::endl;
            std::cout << "  - BF16 precision limitations" << std::endl;
            std::cout << "  - AMX implementation bugs" << std::endl;
            std::cout << "  - Data formatting/alignment issues" << std::endl;
        }
    } else {
        std::cout << "\n✅ " << comparison_name << " accuracy is acceptable" << std::endl;
    }
}

int main() {
    std::cout << "Large-Scale Synthetic AMX Inner Product Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimension: " << dim << std::endl;
    std::cout << "  Data points: " << max_elements << std::endl;
    std::cout << "  Centroids: " << num_centroids << std::endl;
    std::cout << "  Test rounds: " << rounds << std::endl;
    std::cout << "  Data type: Synthetic normalized embeddings" << std::endl << std::endl;

    // Verify AMX constraints
    std::cout << "AMX Constraint Check:" << std::endl;
    std::cout << "  Dimension divisible by 64: " << (dim % 64 == 0 ? "✅" : "❌") << std::endl;
    std::cout << "  Centroids divisible by 16: " << (num_centroids % 16 == 0 ? "✅" : "❌") << std::endl << std::endl;

    // Start Timer for Initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    std::cout << "Generating synthetic embedding data..." << std::endl;
    
    // Generate synthetic data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    
    std::vector<float> data_float;
    std::vector<float> centroids_float;
    
    generate_embeddings(data_float, max_elements, dim, gen);
    generate_embeddings(centroids_float, num_centroids, dim, gen);
    
    std::cout << "Generated " << max_elements << " data vectors" << std::endl;
    std::cout << "Generated " << num_centroids << " centroid vectors" << std::endl;

    // Convert to bfloat16
    std::cout << "Converting to bfloat16 format..." << std::endl;
    
    // Convert data points
    std::vector<bfloat16_t> data_bf16_flat(data_float.size());
    for (size_t i = 0; i < data_float.size(); ++i) {
        data_bf16_flat[i] = float_to_bfloat16(data_float[i]);
    }
    
    // Convert centroids
    std::vector<bfloat16_t> centroids_bf16_flat(centroids_float.size());
    for (size_t i = 0; i < centroids_float.size(); ++i) {
        centroids_bf16_flat[i] = float_to_bfloat16(centroids_float[i]);
    }

    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
    
    std::cout << "Data generation completed in " << init_duration.count() << " ms" << std::endl;
    std::cout << "Data size: " << max_elements << " points × " << dim << " dimensions" << std::endl;
    std::cout << "Centroids: " << num_centroids << " × " << dim << " dimensions" << std::endl;
    std::cout << "Total inner products to compute: " << max_elements * num_centroids << std::endl << std::endl;

    // Prepare result arrays
    size_t result_size = max_elements * num_centroids;
    std::vector<float> scalar_results(result_size);
    std::vector<float> amx_results(result_size);

    // ===== SCALAR COMPUTATION =====
    std::cout << "=== SCALAR COMPUTATION ===" << std::endl;
    
    long scalar_total_time = 0;
    for (int round = 0; round < rounds; ++round) {
        std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t result = compute(data_float.data(), max_elements,
                               centroids_float.data(), num_centroids,
                               dim, scalar_results.data());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        scalar_total_time += duration.count();
        
        std::cout << "  Computed " << result << " distances in " << duration.count() << " μs" << std::endl;
    }

    long avg_scalar_time = scalar_total_time / rounds;
    std::cout << "Scalar average time: " << avg_scalar_time << " μs" << std::endl;

    // ===== AMX COMPUTATION =====
    std::cout << "\n=== AMX COMPUTATION ===" << std::endl;
    
    AMXInnerProductBF16Ptr amx_calc;
    
    // Initialize AMX
    if (!amx_calc.initialize()) {
        std::cout << "❌ AMX initialization failed!" << std::endl;
        std::cout << "AMX requires Intel 4th Gen Xeon Scalable processors or newer" << std::endl;
        std::cout << "Continuing with scalar-only analysis..." << std::endl;
        
        // Show scalar results only
        std::cout << "\n=== SCALAR PERFORMANCE ===" << std::endl;
        std::cout << "Average execution time: " << avg_scalar_time << " μs" << std::endl;
        
        long long total_ops = static_cast<long long>(max_elements) * num_centroids * dim;
        double scalar_gflops = (total_ops * 2.0) / (avg_scalar_time * 1e-6) / 1e9;
        std::cout << "Scalar throughput: " << std::fixed << std::setprecision(2) 
                  << scalar_gflops << " GFLOPS" << std::endl;
        
        return 0;
    }
    
    std::cout << "✅ AMX initialized successfully" << std::endl;
    
    long amx_total_time = 0;
    bool amx_success = true;
    
    for (int round = 0; round < rounds && amx_success; ++round) {
        std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;
        
        amx_calc.reset_timers();
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            size_t result = amx_calc.compute_inner_products(
                data_bf16_flat.data(), max_elements,
                centroids_bf16_flat.data(), num_centroids,
                dim, amx_results.data());
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            amx_total_time += duration.count();
            
            std::cout << "  Computed " << result << " distances in " << duration.count() << " μs" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "❌ AMX computation failed: " << e.what() << std::endl;
            amx_success = false;
        }
    }

    if (!amx_success) {
        std::cout << "AMX computation failed, showing scalar-only results" << std::endl;
        return -1;
    }

    long avg_amx_time = amx_total_time / rounds;
    std::cout << "AMX average time: " << avg_amx_time << " μs" << std::endl;

    // ===== PERFORMANCE COMPARISON =====
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                         PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Scalar average runtime:           " << std::setw(15) << avg_scalar_time << " μs" << std::endl;
    std::cout << "AMX average runtime:              " << std::setw(15) << avg_amx_time << " μs" << std::endl;

    double speedup = static_cast<double>(avg_scalar_time) / avg_amx_time;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nSpeedup (Scalar/AMX):             " << std::setw(15) << speedup << "x" << std::endl;

    if (speedup > 1.0) {
        std::cout << "✅ AMX is " << speedup << "x faster than scalar" << std::endl;
    } else {
        std::cout << "⚠️  Scalar is " << (1.0/speedup) << "x faster than AMX" << std::endl;
    }

    // Calculate throughput
    long long total_ops = static_cast<long long>(max_elements) * num_centroids * dim;
    
    std::cout << "\n=== THROUGHPUT (GFLOPS) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    double scalar_gflops = (total_ops * 2.0) / (avg_scalar_time * 1e-6) / 1e9;
    double amx_gflops = (total_ops * 2.0) / (avg_amx_time * 1e-6) / 1e9;
    
    std::cout << "Scalar throughput:                " << std::setw(15) << scalar_gflops << " GFLOPS" << std::endl;
    std::cout << "AMX throughput:                   " << std::setw(15) << amx_gflops << " GFLOPS" << std::endl;
    std::cout << "Throughput improvement:           " << std::setw(15) << (amx_gflops/scalar_gflops) << "x" << std::endl;

    // Print detailed AMX timing breakdown
    std::cout << "\n=== AMX DETAILED TIMING ===" << std::endl;
    amx_calc.print_timing_stats();

    // ===== ACCURACY COMPARISON =====
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                         ACCURACY ANALYSIS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    accuracyAnalyzer(scalar_results, amx_results, "AMX");

    // ===== SUMMARY =====
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                              SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "Problem size:                     " << max_elements << " × " << num_centroids 
              << " × " << dim << std::endl;
    std::cout << "Total operations:                 " << total_ops << std::endl;
    std::cout << "AMX speedup:                      " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "AMX throughput gain:              " << std::setprecision(2) << (amx_gflops/scalar_gflops) << "x" << std::endl;
    
    // Calculate average absolute difference for summary
    float total_abs_diff = 0.0f;
    for (size_t i = 0; i < scalar_results.size(); ++i) {
        total_abs_diff += std::abs(scalar_results[i] - amx_results[i]);
    }
    float avg_abs_diff = total_abs_diff / scalar_results.size();
    
    std::cout << "Average accuracy difference:      " << std::fixed << std::setprecision(6) 
              << avg_abs_diff << " (absolute)" << std::endl;

    std::cout << std::string(80, '=') << std::endl;
    
    // Data characteristics summary
    std::cout << "\n=== DATA CHARACTERISTICS ===" << std::endl;
    std::cout << "Synthetic data includes:" << std::endl;
    std::cout << "  • Normal distribution components" << std::endl;
    std::cout << "  • Uniform distribution components" << std::endl;
    std::cout << "  • Exponential distribution components" << std::endl;
    std::cout << "  • Correlated sinusoidal components" << std::endl;
    std::cout << "  • All vectors normalized to unit length" << std::endl;
    std::cout << "  • Fixed random seed for reproducibility" << std::endl;

    return 0;
}
