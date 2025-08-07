#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"
#include "arrow/api.h"
#include "arrow/io/api.h"
#include "arrow/ipc/api.h"
#include "parquet/arrow/reader.h"
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
const int max_elements = 96000;    // Maximum number of vectors to load
const int num_centroids = 16;    // Number of centroids - must be multiple of 16 for AMX
const int rounds = 1;             // Number of test rounds for averaging
const std::string dataroot = "/mnt/ceph/district9/dataset/openai/openai_large_5m/";

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

// Convert bfloat16 to float32
static float bfloat16_to_float(bfloat16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// Comprehensive accuracy analysis
static void accuracyAnalyzer(const std::vector<float>& scalar_results, 
                            const std::vector<float>& comparison_results, 
                            const std::string& comparison_name)
{
    if (scalar_results.size() != comparison_results.size()) {
        std::cout << "ERROR: Result size mismatch - Scalar: " << scalar_results.size() 
                  << ", " << comparison_name << ": " << comparison_results.size() << std::endl;
        return;
    }

    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float total_abs_diff = 0.0f;
    float total_rel_diff = 0.0f;
    size_t valid_comparisons = 0;
    size_t significant_errors = 0;
    size_t zero_scalar_count = 0;
    const float tolerance = 0.001f; // 0.1% tolerance
    const float zero_threshold = 1e-4f; // Consider values below this as effectively zero

    for (size_t i = 0; i < scalar_results.size(); ++i) {
        float abs_diff = std::abs(scalar_results[i] - comparison_results[i]);
        
        total_abs_diff += abs_diff;
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        
        // Only calculate relative difference if scalar value is significantly non-zero
        if (std::abs(scalar_results[i]) > zero_threshold) {
            float rel_diff = abs_diff / std::abs(scalar_results[i]);
            total_rel_diff += rel_diff;
            max_rel_diff = std::max(max_rel_diff, rel_diff);
            valid_comparisons++;
            
            if (rel_diff > tolerance) {
                significant_errors++;
            }
        } else {
            zero_scalar_count++;
        }
    }

    float avg_abs_diff = total_abs_diff / scalar_results.size();
    float avg_rel_diff = (valid_comparisons > 0) ? (total_rel_diff / valid_comparisons) : 0.0f;

    std::cout << "\n=== " << comparison_name << " ACCURACY ANALYSIS ===" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Max absolute difference:     " << std::setw(12) << max_abs_diff << std::endl;
    std::cout << "Average absolute difference: " << std::setw(12) << avg_abs_diff << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Max relative difference:     " << std::setw(12) << (max_rel_diff * 100) << "%" << std::endl;
    std::cout << "Average relative difference: " << std::setw(12) << (avg_rel_diff * 100) << "%" << std::endl;
    std::cout << "Significant errors (>" << (tolerance * 100) << "%): " << std::setw(8) << significant_errors 
              << " / " << scalar_results.size() << " (" << std::fixed << std::setprecision(2)
              << (100.0f * significant_errors / scalar_results.size()) << "%)" << std::endl;

    // Show sample of worst differences (excluding zero scalar values)
    std::vector<std::pair<float, size_t>> errors;
    for (size_t i = 0; i < scalar_results.size(); ++i) {
        if (std::abs(scalar_results[i]) > zero_threshold) {
            float rel_diff = std::abs(scalar_results[i] - comparison_results[i]) / std::abs(scalar_results[i]);
            errors.push_back({rel_diff, i});
        }
    }
    
    if (!errors.empty()) {
        std::partial_sort(errors.begin(), errors.begin() + std::min((size_t)5, errors.size()), 
                          errors.end(), std::greater<std::pair<float, size_t>>());

        std::cout << "\nWorst 5 relative errors (excluding near-zero scalar values):" << std::endl;
        std::cout << "Index    | Scalar       | " << comparison_name << std::setw(12-comparison_name.length()) << " | Rel Diff %" << std::endl;
        std::cout << "---------|--------------|--------------|----------" << std::endl;
        
        for (size_t i = 0; i < std::min((size_t)5, errors.size()); ++i) {
            size_t idx = errors[i].second;
            std::cout << std::setw(8) << idx << " | "
                      << std::setw(12) << std::fixed << std::setprecision(6) << scalar_results[idx] << " | "
                      << std::setw(12) << comparison_results[idx] << " | "
                      << std::setw(8) << std::setprecision(4) << (errors[i].first * 100) << "%" << std::endl;
        }
    } else {
        std::cout << "\nNo valid relative error comparisons (all scalar values near zero)" << std::endl;
    }

    // Show worst absolute differences for zero scalar cases
    if (zero_scalar_count > 0) {
        std::vector<std::pair<float, size_t>> zero_errors;
        for (size_t i = 0; i < scalar_results.size(); ++i) {
            if (std::abs(scalar_results[i]) <= zero_threshold) {
                float abs_diff = std::abs(comparison_results[i]);
                zero_errors.push_back({abs_diff, i});
            }
        }
        
        std::partial_sort(zero_errors.begin(), zero_errors.begin() + std::min((size_t)5, zero_errors.size()), 
                          zero_errors.end(), std::greater<std::pair<float, size_t>>());

        std::cout << "\nWorst 5 absolute errors for near-zero scalar values:" << std::endl;
        std::cout << "Index    | Scalar       | " << comparison_name << std::setw(12-comparison_name.length()) << " | Abs Diff" << std::endl;
        std::cout << "---------|--------------|--------------|----------" << std::endl;
        
        for (size_t i = 0; i < std::min((size_t)5, zero_errors.size()); ++i) {
            size_t idx = zero_errors[i].second;
            std::cout << std::setw(8) << idx << " | "
                      << std::setw(12) << std::fixed << std::setprecision(6) << scalar_results[idx] << " | "
                      << std::setw(12) << comparison_results[idx] << " | "
                      << std::setw(10) << std::setprecision(6) << zero_errors[i].first << std::endl;
        }
    }

    // Assessment
    if (avg_rel_diff > 0.001f) { // > 0.1%
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

// Flatten 2D vector results to 1D for easier processing
std::vector<float> flatten_results(const std::vector<std::vector<float>>& results) {
    std::vector<float> flattened;
    flattened.reserve(results.size() * (results.empty() ? 0 : results[0].size()));
    
    for (const auto& row : results) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

int main()
{
    std::cout << "Large-Scale Scalar vs AMX Inner Product Comparison" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimension: " << dim << std::endl;
    std::cout << "  Max elements: " << max_elements << std::endl;
    std::cout << "  Centroids: " << num_centroids << std::endl;
    std::cout << "  Test rounds: " << rounds << std::endl;
    std::cout << "  Data root: " << dataroot << std::endl << std::endl;

    // Verify AMX constraints
    std::cout << "AMX Constraint Check:" << std::endl;
    std::cout << "  Dimension divisible by 64: " << (dim % 64 == 0 ? "✅" : "❌") << std::endl;
    std::cout << "  Centroids divisible by 16: " << (num_centroids % 16 == 0 ? "✅" : "❌") << std::endl << std::endl;

    // Start Timer for Initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    // Load data from parquet files
    std::vector<std::vector<float>> data_float;
    data_float.reserve(max_elements);

    std::cout << "Loading data from parquet files..." << std::endl;
    
    int files_loaded = 0;
    size_t partition_size = 500000;

    for (int file_idx = 0; file_idx < 2 && data_float.size() < max_elements; file_idx++)
    {
        auto pool = arrow::default_memory_pool();
        std::shared_ptr<arrow::io::RandomAccessFile> input;

        std::string path = dataroot + "train-0" + std::to_string(file_idx) + "-of-10.parquet";
        std::cout << "  Loading: " << path << std::endl;

        auto maybe_input = arrow::io::ReadableFile::Open(path);
        if (!maybe_input.ok()) {
            std::cerr << "Error opening file: " << maybe_input.status().ToString() << std::endl;
            continue;
        }
        input = maybe_input.ValueUnsafe();

        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
        if (!status.ok()) {
            std::cerr << "Error opening parquet file: " << status.ToString() << std::endl;
            continue;
        }

        std::shared_ptr<arrow::Table> table;
        status = arrow_reader->ReadTable(&table);
        if (!status.ok()) {
            std::cerr << "Error reading table: " << status.ToString() << std::endl;
            continue;
        }

        auto emb_col = table->column(1);
        if (emb_col->chunks().size() != 1) {
            std::cout << "Multiple chunks found: " << emb_col->chunks().size() << std::endl;
        }

        for (auto &arr : emb_col->chunks()) {
            auto val = std::static_pointer_cast<arrow::DoubleArray>(
                std::static_pointer_cast<arrow::ListArray>(arr)->values());

            for (size_t i = 0; i < partition_size && data_float.size() < max_elements; i++) {
                std::vector<float> vec(dim);
                for (int j = 0; j < dim; j++) {
                    vec[j] = (float)val->Value(i * dim + j);
                }
                data_float.push_back(vec);
            }
        }
        files_loaded++;
    }

    if (data_float.empty()) {
        std::cerr << "ERROR: No data loaded! Check your data path: " << dataroot << std::endl;
        return -1;
    }

    std::cout << "Loaded " << data_float.size() << " vectors from " << files_loaded << " files" << std::endl;

    // Normalize vectors
    std::cout << "Normalizing vectors..." << std::endl;
    for (auto &emb : data_float) {
        float mag = 0.0f;
        for (int d = 0; d < dim; d++) {
            mag += emb[d] * emb[d];
        }
        mag = std::sqrt(mag);

        if (mag > 1e-10f) {
            for (int d = 0; d < dim; d++) {
                emb[d] /= mag;
            }
        }
    }

    // Sample random centroids
    std::cout << "Sampling " << num_centroids << " random centroids..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::vector<std::vector<float>> centroids_float;
    std::sample(data_float.begin(), data_float.end(), 
                std::back_inserter(centroids_float), num_centroids, gen);

    // Convert to bfloat16
    std::cout << "Converting to bfloat16 format..." << std::endl;
    
    // Convert data points
    std::vector<bfloat16_t> data_bf16_flat(data_float.size() * dim);
    for (size_t i = 0; i < data_float.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            data_bf16_flat[i * dim + j] = float_to_bfloat16(data_float[i][j]);
        }
    }
    
    // Convert centroids
    std::vector<bfloat16_t> centroids_bf16_flat(num_centroids * dim);
    for (int i = 0; i < num_centroids; ++i) {
        for (int j = 0; j < dim; ++j) {
            centroids_bf16_flat[i * dim + j] = float_to_bfloat16(centroids_float[i][j]);
        }
    }

    // Keep float versions for scalar computation
    std::vector<float> data_float_flat(data_float.size() * dim);
    std::vector<float> centroids_float_flat(num_centroids * dim);
    
    for (size_t i = 0; i < data_float.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            data_float_flat[i * dim + j] = data_float[i][j];
        }
    }
    
    for (int i = 0; i < num_centroids; ++i) {
        for (int j = 0; j < dim; ++j) {
            centroids_float_flat[i * dim + j] = centroids_float[i][j];
        }
    }

    // Clean up 2D vectors to save memory
    data_float.clear();
    data_float.shrink_to_fit();
    centroids_float.clear();
    centroids_float.shrink_to_fit();

    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
    
    std::cout << "Initialization completed in " << init_duration.count() << " ms" << std::endl;
    std::cout << "Data size: " << data_bf16_flat.size()/dim << " points × " << dim << " dimensions" << std::endl;
    std::cout << "Centroids: " << centroids_bf16_flat.size()/dim << " × " << dim << " dimensions" << std::endl;
    std::cout << "Total inner products to compute: " << (data_bf16_flat.size()/dim) * (centroids_bf16_flat.size()/dim) << std::endl << std::endl;

    // Prepare result arrays
    size_t result_size = (data_bf16_flat.size() / dim) * (centroids_bf16_flat.size() / dim);
    std::vector<float> scalar_results(result_size);
    std::vector<float> amx_results(result_size);

    // ===== SCALAR COMPUTATION =====
    std::cout << "=== SCALAR COMPUTATION ===" << std::endl;
    
    long scalar_total_time = 0;
    for (int round = 0; round < rounds; ++round) {
        std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t result = compute(data_float_flat.data(), data_bf16_flat.size() / dim,
                               centroids_float_flat.data(), centroids_bf16_flat.size() / dim,
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
        
        long long total_ops = static_cast<long long>(data_bf16_flat.size() / dim) * 
                             (centroids_bf16_flat.size() / dim) * dim;
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
                data_bf16_flat.data(), data_bf16_flat.size() / dim,
                centroids_bf16_flat.data(), centroids_bf16_flat.size() / dim,
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
    long long total_ops = static_cast<long long>(data_bf16_flat.size() / dim) * 
                         (centroids_bf16_flat.size() / dim) * dim;
    
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
    
    std::cout << "Problem size:                     " << (data_bf16_flat.size()/dim) << " × " << (centroids_bf16_flat.size()/dim) 
              << " × " << dim << std::endl;
    std::cout << "Total operations:                 " << total_ops << std::endl;
    std::cout << "AMX speedup:                      " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "AMX throughput gain:              " << std::setprecision(2) << (amx_gflops/scalar_gflops) << "x" << std::endl;
    
    // Calculate average relative difference (excluding zero scalars)
    float total_rel_diff = 0.0f;
    size_t valid_comparisons = 0;
    for (size_t i = 0; i < scalar_results.size(); ++i) {
        if (std::abs(scalar_results[i]) > 1e-6f) {
            float rel_diff = std::abs(scalar_results[i] - amx_results[i]) / std::abs(scalar_results[i]);
            total_rel_diff += rel_diff;
            valid_comparisons++;
        }
    }
    float avg_rel_diff = (valid_comparisons > 0) ? (total_rel_diff / valid_comparisons) : 0.0f;
    
    std::cout << "Average accuracy difference:      " << std::fixed << std::setprecision(4) 
              << (avg_rel_diff * 100) << "%" << std::endl;

    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
