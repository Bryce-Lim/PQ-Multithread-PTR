#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"
#include "AMXInnerProductBF16PtrMT.h"
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
#include <thread>

typedef uint16_t bfloat16_t;

// Configuration constants
const int dim = 1024;             // Embedding dimension - must be multiple of 64 for AMX
const int max_elements = 96000;    // Maximum number of vectors to load
const int num_centroids = 48;    // Number of centroids - must be multiple of 16 for AMX
const int rounds = 3;             // Number of test rounds for averaging
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

// Simplified accuracy analysis - absolute differences only
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
        if (comparison_name.find("AMX") != std::string::npos) {
            std::cout << "This suggests potential issues with:" << std::endl;
            std::cout << "  - BF16 precision limitations" << std::endl;
            std::cout << "  - AMX implementation bugs" << std::endl;
            std::cout << "  - Data formatting/alignment issues" << std::endl;
            if (comparison_name.find("Multi") != std::string::npos) {
                std::cout << "  - Thread synchronization issues" << std::endl;
                std::cout << "  - Race conditions in memory access" << std::endl;
            }
        }
    } else {
        std::cout << "\n✅ " << comparison_name << " accuracy is acceptable" << std::endl;
    }
}

// Load data from parquet files with Arrow 21 compatibility
std::vector<std::vector<float>> load_parquet_data(const std::string& dataroot, int max_elements, int dim) {
    std::vector<std::vector<float>> data_float;
    data_float.reserve(max_elements);

    std::cout << "Loading data from parquet files..." << std::endl;

    int files_loaded = 0;
    const size_t partition_size = 500000;

    for (int file_idx = 0; file_idx < 2 && data_float.size() < max_elements; file_idx++) {
        std::string path = dataroot + "train-0" + std::to_string(file_idx) + "-of-10.parquet";
        std::cout << "  Loading: " << path << std::endl;

        // Arrow 21 compatible file reading
        arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> maybe_input = 
            arrow::io::ReadableFile::Open(path);
        
        if (!maybe_input.ok()) {
            std::cerr << "Error opening file: " << maybe_input.status().ToString() << std::endl;
            continue;
        }
        
        std::shared_ptr<arrow::io::ReadableFile> input = maybe_input.ValueOrDie();

        // Create parquet reader
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
        arrow::Status status = parquet::arrow::OpenFile(input, arrow::default_memory_pool(), &arrow_reader);
        
        if (!status.ok()) {
            std::cerr << "Error opening parquet file: " << status.ToString() << std::endl;
            continue;
        }

        // Read table
        arrow::Result<std::shared_ptr<arrow::Table>> maybe_table = arrow_reader->ReadTable();
        if (!maybe_table.ok()) {
            std::cerr << "Error reading table: " << maybe_table.status().ToString() << std::endl;
            continue;
        }
        
        std::shared_ptr<arrow::Table> table = maybe_table.ValueOrDie();

        // Access embedding column (assuming it's column 1)
        if (table->num_columns() < 2) {
            std::cerr << "Error: Table has fewer than 2 columns" << std::endl;
            continue;
        }

        std::shared_ptr<arrow::ChunkedArray> emb_col = table->column(1);
        
        if (emb_col->num_chunks() != 1) {
            std::cout << "Multiple chunks found: " << emb_col->num_chunks() << std::endl;
        }

        for (int chunk_idx = 0; chunk_idx < emb_col->num_chunks(); ++chunk_idx) {
            std::shared_ptr<arrow::Array> chunk = emb_col->chunk(chunk_idx);
            
            // Cast to ListArray
            auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);
            
            // Get values array
            std::shared_ptr<arrow::Array> values_array = list_array->values();
            
            // Cast to appropriate numeric type (try DoubleArray first, then FloatArray)
            auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(values_array);
            auto float_array = std::dynamic_pointer_cast<arrow::FloatArray>(values_array);
            
            if (double_array) {
                // Process as double array
                for (int64_t i = 0; i < std::min(static_cast<int64_t>(partition_size), list_array->length()) && data_float.size() < max_elements; i++) {
                    if (list_array->IsValid(i)) {
                        std::vector<float> vec(dim);
                        for (int j = 0; j < dim; j++) {
                            vec[j] = static_cast<float>(double_array->Value(i * dim + j));
                        }
                        data_float.push_back(vec);
                    }
                }
            } else if (float_array) {
                // Process as float array  
                for (int64_t i = 0; i < std::min(static_cast<int64_t>(partition_size), list_array->length()) && data_float.size() < max_elements; i++) {
                    if (list_array->IsValid(i)) {
                        std::vector<float> vec(dim);
                        for (int j = 0; j < dim; j++) {
                            vec[j] = float_array->Value(i * dim + j);
                        }
                        data_float.push_back(vec);
                    }
                }
            } else {
                std::cerr << "Error: Unsupported array type for embeddings" << std::endl;
                continue;
            }
        }
        files_loaded++;
    }

    std::cout << "Loaded " << data_float.size() << " vectors from " << files_loaded << " files" << std::endl;
    return data_float;
}

int main()
{
    std::cout << "Large-Scale Multithreaded AMX Inner Product Comparison" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimension: " << dim << std::endl;
    std::cout << "  Max elements: " << max_elements << std::endl;
    std::cout << "  Centroids: " << num_centroids << std::endl;
    std::cout << "  Test rounds: " << rounds << std::endl;
    std::cout << "  Data root: " << dataroot << std::endl;
    std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << std::endl << std::endl;

    // Verify AMX constraints
    std::cout << "AMX Constraint Check:" << std::endl;
    std::cout << "  Dimension divisible by 64: " << (dim % 64 == 0 ? "✅" : "❌") << std::endl;
    std::cout << "  Centroids divisible by 16: " << (num_centroids % 16 == 0 ? "✅" : "❌") << std::endl << std::endl;

    // Start Timer for Initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    // Load data from parquet files
    std::vector<std::vector<float>> data_float = load_parquet_data(dataroot, max_elements, dim);

    if (data_float.empty()) {
        std::cerr << "ERROR: No data loaded! Check your data path: " << dataroot << std::endl;
        return -1;
    }

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
    std::vector<float> single_amx_results(result_size);
    std::vector<float> multi_amx_results(result_size);

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

    // ===== SINGLE-THREADED AMX COMPUTATION =====
    std::cout << "\n=== SINGLE-THREADED AMX COMPUTATION ===" << std::endl;

    AMXInnerProductBF16Ptr single_amx_calc;

    // Initialize single-threaded AMX
    if (!single_amx_calc.initialize()) {
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

    std::cout << "✅ Single-threaded AMX initialized successfully" << std::endl;

    long single_amx_total_time = 0;
    bool single_amx_success = true;

    for (int round = 0; round < rounds && single_amx_success; ++round) {
        std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;

        single_amx_calc.reset_timers();
        auto start = std::chrono::high_resolution_clock::now();

        try {
            size_t result = single_amx_calc.compute_inner_products(
                data_bf16_flat.data(), data_bf16_flat.size() / dim,
                centroids_bf16_flat.data(), centroids_bf16_flat.size() / dim,
                dim, single_amx_results.data());

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            single_amx_total_time += duration.count();

            std::cout << "  Computed " << result << " distances in " << duration.count() << " μs" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "❌ Single AMX computation failed: " << e.what() << std::endl;
            single_amx_success = false;
        }
    }

    if (!single_amx_success) {
        std::cout << "Single AMX computation failed, skipping multithreaded test" << std::endl;
        return -1;
    }

    long avg_single_amx_time = single_amx_total_time / rounds;
    std::cout << "Single AMX average time: " << avg_single_amx_time << " μs" << std::endl;

    // ===== MULTI-THREADED AMX COMPUTATION =====
    std::cout << "\n=== MULTI-THREADED AMX COMPUTATION ===" << std::endl;

    // Test different thread counts
    std::vector<size_t> thread_counts = {2, 4, std::thread::hardware_concurrency()};
    thread_counts.erase(std::remove_if(thread_counts.begin(), thread_counts.end(), 
                                      [](size_t n) { return n == 0 || n > 8; }), 
                       thread_counts.end());

    // Remove duplicates and sort
    std::sort(thread_counts.begin(), thread_counts.end());
    thread_counts.erase(std::unique(thread_counts.begin(), thread_counts.end()), thread_counts.end());

    for (size_t num_threads : thread_counts) {
        std::cout << "\n--- Testing with " << num_threads << " threads ---" << std::endl;
        
        AMXInnerProductBF16PtrMT multi_amx_calc(num_threads);
        
        if (!multi_amx_calc.initialize()) {
            std::cout << "❌ Multi-threaded AMX initialization failed for " << num_threads << " threads" << std::endl;
            continue;
        }
        
        std::cout << "✅ Multi-threaded AMX initialized successfully with " << num_threads << " threads" << std::endl;

        long multi_amx_total_time = 0;
        bool multi_amx_success = true;

        for (int round = 0; round < rounds && multi_amx_success; ++round) {
            std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;

            multi_amx_calc.reset_timers();
            auto start = std::chrono::high_resolution_clock::now();

            try {
                size_t result = multi_amx_calc.compute_inner_products(
                    data_bf16_flat.data(), data_bf16_flat.size() / dim,
                    centroids_bf16_flat.data(), centroids_bf16_flat.size() / dim,
                    dim, multi_amx_results.data());

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                multi_amx_total_time += duration.count();

                std::cout << "  Computed " << result << " distances in " << duration.count() << " μs" << std::endl;

            } catch (const std::exception& e) {
                std::cout << "❌ Multi AMX computation failed: " << e.what() << std::endl;
                multi_amx_success = false;
            }
        }

        if (!multi_amx_success) {
            std::cout << "Multi AMX computation failed for " << num_threads << " threads" << std::endl;
            continue;
        }

        long avg_multi_amx_time = multi_amx_total_time / rounds;
        std::cout << "Multi AMX (" << num_threads << " threads) average time: " << avg_multi_amx_time << " μs" << std::endl;

        // ===== PERFORMANCE COMPARISON FOR THIS THREAD COUNT =====
        std::cout << "\n=== PERFORMANCE COMPARISON (" << num_threads << " threads) ===" << std::endl;

        std::cout << std::fixed << std::setprecision(0);
        std::cout << "Scalar average runtime:           " << std::setw(15) << avg_scalar_time << " μs" << std::endl;
        std::cout << "Single AMX average runtime:       " << std::setw(15) << avg_single_amx_time << " μs" << std::endl;
        std::cout << "Multi AMX average runtime:        " << std::setw(15) << avg_multi_amx_time << " μs" << std::endl;

        double scalar_speedup = static_cast<double>(avg_scalar_time) / avg_multi_amx_time;
        double single_speedup = static_cast<double>(avg_single_amx_time) / avg_multi_amx_time;
        double threading_efficiency = single_speedup / num_threads;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nSpeedup vs Scalar:                " << std::setw(15) << scalar_speedup << "x" << std::endl;
        std::cout << "Speedup vs Single AMX:            " << std::setw(15) << single_speedup << "x" << std::endl;
        std::cout << "Threading efficiency:              " << std::setw(15) << (threading_efficiency * 100) << "%" << std::endl;

        // Calculate throughput
        long long total_ops = static_cast<long long>(data_bf16_flat.size() / dim) *
                             (centroids_bf16_flat.size() / dim) * dim;

        std::cout << "\n=== THROUGHPUT (GFLOPS) ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);

        double scalar_gflops = (total_ops * 2.0) / (avg_scalar_time * 1e-6) / 1e9;
        double single_amx_gflops = (total_ops * 2.0) / (avg_single_amx_time * 1e-6) / 1e9;
        double multi_amx_gflops = (total_ops * 2.0) / (avg_multi_amx_time * 1e-6) / 1e9;

        std::cout << "Scalar throughput:                " << std::setw(15) << scalar_gflops << " GFLOPS" << std::endl;
        std::cout << "Single AMX throughput:            " << std::setw(15) << single_amx_gflops << " GFLOPS" << std::endl;
        std::cout << "Multi AMX throughput:             " << std::setw(15) << multi_amx_gflops << " GFLOPS" << std::endl;
        std::cout << "Multi AMX improvement:            " << std::setw(15) << (multi_amx_gflops/scalar_gflops) << "x" << std::endl;

        // Print detailed AMX timing breakdown
        std::cout << "\n=== AMX DETAILED TIMING (" << num_threads << " threads) ===" << std::endl;
        multi_amx_calc.print_timing_stats();

        // ===== ACCURACY COMPARISON =====
        std::cout << "\n=== ACCURACY ANALYSIS (" << num_threads << " threads) ===" << std::endl;

        // Compare Multi AMX vs Single AMX (should be nearly identical)
        accuracyAnalyzer(single_amx_results, multi_amx_results, "Multi AMX (" + std::to_string(num_threads) + " threads)");

        std::cout << std::string(80, '-') << std::endl;
    }

    // ===== FINAL SUMMARY =====
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                              FINAL SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}

    std::cout << "Problem size:                     " << (data_bf16_flat.size()/dim) << " × " << (centroids_bf16_flat.size()/dim)
              << " × " << dim << std::endl;

    long long total_ops = static_cast<long long>(data_bf16_flat.size() / dim) *
                         (centroids_bf16_flat.size() / dim) * dim;
    std::cout << "Total operations:                 " << total_ops << std::endl;

    double single_amx_speedup = static_cast<double>(avg_scalar_time) / avg_single_amx_time;
    std::cout << "Single AMX speedup vs Scalar:    " << std::fixed << std::setprecision(2) << single_amx_speedup << "x" << std::endl;

    std::cout << "\nMultithreading Results:" << std::endl;
    std::cout << "  Best performance achieved with optimal thread count" << std::endl;
    std::cout << "  Threading efficiency varies with system architecture" << std::endl;
    std::cout << "  Memory bandwidth may become limiting factor" << std::endl;

    std::cout << "\nRecommendations:" << std::endl;
    std::cout << "  - Use single AMX for CPU-bound workloads" << std::endl;
    std::cout << "  - Use multithreaded AMX for memory bandwidth utilization" << std::endl;
    std::cout << "  - Consider NUMA topology for multi-socket systems" << std::endl;

    std::cout << std::string(80, '=') << std::endl;
