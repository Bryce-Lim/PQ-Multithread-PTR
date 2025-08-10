#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"
#include "AMXInnerProductBF16PtrMT.h"
#include "HNSWLIBInnerProductPtr.h"
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
#include <map>
#include <numeric>

typedef uint16_t bfloat16_t;

// Configuration constants
const int dim = 1024;             // Embedding dimension - must be multiple of 64 for AMX
const int max_elements = 10240;   // Maximum number of vectors to load
const int num_centroids = 16;     // Number of centroids - must be multiple of 16 for AMX
const int rounds = 10;              // Number of test rounds for averaging
const std::string dataroot = "/mnt/ceph/district9/dataset/openai/openai_large_5m/";

// Validate AMX constraints
static_assert(dim % 64 == 0, "Dimension must be multiple of 64 for AMX");
static_assert(num_centroids % 16 == 0, "Number of centroids must be multiple of 16 for AMX");

// Convert float32 to bfloat16
static bfloat16_t float_to_bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

// Performance metrics structure
struct PerformanceMetrics {
    std::string implementation_name;
    std::vector<double> execution_times_us;  // Microseconds
    double avg_time_us;
    double min_time_us;
    double max_time_us;
    double std_dev_us;
    double throughput_gflops;
    double speedup_vs_scalar;
    bool success;
    
    PerformanceMetrics(const std::string& name) : implementation_name(name), success(false) {}
    
    void calculate_stats(long long total_ops) {
        if (execution_times_us.empty()) {
            success = false;
            return;
        }
        
        success = true;
        avg_time_us = std::accumulate(execution_times_us.begin(), execution_times_us.end(), 0.0) / execution_times_us.size();
        min_time_us = *std::min_element(execution_times_us.begin(), execution_times_us.end());
        max_time_us = *std::max_element(execution_times_us.begin(), execution_times_us.end());
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : execution_times_us) {
            variance += (time - avg_time_us) * (time - avg_time_us);
        }
        std_dev_us = std::sqrt(variance / execution_times_us.size());
        
        // Calculate throughput (GFLOPS)
        throughput_gflops = (total_ops * 2.0) / (avg_time_us * 1e-6) / 1e9;
    }
    
    void set_speedup(double scalar_time_us) {
        speedup_vs_scalar = scalar_time_us / avg_time_us;
    }
};

// Accuracy analysis structure
struct AccuracyMetrics {
    float max_abs_diff;
    float avg_abs_diff;
    float std_dev_diff;
    size_t significant_errors;
    float tolerance;
    bool acceptable;
    
    AccuracyMetrics(float tol = 0.001f) : tolerance(tol), acceptable(false) {}
    
    void analyze(const std::vector<float>& reference, const std::vector<float>& comparison) {
        if (reference.size() != comparison.size()) {
            acceptable = false;
            return;
        }
        
        std::vector<float> abs_diffs;
        abs_diffs.reserve(reference.size());
        
        max_abs_diff = 0.0f;
        float total_abs_diff = 0.0f;
        significant_errors = 0;
        
        for (size_t i = 0; i < reference.size(); ++i) {
            float abs_diff = std::abs(reference[i] - comparison[i]);
            abs_diffs.push_back(abs_diff);
            
            total_abs_diff += abs_diff;
            max_abs_diff = std::max(max_abs_diff, abs_diff);
            
            if (abs_diff > tolerance) {
                significant_errors++;
            }
        }
        
        avg_abs_diff = total_abs_diff / reference.size();
        
        // Calculate standard deviation of differences
        float variance = 0.0f;
        for (float diff : abs_diffs) {
            variance += (diff - avg_abs_diff) * (diff - avg_abs_diff);
        }
        std_dev_diff = std::sqrt(variance / abs_diffs.size());
        
        // Determine if accuracy is acceptable
        acceptable = (avg_abs_diff <= tolerance && max_abs_diff <= tolerance * 10.0f);
    }
};

// Load data from parquet files with Arrow 21 compatibility
std::vector<std::vector<float>> load_parquet_data(const std::string& dataroot, int max_elements, int dim) {
    std::vector<std::vector<float>> data_float;
    data_float.reserve(static_cast<size_t>(max_elements));

    std::cout << "Loading data from parquet files..." << std::endl;

    int files_loaded = 0;
    const size_t partition_size = 500000;

    for (int file_idx = 0; file_idx < 4 && static_cast<int>(data_float.size()) < max_elements; file_idx++) {
        std::string path = dataroot + "train-0" + std::to_string(file_idx) + "-of-10.parquet";
        std::cout << "  Loading: " << path << std::flush;

        // Arrow 21 compatible file reading
        arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> maybe_input = 
            arrow::io::ReadableFile::Open(path);
        
        if (!maybe_input.ok()) {
            std::cerr << " - Error opening file: " << maybe_input.status().ToString() << std::endl;
            continue;
        }
        
        std::shared_ptr<arrow::io::ReadableFile> input = maybe_input.ValueOrDie();

        // Create parquet reader - Updated for Arrow 21
        arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> maybe_reader = 
            parquet::arrow::OpenFile(input, arrow::default_memory_pool());
        
        if (!maybe_reader.ok()) {
            std::cerr << " - Error opening parquet file: " << maybe_reader.status().ToString() << std::endl;
            continue;
        }
        
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader = std::move(maybe_reader).ValueOrDie();

        // Read table - Updated for Arrow 21
        std::shared_ptr<arrow::Table> table;
        arrow::Status status = arrow_reader->ReadTable(&table);
        if (!status.ok()) {
            std::cerr << " - Error reading table: " << status.ToString() << std::endl;
            continue;
        }

        // Access embedding column (assuming it's column 1)
        if (table->num_columns() < 2) {
            std::cerr << " - Error: Table has fewer than 2 columns" << std::endl;
            continue;
        }

        std::shared_ptr<arrow::ChunkedArray> emb_col = table->column(1);
        size_t vectors_from_file = 0;
        
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
                for (int64_t i = 0; i < std::min(static_cast<int64_t>(partition_size), list_array->length()) && static_cast<int>(data_float.size()) < max_elements; i++) {
                    if (list_array->IsValid(i)) {
                        std::vector<float> vec(static_cast<size_t>(dim));
                        for (int j = 0; j < dim; j++) {
                            vec[static_cast<size_t>(j)] = static_cast<float>(double_array->Value(i * dim + j));
                        }
                        data_float.push_back(vec);
                        vectors_from_file++;
                    }
                }
            } else if (float_array) {
                // Process as float array  
                for (int64_t i = 0; i < std::min(static_cast<int64_t>(partition_size), list_array->length()) && static_cast<int>(data_float.size()) < max_elements; i++) {
                    if (list_array->IsValid(i)) {
                        std::vector<float> vec(static_cast<size_t>(dim));
                        for (int j = 0; j < dim; j++) {
                            vec[static_cast<size_t>(j)] = float_array->Value(i * dim + j);
                        }
                        data_float.push_back(vec);
                        vectors_from_file++;
                    }
                }
            } else {
                std::cerr << " - Error: Unsupported array type for embeddings" << std::endl;
                continue;
            }
        }
        
        std::cout << " (" << vectors_from_file << " vectors)" << std::endl;
        files_loaded++;
    }

    std::cout << "Total loaded: " << data_float.size() << " vectors from " << files_loaded << " files" << std::endl;
    return data_float;
}

// Print performance summary table
void print_performance_table(const std::vector<PerformanceMetrics>& metrics, long long total_ops) {
    std::cout << "\n" << std::string(130, '=') << std::endl;
    std::cout << "                                    PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(130, '=') << std::endl;
    
    std::cout << std::left 
              << std::setw(30) << "Implementation"
              << std::setw(12) << "Avg Time"
              << std::setw(12) << "Min Time" 
              << std::setw(12) << "Max Time"
              << std::setw(12) << "Std Dev"
              << std::setw(15) << "Throughput"
              << std::setw(10) << "Speedup"
              << std::setw(8) << "Status" << std::endl;
              
    std::cout << std::string(30, '-') << " "
              << std::string(11, '-') << " "
              << std::string(11, '-') << " "
              << std::string(11, '-') << " "
              << std::string(11, '-') << " "
              << std::string(14, '-') << " "
              << std::string(9, '-') << " "
              << std::string(7, '-') << std::endl;
    
    for (const auto& metric : metrics) {
        if (!metric.success) {
            std::cout << std::left << std::setw(30) << metric.implementation_name
                      << std::setw(75) << "FAILED"
                      << std::setw(8) << "❌" << std::endl;
            continue;
        }
        
        std::cout << std::left << std::setw(30) << metric.implementation_name
                  << std::right << std::fixed << std::setprecision(0)
                  << std::setw(10) << metric.avg_time_us << "μs "
                  << std::setw(10) << metric.min_time_us << "μs "
                  << std::setw(10) << metric.max_time_us << "μs "
                  << std::setw(10) << metric.std_dev_us << "μs "
                  << std::setw(12) << std::setprecision(2) << metric.throughput_gflops << " GFLOPS "
                  << std::setw(8) << std::setprecision(2) << metric.speedup_vs_scalar << "x "
                  << std::left << std::setw(8) << "✅" << std::endl;
    }
    
    std::cout << std::string(130, '=') << std::endl;
}

// Print accuracy summary table
void print_accuracy_table(const std::map<std::string, AccuracyMetrics>& accuracy_map) {
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                                ACCURACY ANALYSIS" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    std::cout << std::left 
              << std::setw(30) << "Implementation"
              << std::setw(15) << "Max Abs Diff"
              << std::setw(15) << "Avg Abs Diff"
              << std::setw(15) << "Std Dev Diff"
              << std::setw(12) << "Sig Errors"
              << std::setw(8) << "Status" << std::endl;
              
    std::cout << std::string(30, '-') << " "
              << std::string(14, '-') << " "
              << std::string(14, '-') << " "
              << std::string(14, '-') << " "
              << std::string(11, '-') << " "
              << std::string(7, '-') << std::endl;
    
    for (const auto& pair : accuracy_map) {
        const std::string& name = pair.first;
        const AccuracyMetrics& acc = pair.second;
        
        std::cout << std::left << std::setw(30) << name
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(14) << acc.max_abs_diff << " "
                  << std::setw(14) << acc.avg_abs_diff << " "
                  << std::setw(14) << acc.std_dev_diff << " "
                  << std::setw(8) << acc.significant_errors << "/"
                  << std::left << std::setw(3) << "total "
                  << std::setw(8) << (acc.acceptable ? "✅" : "⚠️") << std::endl;
    }
    
    std::cout << std::string(100, '=') << std::endl;
}

int main()
{
    std::cout << "Comprehensive Implementation Comparison (Including HNSWLIB)" << std::endl;
    std::cout << "============================================================" << std::endl;
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
    std::cout << "  Centroids divisible by 16: " << (num_centroids % 16 == 0 ? "✅" : "❌") << std::endl;
    std::cout << "  Max elements divisible by 32: " << (max_elements % 32 == 0 ? "✅" : "❌") << std::endl << std::endl;

    // Data loading and preparation
    auto init_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<float>> data_float = load_parquet_data(dataroot, max_elements, dim);

    if (data_float.empty()) {
        std::cerr << "ERROR: No data loaded! Check your data path: " << dataroot << std::endl;
        return -1;
    }

    // Normalize vectors
    std::cout << "\nNormalizing vectors..." << std::endl;
    size_t normalized_count = 0;
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
            normalized_count++;
        }
    }
    std::cout << "  Normalized " << normalized_count << " vectors" << std::endl;

    // Sample random centroids
    std::cout << "Sampling " << num_centroids << " random centroids..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::vector<std::vector<float>> centroids_float;
    std::sample(data_float.begin(), data_float.end(),
                std::back_inserter(centroids_float), num_centroids, gen);

    // Ensure data size is compatible with AMX constraints
    size_t amx_compatible_size = (data_float.size() / 32) * 32;  // Round down to multiple of 32
    if (amx_compatible_size != data_float.size()) {
        std::cout << "Adjusting data size from " << data_float.size() 
                  << " to " << amx_compatible_size << " for AMX compatibility" << std::endl;
        data_float.resize(amx_compatible_size);
    }

    // Convert to flat arrays
    std::cout << "Converting to flat arrays..." << std::endl;

    // Float versions for scalar and HNSWLIB computation
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

    // BF16 versions for AMX computation
    std::vector<bfloat16_t> data_bf16_flat(data_float.size() * dim);
    std::vector<bfloat16_t> centroids_bf16_flat(num_centroids * dim);

    for (size_t i = 0; i < data_float.size(); ++i) {
        for (int j = 0; j < dim; ++j) {
            data_bf16_flat[i * dim + j] = float_to_bfloat16(data_float[i][j]);
        }
    }

    for (int i = 0; i < num_centroids; ++i) {
        for (int j = 0; j < dim; ++j) {
            centroids_bf16_flat[i * dim + j] = float_to_bfloat16(centroids_float[i][j]);
        }
    }

    // Clean up 2D vectors to save memory
    data_float.clear();
    data_float.shrink_to_fit();
    centroids_float.clear();
    centroids_float.shrink_to_fit();

    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);

    std::cout << "Data preparation completed in " << init_duration.count() << " ms" << std::endl;
    std::cout << "Final dataset: " << (data_bf16_flat.size()/dim) << " points × " << dim << " dimensions" << std::endl;
    std::cout << "Total inner products to compute: " 
              << (data_bf16_flat.size()/dim) * (centroids_bf16_flat.size()/dim) << std::endl << std::endl;

    // Calculate total operations for throughput calculation
    long long total_ops = static_cast<long long>(data_bf16_flat.size() / dim) *
                         (centroids_bf16_flat.size() / dim) * dim;

    // Prepare result arrays
    size_t result_size = (data_bf16_flat.size() / dim) * (centroids_bf16_flat.size() / dim);
    std::vector<float> scalar_results(result_size);
    std::vector<float> hnswlib_results_1t(result_size);
    std::vector<float> hnswlib_results_4t(result_size);
    std::vector<float> hnswlib_results_8t(result_size);
    std::vector<float> hnswlib_results_opt(result_size);
    std::vector<float> single_amx_results(result_size);
    std::vector<float> multi_amx_results(result_size);

    // Performance tracking
    std::vector<PerformanceMetrics> performance_metrics;
    std::map<std::string, AccuracyMetrics> accuracy_metrics;

    // ===== SCALAR COMPUTATION =====
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "SCALAR COMPUTATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    PerformanceMetrics scalar_perf("Scalar (FP32)");
    
    for (int round = 0; round < 2; ++round) {
        std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        size_t result = compute(data_float_flat.data(), data_bf16_flat.size() / dim,
                               centroids_float_flat.data(), centroids_bf16_flat.size() / dim,
                               dim, scalar_results.data());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        scalar_perf.execution_times_us.push_back(static_cast<double>(duration.count()));
        std::cout << " " << duration.count() << "μs" << std::endl;
    }
    
    scalar_perf.calculate_stats(total_ops);
    scalar_perf.speedup_vs_scalar = 1.0; // Reference
    performance_metrics.push_back(scalar_perf);

    // ===== HNSWLIB COMPUTATION =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "HNSWLIB COMPUTATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Initialize HNSWLIB calculator
    HNSWLIBInnerProductPtr hnswlib_calc(dim);
    hnswlib_calc.print_thread_info();

    // Test different thread configurations
    std::vector<std::pair<int, std::string>> hnswlib_configs = {
        {1, "HNSWLIB (1 thread)"},
        {4, "HNSWLIB (4 threads)"},
        {8, "HNSWLIB (8 threads)"},
        {0, "HNSWLIB (optimized)"}
    };

    std::vector<std::vector<float>*> hnswlib_result_arrays = {
        &hnswlib_results_1t, &hnswlib_results_4t, &hnswlib_results_8t, &hnswlib_results_opt
    };

    for (size_t config_idx = 0; config_idx < hnswlib_configs.size(); ++config_idx) {
        int num_threads = hnswlib_configs[config_idx].first;
        std::string config_name = hnswlib_configs[config_idx].second;
        
        std::cout << "\n--- " << config_name << " ---" << std::endl;
        
        PerformanceMetrics hnswlib_perf(config_name);
        
        try {
            hnswlib_calc.reset_timers();
            
            for (int round = 0; round < rounds; ++round) {
                std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::flush;

                auto start = std::chrono::high_resolution_clock::now();

                size_t result;
                if (num_threads == 0) {
                    // Optimized version with automatic thread selection
                    result = hnswlib_calc.compute_inner_products_optimized(
                        data_float_flat.data(), data_bf16_flat.size() / dim,
                        centroids_float_flat.data(), centroids_bf16_flat.size() / dim,
                        dim, hnswlib_result_arrays[config_idx]->data());
                } else {
                    // Specific thread count
                    result = hnswlib_calc.compute_inner_products_multi_threaded(
                        data_float_flat.data(), data_bf16_flat.size() / dim,
                        centroids_float_flat.data(), centroids_bf16_flat.size() / dim,
                        dim, hnswlib_result_arrays[config_idx]->data(), num_threads);
                }

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                hnswlib_perf.execution_times_us.push_back(static_cast<double>(duration.count()));
                std::cout << " " << duration.count() << "μs" << std::endl;
            }
            
            hnswlib_perf.calculate_stats(total_ops);
            hnswlib_perf.set_speedup(scalar_perf.avg_time_us);
            
            // Print detailed timing for this configuration
            hnswlib_calc.print_timing_stats();
            
        } catch (const std::exception& e) {
            std::cout << " FAILED: " << e.what() << std::endl;
            hnswlib_perf.success = false;
        }
        
        performance_metrics.push_back(hnswlib_perf);
    }

    // ===== SINGLE-THREADED AMX COMPUTATION =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "SINGLE-THREADED AMX COMPUTATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    PerformanceMetrics single_amx_perf("Single AMX (BF16)");
    AMXInnerProductBF16Ptr single_amx_calc;

    if (!single_amx_calc.initialize()) {
        std::cout << "❌ AMX initialization failed!" << std::endl;
        single_amx_perf.success = false;
    } else {
        std::cout << "✅ Single-threaded AMX initialized successfully" << std::endl;

        for (int round = 0; round < rounds; ++round) {
            std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::flush;

            single_amx_calc.reset_timers();
            auto start = std::chrono::high_resolution_clock::now();

            try {
                size_t result = single_amx_calc.compute_inner_products(
                    data_bf16_flat.data(), data_bf16_flat.size() / dim,
                    centroids_bf16_flat.data(), centroids_bf16_flat.size() / dim,
                    dim, single_amx_results.data());

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                single_amx_perf.execution_times_us.push_back(static_cast<double>(duration.count()));
                std::cout << " " << duration.count() << "μs" << std::endl;

            } catch (const std::exception& e) {
                std::cout << " FAILED: " << e.what() << std::endl;
                single_amx_perf.success = false;
                break;
            }
        }
        
        single_amx_perf.calculate_stats(total_ops);
        single_amx_perf.set_speedup(scalar_perf.avg_time_us);
    }
    
    performance_metrics.push_back(single_amx_perf);

    // ===== MULTI-THREADED AMX COMPUTATION =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "MULTI-THREADED AMX COMPUTATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    PerformanceMetrics multi_amx_perf("Multi AMX (8 threads)");
    AMXInnerProductBF16PtrMT multi_amx_calc(8);

    if (!multi_amx_calc.initialize()) {
        std::cout << "❌ Multi-threaded AMX initialization failed!" << std::endl;
        multi_amx_perf.success = false;
    } else {
        std::cout << "✅ Multi-threaded AMX initialized successfully" << std::endl;

        for (int round = 0; round < rounds; ++round) {
            std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::flush;

            multi_amx_calc.reset_timers();
            auto start = std::chrono::high_resolution_clock::now();

            try {
                size_t result = multi_amx_calc.compute_inner_products(
                    data_bf16_flat.data(), data_bf16_flat.size() / dim,
                    centroids_bf16_flat.data(), centroids_bf16_flat.size() / dim,
                    dim, multi_amx_results.data());

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                
                multi_amx_perf.execution_times_us.push_back(static_cast<double>(duration.count()));
                std::cout << " " << duration.count() << "μs" << std::endl;

            } catch (const std::exception& e) {
                std::cout << " FAILED: " << e.what() << std::endl;
                multi_amx_perf.success = false;
                break;
            }
        }
        
        multi_amx_perf.calculate_stats(total_ops);
        multi_amx_perf.set_speedup(scalar_perf.avg_time_us);
    }
    
    performance_metrics.push_back(multi_amx_perf);

    // ===== ACCURACY ANALYSIS =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ACCURACY ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Compare HNSWLIB implementations against scalar reference
    for (size_t config_idx = 0; config_idx < hnswlib_configs.size(); ++config_idx) {
        std::string config_name = hnswlib_configs[config_idx].second;
        auto& perf_metric = performance_metrics[1 + config_idx]; // Skip scalar (index 0)
        
        if (perf_metric.success) {
            std::cout << "Analyzing " << config_name << " vs Scalar..." << std::endl;
            AccuracyMetrics hnswlib_acc(0.001f);
            hnswlib_acc.analyze(scalar_results, *hnswlib_result_arrays[config_idx]);
            accuracy_metrics[config_name] = hnswlib_acc;
        }
    }

    // Compare AMX implementations against scalar reference
    if (single_amx_perf.success) {
        std::cout << "Analyzing Single AMX vs Scalar..." << std::endl;
        AccuracyMetrics single_amx_acc(0.001f);
        single_amx_acc.analyze(scalar_results, single_amx_results);
        accuracy_metrics["Single AMX (BF16)"] = single_amx_acc;
    }

    if (multi_amx_perf.success) {
        std::cout << "Analyzing Multi AMX vs Scalar..." << std::endl;
        AccuracyMetrics multi_amx_acc(0.001f);
        multi_amx_acc.analyze(scalar_results, multi_amx_results);
        accuracy_metrics["Multi AMX (4 threads)"] = multi_amx_acc;
    }

    // Cross-compare HNSWLIB vs AMX for consistency
    if (single_amx_perf.success && performance_metrics[1].success) { // HNSWLIB 1 thread
        std::cout << "Analyzing HNSWLIB vs Single AMX..." << std::endl;
        AccuracyMetrics cross_acc(0.01f); // More lenient tolerance for different precision
        cross_acc.analyze(single_amx_results, hnswlib_results_1t);
        accuracy_metrics["HNSWLIB vs AMX (consistency)"] = cross_acc;
    }

    // ===== DETAILED TIMING ANALYSIS =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "DETAILED TIMING ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // HNSWLIB detailed timing
    std::cout << "\nHNSWLIB Detailed Analysis:" << std::endl;
    hnswlib_calc.print_timing_stats();

    // AMX detailed timing
    if (single_amx_perf.success) {
        std::cout << "\nSingle AMX Detailed Breakdown:" << std::endl;
        single_amx_calc.print_timing_stats();
    }

    if (multi_amx_perf.success) {
        std::cout << "\nMulti AMX Detailed Breakdown:" << std::endl;
        multi_amx_calc.print_timing_stats();
    }

    // Threading efficiency analysis
    std::cout << "\n--- Threading Efficiency Analysis ---" << std::endl;
    
    // Compare HNSWLIB threading efficiency
    if (performance_metrics[1].success && performance_metrics[2].success) { // 1 thread vs 4 threads
        double hnswlib_speedup = performance_metrics[1].avg_time_us / performance_metrics[2].avg_time_us;
        double hnswlib_efficiency = hnswlib_speedup / 4.0;
        
        std::cout << "HNSWLIB 4-thread speedup: " << std::fixed << std::setprecision(2) 
                  << hnswlib_speedup << "x" << std::endl;
        std::cout << "HNSWLIB 4-thread efficiency: " << std::setprecision(1) 
                  << (hnswlib_efficiency * 100) << "%" << std::endl;
    }

    // Compare AMX threading efficiency
    if (single_amx_perf.success && multi_amx_perf.success) {
        double amx_speedup = single_amx_perf.avg_time_us / multi_amx_perf.avg_time_us;
        double amx_efficiency = amx_speedup / 4.0;
        
        std::cout << "AMX 4-thread speedup: " << std::fixed << std::setprecision(2) 
                  << amx_speedup << "x" << std::endl;
        std::cout << "AMX 4-thread efficiency: " << std::setprecision(1) 
                  << (amx_efficiency * 100) << "%" << std::endl;
    }

    // ===== COMPREHENSIVE RESULTS =====
    print_performance_table(performance_metrics, total_ops);
    print_accuracy_table(accuracy_metrics);

    // ===== RECOMMENDATIONS =====
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                                RECOMMENDATIONS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // Find best performing implementation
    double best_throughput = 0.0;
    std::string best_impl = "None";
    
    for (const auto& metric : performance_metrics) {
        if (metric.success && metric.throughput_gflops > best_throughput) {
            best_throughput = metric.throughput_gflops;
            best_impl = metric.implementation_name;
        }
    }

    std::cout << "\nPerformance Analysis:" << std::endl;
    std::cout << "  Best performing implementation: " << best_impl << std::endl;
    std::cout << "  Peak throughput achieved: " << std::fixed << std::setprecision(2) 
              << best_throughput << " GFLOPS" << std::endl;

    // Implementation comparison
    std::cout << "\nImplementation Comparison:" << std::endl;
    
    // HNSWLIB vs Scalar
    if (performance_metrics.size() > 1 && performance_metrics[1].success) {
        std::cout << "  HNSWLIB vs Scalar speedup: " << std::setprecision(2) 
                  << performance_metrics[1].speedup_vs_scalar << "x" << std::endl;
    }
    
    // AMX vs Scalar
    if (single_amx_perf.success) {
        std::cout << "  Single AMX vs Scalar speedup: " << std::setprecision(2) 
                  << single_amx_perf.speedup_vs_scalar << "x" << std::endl;
    }
    
    // AMX vs HNSWLIB comparison
    if (single_amx_perf.success && performance_metrics.size() > 1 && performance_metrics[1].success) {
        double amx_vs_hnswlib = performance_metrics[1].avg_time_us / single_amx_perf.avg_time_us;
        std::cout << "  Single AMX vs HNSWLIB (1 thread): " << std::setprecision(2) 
                  << amx_vs_hnswlib << "x" << std::endl;
    }

    // Memory bandwidth analysis
    size_t total_memory_gb = (data_bf16_flat.size() * sizeof(bfloat16_t) + 
                             centroids_bf16_flat.size() * sizeof(bfloat16_t) +
                             result_size * sizeof(float)) / (1024 * 1024 * 1024);
    
    std::cout << "\nMemory Analysis:" << std::endl;
    std::cout << "  Total memory footprint: ~" << total_memory_gb << " GB" << std::endl;
    std::cout << "  Data points memory: " << (data_bf16_flat.size() * sizeof(bfloat16_t) / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Centroids memory: " << (centroids_bf16_flat.size() * sizeof(bfloat16_t) / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Results memory: " << (result_size * sizeof(float) / (1024*1024)) << " MB" << std::endl;

    return 0;
}
