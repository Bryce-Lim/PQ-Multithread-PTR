#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"
#include "AMXInnerProductBF16PtrMTEnhanced.h"
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

// ==================== Configuration Constants ====================

const int dim = 1024;              // Embedding dimension - must be multiple of 64 for AMX
const int max_elements = 102400;   // Maximum number of vectors to load (large scale)
const int num_centroids = 16;      // Number of centroids - must be multiple of 16 for AMX
const int rounds = 10;             // Number of test rounds for averaging
const std::string dataroot = "/mnt/ceph/district9/dataset/openai/openai_large_5m/";

// Validate AMX constraints at compile time
static_assert(dim % 32 == 0, "Dimension must be multiple of 32 for AMX");
static_assert(num_centroids % 16 == 0, "Number of centroids must be multiple of 16 for AMX");
static_assert(max_elements % 16 == 0, "Number of data vectors must be a multiple of 16 for AMX");

// ==================== Data Type Conversion ====================

/**
 * @brief Convert float32 to bfloat16 with proper rounding
 * @param f Input float value
 * @return Converted bfloat16 value
 */
static bfloat16_t float_to_bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

// ==================== Performance Metrics ====================

/**
 * @brief Comprehensive performance metrics for implementation comparison
 */
struct PerformanceMetrics {
    std::string implementation_name;
    std::vector<double> execution_times_us;  // Execution times in microseconds
    double avg_time_us;
    double min_time_us;
    double max_time_us;
    double std_dev_us;
    double throughput_gflops;
    double speedup_vs_scalar;
    bool success;
    
    explicit PerformanceMetrics(const std::string& name) 
        : implementation_name(name), success(false) {}
    
    /**
     * @brief Calculate statistical metrics from execution times
     * @param total_ops Total number of operations performed
     */
    void calculate_stats(long long total_ops) {
        if (execution_times_us.empty()) {
            success = false;
            return;
        }
        
        success = true;
        
        // Calculate basic statistics
        avg_time_us = std::accumulate(execution_times_us.begin(), execution_times_us.end(), 0.0) / execution_times_us.size();
        min_time_us = *std::min_element(execution_times_us.begin(), execution_times_us.end());
        max_time_us = *std::max_element(execution_times_us.begin(), execution_times_us.end());
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : execution_times_us) {
            variance += (time - avg_time_us) * (time - avg_time_us);
        }
        std_dev_us = std::sqrt(variance / execution_times_us.size());
        
        // Calculate throughput in GFLOPS (multiply-add operations)
        throughput_gflops = (total_ops * 2.0) / (avg_time_us * 1e-6) / 1e9;
    }
    
    /**
     * @brief Set speedup relative to scalar reference implementation
     * @param scalar_time_us Reference scalar execution time in microseconds
     */
    void set_speedup(double scalar_time_us) {
        speedup_vs_scalar = scalar_time_us / avg_time_us;
    }
};

/**
 * @brief Accuracy analysis metrics for validation between implementations
 */
struct AccuracyMetrics {
    float max_abs_diff;
    float avg_abs_diff;
    float std_dev_diff;
    size_t significant_errors;
    float tolerance;
    bool acceptable;
    
    explicit AccuracyMetrics(float tol = 0.001f) : tolerance(tol), acceptable(false) {}
    
    /**
     * @brief Analyze accuracy between reference and comparison results
     * @param reference Reference implementation results
     * @param comparison Comparison implementation results
     */
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
        
        // Calculate absolute differences
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

// ==================== Data Loading ====================

/**
 * @brief Load embedding data from parquet files with Arrow compatibility
 * @param dataroot Root directory containing parquet files
 * @param max_elements Maximum number of vectors to load
 * @param dim Dimension of each vector
 * @return Vector of loaded embedding vectors
 */
std::vector<std::vector<float>> load_parquet_data(const std::string& dataroot, int max_elements, int dim) {
    std::vector<std::vector<float>> data_float;
    data_float.reserve(static_cast<size_t>(max_elements));

    std::cout << "Loading data from parquet files..." << std::endl;

    int files_loaded = 0;
    const size_t partition_size = 500000;

    for (int file_idx = 0; file_idx < 4 && static_cast<int>(data_float.size()) < max_elements; file_idx++) {
        std::string path = dataroot + "train-0" + std::to_string(file_idx) + "-of-10.parquet";
        std::cout << "  Loading: " << path << std::flush;

        // Arrow compatible file reading
        arrow::Result<std::shared_ptr<arrow::io::ReadableFile>> maybe_input = 
            arrow::io::ReadableFile::Open(path);
        
        if (!maybe_input.ok()) {
            std::cerr << " - Error opening file: " << maybe_input.status().ToString() << std::endl;
            continue;
        }
        
        std::shared_ptr<arrow::io::ReadableFile> input = maybe_input.ValueOrDie();

        // Create parquet reader
        arrow::Result<std::unique_ptr<parquet::arrow::FileReader>> maybe_reader = 
            parquet::arrow::OpenFile(input, arrow::default_memory_pool());
        
        if (!maybe_reader.ok()) {
            std::cerr << " - Error opening parquet file: " << maybe_reader.status().ToString() << std::endl;
            continue;
        }
        
        std::unique_ptr<parquet::arrow::FileReader> arrow_reader = std::move(maybe_reader).ValueOrDie();

        // Read table
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
        
        // Process each chunk in the column
        for (int chunk_idx = 0; chunk_idx < emb_col->num_chunks(); ++chunk_idx) {
            std::shared_ptr<arrow::Array> chunk = emb_col->chunk(chunk_idx);
            auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);
            std::shared_ptr<arrow::Array> values_array = list_array->values();
            
            // Handle both double and float arrays
            auto double_array = std::dynamic_pointer_cast<arrow::DoubleArray>(values_array);
            auto float_array = std::dynamic_pointer_cast<arrow::FloatArray>(values_array);
            
            if (double_array) {
                // Process double array
                for (int64_t i = 0; i < std::min(static_cast<int64_t>(partition_size), list_array->length()) 
                     && static_cast<int>(data_float.size()) < max_elements; i++) {
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
                // Process float array
                for (int64_t i = 0; i < std::min(static_cast<int64_t>(partition_size), list_array->length()) 
                     && static_cast<int>(data_float.size()) < max_elements; i++) {
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

// ==================== Output Functions ====================

/**
 * @brief Print comprehensive performance comparison table
 * @param metrics Vector of performance metrics for different implementations
 * @param total_ops Total number of operations performed
 */
void print_performance_table(const std::vector<PerformanceMetrics>& metrics, long long total_ops) {
    std::cout << "\n" << std::string(130, '=') << std::endl;
    std::cout << "                                    PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(130, '=') << std::endl;
    
    // Table header
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
    
    // Table rows
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

/**
 * @brief Print detailed accuracy analysis table
 * @param accuracy_map Map of implementation names to accuracy metrics
 */
void print_accuracy_table(const std::map<std::string, AccuracyMetrics>& accuracy_map) {
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                                ACCURACY ANALYSIS" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    // Table header
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
    
    // Table rows
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

// ==================== Main Program ====================

int main() {
    std::cout << "Comprehensive Implementation Comparison (AMX vs HNSWLIB)" << std::endl;
    std::cout << "========================================================" << std::endl;
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

    // ==================== Data Loading and Preparation ====================
    
    auto init_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<float>> data_float = load_parquet_data(dataroot, max_elements, dim);

    if (data_float.empty()) {
        std::cerr << "ERROR: No data loaded! Check your data path: " << dataroot << std::endl;
        return -1;
    }

    // Vector normalization for consistent similarity computation
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

    // Random centroid sampling for consistent testing
    std::cout << "Sampling " << num_centroids << " random centroids..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::vector<std::vector<float>> centroids_float;
    std::sample(data_float.begin(), data_float.end(),
                std::back_inserter(centroids_float), num_centroids, gen);

    // Ensure data size compatibility with AMX constraints
    size_t amx_compatible_size = (data_float.size() / 32) * 32;  // Round down to multiple of 32
    if (amx_compatible_size != data_float.size()) {
        std::cout << "Adjusting data size from " << data_float.size() 
                  << " to " << amx_compatible_size << " for AMX compatibility" << std::endl;
        data_float.resize(amx_compatible_size);
    }

    // Convert to flat arrays for computation
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

    // Prepare result arrays for different implementations
    size_t result_size = (data_bf16_flat.size() / dim) * (centroids_bf16_flat.size() / dim);
    std::vector<float> scalar_results(result_size);
    std::vector<float> hnswlib_results_56t(result_size);
    std::vector<float> hnswlib_results_84t(result_size);
    std::vector<float> hnswlib_results_112t(result_size);
    std::vector<float> hnswlib_results_224t(result_size);
    std::vector<float> single_amx_results(result_size);
    std::vector<float> multi_amx_results(result_size);

    // Performance and accuracy tracking
    std::vector<PerformanceMetrics> performance_metrics;
    std::map<std::string, AccuracyMetrics> accuracy_metrics;

    // ==================== Scalar Computation ====================
    
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "SCALAR COMPUTATION (BASELINE)" << std::endl;
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
    scalar_perf.speedup_vs_scalar = 1.0; // Reference implementation
    performance_metrics.push_back(scalar_perf);

    // ==================== HNSWLIB Computation ====================
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "HNSWLIB COMPUTATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Initialize HNSWLIB calculator
    HNSWLIBInnerProductPtr hnswlib_calc(dim);
    hnswlib_calc.print_thread_info();

    // Test different thread configurations for scalability analysis
    std::vector<std::pair<int, std::string>> hnswlib_configs = {
        {56, "HNSWLIB (56 threads)"},
        {84, "HNSWLIB (84 threads)"},
        {112, "HNSWLIB (112 threads)"},
        {224, "HNSWLIB (224 threads)"}
    };

    std::vector<std::vector<float>*> hnswlib_result_arrays = {
        &hnswlib_results_56t, &hnswlib_results_84t, &hnswlib_results_112t, &hnswlib_results_224t
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

                size_t result = hnswlib_calc.compute_inner_products_multi_threaded(
                    data_float_flat.data(), data_bf16_flat.size() / dim,
                    centroids_float_flat.data(), centroids_bf16_flat.size() / dim,
                    dim, hnswlib_result_arrays[config_idx]->data(), num_threads);

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

    // ==================== Single-Threaded AMX Computation ====================
    
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

    // ==================== Multi-Threaded AMX Computation ====================
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "MULTI-THREADED AMX COMPUTATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    PerformanceMetrics multi_amx_perf("Multi AMX (32 threads)");
    AMXInnerProductBF16PtrMTEnhanced multi_amx_calc(32);

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

    // ==================== Accuracy Analysis ====================
    
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
        accuracy_metrics["Multi AMX (32 threads)"] = multi_amx_acc;
    }

    // Cross-compare HNSWLIB vs AMX for consistency analysis
    if (single_amx_perf.success && performance_metrics[1].success) { // HNSWLIB first config
        std::cout << "Analyzing HNSWLIB vs Single AMX..." << std::endl;
        AccuracyMetrics cross_acc(0.01f); // More lenient tolerance for different precision
        cross_acc.analyze(single_amx_results, hnswlib_results_56t);
        accuracy_metrics["HNSWLIB vs AMX (consistency)"] = cross_acc;
    }

    // ==================== Detailed Timing Analysis ====================
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "DETAILED TIMING ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // HNSWLIB detailed timing analysis
    std::cout << "\nHNSWLIB Comprehensive Analysis:" << std::endl;
    hnswlib_calc.print_timing_stats();

    // AMX detailed timing analysis
    if (single_amx_perf.success) {
        std::cout << "\nSingle AMX Detailed Breakdown:" << std::endl;
        single_amx_calc.print_timing_stats();
    }

    if (multi_amx_perf.success) {
        std::cout << "\nMulti AMX Comprehensive Breakdown:" << std::endl;
        multi_amx_calc.print_comprehensive_timing_stats();
    }

    // Threading efficiency analysis
    std::cout << "\n--- Threading Efficiency Analysis ---" << std::endl;
    
    // Compare HNSWLIB threading efficiency across different thread counts
    if (performance_metrics.size() >= 3 && performance_metrics[1].success && performance_metrics[2].success) {
        double hnswlib_56_to_84_speedup = performance_metrics[1].avg_time_us / performance_metrics[2].avg_time_us;
        double threading_efficiency = hnswlib_56_to_84_speedup / (84.0 / 56.0);
        
        std::cout << "HNSWLIB 56→84 thread speedup: " << std::fixed << std::setprecision(2) 
                  << hnswlib_56_to_84_speedup << "x" << std::endl;
        std::cout << "HNSWLIB threading efficiency: " << std::setprecision(1) 
                  << (threading_efficiency * 100) << "%" << std::endl;
    }

    // Compare AMX threading efficiency
    if (single_amx_perf.success && multi_amx_perf.success) {
        double amx_speedup = single_amx_perf.avg_time_us / multi_amx_perf.avg_time_us;
        double amx_efficiency = amx_speedup / 32.0;
        
        std::cout << "AMX 32-thread speedup: " << std::fixed << std::setprecision(2) 
                  << amx_speedup << "x" << std::endl;
        std::cout << "AMX threading efficiency: " << std::setprecision(1) 
                  << (amx_efficiency * 100) << "%" << std::endl;
    }

    // ==================== Comprehensive Results ====================
    
    print_performance_table(performance_metrics, total_ops);
    print_accuracy_table(accuracy_metrics);

    // ==================== Implementation Comparison Analysis ====================
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                          IMPLEMENTATION COMPARISON ANALYSIS" << std::endl;
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

    // Implementation-specific comparisons
    std::cout << "\nImplementation Comparison:" << std::endl;
    
    // HNSWLIB vs Scalar analysis
    if (performance_metrics.size() > 1 && performance_metrics[1].success) {
        std::cout << "  Best HNSWLIB vs Scalar speedup: " << std::setprecision(2) 
                  << performance_metrics[1].speedup_vs_scalar << "x" << std::endl;
    }
    
    // AMX vs Scalar analysis
    if (single_amx_perf.success) {
        std::cout << "  Single AMX vs Scalar speedup: " << std::setprecision(2) 
                  << single_amx_perf.speedup_vs_scalar << "x" << std::endl;
    }
    
    // Direct AMX vs HNSWLIB comparison
    if (single_amx_perf.success && performance_metrics.size() > 1 && performance_metrics[1].success) {
        double amx_vs_hnswlib = performance_metrics[1].avg_time_us / single_amx_perf.avg_time_us;
        std::cout << "  Single AMX vs Best HNSWLIB: " << std::setprecision(2) 
                  << amx_vs_hnswlib << "x" << std::endl;
        
        if (amx_vs_hnswlib > 1.5) {
            std::cout << "  ✅ AMX provides significant advantage over HNSWLIB" << std::endl;
        } else if (amx_vs_hnswlib > 1.1) {
            std::cout << "  ✅ AMX provides moderate advantage over HNSWLIB" << std::endl;
        } else {
            std::cout << "  ⚠️  AMX and HNSWLIB performance are comparable" << std::endl;
        }
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

    // Threading scalability analysis
    std::cout << "\nThreading Scalability Analysis:" << std::endl;
    if (performance_metrics.size() >= 4) {
        std::cout << "  HNSWLIB scaling (56→224 threads):" << std::endl;
        for (size_t i = 1; i < 5 && i < performance_metrics.size(); ++i) {
            if (performance_metrics[i].success) {
                double relative_speedup = performance_metrics[1].avg_time_us / performance_metrics[i].avg_time_us;
                std::cout << "    " << performance_metrics[i].implementation_name 
                          << ": " << std::setprecision(2) << relative_speedup << "x vs 56 threads" << std::endl;
            }
        }
    }

    // ==================== Final Summary ====================
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                                 FINAL SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "Dataset Configuration:" << std::endl;
    std::cout << "  " << (data_bf16_flat.size()/dim) << " points × "
              << (centroids_bf16_flat.size()/dim) << " centroids × " << dim << " dimensions" << std::endl;
    std::cout << "  Total operations: " << total_ops << " (≈" << (total_ops / 1e9) << "B ops)" << std::endl;
    std::cout << "  Test configuration: " << rounds << " rounds per implementation" << std::endl;

    // Success summary
    int successful_implementations = 0;
    for (const auto& metric : performance_metrics) {
        if (metric.success) successful_implementations++;
    }

    std::cout << "\nImplementation Status: " << successful_implementations << "/"
              << static_cast<int>(performance_metrics.size()) << " successful" << std::endl;

    if (successful_implementations == 0) {
        std::cout << "❌ No implementations completed successfully" << std::endl;
        std::cout << "   Check hardware support and data compatibility" << std::endl;
        return -1;
    } else if (successful_implementations == static_cast<int>(performance_metrics.size())) {
        std::cout << "✅ All implementations completed successfully" << std::endl;
    } else {
        std::cout << "⚠️  Some implementations failed - check error messages above" << std::endl;
    }

    // Performance range analysis
    if (successful_implementations > 1) {
        double min_throughput = std::numeric_limits<double>::max();
        double max_throughput = 0.0;

        for (const auto& metric : performance_metrics) {
            if (metric.success) {
                min_throughput = std::min(min_throughput, metric.throughput_gflops);
                max_throughput = std::max(max_throughput, metric.throughput_gflops);
            }
        }

        std::cout << "Performance range: " << std::fixed << std::setprecision(2)
                  << min_throughput << " - " << max_throughput << " GFLOPS" << std::endl;
        std::cout << "Performance variation: " << std::setprecision(1)
                  << ((max_throughput / min_throughput - 1.0) * 100) << "%" << std::endl;
    }

    std::cout << "\n🎯 Comprehensive testing completed successfully!" << std::endl;
    std::cout << "   Results demonstrate relative performance characteristics of AMX vs HNSWLIB implementations." << std::endl;
    std::cout << "   Use the analysis above to select the optimal approach for your specific workload." << std::endl;

    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
