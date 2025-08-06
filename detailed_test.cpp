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

// Define these constants based on your data
const int dim = 1024;             // Adjust to your embedding dimension
const int max_elements = 960000;  // Maximum number of vectors to load
const int num_centroids = 256;
const int rounds = 3;             // Number of test rounds for averaging
const std::string dataroot = "/mnt/ceph/district9/dataset/openai/openai_large_5m/"; // Set your data directory

// Convert float32 to bfloat16
static bfloat16_t float_to_bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    // Round to nearest even and truncate to bfloat16
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

// Convert bfloat16 to float32 for debugging/printing
static float bfloat16_to_float(bfloat16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// Simple validation function to check results make sense
bool validate_results(const float* distances, size_t data_count, size_t centroid_count, size_t dimension) {
    bool valid = true;
    float min_val = distances[0], max_val = distances[0];
    size_t nan_count = 0, inf_count = 0;
    
    for (size_t i = 0; i < data_count * centroid_count; ++i) {
        if (std::isnan(distances[i])) {
            nan_count++;
            valid = false;
        } else if (std::isinf(distances[i])) {
            inf_count++;
            valid = false;
        } else {
            min_val = std::min(min_val, distances[i]);
            max_val = std::max(max_val, distances[i]);
        }
    }
    
    std::cout << "\n=== RESULT VALIDATION ===" << std::endl;
    std::cout << "Valid results:                         " << (valid ? "YES" : "NO") << std::endl;
    std::cout << "Result range:                          [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "NaN values found:                      " << nan_count << std::endl;
    std::cout << "Inf values found:                      " << inf_count << std::endl;
    
    // For normalized vectors, inner products should be in [-1, 1]
    if (max_val > 1.1f || min_val < -1.1f) {
        std::cout << "WARNING: Results outside expected range [-1, 1] for normalized vectors" << std::endl;
        valid = false;
    }
    
    return valid;
}

// Print sample results for inspection
void print_sample_results(const float* distances, size_t data_count, size_t centroid_count, 
                         const bfloat16_t* data_ptr, const bfloat16_t* centroids_ptr, size_t dimension) {
    std::cout << "\n=== SAMPLE RESULTS ===" << std::endl;
    
    // Print first few results
    std::cout << "First 5x5 distance matrix:" << std::endl;
    for (int i = 0; i < std::min(5, (int)data_count); ++i) {
        std::cout << "Data " << i << ": ";
        for (int j = 0; j < std::min(5, (int)centroid_count); ++j) {
            std::cout << std::setw(8) << std::setprecision(4) << std::fixed 
                     << distances[i * centroid_count + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Manual verification of first result
    if (data_count > 0 && centroid_count > 0) {
        float manual_result = 0.0f;
        for (size_t d = 0; d < dimension; ++d) {
            float data_val = bfloat16_to_float(data_ptr[d]);
            float centroid_val = bfloat16_to_float(centroids_ptr[d]);
            manual_result += data_val * centroid_val;
        }
        
        std::cout << "\nManual verification:" << std::endl;
        std::cout << "AMX result [0,0]:                     " << distances[0] << std::endl;
        std::cout << "Manual calculation [0,0]:             " << manual_result << std::endl;
        std::cout << "Difference:                           " << std::abs(distances[0] - manual_result) << std::endl;
    }
}

int main()
{
    std::cout << "=== AMX Inner Product Pointer Implementation Test ===" << std::endl;
    std::cout << "Testing with " << rounds << " rounds for performance averaging" << std::endl;

    // Start Timer for Initialization
    auto init_start = std::chrono::high_resolution_clock::now();

    // Reading parquet files (0, 1 - 1m size)
    std::vector<std::vector<float>> data_float;  // Temporary float storage
    data_float.reserve(max_elements);

    int cnt = 0;
    size_t partition_size = 500000;

    std::cout << "\nLoading data from parquet files..." << std::endl;

    for (int file_idx = 0; file_idx < 2; file_idx++)
    {
        auto pool = arrow::default_memory_pool();
        std::shared_ptr<arrow::io::RandomAccessFile> input;

        std::string path = dataroot + "train-0";
        path += std::to_string(file_idx);
        path += "-of-10.parquet";

        std::cout << "Loading file: " << path << std::endl;

        auto maybe_input = arrow::io::ReadableFile::Open(path);
        if (!maybe_input.ok())
        {
            std::cerr << "Error opening file: " << maybe_input.status().ToString() << std::endl;
            return -1;
        }
        input = maybe_input.ValueUnsafe();

        auto maybe_arrow_reader = parquet::arrow::OpenFile(input, pool);
        if (!maybe_arrow_reader.ok())
        {
            std::cerr << "Error opening parquet file: " << maybe_arrow_reader.status().ToString() << std::endl;
            return -2;
        }
        auto arrow_reader = std::move(maybe_arrow_reader).ValueUnsafe();

        std::shared_ptr<arrow::Table> table;
        auto status = arrow_reader->ReadTable(&table);
        if (!status.ok())
        {
            std::cerr << "Error reading table: " << status.ToString() << std::endl;
            return -3;
        }

        auto emb_col = table->column(1);
        if (emb_col->chunks().size() != 1)
        {
            std::cout << "Multiple chunks found: " << emb_col->chunks().size() << std::endl;
        }

        for (auto &arr : emb_col->chunks())
        {
            auto val = std::static_pointer_cast<arrow::DoubleArray>(
                std::static_pointer_cast<arrow::ListArray>(arr)->values());

            for (size_t i = 0; i < partition_size && data_float.size() < max_elements; i++)
            {
                std::vector<float> vec(dim);
                for (int j = 0; j < dim; j++)
                {
                    vec[j] = (float)val->Value(i * dim + j);
                }
                data_float.push_back(vec);
            }
        }
        cnt++;
    }

    std::cout << "Loaded " << data_float.size() << " vectors" << std::endl;

    // Normalize vectors (still in float)
    std::cout << "Normalizing vectors..." << std::endl;
    for (auto &emb : data_float)
    {
        float mag = 0;
        for (int d = 0; d < dim; d++)
        {
            mag += emb[d] * emb[d];
        }
        mag = sqrt(mag);

        if (mag > 0)
        { // Avoid division by zero
            for (int d = 0; d < dim; d++)
            {
                emb[d] /= mag;
            }
        }
    }

    // Sample random centroids (still in float)
    std::cout << "Sampling " << num_centroids << " random centroids..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::vector<float>> random_centroids_float;
    std::sample(data_float.begin(), data_float.end(), std::back_inserter(random_centroids_float), num_centroids, gen);

    // Set up dimensions
    size_t data_count = data_float.size();
    size_t centroid_count = random_centroids_float.size();
    size_t dimension = dim;

    // Allocate contiguous arrays
    std::cout << "Converting to bfloat16 and creating contiguous arrays..." << std::endl;
    bfloat16_t* data_ptr = new bfloat16_t[data_count * dimension];
    bfloat16_t* centroids_ptr = new bfloat16_t[centroid_count * dimension];
    float* distances_ptr = new float[data_count * centroid_count];

    // Convert and copy data to contiguous arrays (row-major order)
    for (size_t i = 0; i < data_count; ++i) {
        for (size_t j = 0; j < dimension; ++j) {
            data_ptr[i * dimension + j] = float_to_bfloat16(data_float[i][j]);
        }
    }

    for (size_t i = 0; i < centroid_count; ++i) {
        for (size_t j = 0; j < dimension; ++j) {
            centroids_ptr[i * dimension + j] = float_to_bfloat16(random_centroids_float[i][j]);
        }
    }

    // Clean up float vectors to save memory
    data_float.clear();
    data_float.shrink_to_fit();
    random_centroids_float.clear();
    random_centroids_float.shrink_to_fit();

    auto init_end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start);

    std::cout << "\n=== INITIALIZATION COMPLETE ===" << std::endl;
    std::cout << "Data vectors:                          " << data_count << std::endl;
    std::cout << "Centroids:                             " << centroid_count << std::endl;
    std::cout << "Dimension:                             " << dimension << std::endl;
    std::cout << "Total inner products to compute:       " << data_count * centroid_count << std::endl;
    std::cout << "Data memory usage:                     " << (data_count * dimension * sizeof(bfloat16_t)) / (1024*1024) << " MB" << std::endl;
    std::cout << "Centroid memory usage:                 " << (centroid_count * dimension * sizeof(bfloat16_t)) / (1024*1024) << " MB" << std::endl;
    std::cout << "Results memory usage:                  " << (data_count * centroid_count * sizeof(float)) / (1024*1024) << " MB" << std::endl;
    std::cout << "Preprocessing time:                    " << init_duration.count() << " microseconds" << std::endl;

    // ===== AMX COMPUTATION =====
    std::cout << "\n=== INITIALIZING AMX ===" << std::endl;
    AMXInnerProductBF16Ptr amx_calculator;
    
    if (!amx_calculator.initialize()) {
        std::cerr << "ERROR: Failed to initialize AMX! Check if AMX is available on this system." << std::endl;
        delete[] data_ptr;
        delete[] centroids_ptr;
        delete[] distances_ptr;
        return -1;
    }
    std::cout << "AMX initialized successfully!" << std::endl;

    std::cout << "\n=== RUNNING AMX COMPUTATION ===" << std::endl;
    std::vector<long> computation_times;
    
    for (int round = 0; round < rounds; round++)
    {
        std::cout << "\nRound " << (round + 1) << "/" << rounds << ":" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        amx_calculator.reset_timers();
        
        // Clear distances array
        std::fill(distances_ptr, distances_ptr + data_count * centroid_count, 0.0f);
        
        size_t result_count = amx_calculator.compute_inner_products(
            data_ptr, data_count, centroids_ptr, centroid_count, dimension, distances_ptr);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        computation_times.push_back(duration.count());
        
        std::cout << "  Computation time:                    " << duration.count() << " μs" << std::endl;
        std::cout << "  Results computed:                    " << result_count << std::endl;
        
        // Validate results on first round
        if (round == 0) {
            validate_results(distances_ptr, data_count, centroid_count, dimension);
        }
    }

    // ===== PERFORMANCE ANALYSIS =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                PERFORMANCE ANALYSIS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Calculate statistics
    long total_time = 0;
    for (long time : computation_times) {
        total_time += time;
    }
    long avg_time = total_time / rounds;
    
    long min_time = *std::min_element(computation_times.begin(), computation_times.end());
    long max_time = *std::max_element(computation_times.begin(), computation_times.end());

    std::cout << "Average computation time:              " << std::setw(10) << avg_time << " μs" << std::endl;
    std::cout << "Minimum computation time:              " << std::setw(10) << min_time << " μs" << std::endl;
    std::cout << "Maximum computation time:              " << std::setw(10) << max_time << " μs" << std::endl;
    std::cout << "Time variance:                         " << std::setw(10) << (max_time - min_time) << " μs" << std::endl;

    // Throughput calculations
    long long total_ops = static_cast<long long>(data_count) * centroid_count * dimension;
    double avg_gflops = (total_ops * 2.0) / (avg_time * 1e-6) / 1e9;  // *2 for multiply-add
    double peak_gflops = (total_ops * 2.0) / (min_time * 1e-6) / 1e9;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nThroughput Analysis:" << std::endl;
    std::cout << "Total multiply-add operations:         " << total_ops << std::endl;
    std::cout << "Average throughput:                    " << std::setw(10) << avg_gflops << " GFLOPS" << std::endl;
    std::cout << "Peak throughput:                       " << std::setw(10) << peak_gflops << " GFLOPS" << std::endl;

    std::cout << "\nPer-operation metrics:" << std::endl;
    std::cout << "Average time per inner product:        " << std::setw(10) << (avg_time / (data_count * centroid_count)) << " ns" << std::endl;
    std::cout << "Inner products per second:             " << std::setw(10) << (data_count * centroid_count * 1e6 / avg_time) << std::endl;

    // Print detailed timing breakdown
    std::cout << "\n=== DETAILED TIMING BREAKDOWN ===" << std::endl;
    amx_calculator.print_timing_stats();

    // Print sample results
    print_sample_results(distances_ptr, data_count, centroid_count, data_ptr, centroids_ptr, dimension);

    // Additional performance insights
    std::cout << "\n=== PERFORMANCE INSIGHTS ===" << std::endl;
    double data_throughput_gb_s = (data_count * dimension * sizeof(bfloat16_t) / (1024.0*1024.0*1024.0)) / (avg_time * 1e-6);
    double centroid_throughput_gb_s = (centroid_count * dimension * sizeof(bfloat16_t) / (1024.0*1024.0*1024.0)) / (avg_time * 1e-6);
    
    std::cout << "Data throughput:                       " << std::setw(10) << data_throughput_gb_s << " GB/s" << std::endl;
    std::cout << "Centroid throughput:                   " << std::setw(10) << centroid_throughput_gb_s << " GB/s" << std::endl;
    
    // Memory bandwidth utilization (rough estimate)
    double total_memory_accessed = (data_count + centroid_count) * dimension * sizeof(bfloat16_t) + data_count * centroid_count * sizeof(float);
    double memory_bandwidth = (total_memory_accessed / (1024.0*1024.0*1024.0)) / (avg_time * 1e-6);
    std::cout << "Estimated memory bandwidth:            " << std::setw(10) << memory_bandwidth << " GB/s" << std::endl;

    std::cout << "\n=== TEST COMPLETED SUCCESSFULLY ===" << std::endl;

    // Clean up
    delete[] data_ptr;
    delete[] centroids_ptr;
    delete[] distances_ptr;

    return 0;
}
