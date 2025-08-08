// multithread_test.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <cstring>
#include "ScalarInnerProduct.h"
#include "AMXInnerProductBF16Ptr.h"
#include "AMXInnerProductBF16PtrMT.h"

// Helper function to convert float to bfloat16
uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

int main() {
    std::cout << "Multithreaded AMX Inner Product Performance Test" << std::endl;
    std::cout << "===============================================" << std::endl;

    // Test configuration
    const size_t point_count = 960000;
    const size_t centroid_count = 32;
    const size_t dimension = 1024;
    const int rounds = 3;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data points: " << point_count << std::endl;
    std::cout << "  Centroids: " << centroid_count << std::endl;
    std::cout << "  Dimension: " << dimension << std::endl;
    std::cout << "  Test rounds: " << rounds << std::endl;
    std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << std::endl;

    // Generate test data
    std::cout << "\nGenerating test data..." << std::endl;
    
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

    // Result arrays
    std::vector<float> scalar_results(point_count * centroid_count);
    std::vector<float> single_amx_results(point_count * centroid_count);
    std::vector<float> multi_amx_results(point_count * centroid_count);

    // ===== SCALAR BASELINE =====
    std::cout << "\n=== SCALAR BASELINE ===" << std::endl;
    
    long scalar_total_time = 0;
    for (int round = 0; round < rounds; ++round) {
        std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t result = compute(points_float.data(), point_count,
                               centroids_float.data(), centroid_count,
                               dimension, scalar_results.data());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        scalar_total_time += duration.count();
        
        std::cout << "  Computed " << result << " distances in " << duration.count() << " μs" << std::endl;
    }
    
    long avg_scalar_time = scalar_total_time / rounds;
    std::cout << "Scalar average time: " << avg_scalar_time << " μs" << std::endl;

    // ===== SINGLE-THREADED AMX =====
    std::cout << "\n=== SINGLE-THREADED AMX ===" << std::endl;
    
    AMXInnerProductBF16Ptr single_amx;
    if (!single_amx.initialize()) {
        std::cout << "❌ AMX not available on this system" << std::endl;
        return 1;
    }
    
    std::cout << "✅ AMX initialized successfully" << std::endl;
    
    long single_amx_total_time = 0;
    for (int round = 0; round < rounds; ++round) {
        std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;
        
        single_amx.reset_timers();
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t result = single_amx.compute_inner_products(
            reinterpret_cast<const uint16_t*>(points_bf16.data()), point_count,
            reinterpret_cast<const uint16_t*>(centroids_bf16.data()), centroid_count,
            dimension, single_amx_results.data());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        single_amx_total_time += duration.count();
        
        std::cout << "  Computed " << result << " distances in " << duration.count() << " μs" << std::endl;
    }
    
    long avg_single_amx_time = single_amx_total_time / rounds;
    std::cout << "Single AMX average time: " << avg_single_amx_time << " μs" << std::endl;
    
    // ===== MULTITHREADED AMX =====
    std::cout << "\n=== MULTITHREADED AMX ===" << std::endl;
    
    // Test different thread counts
    std::vector<size_t> thread_counts = {1, 2, 4, 8, std::thread::hardware_concurrency()};
    
    for (size_t num_threads : thread_counts) {
        if (num_threads == 0) continue;
        
        std::cout << "\n--- Testing with " << num_threads << " threads ---" << std::endl;
        
        AMXInnerProductBF16PtrMT multi_amx(num_threads);
        if (!multi_amx.initialize()) {
            std::cout << "❌ Multithreaded AMX initialization failed" << std::endl;
            continue;
        }
        
        long multi_amx_total_time = 0;
        bool success = true;
        
        for (int round = 0; round < rounds && success; ++round) {
            std::cout << "Round " << (round + 1) << "/" << rounds << "..." << std::endl;
            
            multi_amx.reset_timers();
            auto start = std::chrono::high_resolution_clock::now();
            
            try {
                size_t result = multi_amx.compute_inner_products(
                    reinterpret_cast<const uint16_t*>(points_bf16.data()), point_count,
                    reinterpret_cast<const uint16_t*>(centroids_bf16.data()), centroid_count,
                    dimension, multi_amx_results.data());
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                multi_amx_total_time += duration.count();
                
                std::cout << "  Computed " << result << " distances in " << duration.count() << " μs" << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "❌ Multithreaded AMX failed: " << e.what() << std::endl;
                success = false;
            }
        }
        
        if (!success) continue;
        
        long avg_multi_amx_time = multi_amx_total_time / rounds;
        std::cout << "Multi AMX (" << num_threads << " threads) average time: " << avg_multi_amx_time << " μs" << std::endl;
        
        // Performance comparison
        double speedup_vs_scalar = static_cast<double>(avg_scalar_time) / avg_multi_amx_time;
        double speedup_vs_single_amx = static_cast<double>(avg_single_amx_time) / avg_multi_amx_time;
        double efficiency = speedup_vs_single_amx / num_threads;
        
        std::cout << "Speedup vs Scalar: " << std::fixed << std::setprecision(2) << speedup_vs_scalar << "x" << std::endl;
        std::cout << "Speedup vs Single AMX: " << std::setprecision(2) << speedup_vs_single_amx << "x" << std::endl;
        std::cout << "Threading efficiency: " << std::setprecision(1) << (efficiency * 100) << "%" << std::endl;
        
        // Show detailed timing
        multi_amx.print_timing_stats();
        
        // Quick accuracy check against scalar
        std::cout << "Accuracy check..." << std::endl;
        float max_diff = 0.0f;
        for (size_t i = 0; i < std::min((size_t)100, scalar_results.size()); ++i) {
            float diff = std::abs(scalar_results[i] - multi_amx_results[i]);
            max_diff = std::max(max_diff, diff);
        }
        std::cout << "Max difference vs scalar (first 100): " << std::scientific << std::setprecision(3) << max_diff << std::endl;
    }

    // ===== PERFORMANCE SUMMARY =====
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                  PERFORMANCE SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Scalar runtime:              " << std::setw(10) << avg_scalar_time << " μs" << std::endl;
    std::cout << "Single AMX runtime:          " << std::setw(10) << avg_single_amx_time << " μs" << std::endl;
    
    double single_amx_speedup = static_cast<double>(avg_scalar_time) / avg_single_amx_time;
    std::cout << "Single AMX speedup:          " << std::setw(10) << std::setprecision(2) << single_amx_speedup << "x" << std::endl;

    // Calculate throughput
    long long total_ops = static_cast<long long>(point_count) * centroid_count * dimension;
    double scalar_gflops = (total_ops * 2.0) / (avg_scalar_time * 1e-6) / 1e9;
    double single_amx_gflops = (total_ops * 2.0) / (avg_single_amx_time * 1e-6) / 1e9;
    
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "Scalar:                      " << std::setw(10) << std::setprecision(3) << scalar_gflops << " GFLOPS" << std::endl;
    std::cout << "Single AMX:                  " << std::setw(10) << single_amx_gflops << " GFLOPS" << std::endl;

    std::cout << "\nRecommendations:" << std::endl;
    std::cout << "- For maximum single-core performance, use single-threaded AMX" << std::endl;
    std::cout << "- For large datasets, multithreading can provide additional speedup" << std::endl;
    std::cout << "- Monitor memory bandwidth - it may become the bottleneck with multiple threads" << std::endl;
    std::cout << "- Consider NUMA topology for systems with multiple sockets" << std::endl;

    return 0;
}
