#include "HNSWLIBInnerProductPtr.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <cmath>
#include <cstring>
#include <climits>
#include <omp.h>

// Constructor
HNSWLIBInnerProductPtr::HNSWLIBInnerProductPtr(size_t dim) 
    : dimension(dim), current_thread_count(0), total_operations_performed(0), max_threads_observed(0) {
    
    if (dim == 0) {
        throw std::invalid_argument("Dimension must be greater than 0");
    }
    
    // Initialize HNSWLIB space
    space = new hnswlib::InnerProductSpace(static_cast<int>(dimension));
    dist_func = space->get_dist_func();
    dist_func_param = space->get_dist_func_param();
    
    // Set default thread count
    current_thread_count = std::thread::hardware_concurrency();
    if (current_thread_count <= 0) current_thread_count = 1;
    
    reset_timers();
    
    std::cout << "Initialized HNSWLIBInnerProductPtr for " << dimension 
              << "-dimensional vectors with " << current_thread_count 
              << " threads" << std::endl;
}

// Destructor
HNSWLIBInnerProductPtr::~HNSWLIBInnerProductPtr() {
    delete space;
}

// Reset timing counters
void HNSWLIBInnerProductPtr::reset_timers() {
    std::lock_guard<std::mutex> lock(timing_mutex);
    total_compute_time = std::chrono::duration<double>::zero();
    thread_overhead_time = std::chrono::duration<double>::zero();
    memory_access_time = std::chrono::duration<double>::zero();
    total_operations_performed = 0;
    max_threads_observed = 0;
}

// Validate inputs
bool HNSWLIBInnerProductPtr::validate_inputs(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, const float* distances) const {
    
    if (!data_points || !centroids || !distances) {
        std::cerr << "Error: Null pointer provided" << std::endl;
        return false;
    }
    
    if (data_count == 0 || centroid_count == 0 || dimension == 0) {
        std::cerr << "Error: Zero count provided" << std::endl;
        return false;
    }
    
    if (dimension != this->dimension) {
        std::cerr << "Error: Dimension mismatch. Expected " << this->dimension 
                  << ", got " << dimension << std::endl;
        return false;
    }
    
    return true;
}

// Main computation interface
size_t HNSWLIBInnerProductPtr::compute_inner_products_fp32(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances) {
    
    return compute_inner_products_optimized(data_points, data_count, centroids, 
                                          centroid_count, dimension, distances, 
                                          current_thread_count);
}

// Single-threaded computation
size_t HNSWLIBInnerProductPtr::compute_inner_products_single_threaded(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances) {
    
    if (!validate_inputs(data_points, data_count, centroids, centroid_count, dimension, distances)) {
        return 0;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simple row-major computation
    compute_row_major(data_points, data_count, centroids, centroid_count, dimension, distances);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(timing_mutex);
    total_compute_time += end_time - start_time;
    total_operations_performed += data_count * centroid_count;
    max_threads_observed = std::max(max_threads_observed.load(), 1);
    
    return data_count * centroid_count;
}

// Multi-threaded computation
size_t HNSWLIBInnerProductPtr::compute_inner_products_multi_threaded(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances,
    int num_threads) {
    
    if (!validate_inputs(data_points, data_count, centroids, centroid_count, dimension, distances)) {
        return 0;
    }
    
    if (num_threads <= 0) {
        num_threads = current_thread_count;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto thread_start = std::chrono::high_resolution_clock::now();
    
    // Set OpenMP thread count
    int original_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    auto thread_end = std::chrono::high_resolution_clock::now();
    
    // Use blocked computation with OpenMP
    compute_blocked(data_points, data_count, centroids, centroid_count, dimension, distances);
    
    // Restore original thread count
    omp_set_num_threads(original_threads);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock(timing_mutex);
    total_compute_time += end_time - start_time;
    thread_overhead_time += thread_end - thread_start;
    total_operations_performed += data_count * centroid_count;
    max_threads_observed = std::max(max_threads_observed.load(), num_threads);
    
    return data_count * centroid_count;
}

// Optimized computation with automatic thread selection
size_t HNSWLIBInnerProductPtr::compute_inner_products_optimized(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances,
    int num_threads) {
    
    if (!validate_inputs(data_points, data_count, centroids, centroid_count, dimension, distances)) {
        return 0;
    }
    
    // Automatically choose threading strategy based on problem size
    size_t total_operations = data_count * centroid_count;
    const size_t threading_threshold = 10000; // Threshold for using multiple threads
    
    if (num_threads <= 0) {
        num_threads = current_thread_count;
    }
    
    if (total_operations < threading_threshold || num_threads == 1) {
        return compute_inner_products_single_threaded(data_points, data_count, centroids, 
                                                    centroid_count, dimension, distances);
    } else {
        return compute_inner_products_multi_threaded(data_points, data_count, centroids, 
                                                   centroid_count, dimension, distances, num_threads);
    }
}

// Row-major computation implementation
void HNSWLIBInnerProductPtr::compute_row_major(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances) {
    
    // Iterate through data points in outer loop for better cache locality
    for (size_t i = 0; i < data_count; ++i) {
        const float* data_vec = data_points + i * dimension;
        
        for (size_t j = 0; j < centroid_count; ++j) {
            const float* centroid_vec = centroids + j * dimension;
            
            // Use HNSWLIB's optimized distance function
            // Note: HNSWLIB's inner product space returns (1 - inner_product)
            float distance = dist_func(data_vec, centroid_vec, dist_func_param);
            distances[i * centroid_count + j] = 1.0f - distance;
        }
    }
}

// Blocked computation with OpenMP
void HNSWLIBInnerProductPtr::compute_blocked(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances,
    size_t block_size_data, size_t block_size_centroids) {
    
    std::atomic<int> active_threads(0);
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (size_t di = 0; di < data_count; di += block_size_data) {
        for (size_t cj = 0; cj < centroid_count; cj += block_size_centroids) {
            
            // Monitor thread activity
            int current_threads = ++active_threads;
            int prev_max = max_threads_observed.load();
            while (current_threads > prev_max && 
                   !max_threads_observed.compare_exchange_weak(prev_max, current_threads)) {
                prev_max = max_threads_observed.load();
            }
            
            size_t di_end = std::min(di + block_size_data, data_count);
            size_t cj_end = std::min(cj + block_size_centroids, centroid_count);
            
            // Process block
            for (size_t i = di; i < di_end; ++i) {
                const float* data_vec = data_points + i * dimension;
                
                for (size_t j = cj; j < cj_end; ++j) {
                    const float* centroid_vec = centroids + j * dimension;
                    
                    float distance = dist_func(data_vec, centroid_vec, dist_func_param);
                    distances[i * centroid_count + j] = 1.0f - distance;
                }
            }
            
            --active_threads;
        }
    }
}

// Thread management
void HNSWLIBInnerProductPtr::set_thread_count(int num_threads) {
    if (num_threads <= 0) {
        current_thread_count = std::thread::hardware_concurrency();
        if (current_thread_count <= 0) current_thread_count = 1;
        std::cout << "HNSWLIB threads reset to default (" << current_thread_count << ")" << std::endl;
    } else {
        current_thread_count = num_threads;
        std::cout << "HNSWLIB threads set to: " << current_thread_count << std::endl;
    }
}

int HNSWLIBInnerProductPtr::get_thread_count() const {
    return current_thread_count;
}

void HNSWLIBInnerProductPtr::print_thread_info() const {
    std::cout << "\n=== HNSWLIB Threading Information ===" << std::endl;
    
    #ifdef _OPENMP
    std::cout << "OpenMP: ENABLED" << std::endl;
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
    std::cout << "OpenMP num procs: " << omp_get_num_procs() << std::endl;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "OpenMP threads in parallel region: " << omp_get_num_threads() << std::endl;
        }
    }
    #else
    std::cout << "OpenMP: DISABLED" << std::endl;
    #endif
    
    std::cout << "Current thread setting: " << current_thread_count << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << std::endl;
    std::cout << "Max threads observed: " << max_threads_observed.load() << std::endl;
    std::cout << "=========================================" << std::endl;
}

// Find optimal thread count
int HNSWLIBInnerProductPtr::find_optimal_thread_count(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* temp_distances) {
    
    std::cout << "\n=== Finding Optimal Thread Count for HNSWLIB ===" << std::endl;
    
    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12, 16};
    int best_threads = 1;
    long best_time = LONG_MAX;
    
    for (int num_threads : thread_counts) {
        if (num_threads > static_cast<int>(std::thread::hardware_concurrency())) {
            continue;
        }
        
        std::cout << "Testing " << num_threads << " threads..." << std::flush;
        
        // Reset timers for clean measurement
        reset_timers();
        
        auto start = std::chrono::high_resolution_clock::now();
        compute_inner_products_multi_threaded(data_points, data_count, centroids, 
                                            centroid_count, dimension, temp_distances, num_threads);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        long time_us = duration.count();
        
        std::cout << " " << time_us << " μs";
        
        if (time_us < best_time) {
            best_time = time_us;
            best_threads = num_threads;
            std::cout << " (NEW BEST)";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Optimal thread count: " << best_threads << " (" << best_time << " μs)" << std::endl;
    current_thread_count = best_threads;
    return best_threads;
}

// Benchmark thread scaling
void HNSWLIBInnerProductPtr::benchmark_thread_scaling(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* temp_distances) {
    
    std::cout << "\n=== HNSWLIB Thread Scaling Benchmark ===" << std::endl;
    std::cout << "Data: " << data_count << " x " << dimension << std::endl;
    std::cout << "Centroids: " << centroid_count << " x " << dimension << std::endl;
    std::cout << "Total operations: " << data_count * centroid_count << std::endl;
    
    std::vector<int> thread_counts = {1, 2, 4, 6, 8, 12, 16};
    long baseline_time = 0;
    
    for (int num_threads : thread_counts) {
        if (num_threads > static_cast<int>(std::thread::hardware_concurrency())) {
            continue;
        }
        
        std::cout << "\nTesting " << num_threads << " threads..." << std::endl;
        
        reset_timers();
        auto start = std::chrono::high_resolution_clock::now();
        
        compute_inner_products_multi_threaded(data_points, data_count, centroids,
                                            centroid_count, dimension, temp_distances, num_threads);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        long time_us = duration.count();
        
        if (num_threads == 1) {
            baseline_time = time_us;
        }
        
        double speedup = baseline_time > 0 ? static_cast<double>(baseline_time) / time_us : 1.0;
        double efficiency = speedup / num_threads;
        
        long long total_ops = static_cast<long long>(data_count) * centroid_count * dimension;
        double throughput = calculate_throughput_gflops(total_ops * 2, time_us / 1000.0); // *2 for multiply-add
        
        std::cout << "  Time: " << time_us << " μs" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << std::setprecision(1) << (efficiency * 100) << "%" << std::endl;
        std::cout << "  Throughput: " << std::setprecision(2) << throughput << " GFLOPS" << std::endl;
        std::cout << "  Max threads observed: " << max_threads_observed.load() << std::endl;
    }
}

// Timing methods
double HNSWLIBInnerProductPtr::get_total_compute_time_ms() const {
    return total_compute_time.count() * 1000.0;
}

double HNSWLIBInnerProductPtr::get_thread_overhead_time_ms() const {
    return thread_overhead_time.count() * 1000.0;
}

double HNSWLIBInnerProductPtr::get_memory_access_time_ms() const {
    return memory_access_time.count() * 1000.0;
}

void HNSWLIBInnerProductPtr::print_timing_stats() const {
    std::cout << "\n=== HNSWLIB Timing Statistics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << " Total compute time:         " << std::setw(8) << get_total_compute_time_ms() << " ms" << std::endl;
    std::cout << " Thread overhead:            " << std::setw(8) << get_thread_overhead_time_ms() << " ms" << std::endl;
    std::cout << " Memory access time:         " << std::setw(8) << get_memory_access_time_ms() << " ms" << std::endl;
    
    std::cout << " Total operations:           " << std::setw(8) << total_operations_performed.load() << std::endl;
    std::cout << " Max threads observed:       " << std::setw(8) << max_threads_observed.load() << std::endl;
    
    if (total_operations_performed > 0 && total_compute_time.count() > 0) {
        double ops_per_second = total_operations_performed.load() / total_compute_time.count();
        std::cout << " Operations per second:      " << std::setw(8) << std::setprecision(0) << ops_per_second << std::endl;
        
        long long flops = static_cast<long long>(total_operations_performed.load()) * dimension * 2; // multiply-add
        double gflops = flops / (total_compute_time.count() * 1e9);
        std::cout << " Throughput:                 " << std::setw(8) << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    }
    
    std::cout << "===================================" << std::endl;
}

// Performance metrics
size_t HNSWLIBInnerProductPtr::get_total_operations() const {
    return total_operations_performed.load();
}

int HNSWLIBInnerProductPtr::get_max_threads_observed() const {
    return max_threads_observed.load();
}

double HNSWLIBInnerProductPtr::calculate_throughput_gflops(size_t operations, double time_ms) const {
    if (time_ms <= 0) return 0.0;
    return operations / (time_ms * 1e6); // Convert ms to seconds, then to GFLOPS
}

// Utility methods
bool HNSWLIBInnerProductPtr::verify_results(
    const float* result1, const float* result2,
    size_t count, float tolerance) const {
    
    for (size_t i = 0; i < count; ++i) {
        if (std::abs(result1[i] - result2[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": " << result1[i] 
                      << " vs " << result2[i] << " (diff: " 
                      << std::abs(result1[i] - result2[i]) << ")" << std::endl;
            return false;
        }
    }
    return true;
}

void HNSWLIBInnerProductPtr::naive_compute(
    const float* data_points, size_t data_count,
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances) const {
    
    for (size_t i = 0; i < data_count; ++i) {
        for (size_t j = 0; j < centroid_count; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < dimension; ++k) {
                sum += data_points[i * dimension + k] * centroids[j * dimension + k];
            }
            distances[i * centroid_count + j] = sum;
        }
    }
}
