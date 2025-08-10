#ifndef HNSWLIB_INNER_PRODUCT_PTR_H
#define HNSWLIB_INNER_PRODUCT_PTR_H

#include <vector>
#include <chrono>
#include <string>
#include <atomic>
#include <mutex>

// HNSWLIB includes
#include "hnswlib/hnswlib.h"

class HNSWLIBInnerProductPtr {
private:
    size_t dimension;
    hnswlib::InnerProductSpace* space;
    hnswlib::DISTFUNC<float> dist_func;
    void* dist_func_param;
    
    // Thread management
    int current_thread_count;
    
    // Timing tracking
    std::chrono::duration<double> total_compute_time;
    std::chrono::duration<double> thread_overhead_time;
    std::chrono::duration<double> memory_access_time;
    std::mutex timing_mutex;
    
    // Performance monitoring
    std::atomic<size_t> total_operations_performed;
    std::atomic<int> max_threads_observed;
    
    // Internal computation methods
    void compute_block(
        const float* data_points, size_t data_start, size_t data_end,
        const float* centroids, size_t centroid_start, size_t centroid_end,
        size_t dimension, float* distances, size_t centroid_count,
        int thread_id
    );
    
    void compute_row_major(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );
    
    void compute_blocked(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances,
        size_t block_size_data = 64, size_t block_size_centroids = 8
    );

public:
    // Constructor and Destructor
    explicit HNSWLIBInnerProductPtr(size_t dim);
    ~HNSWLIBInnerProductPtr();
    
    // Main computation interface
    size_t compute_inner_products_fp32(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );
    
    // Computation variants
    size_t compute_inner_products_single_threaded(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );
    
    size_t compute_inner_products_multi_threaded(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances,
        int num_threads
    );
    
    size_t compute_inner_products_optimized(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances,
        int num_threads = 0
    );
    
    // Thread management
    void set_thread_count(int num_threads);
    int get_thread_count() const;
    void print_thread_info() const;
    
    // Performance analysis
    int find_optimal_thread_count(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* temp_distances
    );
    
    void benchmark_thread_scaling(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* temp_distances
    );
    
    // Timing methods
    void reset_timers();
    void print_timing_stats() const;
    double get_total_compute_time_ms() const;
    double get_thread_overhead_time_ms() const;
    double get_memory_access_time_ms() const;
    
    // Performance metrics
    size_t get_total_operations() const;
    int get_max_threads_observed() const;
    double calculate_throughput_gflops(size_t operations, double time_ms) const;
    
    // Utility methods
    bool verify_results(
        const float* result1, const float* result2,
        size_t count, float tolerance = 1e-5f
    ) const;
    
    void naive_compute(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances
    ) const;
    
    // Validation
    bool validate_inputs(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, const float* distances
    ) const;
    
    // Information
    size_t get_dimension() const { return dimension; }
    std::string get_implementation_name() const { return "HNSWLIB Pointer-based"; }
};

#endif // HNSWLIB_INNER_PRODUCT_PTR_H

