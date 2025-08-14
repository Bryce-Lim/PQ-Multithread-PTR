#ifndef AMX_INNER_PRODUCT_BF16_PTR_MT_ENHANCED_H
#define AMX_INNER_PRODUCT_BF16_PTR_MT_ENHANCED_H

#include "AMXCommon.h"
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

/**
 * @brief Per-thread timing data structure for detailed performance analysis
 */
struct ThreadTiming {
    std::chrono::duration<double> thread_init_time;
    std::chrono::duration<double> centroid_formatting_time;
    std::chrono::duration<double> data_formatting_time;
    std::chrono::duration<double> tile_loading_time;
    std::chrono::duration<double> actual_computation_time;
    std::chrono::duration<double> result_merging_time;
    std::chrono::duration<double> total_thread_time;
    
    size_t data_points_processed;
    size_t total_amx_operations;
    
    ThreadTiming() : data_points_processed(0), total_amx_operations(0) {}
};

/**
 * @brief Enhanced multi-threaded AMX inner product calculator with comprehensive timing analysis
 * 
 * This class provides high-performance inner product computation using Intel AMX instructions
 * with detailed per-thread timing analysis, load balancing metrics, and performance bottleneck
 * identification. Optimized for large-scale vector similarity computations.
 */
class AMXInnerProductBF16PtrMTEnhanced {
private:
    // Core state
    bool amx_initialized;
    size_t num_threads;
    
    // Aggregate timing tracking
    std::chrono::duration<double> total_compute_time;
    std::chrono::duration<double> thread_spawn_time;
    std::chrono::duration<double> thread_join_time;
    std::chrono::duration<double> memory_allocation_time;
    std::chrono::duration<double> initialization_time;
    
    // Per-thread timing collection
    std::vector<ThreadTiming> thread_timings;
    std::mutex timing_mutex;
    
    // Performance counters
    std::atomic<size_t> total_tile_loads{0};
    std::atomic<size_t> total_computations{0};
    
    // AMX initialization and configuration
    static bool initialize_thread_amx_enhanced();
    static void init_thread_tile_config_enhanced(__tilecfg* tileinfo);
    
    // Core computation worker
    void worker_thread_enhanced(
        const bfloat16_t* data_points, 
        const bfloat16_t* centroids,
        size_t centroid_count,
        size_t dimension,
        float* distances,
        size_t data_start,
        size_t data_end,
        size_t thread_id
    );

    // Helper functions
    static bfloat16_t float_to_bfloat16_enhanced(float f);
    
    // Data formatting helpers
    void format_centroids_for_amx(bfloat16_t* centroid_chunks, const bfloat16_t* centroids, 
                                  size_t centroid_count, size_t dimension);
    void format_data_chunk(bfloat16_t* data_chunk, const bfloat16_t* data, 
                          size_t actual_data_size, size_t dimension, 
                          size_t data_offset, size_t d_offset);
    void merge_results_to_output(const float* results_chunk, float* distances,
                                size_t actual_data_size, size_t actual_centroid_size,
                                size_t global_data_offset, size_t centroid_offset, 
                                size_t centroid_count);
    
    // Performance analysis functions
    void analyze_load_balancing() const;
    void analyze_memory_efficiency() const;
    void analyze_threading_efficiency() const;

public:
    /**
     * @brief Constructor
     * @param num_threads Number of threads to use (0 = auto-detect)
     */
    explicit AMXInnerProductBF16PtrMTEnhanced(size_t num_threads = 0);
    
    /**
     * @brief Destructor - releases AMX resources
     */
    ~AMXInnerProductBF16PtrMTEnhanced();

    /**
     * @brief Initialize AMX functionality
     * @return true if initialization successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Compute inner products between data points and centroids
     * @param data_points Input vectors in BF16 format
     * @param data_count Number of data vectors
     * @param centroids Centroid vectors in BF16 format
     * @param centroid_count Number of centroids
     * @param dimension Vector dimension (must be multiple of 64)
     * @param distances Output array for results
     * @return Number of computed distances
     */
    size_t compute_inner_products(
        const bfloat16_t* data_points, size_t data_count,
        const bfloat16_t* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );

    // Timing and analysis methods
    void reset_timers();
    void print_comprehensive_timing_stats() const;
    void print_per_thread_breakdown() const;
    void print_performance_analysis() const;
    
    // Individual timing getters (in milliseconds)
    double get_total_compute_time_ms() const;
    double get_thread_spawn_time_ms() const;
    double get_thread_join_time_ms() const;
    double get_memory_allocation_time_ms() const;
    double get_initialization_time_ms() const;
    double get_avg_thread_time_ms() const;
    double get_avg_actual_computation_time_ms() const;
    
    // Compatibility getters
    double get_actual_amx_time_ms() const { return get_avg_actual_computation_time_ms(); }

    // Configuration getters
    size_t get_num_threads() const { return num_threads; }
    bool is_initialized() const { return amx_initialized; }

    /**
     * @brief Convert float array to bfloat16 array using vectorized instructions
     * @param input_buffer Source float array
     * @param output_buffer Destination bfloat16 array
     * @param count Number of elements to convert
     */
    void float_to_bfloat16(const float* input_buffer, bfloat16_t* output_buffer, size_t count);
};

#endif // AMX_INNER_PRODUCT_BF16_PTR_MT_ENHANCED_H
