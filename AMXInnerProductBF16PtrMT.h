#ifndef AMX_INNER_PRODUCT_BF16_PTR_MT_H
#define AMX_INNER_PRODUCT_BF16_PTR_MT_H

#include "AMXCommon.h"
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>

class AMXInnerProductBF16PtrMT
{
private:
    bool amx_initialized;
    size_t num_threads;
    
    // Timing tracking
    std::chrono::duration<double> total_compute_time;
    std::chrono::duration<double> actual_amx_time;
    std::mutex timing_mutex;

    // AMX initialization per thread
    static bool initialize_thread_amx();
    static void init_thread_tile_config(__tilecfg *tileinfo);
    
    // Simple worker that replicates single-threaded logic exactly
    void worker_thread_simple(
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
    static bfloat16_t float_to_bfloat16(float f);

public:
    AMXInnerProductBF16PtrMT(size_t num_threads = 0);
    ~AMXInnerProductBF16PtrMT();

    bool initialize();
    
    size_t compute_inner_products(
        const bfloat16_t* data_points, size_t data_count,
        const bfloat16_t* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );

    void reset_timers();
    void print_timing_stats() const;
    double get_total_compute_time_ms() const;
    double get_actual_amx_time_ms() const;

    size_t get_num_threads() const { return num_threads; }
    bool is_initialized() const { return amx_initialized; }

    void float_to_bfloat16(const float* input_buffer, bfloat16_t* output_buffer, size_t count);
};

#endif

