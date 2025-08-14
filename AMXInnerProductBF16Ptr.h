#ifndef AMX_INNER_PRODUCT_BF16_PTR_H
#define AMX_INNER_PRODUCT_BF16_PTR_H

#include "AMXCommon.h"
#include <vector>
#include <chrono>

/**
 * @brief Single-threaded AMX inner product calculator with detailed timing analysis
 * 
 * This class provides high-performance inner product computation using Intel AMX instructions
 * in a single-threaded configuration. Optimized for scenarios where threading overhead
 * exceeds benefits or when precise timing analysis is required without thread coordination.
 * 
 * Features:
 * - BF16 precision for optimal AMX performance
 * - Detailed timing breakdown of computation phases
 * - Automatic memory layout optimization for AMX tiles
 * - Comprehensive error handling and validation
 */
class AMXInnerProductBF16Ptr {
private:
    // Core state
    bool amx_initialized;

    // Detailed timing tracking
    std::chrono::duration<double> total_compute_time;
    std::chrono::duration<double> padding_time;
    std::chrono::duration<double> conversion_time;
    std::chrono::duration<double> chunking_time;
    std::chrono::duration<double> merging_time;
    std::chrono::duration<double> actual_amx_time;

    // AMX configuration and initialization
    static void init_tile_config(__tilecfg* tileinfo);

    // Core computation engine
    void main_compute(
        const bfloat16_t* data_points, size_t data_count,
        const bfloat16_t* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );

    // Data formatting for AMX tiles
    void centroid_format(
        bfloat16_t* centroid_chunks, const bfloat16_t* centroids, 
        size_t centroid_count, size_t dimension
    );
    
    void data_format(
        bfloat16_t* data_chunk, const bfloat16_t* data, 
        size_t data_count, size_t dimension, 
        int data_offset, int d_offset
    );

    // Precision conversion utilities
    static bfloat16_t float_to_bfloat16(float f);
    static float bfloat16_to_float(bfloat16_t bf16);

public:
    /**
     * @brief Constructor - initializes timing counters
     */
    AMXInnerProductBF16Ptr();

    /**
     * @brief Destructor - releases AMX resources if initialized
     */
    ~AMXInnerProductBF16Ptr();

    /**
     * @brief Initialize AMX functionality
     * @return true if initialization successful, false otherwise
     * 
     * Must be called before any computation. Checks hardware AMX support
     * and configures tile registers for optimal performance.
     */
    bool initialize();

    /**
     * @brief Compute inner products using BF16 precision
     * @param data_points Input vectors in BF16 format
     * @param data_count Number of data vectors
     * @param centroids Centroid vectors in BF16 format  
     * @param centroid_count Number of centroids
     * @param dimension Vector dimension (must be multiple of 64)
     * @param distances Output array for results
     * @return Number of computed distances
     * 
     * @throws std::runtime_error if AMX not initialized or invalid parameters
     * 
     * Constraints for optimal performance:
     * - dimension must be divisible by 64
     * - centroid_count must be divisible by 16
     * - data_count must be divisible by 32
     */
    size_t compute_inner_products(
        const bfloat16_t* data_points, size_t data_count,
        const bfloat16_t* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );

    /**
     * @brief Compute inner products from FP32 input (with automatic conversion)
     * @param data_points Input vectors in FP32 format
     * @param data_count Number of data vectors
     * @param centroids Centroid vectors in FP32 format
     * @param centroid_count Number of centroids
     * @param dimension Vector dimension
     * @param distances Output array for results
     * @return Number of computed distances
     * 
     * Convenience method that handles FP32 to BF16 conversion internally.
     * Slightly slower due to conversion overhead.
     */
    size_t compute_inner_products_fp32(
        const float* data_points, size_t data_count,
        const float* centroids, size_t centroid_count,
        size_t dimension, float* distances
    );

    // Timing management
    void reset_timers();
    void print_timing_stats() const;

    // Individual timing getters (in milliseconds)
    double get_total_compute_time_ms() const;
    double get_padding_time_ms() const;
    double get_conversion_time_ms() const;
    double get_chunking_time_ms() const;
    double get_merging_time_ms() const;
    double get_actual_amx_time_ms() const;

    /**
     * @brief Convert float array to bfloat16 using vectorized instructions
     * @param input_buffer Source float array
     * @param output_buffer Destination bfloat16 array
     * @param count Number of elements to convert
     * 
     * Uses AVX-512 instructions for optimal conversion performance.
     */
    void float_to_bfloat16(const float* input_buffer, bfloat16_t* output_buffer, size_t count);

    // Status checking
    bool is_initialized() const { return amx_initialized; }
};

#endif // AMX_INNER_PRODUCT_BF16_PTR_H
