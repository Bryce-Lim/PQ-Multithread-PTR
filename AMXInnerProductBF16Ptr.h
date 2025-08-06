#ifndef AMX_INNER_PRODUCT_BF16_PTR_H
#define AMX_INNER_PRODUCT_BF16_PTR_H

#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>
#include <chrono>

#define MAX_SIZE 16
#define MAX_COLS 32
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

// Define bfloat16 type
typedef uint16_t bfloat16_t;

// Define tile config data structure
typedef struct __tile_config
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

class AMXInnerProductBF16Ptr
{
private:
    bool amx_initialized;

    // Timing instance variables
    std::chrono::duration<double> total_compute_time;
    std::chrono::duration<double> padding_time;
    std::chrono::duration<double> conversion_time;
    std::chrono::duration<double> chunking_time;
    std::chrono::duration<double> merging_time;
    std::chrono::duration<double> actual_amx_time;

    // Working buffers to avoid frequent allocations
    bfloat16_t* padded_data_buffer;
    bfloat16_t* padded_centroids_buffer;
    bfloat16_t* data_chunk_buffer;
    bfloat16_t* centroid_chunk_buffer;
    float* results_chunk_buffer;
    
    size_t padded_data_buffer_size;
    size_t padded_centroids_buffer_size;
    size_t padded_dimension;

    // Initialization methods
    static void init_tile_config(__tilecfg *tileinfo);

    // Helper functions for type conversion
    static bfloat16_t float_to_bfloat16(float f);
    static float bfloat16_to_float(bfloat16_t bf16);

    // Internal computation functions
    void prepare_padded_data(const bfloat16_t* data_points, size_t data_count, size_t dimension);
    void prepare_padded_centroids(const bfloat16_t* centroids, size_t centroid_count, size_t dimension);
    void format_centroid_chunk(const bfloat16_t* padded_centroids, size_t centroid_count, 
                              size_t centroid_offset, size_t dim_offset);
    void format_data_chunk(const bfloat16_t* padded_data, size_t data_count,
                          size_t data_offset, size_t dim_offset);
    void main_multiply(const bfloat16_t* padded_data, size_t data_count,
                      const bfloat16_t* padded_centroids, size_t centroid_count,
                      size_t dimension, float* distances);

    // Buffer management
    void allocate_buffers(size_t data_count, size_t centroid_count, size_t dimension);
    void deallocate_buffers();

public:    
    // Constructor and Destructor
    AMXInnerProductBF16Ptr();
    ~AMXInnerProductBF16Ptr();

    // Main function - computes inner products between data points and centroids
    // Returns the number of distances computed (data_count * centroid_count)
    // distances array must be pre-allocated with size data_count * centroid_count
    // distances[i * centroid_count + j] = inner_product(data_points[i], centroids[j])
    size_t compute_inner_products(const bfloat16_t* data_points, size_t data_count,
                                 const bfloat16_t* centroids, size_t centroid_count,
                                 size_t dimension, float* distances);

    // Initialization
    bool initialize();

    // Utility functions
    static void convert_float_to_bfloat16_array(const float* input, bfloat16_t* output, size_t count);
    static void convert_bfloat16_to_float_array(const bfloat16_t* input, float* output, size_t count);
    
    // Print functions for debugging
    void print_bfloat16_array(const bfloat16_t* arr, size_t rows, size_t cols, const char* name = "Array");
    void print_float_array(const float* arr, size_t rows, size_t cols, const char* name = "Array");

    // Timing methods
    void reset_timers();
    void print_timing_stats() const;

    // Individual timing getters (in milliseconds)
    double get_total_compute_time_ms() const;
    double get_padding_time_ms() const;
    double get_conversion_time_ms() const;
    double get_chunking_time_ms() const;
    double get_merging_time_ms() const;
    double get_actual_amx_time_ms() const;

    // Check if AMX is properly initialized
    bool is_initialized() const { return amx_initialized; }
};

#endif
