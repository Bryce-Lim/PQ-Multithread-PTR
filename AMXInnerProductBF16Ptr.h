#ifndef AMX_INNER_PRODUCT_BF16_PTR_H
#define AMX_INNER_PRODUCT_BF16_PTR_H

#include <vector>
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
    std::chrono::duration<double> multiplication_time;
    std::chrono::duration<double> merging_time;
    std::chrono::duration<double> actual_amx_time;
    std::chrono::duration<double> tile_load_time;

    // Initialization methods
    static void init_tile_config(__tilecfg *tileinfo);

    // Helper functions for type conversion
    static bfloat16_t float_to_bfloat16(float f);
    static float bfloat16_to_float(bfloat16_t bf16);

    // Formatting functions
    // void padVectors(std::vector<std::vector<bfloat16_t>> &vectors);
    void centroid_format(bfloat16_t* centroid_chunks, const bfloat16_t* centroids, size_t centroid_count, size_t dimension);
    void data_format(bfloat16_t* data_chunk, const bfloat16_t* data, size_t data_count, size_t dimension, int data_offset, int d_offset);

    // Computation and merging functions
    void main_compute(const bfloat16_t* data_points, size_t data_count, const bfloat16_t* centroids, size_t centroid_count, size_t dimension, float* distances);

    // Printing methods
    void print_bfloat16_vectors(const std::vector<std::vector<bfloat16_t>> &vecs);

public:
    // Constructor and Destructor
    AMXInnerProductBF16Ptr();
    ~AMXInnerProductBF16Ptr();

    // Main functions
    bool initialize();
    size_t compute_inner_products(const bfloat16_t* data_points, size_t data_count, const bfloat16_t* centroids, size_t centroid_count, size_t dimension, float* distances);
    size_t compute_inner_products_fp32(const float *data_points, size_t data_count, const float *centroids, size_t centroid_count, size_t dimension, float *distances);

    // Timing methods
    void reset_timers();
    void print_timing_stats() const;

    // Individual timing getters (in milliseconds)
    double get_total_compute_time_ms() const;
    double get_padding_time_ms() const;
    double get_conversion_time_ms() const;
    double get_chunking_time_ms() const;
    double get_multiplication_time_ms() const;
    double get_merging_time_ms() const;
    double get_actual_amx_time_ms() const;
    double get_tile_load_time_ms() const;

    // Check if AMX is properly initialized
    bool is_initialized() const { return amx_initialized; }

    void float_to_bfloat16(const float* input_buffer, bfloat16_t* output_buffer, size_t count);

};
#endif
