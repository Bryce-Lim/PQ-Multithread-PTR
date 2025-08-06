//==============================================================//
// Author: Bryce Lim, 2025                                      //
// SPDX-License-Identifier: MIT                                 //
//                                                              //
// Pointer-based AMX Inner Product Implementation               //
// Modified from vector-based version for better performance    //
//==============================================================//

#include "AMXInnerProductBF16Ptr.h"
#include <iostream>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <cstdint>

// Constructor
AMXInnerProductBF16Ptr::AMXInnerProductBF16Ptr() 
    : amx_initialized(false),
      padded_data_buffer(nullptr),
      padded_centroids_buffer(nullptr),
      data_chunk_buffer(nullptr),
      centroid_chunk_buffer(nullptr),
      results_chunk_buffer(nullptr),
      padded_data_buffer_size(0),
      padded_centroids_buffer_size(0),
      padded_dimension(0)
{
    reset_timers();
}

// Destructor
AMXInnerProductBF16Ptr::~AMXInnerProductBF16Ptr()
{
    if (amx_initialized)
    {
        _tile_release();
    }
    deallocate_buffers();
}

// Initialize AMX functionality
bool AMXInnerProductBF16Ptr::initialize()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
    {
        amx_initialized = false;
        return false;
    }

    amx_initialized = true;
    return true;
}

// Buffer management
void AMXInnerProductBF16Ptr::allocate_buffers(size_t data_count, size_t centroid_count, size_t dimension)
{
    // Calculate padded dimensions
    padded_dimension = ((dimension + MAX_COLS - 1) / MAX_COLS) * MAX_COLS;
    size_t padded_data_count = ((data_count + MAX_SIZE - 1) / MAX_SIZE) * MAX_SIZE;
    size_t padded_centroid_count = ((centroid_count + MAX_SIZE - 1) / MAX_SIZE) * MAX_SIZE;

    // Calculate required buffer sizes
    size_t new_data_buffer_size = padded_data_count * padded_dimension;
    size_t new_centroids_buffer_size = padded_centroid_count * padded_dimension;

    // Reallocate only if needed
    if (new_data_buffer_size > padded_data_buffer_size)
    {
        delete[] padded_data_buffer;
        padded_data_buffer = new bfloat16_t[new_data_buffer_size]();
        padded_data_buffer_size = new_data_buffer_size;
    }

    if (new_centroids_buffer_size > padded_centroids_buffer_size)
    {
        delete[] padded_centroids_buffer;
        padded_centroids_buffer = new bfloat16_t[new_centroids_buffer_size]();
        padded_centroids_buffer_size = new_centroids_buffer_size;
    }

    // Allocate working buffers if not already allocated
    if (!data_chunk_buffer)
    {
        data_chunk_buffer = new bfloat16_t[MAX_SIZE * MAX_COLS]();
    }
    
    if (!centroid_chunk_buffer)
    {
        centroid_chunk_buffer = new bfloat16_t[MAX_SIZE * MAX_COLS]();
    }
    
    if (!results_chunk_buffer)
    {
        results_chunk_buffer = new float[MAX_SIZE * MAX_SIZE]();
    }
}

void AMXInnerProductBF16Ptr::deallocate_buffers()
{
    delete[] padded_data_buffer;
    delete[] padded_centroids_buffer;
    delete[] data_chunk_buffer;
    delete[] centroid_chunk_buffer;
    delete[] results_chunk_buffer;
    
    padded_data_buffer = nullptr;
    padded_centroids_buffer = nullptr;
    data_chunk_buffer = nullptr;
    centroid_chunk_buffer = nullptr;
    results_chunk_buffer = nullptr;
    
    padded_data_buffer_size = 0;
    padded_centroids_buffer_size = 0;
}

// Timing getter methods
double AMXInnerProductBF16Ptr::get_total_compute_time_ms() const { return total_compute_time.count() * 1000.0; }
double AMXInnerProductBF16Ptr::get_padding_time_ms() const { return padding_time.count() * 1000.0; }
double AMXInnerProductBF16Ptr::get_conversion_time_ms() const { return conversion_time.count() * 1000.0; }
double AMXInnerProductBF16Ptr::get_chunking_time_ms() const { return chunking_time.count() * 1000.0; }
double AMXInnerProductBF16Ptr::get_merging_time_ms() const { return merging_time.count() * 1000.0; }
double AMXInnerProductBF16Ptr::get_actual_amx_time_ms() const { return actual_amx_time.count() * 1000.0; }

// Reset all timing counters
void AMXInnerProductBF16Ptr::reset_timers()
{
    total_compute_time = std::chrono::duration<double>::zero();
    padding_time = std::chrono::duration<double>::zero();
    conversion_time = std::chrono::duration<double>::zero();
    chunking_time = std::chrono::duration<double>::zero();
    merging_time = std::chrono::duration<double>::zero();
    actual_amx_time = std::chrono::duration<double>::zero();
}

// Print comprehensive timing statistics
void AMXInnerProductBF16Ptr::print_timing_stats() const
{
    std::cout << "\n=== AMX Inner Product Timing Statistics (Pointer Version) ===\n";
    std::cout << std::fixed << std::setprecision(3);

    std::cout << " Total compute time:          " << std::setw(8) << get_total_compute_time_ms() << " ms\n";
    std::cout << " - Padding time:              " << std::setw(8) << get_padding_time_ms() << " ms\n";
    std::cout << " - Chunking time:             " << std::setw(8) << get_chunking_time_ms() << " ms\n";
    std::cout << "     - Tile loading time:     " << std::setw(8) << get_conversion_time_ms() << " ms\n";
    std::cout << "     - Result merging time:   " << std::setw(8) << get_merging_time_ms() << " ms\n";
    std::cout << "     - Actual AMX time:       " << std::setw(8) << get_actual_amx_time_ms() << " ms\n";

    std::cout << "==========================================================\n\n";
}

// Initialize tile config
void AMXInnerProductBF16Ptr::init_tile_config(__tilecfg *tileinfo)
{
    int i;
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    // Tile 1: accumulator (float32)
    tileinfo->colsb[0] = MAX_SIZE * sizeof(float);
    tileinfo->rows[0] = MAX_SIZE;

    // Tiles 2,3: bfloat16 operands
    for (i = 1; i < 4; ++i)
    {
        tileinfo->colsb[i] = MAX_COLS * sizeof(bfloat16_t);
        tileinfo->rows[i] = MAX_SIZE;
    }

    _tile_loadconfig(tileinfo);
}

// Main public interface for computing inner products
size_t AMXInnerProductBF16Ptr::compute_inner_products(const bfloat16_t* data_points, size_t data_count,
                                                      const bfloat16_t* centroids, size_t centroid_count,
                                                      size_t dimension, float* distances)
{
    auto start_total = std::chrono::high_resolution_clock::now();

    if (!amx_initialized)
    {
        throw std::runtime_error("AMX not initialized. Call initialize() first.");
    }

    if (!data_points || !centroids || !distances || data_count == 0 || centroid_count == 0 || dimension == 0)
    {
        return 0;
    }

    // Allocate buffers
    allocate_buffers(data_count, centroid_count, dimension);

    // Initialize output array
    size_t total_distances = data_count * centroid_count;
    std::fill(distances, distances + total_distances, 0.0f);

    // Time padding and data preparation
    auto start_padding = std::chrono::high_resolution_clock::now();
    prepare_padded_data(data_points, data_count, dimension);
    prepare_padded_centroids(centroids, centroid_count, dimension);
    auto end_padding = std::chrono::high_resolution_clock::now();
    padding_time += end_padding - start_padding;

    // Perform the computation
    main_multiply(padded_data_buffer, data_count, padded_centroids_buffer, 
                 centroid_count, dimension, distances);

    auto end_total = std::chrono::high_resolution_clock::now();
    total_compute_time += end_total - start_total;

    return total_distances;
}

void AMXInnerProductBF16Ptr::prepare_padded_data(const bfloat16_t* data_points, size_t data_count, size_t dimension)
{
    // Clear the buffer
    std::fill(padded_data_buffer, padded_data_buffer + padded_data_buffer_size, 0);

    // Copy data with padding
    for (size_t i = 0; i < data_count; ++i)
    {
        std::memcpy(padded_data_buffer + i * padded_dimension,
                   data_points + i * dimension,
                   dimension * sizeof(bfloat16_t));
    }
}

void AMXInnerProductBF16Ptr::prepare_padded_centroids(const bfloat16_t* centroids, size_t centroid_count, size_t dimension)
{
    // Clear the buffer
    std::fill(padded_centroids_buffer, padded_centroids_buffer + padded_centroids_buffer_size, 0);

    // Copy centroids with padding
    for (size_t i = 0; i < centroid_count; ++i)
    {
        std::memcpy(padded_centroids_buffer + i * padded_dimension,
                   centroids + i * dimension,
                   dimension * sizeof(bfloat16_t));
    }
}

void AMXInnerProductBF16Ptr::main_multiply(const bfloat16_t* padded_data, size_t data_count,
                                          const bfloat16_t* padded_centroids, size_t centroid_count,
                                          size_t dimension, float* distances)
{
    auto start_chunking = std::chrono::high_resolution_clock::now();

    // Tile init
    __tilecfg tile_data = {};
    init_tile_config(&tile_data);

    // Iterate through data chunks
    for (size_t data_offset = 0; data_offset < data_count; data_offset += MAX_SIZE)
    {
        for (size_t dim_offset = 0; dim_offset < padded_dimension; dim_offset += MAX_COLS)
        {
            // Format data chunk
            auto start_conversion = std::chrono::high_resolution_clock::now();
            format_data_chunk(padded_data, data_count, data_offset, dim_offset);
            auto end_conversion = std::chrono::high_resolution_clock::now();
            conversion_time += end_conversion - start_conversion;

            for (size_t centroid_offset = 0; centroid_offset < centroid_count; centroid_offset += MAX_SIZE)
            {
                // Format centroid chunk
                start_conversion = std::chrono::high_resolution_clock::now();
                format_centroid_chunk(padded_centroids, centroid_count, centroid_offset, dim_offset);
                end_conversion = std::chrono::high_resolution_clock::now();
                conversion_time += end_conversion - start_conversion;

                auto start_AMX = std::chrono::high_resolution_clock::now();
                _tile_zero(1);
                _tile_loadd(2, centroid_chunk_buffer, STRIDE);
                _tile_loadd(3, data_chunk_buffer, STRIDE);

                _tile_dpbf16ps(1, 3, 2);
                _tile_stored(1, results_chunk_buffer, STRIDE);
                auto end_AMX = std::chrono::high_resolution_clock::now();
                actual_amx_time += end_AMX - start_AMX;

                // Merge results
                auto start_merge = std::chrono::high_resolution_clock::now();

                size_t actual_centroid_size = std::min(static_cast<size_t>(MAX_SIZE), centroid_count - centroid_offset);
                size_t actual_data_size = std::min(static_cast<size_t>(MAX_SIZE), data_count - data_offset);

                // Cache-optimized merging
                const int BLOCK_SIZE = 8;

                for (size_t centroid_block = 0; centroid_block < actual_centroid_size; centroid_block += BLOCK_SIZE)
                {
                    size_t block_end = std::min(centroid_block + BLOCK_SIZE, actual_centroid_size);

                    // Prefetch next block
                    if (centroid_block + BLOCK_SIZE < actual_centroid_size)
                    {
                        for (size_t prefetch_row = 0; prefetch_row < BLOCK_SIZE && 
                             centroid_block + BLOCK_SIZE + prefetch_row < actual_centroid_size; ++prefetch_row)
                        {
                            size_t prefetch_result_row = data_offset + centroid_block + BLOCK_SIZE + prefetch_row;
                            if (prefetch_result_row < data_count)
                            {
                                _mm_prefetch(reinterpret_cast<const char*>(&distances[prefetch_result_row * centroid_count + centroid_offset]), _MM_HINT_T0);
                            }
                        }
                    }

                    for (size_t centroid_row = centroid_block; centroid_row < block_end; ++centroid_row)
                    {
                        size_t result_centroid_idx = centroid_offset + centroid_row;

                        // Process 16 elements at a time using AVX-512
                        size_t data_col = 0;

                        // Main vectorized loop
                        for (; data_col + 16 <= actual_data_size; data_col += 16)
                        {
                            // Prefetch next chunk
                            if (data_col + 32 < actual_data_size)
                            {
                                size_t prefetch_data_idx = data_offset + data_col + 32;
                                if (prefetch_data_idx < data_count)
                                {
                                    _mm_prefetch(reinterpret_cast<const char*>(&distances[prefetch_data_idx * centroid_count + result_centroid_idx]), _MM_HINT_T0);
                                }
                            }

                            // Gather transposed values from results_chunk
                            __m512i indices = _mm512_setr_epi32(
                                centroid_row + (data_col + 0) * MAX_SIZE,
                                centroid_row + (data_col + 1) * MAX_SIZE,
                                centroid_row + (data_col + 2) * MAX_SIZE,
                                centroid_row + (data_col + 3) * MAX_SIZE,
                                centroid_row + (data_col + 4) * MAX_SIZE,
                                centroid_row + (data_col + 5) * MAX_SIZE,
                                centroid_row + (data_col + 6) * MAX_SIZE,
                                centroid_row + (data_col + 7) * MAX_SIZE,
                                centroid_row + (data_col + 8) * MAX_SIZE,
                                centroid_row + (data_col + 9) * MAX_SIZE,
                                centroid_row + (data_col + 10) * MAX_SIZE,
                                centroid_row + (data_col + 11) * MAX_SIZE,
                                centroid_row + (data_col + 12) * MAX_SIZE,
                                centroid_row + (data_col + 13) * MAX_SIZE,
                                centroid_row + (data_col + 14) * MAX_SIZE,
                                centroid_row + (data_col + 15) * MAX_SIZE
                            );

                            __m512 transposed_values = _mm512_i32gather_ps(indices, results_chunk_buffer, sizeof(float));

                            // Add to existing distances
                            for (int k = 0; k < 16; ++k)
                            {
                                size_t data_idx = data_offset + data_col + k;
                                if (data_idx < data_count)
                                {
                                    distances[data_idx * centroid_count + result_centroid_idx] += 
                                        reinterpret_cast<float*>(&transposed_values)[k];
                                }
                            }
                        }

                        // Handle remaining elements
                        for (; data_col < actual_data_size; ++data_col)
                        {
                            size_t data_idx = data_offset + data_col;
                            if (data_idx < data_count)
                            {
                                distances[data_idx * centroid_count + result_centroid_idx] += 
                                    results_chunk_buffer[data_col * MAX_SIZE + centroid_row];
                            }
                        }
                    }
                }
                auto end_merge = std::chrono::high_resolution_clock::now();
                merging_time += end_merge - start_merge;
            }
        }
    }
    auto end_chunking = std::chrono::high_resolution_clock::now();
    chunking_time += end_chunking - start_chunking;
}

void AMXInnerProductBF16Ptr::format_centroid_chunk(const bfloat16_t* padded_centroids, size_t centroid_count,
                                                   size_t centroid_offset, size_t dim_offset)
{
    // Clear the chunk buffer
    std::fill(centroid_chunk_buffer, centroid_chunk_buffer + MAX_COLS * MAX_SIZE, 0);

    size_t k = 0;
    // Process pairs of columns for BF16 packing
    for (int i = 0; i < MAX_COLS; i += 2)
    {
        for (int j = 0; j < MAX_SIZE; j++)
        {
            size_t centroid_idx = centroid_offset + j;
            size_t dim_idx1 = dim_offset + i;
            size_t dim_idx2 = dim_offset + i + 1;

            if (centroid_idx < centroid_count)
            {
                if (dim_idx1 < padded_dimension)
                {
                    centroid_chunk_buffer[k] = padded_centroids[centroid_idx * padded_dimension + dim_idx1];
                }
                k++;

                if (dim_idx2 < padded_dimension)
                {
                    centroid_chunk_buffer[k] = padded_centroids[centroid_idx * padded_dimension + dim_idx2];
                }
                k++;
            }
            else
            {
                k += 2;
            }
        }
    }
}

void AMXInnerProductBF16Ptr::format_data_chunk(const bfloat16_t* padded_data, size_t data_count,
                                               size_t data_offset, size_t dim_offset)
{
    // Clear the data chunk buffer
    std::fill(data_chunk_buffer, data_chunk_buffer + MAX_SIZE * MAX_COLS, 0);

    for (int i = 0; i < MAX_SIZE; i++)
    {
        size_t data_idx = data_offset + i;
        if (data_idx < data_count)
        {
            size_t elements_to_copy = std::min(static_cast<size_t>(MAX_COLS), padded_dimension - dim_offset);
            if (elements_to_copy > 0)
            {
                std::memcpy(&data_chunk_buffer[i * MAX_COLS],
                           &padded_data[data_idx * padded_dimension + dim_offset],
                           elements_to_copy * sizeof(bfloat16_t));
            }
        }
    }
}

// Utility functions
bfloat16_t AMXInnerProductBF16Ptr::float_to_bfloat16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    // Round to nearest even and truncate to bfloat16
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

float AMXInnerProductBF16Ptr::bfloat16_to_float(bfloat16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

void AMXInnerProductBF16Ptr::convert_float_to_bfloat16_array(const float* input, bfloat16_t* output, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        output[i] = float_to_bfloat16(input[i]);
    }
}

void AMXInnerProductBF16Ptr::convert_bfloat16_to_float_array(const bfloat16_t* input, float* output, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        output[i] = bfloat16_to_float(input[i]);
    }
}

void AMXInnerProductBF16Ptr::print_bfloat16_array(const bfloat16_t* arr, size_t rows, size_t cols, const char* name)
{
    std::cout << name << " (bfloat16, " << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; ++i)
    {
        std::cout << "Row " << i << ":\t[";
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << bfloat16_to_float(arr[i * cols + j]);
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

void AMXInnerProductBF16Ptr::print_float_array(const float* arr, size_t rows, size_t cols, const char* name)
{
    std::cout << name << " (float, " << rows << "x" << cols << "):\n";
    for (size_t i = 0; i < rows; ++i)
    {
        std::cout << "Row " << i << ":\t[";
        for (size_t j = 0; j < cols; ++j)
        {
            std::cout << arr[i * cols + j];
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}
