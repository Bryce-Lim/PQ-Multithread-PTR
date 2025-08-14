#include "AMXInnerProductBF16Ptr.h"
#include <iostream>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <immintrin.h>
#include <cstdint>

// ==================== Constructor/Destructor ====================

AMXInnerProductBF16Ptr::AMXInnerProductBF16Ptr() : amx_initialized(false) {
    reset_timers();
}

AMXInnerProductBF16Ptr::~AMXInnerProductBF16Ptr() {
    if (amx_initialized) {
        _tile_release();
    }
}

// ==================== Initialization ====================

bool AMXInnerProductBF16Ptr::initialize() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        amx_initialized = false;
        return false;
    }

    amx_initialized = true;
    return true;
}

void AMXInnerProductBF16Ptr::init_tile_config(__tilecfg* tileinfo) {
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    // Tile 0: accumulator (float32)
    tileinfo->colsb[0] = MAX_SIZE * sizeof(float);
    tileinfo->rows[0] = MAX_SIZE;

    // Tiles 1-3: bfloat16 operands
    for (int i = 1; i < 4; ++i) {
        tileinfo->colsb[i] = MAX_COLS * sizeof(bfloat16_t);
        tileinfo->rows[i] = MAX_SIZE;
    }

    _tile_loadconfig(tileinfo);
}

// ==================== Main Computation Interface ====================

size_t AMXInnerProductBF16Ptr::compute_inner_products_fp32(
    const float* data_points, size_t data_count, 
    const float* centroids, size_t centroid_count,
    size_t dimension, float* distances) {
    
    // Convert FP32 data to BF16 format
    std::vector<bfloat16_t> data_bfloat16(data_count * dimension);
    std::vector<bfloat16_t> centroids_bfloat16(centroid_count * dimension);

    float_to_bfloat16(data_points, data_bfloat16.data(), data_count * dimension);
    float_to_bfloat16(centroids, centroids_bfloat16.data(), centroid_count * dimension);

    return compute_inner_products(
        data_bfloat16.data(), data_count, 
        centroids_bfloat16.data(), centroid_count, 
        dimension, distances
    );
}

size_t AMXInnerProductBF16Ptr::compute_inner_products(
    const bfloat16_t* data_points, size_t data_count,
    const bfloat16_t* centroids, size_t centroid_count,
    size_t dimension, float* distances) {
    
    auto start_total = std::chrono::high_resolution_clock::now();

    // Validate initialization and parameters
    if (!amx_initialized) { 
        throw std::runtime_error("AMX not initialized. Call initialize() first."); 
    }
    if (!data_points || !centroids || !distances || data_count == 0 || centroid_count == 0 || dimension == 0) { 
        return 0; 
    }
    
    // Validate AMX constraints
    if (dimension % 64 != 0) { 
        throw std::runtime_error("Dimension must be divisible by 64 for optimal AMX performance"); 
    }
    if (centroid_count % 16 != 0) { 
        throw std::runtime_error("Number of centroids must be divisible by 16 for optimal AMX performance"); 
    }
    if (data_count % 32 != 0) { 
        throw std::runtime_error("Number of data vectors must be divisible by 32 for optimal AMX performance"); 
    }

    // Initialize output array
    size_t total_distances = data_count * centroid_count;
    memset(distances, 0, total_distances * sizeof(float));

    // Perform the core computation
    auto start_chunk = std::chrono::high_resolution_clock::now();
    main_compute(data_points, data_count, centroids, centroid_count, dimension, distances);
    auto end_chunk = std::chrono::high_resolution_clock::now();

    // Update timing statistics
    auto end_total = std::chrono::high_resolution_clock::now();
    total_compute_time += end_total - start_total;
    chunking_time += end_chunk - start_chunk;

    return total_distances;
}

// ==================== Core Computation Engine ====================

void AMXInnerProductBF16Ptr::main_compute(
    const bfloat16_t* data_points, size_t data_count,
    const bfloat16_t* centroids, size_t centroid_count,
    size_t dimension, float* distances) {

    // Prepare centroid data in AMX-optimized format
    std::vector<bfloat16_t> centroid_chunks(centroid_count * dimension);
    auto start_conversion = std::chrono::high_resolution_clock::now();
    centroid_format(centroid_chunks.data(), centroids, centroid_count, dimension);
    auto end_conversion = std::chrono::high_resolution_clock::now();
    conversion_time += end_conversion - start_conversion;

    // Initialize AMX tile configuration
    __tilecfg tile_data = {0};
    init_tile_config(&tile_data);

    // Allocate working memory for computation chunks
    std::vector<bfloat16_t> data_chunk(MAX_SIZE * MAX_COLS);
    std::vector<float> results_chunk(MAX_SIZE * MAX_SIZE);

    // Initialize distance array
    std::memset(distances, 0, sizeof(float) * data_count * centroid_count);

    // Calculate chunking parameters
    size_t dim_chunks = (dimension + MAX_COLS - 1) / MAX_COLS;

    // Main computation loop - process data in AMX-sized chunks
    for (size_t data_offset = 0; data_offset < data_count; data_offset += MAX_SIZE) {
        size_t actual_data_size = std::min(static_cast<size_t>(MAX_SIZE), data_count - data_offset);

        // Prefetch next data batch for improved cache performance
        if (data_offset + MAX_SIZE < data_count) {
            const bfloat16_t* next_data_start = data_points + (data_offset + MAX_SIZE) * dimension;
            for (size_t pf_offset = 0; pf_offset < MAX_SIZE * dimension; pf_offset += 32) {
                _mm_prefetch(reinterpret_cast<const char*>(next_data_start + pf_offset), _MM_HINT_T1);
            }
        }

        for (size_t centroid_offset = 0; centroid_offset < centroid_count; centroid_offset += MAX_SIZE) {
            size_t actual_centroid_size = std::min(static_cast<size_t>(MAX_SIZE), centroid_count - centroid_offset);

            for (size_t d_offset = 0; d_offset < dimension; d_offset += MAX_COLS) {
                size_t actual_dim_size = std::min(static_cast<size_t>(MAX_COLS), dimension - d_offset);

                // Calculate centroid chunk indices
                size_t centroid_chunk_row = centroid_offset / MAX_SIZE;
                size_t dim_chunk_col = d_offset / MAX_COLS;
                size_t centroid_chunk_idx = centroid_chunk_row * dim_chunks + dim_chunk_col;

                // Prefetch next centroid chunk for better memory bandwidth utilization
                if (d_offset + MAX_COLS < dimension) {
                    size_t next_dim_chunk_col = (d_offset + MAX_COLS) / MAX_COLS;
                    size_t next_centroid_chunk_idx = centroid_chunk_row * dim_chunks + next_dim_chunk_col;
                    const bfloat16_t* next_centroid_chunk = centroid_chunks.data() + (MAX_SIZE * MAX_COLS * next_centroid_chunk_idx);

                    for (size_t pf_offset = 0; pf_offset < MAX_SIZE * MAX_COLS; pf_offset += 32) {
                        _mm_prefetch(reinterpret_cast<const char*>(next_centroid_chunk + pf_offset), _MM_HINT_T0);
                    }
                }

                // Prefetch next data section
                if (d_offset + MAX_COLS < dimension) {
                    const bfloat16_t* next_data_section = data_points + data_offset * dimension + (d_offset + MAX_COLS);
                    for (size_t i = 0; i < actual_data_size; i += 8) {
                        _mm_prefetch(reinterpret_cast<const char*>(next_data_section + i * dimension), _MM_HINT_T0);
                    }
                }

                // Format data chunk for AMX processing
                auto start_data_conversion = std::chrono::high_resolution_clock::now();
                data_format(data_chunk.data(), data_points, actual_data_size, dimension, data_offset, d_offset);
                auto end_data_conversion = std::chrono::high_resolution_clock::now();
                conversion_time += end_data_conversion - start_data_conversion;

                // Prefetch result merging destination for the final dimension chunk
                if (d_offset + MAX_COLS >= dimension) {
                    for (size_t i = 0; i < actual_data_size; i += 8) {
                        size_t global_data_idx = data_offset + i;
                        size_t distance_base_idx = global_data_idx * centroid_count + centroid_offset;
                        _mm_prefetch(reinterpret_cast<const char*>(distances + distance_base_idx), _MM_HINT_T0);
                    }
                }

                // Core AMX operations
                auto start_AMX = std::chrono::high_resolution_clock::now();

                if (d_offset == 0) {
                    _tile_zero(1); // Zero accumulator for first dimension chunk
                } else {
                    _tile_loadd(1, results_chunk.data(), STRIDE); // Load previous partial results
                }

                _tile_loadd(2, centroid_chunks.data() + (MAX_SIZE * MAX_COLS * centroid_chunk_idx), STRIDE);
                _tile_loadd(3, data_chunk.data(), STRIDE);

                _tile_dpbf16ps(1, 3, 2); // Core matrix multiplication: tile1 += tile3 * tile2
                _tile_stored(1, results_chunk.data(), STRIDE);
                
                auto end_AMX = std::chrono::high_resolution_clock::now();
                actual_amx_time += end_AMX - start_AMX;
            }

            // Merge results back into final distance array
            auto start_merge = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < actual_data_size; ++i) {
                size_t global_data_idx = data_offset + i;

                // Prefetch next distance row for improved cache performance
                if (i + 4 < actual_data_size) {
                    size_t prefetch_data_idx = data_offset + i + 4;
                    size_t prefetch_distance_idx = prefetch_data_idx * centroid_count + centroid_offset;
                    _mm_prefetch(reinterpret_cast<const char*>(distances + prefetch_distance_idx), _MM_HINT_T0);
                }

                for (size_t j = 0; j < actual_centroid_size; ++j) {
                    size_t global_centroid_idx = centroid_offset + j;
                    size_t distance_idx = global_data_idx * centroid_count + global_centroid_idx;
                    size_t result_idx = i * MAX_SIZE + j;

                    distances[distance_idx] += results_chunk[result_idx];
                }
            }
            
            auto end_merge = std::chrono::high_resolution_clock::now();
            merging_time += end_merge - start_merge;
        }
    }
}

// ==================== Data Formatting Functions ====================

void AMXInnerProductBF16Ptr::centroid_format(
    bfloat16_t* centroid_chunks, const bfloat16_t* centroids, 
    size_t centroid_count, size_t dimension) {
    
    for (size_t centroid_offset = 0; centroid_offset < centroid_count / MAX_SIZE; centroid_offset++) {
        for (size_t d_offset = 0; d_offset < dimension / MAX_COLS; d_offset++) {
            size_t k = 0;
            size_t chunk_idx = centroid_offset * (dimension / MAX_COLS) + d_offset;
            
            // Format data in 2-wide chunks for optimal AMX tile layout
            for (size_t i = 0; i < MAX_COLS; i += 2) {
                for (size_t j = 0; j < MAX_SIZE; j++) {
                    size_t centroid_idx = centroid_offset * MAX_SIZE + j;
                    size_t dim_idx = d_offset * MAX_COLS + i;

                    centroid_chunks[chunk_idx * MAX_COLS * MAX_SIZE + k] = 
                        centroids[centroid_idx * dimension + dim_idx];
                    k++;
                    centroid_chunks[chunk_idx * MAX_COLS * MAX_SIZE + k] = 
                        centroids[centroid_idx * dimension + dim_idx + 1];
                    k++;
                }
            }
        }
    }
}

void AMXInnerProductBF16Ptr::data_format(
    bfloat16_t* data_chunk, const bfloat16_t* data, 
    size_t data_count, size_t dimension, 
    int data_offset, int d_offset) {
    
    const int AVX512_BF16_COUNT = 32;
    
    for (int i = 0; i < MAX_SIZE; i++) {
        const bfloat16_t* src = &data[(data_offset + i) * dimension + d_offset];
        bfloat16_t* dst = &data_chunk[i * MAX_COLS];

        // Process 32 bfloat16 values at a time using AVX-512
        for (int j = 0; j < MAX_COLS; j += AVX512_BF16_COUNT) {
            __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + j));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + j), vec);
        }
    }
}

// ==================== Precision Conversion Functions ====================

bfloat16_t AMXInnerProductBF16Ptr::float_to_bfloat16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    // Apply round-to-nearest-even and truncate to bfloat16
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

float AMXInnerProductBF16Ptr::bfloat16_to_float(bfloat16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

void AMXInnerProductBF16Ptr::float_to_bfloat16(
    const float* input_buffer, bfloat16_t* output_buffer, size_t count) {
    
    const size_t simd_width = 32; // Process 32 floats at a time for maximum throughput
    size_t simd_count = (count / simd_width) * simd_width;
    
    // Vectorized conversion using AVX-512 BF16 instructions
    for (size_t i = 0; i < simd_count; i += simd_width) {
        // Load two chunks of 16 floats each
        __m512 float_vec1 = _mm512_loadu_ps(&input_buffer[i]);
        __m512 float_vec2 = _mm512_loadu_ps(&input_buffer[i + 16]);
        
        // Convert both chunks to BF16
        __m256i bf16_vec1 = _mm512_cvtneps_pbh(float_vec1);
        __m256i bf16_vec2 = _mm512_cvtneps_pbh(float_vec2);
        
        // Store both results
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output_buffer[i]), bf16_vec1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output_buffer[i + 16]), bf16_vec2);
    }
    
    // Handle remaining elements with scalar conversion
    for (size_t i = simd_count; i < count; ++i) {
        output_buffer[i] = float_to_bfloat16(input_buffer[i]);
    }
}

// ==================== Timing and Analysis Methods ====================

void AMXInnerProductBF16Ptr::reset_timers() {
    total_compute_time = std::chrono::duration<double>::zero();
    padding_time = std::chrono::duration<double>::zero();
    conversion_time = std::chrono::duration<double>::zero();
    chunking_time = std::chrono::duration<double>::zero();
    merging_time = std::chrono::duration<double>::zero();
    actual_amx_time = std::chrono::duration<double>::zero();
}

void AMXInnerProductBF16Ptr::print_timing_stats() const {
    std::cout << "\n=== Single-Threaded AMX Timing Statistics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    std::cout << " Total compute time:          " << std::setw(8) << get_total_compute_time_ms() << " ms" << std::endl;
    std::cout << " ├─ Data formatting time:     " << std::setw(8) << get_conversion_time_ms() << " ms" << std::endl;
    std::cout << " ├─ Core computation time:    " << std::setw(8) << get_chunking_time_ms() << " ms" << std::endl;
    std::cout << " │  └─ Actual AMX time:       " << std::setw(8) << get_actual_amx_time_ms() << " ms" << std::endl;
    std::cout << " └─ Result merging time:      " << std::setw(8) << get_merging_time_ms() << " ms" << std::endl;

    // Calculate efficiency metrics
    double total_time = get_total_compute_time_ms();
    double amx_time = get_actual_amx_time_ms();
    
    if (total_time > 0) {
        double amx_efficiency = (amx_time / total_time) * 100.0;
        std::cout << "\n Performance Analysis:" << std::endl;
        std::cout << " AMX computation efficiency:  " << std::setw(8) << std::setprecision(1) << amx_efficiency << " %" << std::endl;
        
        if (amx_efficiency > 80.0) {
            std::cout << " ✅ Excellent AMX utilization" << std::endl;
        } else if (amx_efficiency > 60.0) {
            std::cout << " ✅ Good AMX utilization" << std::endl;
        } else {
            std::cout << " ⚠️  Low AMX utilization - check data formatting overhead" << std::endl;
        }
    }

    std::cout << "=============================================" << std::endl;
}

// ==================== Timing Getters ====================

double AMXInnerProductBF16Ptr::get_total_compute_time_ms() const { 
    return total_compute_time.count() * 1000.0; 
}

double AMXInnerProductBF16Ptr::get_padding_time_ms() const { 
    return padding_time.count() * 1000.0; 
}

double AMXInnerProductBF16Ptr::get_conversion_time_ms() const { 
    return conversion_time.count() * 1000.0; 
}

double AMXInnerProductBF16Ptr::get_chunking_time_ms() const { 
    return chunking_time.count() * 1000.0; 
}

double AMXInnerProductBF16Ptr::get_merging_time_ms() const { 
    return merging_time.count() * 1000.0; 
}

double AMXInnerProductBF16Ptr::get_actual_amx_time_ms() const { 
    return actual_amx_time.count() * 1000.0; 
}
