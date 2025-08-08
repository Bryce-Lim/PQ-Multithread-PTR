#include "AMXInnerProductBF16PtrMT.h"
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

// Thread-local storage
thread_local bool thread_amx_initialized = false;
thread_local __tilecfg thread_tile_config = {0};

AMXInnerProductBF16PtrMT::AMXInnerProductBF16PtrMT(size_t num_threads)
    : amx_initialized(false), num_threads(num_threads)
{
    if (this->num_threads == 0) {
        this->num_threads = std::thread::hardware_concurrency();
        if (this->num_threads == 0) this->num_threads = 1;
    }
    this->num_threads = std::min(this->num_threads, static_cast<size_t>(4)); // Conservative limit
    reset_timers();
}

AMXInnerProductBF16PtrMT::~AMXInnerProductBF16PtrMT()
{
    if (thread_amx_initialized) {
        _tile_release();
        thread_amx_initialized = false;
    }
}

bool AMXInnerProductBF16PtrMT::initialize()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        amx_initialized = false;
        return false;
    }
    amx_initialized = true;
    return true;
}

bool AMXInnerProductBF16PtrMT::initialize_thread_amx()
{
    if (thread_amx_initialized) return true;

    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        return false;
    }

    init_thread_tile_config(&thread_tile_config);
    thread_amx_initialized = true;
    return true;
}

void AMXInnerProductBF16PtrMT::init_thread_tile_config(__tilecfg *tileinfo)
{
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    tileinfo->colsb[0] = MAX_SIZE * sizeof(float);
    tileinfo->rows[0] = MAX_SIZE;

    for (int i = 1; i < 4; ++i) {
        tileinfo->colsb[i] = MAX_COLS * sizeof(bfloat16_t);
        tileinfo->rows[i] = MAX_SIZE;
    }

    _tile_loadconfig(tileinfo);
}

size_t AMXInnerProductBF16PtrMT::compute_inner_products(
    const bfloat16_t* data_points, size_t data_count,
    const bfloat16_t* centroids, size_t centroid_count,
    size_t dimension, float* distances)
{
    auto start_total = std::chrono::high_resolution_clock::now();

    if (!amx_initialized) {
        throw std::runtime_error("AMX not initialized");
    }
    if (!data_points || !centroids || !distances) {
        return 0;
    }

    // Initialize output
    memset(distances, 0, data_count * centroid_count * sizeof(float));

    // Simple data partitioning - divide data points among threads
    size_t chunk_size = (data_count + num_threads - 1) / num_threads;

    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        if (start >= data_count) break;

        size_t end = std::min(start + chunk_size, data_count);

        threads.emplace_back([this, data_points, centroids, centroid_count, dimension, distances, start, end, t]() {
            this->worker_thread_simple(data_points, centroids, centroid_count, dimension, distances, start, end, t);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_total = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(timing_mutex);
    total_compute_time += end_total - start_total;

    return data_count * centroid_count;
}

void AMXInnerProductBF16PtrMT::worker_thread_simple(
    const bfloat16_t* data_points,
    const bfloat16_t* centroids,
    size_t centroid_count,
    size_t dimension,
    float* distances,
    size_t data_start,
    size_t data_end,
    size_t thread_id)
{
    if (!initialize_thread_amx()) {
        std::cerr << "Thread " << thread_id << " failed to initialize AMX" << std::endl;
        return;
    }

    auto start_amx = std::chrono::high_resolution_clock::now();

    // EXACT copy of the single-threaded logic, but only for data range [data_start, data_end)

    // Allocate working memory (same as single-threaded)
    bfloat16_t* centroid_chunks = new bfloat16_t[centroid_count * dimension];
    bfloat16_t* data_chunk = new bfloat16_t[MAX_SIZE * MAX_COLS];
    float* results_chunk = new float[MAX_SIZE * MAX_SIZE];

    // Format centroids EXACTLY like single-threaded version
    for (size_t centroid_offset = 0; centroid_offset < centroid_count / MAX_SIZE; centroid_offset++) {
        for (size_t d_offset = 0; d_offset < dimension / MAX_COLS; d_offset++) {
            size_t k = 0;
            size_t chunk_idx = centroid_offset * (dimension / MAX_COLS) + d_offset;

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

    size_t dim_chunks = (dimension + MAX_COLS - 1) / MAX_COLS;
    size_t thread_data_count = data_end - data_start;

    // Process ONLY this thread's data points using EXACT single-threaded logic
    for (size_t point_idx = 0; point_idx < thread_data_count; point_idx += MAX_SIZE) {
        size_t actual_data_size = std::min((size_t)MAX_SIZE, thread_data_count - point_idx);

        for (size_t centroid_offset = 0; centroid_offset < centroid_count; centroid_offset += MAX_SIZE) {
            size_t actual_centroid_size = std::min((size_t)MAX_SIZE, centroid_count - centroid_offset);

            for (size_t d_offset = 0; d_offset < dimension; d_offset += MAX_COLS) {
                size_t centroid_chunk_row = centroid_offset / MAX_SIZE;
                size_t dim_chunk_col = d_offset / MAX_COLS;
                size_t centroid_chunk_idx = centroid_chunk_row * dim_chunks + dim_chunk_col;

                // Format data chunk EXACTLY like single-threaded
                for (size_t i = 0; i < MAX_SIZE; i++) {
                    if (i < actual_data_size) {
                        // Global data index = thread's start + local offset
                        size_t global_point_idx = data_start + point_idx + i;
                        const bfloat16_t* src = &data_points[global_point_idx * dimension + d_offset];
                        bfloat16_t* dst = &data_chunk[i * MAX_COLS];

                        // Copy MAX_COLS elements
                        for (size_t j = 0; j < MAX_COLS; j += 32) {
                            __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + j));
                            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + j), vec);
                        }
                    } else {
                        // Zero unused rows
                        memset(&data_chunk[i * MAX_COLS], 0, MAX_COLS * sizeof(bfloat16_t));
                    }
                }

                // AMX computation EXACTLY like single-threaded
                if (d_offset == 0) {
                    _tile_zero(1);
                } else {
                    _tile_loadd(1, results_chunk, STRIDE);
                }

                _tile_loadd(2, centroid_chunks + (MAX_SIZE * MAX_COLS * centroid_chunk_idx), STRIDE);
                _tile_loadd(3, data_chunk, STRIDE);
                _tile_dpbf16ps(1, 3, 2);
                _tile_stored(1, results_chunk, STRIDE);
            }

            // Store results in global distances array
            for (size_t i = 0; i < actual_data_size; ++i) {
                size_t global_point_idx = data_start + point_idx + i;

                for (size_t j = 0; j < actual_centroid_size; ++j) {
                    size_t global_centroid_idx = centroid_offset + j;
                    size_t distance_idx = global_point_idx * centroid_count + global_centroid_idx;
                    size_t result_idx = i * MAX_SIZE + j;

                    distances[distance_idx] = results_chunk[result_idx];
                }
            }
        }
    }

    delete[] centroid_chunks;
    delete[] data_chunk;
    delete[] results_chunk;

    auto end_amx = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(timing_mutex);
    actual_amx_time += end_amx - start_amx;

    _tile_release();
    thread_amx_initialized = false;
}

void AMXInnerProductBF16PtrMT::reset_timers()
{
    std::lock_guard<std::mutex> lock(timing_mutex);
    total_compute_time = std::chrono::duration<double>::zero();
    actual_amx_time = std::chrono::duration<double>::zero();
}

double AMXInnerProductBF16PtrMT::get_total_compute_time_ms() const
{
    return total_compute_time.count() * 1000.0;
}

double AMXInnerProductBF16PtrMT::get_actual_amx_time_ms() const
{
    return actual_amx_time.count() * 1000.0;
}

void AMXInnerProductBF16PtrMT::print_timing_stats() const
{
    std::cout << "\n=== Simple Multithreaded AMX Timing ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << " Total compute time:     " << std::setw(10) << get_total_compute_time_ms() << " ms\n";
    std::cout << " Actual AMX time:        " << std::setw(10) << get_actual_amx_time_ms() << " ms\n";
    std::cout << " Number of threads:      " << std::setw(10) << num_threads << "\n";
    std::cout << "======================================\n\n";
}

bfloat16_t AMXInnerProductBF16PtrMT::float_to_bfloat16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

void AMXInnerProductBF16PtrMT::float_to_bfloat16(const float* input_buffer, bfloat16_t* output_buffer, size_t count)
{
    const size_t simd_width = 32;
    size_t simd_count = (count / simd_width) * simd_width;

    for (size_t i = 0; i < simd_count; i += simd_width) {
        __m512 float_vec1 = _mm512_loadu_ps(&input_buffer[i]);
        __m512 float_vec2 = _mm512_loadu_ps(&input_buffer[i + 16]);

        __m256i bf16_vec1 = _mm512_cvtneps_pbh(float_vec1);
        __m256i bf16_vec2 = _mm512_cvtneps_pbh(float_vec2);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output_buffer[i]), bf16_vec1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output_buffer[i + 16]), bf16_vec2);
    }

    for (size_t i = simd_count; i < count; ++i) {
        output_buffer[i] = float_to_bfloat16(input_buffer[i]);
    }
}
