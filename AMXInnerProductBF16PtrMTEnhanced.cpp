#include "AMXInnerProductBF16PtrMTEnhanced.h"
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
#include <numeric>
#include <thread>
#include <mutex>
#include <atomic>

// Enhanced-specific thread-local storage to avoid conflicts
namespace {
    thread_local bool enhanced_thread_amx_initialized = false;
    thread_local __tilecfg enhanced_thread_tile_config = {0};
}

AMXInnerProductBF16PtrMTEnhanced::AMXInnerProductBF16PtrMTEnhanced(size_t num_threads)
    : amx_initialized(false), num_threads(num_threads)
{
    if (this->num_threads == 0) {
        this->num_threads = std::thread::hardware_concurrency();
        if (this->num_threads == 0) this->num_threads = 1;
    }
    this->num_threads = std::min(this->num_threads, static_cast<size_t>(32));
    reset_timers();
}

AMXInnerProductBF16PtrMTEnhanced::~AMXInnerProductBF16PtrMTEnhanced()
{
    if (enhanced_thread_amx_initialized) {
        _tile_release();
        enhanced_thread_amx_initialized = false;
    }
}

bool AMXInnerProductBF16PtrMTEnhanced::initialize()
{
    auto start_init = std::chrono::high_resolution_clock::now();
    
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        amx_initialized = false;
        return false;
    }
    amx_initialized = true;
    
    auto end_init = std::chrono::high_resolution_clock::now();
    initialization_time += end_init - start_init;
    
    return true;
}

bool AMXInnerProductBF16PtrMTEnhanced::initialize_thread_amx_enhanced()
{
    if (enhanced_thread_amx_initialized) return true;

    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        return false;
    }

    init_thread_tile_config_enhanced(&enhanced_thread_tile_config);
    enhanced_thread_amx_initialized = true;
    return true;
}

void AMXInnerProductBF16PtrMTEnhanced::init_thread_tile_config_enhanced(__tilecfg *tileinfo)
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

size_t AMXInnerProductBF16PtrMTEnhanced::compute_inner_products(
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

    // Memory allocation timing
    auto start_memory = std::chrono::high_resolution_clock::now();
    memset(distances, 0, data_count * centroid_count * sizeof(float));
    
    // Prepare per-thread timing collection
    thread_timings.clear();
    thread_timings.resize(num_threads);
    auto end_memory = std::chrono::high_resolution_clock::now();
    memory_allocation_time += end_memory - start_memory;

    // Data partitioning
    size_t chunk_size = (data_count + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    
    // Thread spawning timing
    auto start_spawn = std::chrono::high_resolution_clock::now();
    
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        if (start >= data_count) break;

        size_t end = std::min(start + chunk_size, data_count);

        threads.emplace_back([this, data_points, centroids, centroid_count, dimension, distances, start, end, t]() {
            this->worker_thread_enhanced(data_points, centroids, centroid_count, dimension, distances, start, end, t);
        });
    }
    
    auto end_spawn = std::chrono::high_resolution_clock::now();
    thread_spawn_time += end_spawn - start_spawn;

    // Thread joining timing
    auto start_join = std::chrono::high_resolution_clock::now();
    for (auto& thread : threads) {
        thread.join();
    }
    auto end_join = std::chrono::high_resolution_clock::now();
    thread_join_time += end_join - start_join;

    auto end_total = std::chrono::high_resolution_clock::now();
    total_compute_time += end_total - start_total;

    return data_count * centroid_count;
}

void AMXInnerProductBF16PtrMTEnhanced::worker_thread_enhanced(
    const bfloat16_t* data_points,
    const bfloat16_t* centroids,
    size_t centroid_count,
    size_t dimension,
    float* distances,
    size_t data_start,
    size_t data_end,
    size_t thread_id)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    
    // Initialize thread timing
    ThreadTiming& timing = thread_timings[thread_id];
    timing = {};
    timing.data_points_processed = data_end - data_start;
    
    // Thread AMX initialization timing
    auto init_start = std::chrono::high_resolution_clock::now();
    if (!initialize_thread_amx_enhanced()) {
        std::cerr << "Enhanced thread " << thread_id << " failed to initialize AMX" << std::endl;
        return;
    }
    auto init_end = std::chrono::high_resolution_clock::now();
    timing.thread_init_time = init_end - init_start;

    // Allocate working memory
    bfloat16_t* centroid_chunks = new bfloat16_t[centroid_count * dimension];
    bfloat16_t* data_chunk = new bfloat16_t[MAX_SIZE * MAX_COLS];
    float* results_chunk = new float[MAX_SIZE * MAX_SIZE];

    // Centroid formatting timing
    auto format_start = std::chrono::high_resolution_clock::now();
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
    auto format_end = std::chrono::high_resolution_clock::now();
    timing.centroid_formatting_time = format_end - format_start;

    size_t dim_chunks = (dimension + MAX_COLS - 1) / MAX_COLS;
    size_t thread_data_count = data_end - data_start;
    size_t local_tile_loads = 0;
    size_t local_computations = 0;

    // Main computation loop with detailed timing
    for (size_t point_idx = 0; point_idx < thread_data_count; point_idx += MAX_SIZE) {
        size_t actual_data_size = std::min((size_t)MAX_SIZE, thread_data_count - point_idx);

        for (size_t centroid_offset = 0; centroid_offset < centroid_count; centroid_offset += MAX_SIZE) {
            size_t actual_centroid_size = std::min((size_t)MAX_SIZE, centroid_count - centroid_offset);

            for (size_t d_offset = 0; d_offset < dimension; d_offset += MAX_COLS) {
                size_t centroid_chunk_row = centroid_offset / MAX_SIZE;
                size_t dim_chunk_col = d_offset / MAX_COLS;
                size_t centroid_chunk_idx = centroid_chunk_row * dim_chunks + dim_chunk_col;

                // Data formatting timing
                auto data_format_start = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < MAX_SIZE; i++) {
                    if (i < actual_data_size) {
                        size_t global_point_idx = data_start + point_idx + i;
                        const bfloat16_t* src = &data_points[global_point_idx * dimension + d_offset];
                        bfloat16_t* dst = &data_chunk[i * MAX_COLS];

                        for (size_t j = 0; j < MAX_COLS; j += 32) {
                            __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(src + j));
                            _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + j), vec);
                        }
                    } else {
                        memset(&data_chunk[i * MAX_COLS], 0, MAX_COLS * sizeof(bfloat16_t));
                    }
                }
                auto data_format_end = std::chrono::high_resolution_clock::now();
                timing.data_formatting_time += data_format_end - data_format_start;

                // Tile loading timing
                auto tile_load_start = std::chrono::high_resolution_clock::now();
                if (d_offset == 0) {
                    _tile_zero(1);
                } else {
                    _tile_loadd(1, results_chunk, STRIDE);
                }

                _tile_loadd(2, centroid_chunks + (MAX_SIZE * MAX_COLS * centroid_chunk_idx), STRIDE);
                _tile_loadd(3, data_chunk, STRIDE);
                auto tile_load_end = std::chrono::high_resolution_clock::now();
                timing.tile_loading_time += tile_load_end - tile_load_start;
                
                local_tile_loads += 3;

                // Actual computation timing
                auto computation_start = std::chrono::high_resolution_clock::now();
                _tile_dpbf16ps(1, 3, 2);
                _tile_stored(1, results_chunk, STRIDE);
                auto computation_end = std::chrono::high_resolution_clock::now();
                timing.actual_computation_time += computation_end - computation_start;
                
                local_computations++;
            }

            // Result merging timing
            auto merge_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < actual_data_size; ++i) {
                size_t global_point_idx = data_start + point_idx + i;

                for (size_t j = 0; j < actual_centroid_size; ++j) {
                    size_t global_centroid_idx = centroid_offset + j;
                    size_t distance_idx = global_point_idx * centroid_count + global_centroid_idx;
                    size_t result_idx = i * MAX_SIZE + j;

                    distances[distance_idx] = results_chunk[result_idx];
                }
            }
            auto merge_end = std::chrono::high_resolution_clock::now();
            timing.result_merging_time += merge_end - merge_start;
        }
    }

    // Update global counters
    total_tile_loads.fetch_add(local_tile_loads);
    total_computations.fetch_add(local_computations);
    timing.total_amx_operations = local_computations;

    delete[] centroid_chunks;
    delete[] data_chunk;
    delete[] results_chunk;

    auto thread_end = std::chrono::high_resolution_clock::now();
    timing.total_thread_time = thread_end - thread_start;

    _tile_release();
    enhanced_thread_amx_initialized = false;
}

void AMXInnerProductBF16PtrMTEnhanced::reset_timers()
{
    std::lock_guard<std::mutex> lock(timing_mutex);
    total_compute_time = std::chrono::duration<double>::zero();
    thread_spawn_time = std::chrono::duration<double>::zero();
    thread_join_time = std::chrono::duration<double>::zero();
    memory_allocation_time = std::chrono::duration<double>::zero();
    initialization_time = std::chrono::duration<double>::zero();
    thread_timings.clear();
    total_tile_loads.store(0);
    total_computations.store(0);
}

void AMXInnerProductBF16PtrMTEnhanced::print_comprehensive_timing_stats() const
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "           COMPREHENSIVE MULTITHREADED AMX TIMING ANALYSIS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << std::fixed << std::setprecision(3);

    // Main timing breakdown
    std::cout << "\n📊 Main Process Timing:" << std::endl;
    std::cout << "  Total compute time:           " << std::setw(10) << get_total_compute_time_ms() << " ms" << std::endl;
    std::cout << "  ├─ Initialization:            " << std::setw(10) << get_initialization_time_ms() << " ms" << std::endl;
    std::cout << "  ├─ Memory allocation:         " << std::setw(10) << get_memory_allocation_time_ms() << " ms" << std::endl;
    std::cout << "  ├─ Thread spawning:           " << std::setw(10) << get_thread_spawn_time_ms() << " ms" << std::endl;
    std::cout << "  ├─ Thread joining:            " << std::setw(10) << get_thread_join_time_ms() << " ms" << std::endl;
    std::cout << "  └─ Actual parallel work:      " << std::setw(10) << get_avg_thread_time_ms() << " ms (avg)" << std::endl;

    // Calculate overhead
    double total_overhead = get_initialization_time_ms() + get_memory_allocation_time_ms() + 
                           get_thread_spawn_time_ms() + get_thread_join_time_ms();
    double overhead_percentage = (total_overhead / get_total_compute_time_ms()) * 100.0;
    
    std::cout << "\n⚙️  Threading Overhead Analysis:" << std::endl;
    std::cout << "  Total overhead:               " << std::setw(10) << total_overhead << " ms" << std::endl;
    std::cout << "  Overhead percentage:          " << std::setw(9) << overhead_percentage << " %" << std::endl;
    
    if (overhead_percentage > 20.0) {
        std::cout << "  ⚠️  High overhead detected - consider larger workloads or fewer threads" << std::endl;
    } else if (overhead_percentage < 5.0) {
        std::cout << "  ✅ Low overhead - good threading efficiency" << std::endl;
    }

    // Thread timing statistics
    if (!thread_timings.empty()) {
        std::cout << "\n🧵 Per-Thread Timing Statistics:" << std::endl;
        std::cout << "  Number of threads:            " << std::setw(10) << thread_timings.size() << std::endl;
        
        // Calculate min/max/avg for thread times
        std::vector<double> thread_times;
        for (const auto& timing : thread_timings) {
            thread_times.push_back(timing.total_thread_time.count() * 1000.0);
        }
        
        double min_time = *std::min_element(thread_times.begin(), thread_times.end());
        double max_time = *std::max_element(thread_times.begin(), thread_times.end());
        double avg_time = std::accumulate(thread_times.begin(), thread_times.end(), 0.0) / thread_times.size();
        
        std::cout << "  Min thread time:              " << std::setw(10) << min_time << " ms" << std::endl;
        std::cout << "  Max thread time:              " << std::setw(10) << max_time << " ms" << std::endl;
        std::cout << "  Avg thread time:              " << std::setw(10) << avg_time << " ms" << std::endl;
        std::cout << "  Load imbalance:               " << std::setw(9) << ((max_time - min_time) / avg_time * 100.0) << " %" << std::endl;
    }

    // Performance counters
    std::cout << "\n📈 Performance Counters:" << std::endl;
    std::cout << "  Total tile loads:             " << std::setw(10) << total_tile_loads.load() << std::endl;
    std::cout << "  Total AMX operations:         " << std::setw(10) << total_computations.load() << std::endl;
    
    if (total_computations.load() > 0) {
        double ops_per_ms = total_computations.load() / get_avg_actual_computation_time_ms();
        std::cout << "  AMX ops per millisecond:      " << std::setw(10) << ops_per_ms << std::endl;
    }

    std::cout << std::string(80, '=') << std::endl;
}

void AMXInnerProductBF16PtrMTEnhanced::print_per_thread_breakdown() const
{
    if (thread_timings.empty()) {
        std::cout << "No per-thread timing data available." << std::endl;
        return;
    }

    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                           PER-THREAD TIMING BREAKDOWN" << std::endl;
    std::cout << std::string(100, '=') << std::endl;

    std::cout << std::left 
              << std::setw(6) << "Thread"
              << std::setw(10) << "Total"
              << std::setw(8) << "Init"
              << std::setw(10) << "Format"
              << std::setw(12) << "DataFmt"
              << std::setw(10) << "TileLoad"
              << std::setw(10) << "Compute"
              << std::setw(10) << "Merge"
              << std::setw(8) << "Points"
              << std::setw(6) << "Ops" << std::endl;
              
    std::cout << std::string(6, '-') << " "
              << std::string(9, '-') << " "
              << std::string(7, '-') << " "
              << std::string(9, '-') << " "
              << std::string(11, '-') << " "
              << std::string(9, '-') << " "
              << std::string(9, '-') << " "
              << std::string(9, '-') << " "
              << std::string(7, '-') << " "
              << std::string(5, '-') << std::endl;

    std::cout << std::fixed << std::setprecision(1);
    
    for (size_t i = 0; i < thread_timings.size(); ++i) {
        const ThreadTiming& timing = thread_timings[i];
        
        std::cout << std::left << std::setw(6) << i
                  << std::right
                  << std::setw(8) << (timing.total_thread_time.count() * 1000.0) << "ms "
                  << std::setw(6) << (timing.thread_init_time.count() * 1000.0) << "ms "
                  << std::setw(8) << (timing.centroid_formatting_time.count() * 1000.0) << "ms "
                  << std::setw(10) << (timing.data_formatting_time.count() * 1000.0) << "ms "
                  << std::setw(8) << (timing.tile_loading_time.count() * 1000.0) << "ms "
                  << std::setw(8) << (timing.actual_computation_time.count() * 1000.0) << "ms "
                  << std::setw(8) << (timing.result_merging_time.count() * 1000.0) << "ms "
                  << std::setw(7) << timing.data_points_processed << " "
                  << std::setw(5) << timing.total_amx_operations << std::endl;
    }

    std::cout << std::string(100, '=') << std::endl;
}

void AMXInnerProductBF16PtrMTEnhanced::print_performance_analysis() const
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                         PERFORMANCE ANALYSIS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    analyze_load_balancing();
    analyze_memory_efficiency();
    analyze_threading_efficiency();
}

void AMXInnerProductBF16PtrMTEnhanced::analyze_load_balancing() const
{
    if (thread_timings.empty()) return;

    std::cout << "\n🎯 Load Balancing Analysis:" << std::endl;
    
    std::vector<double> thread_times;
    std::vector<size_t> points_processed;
    
    for (const auto& timing : thread_timings) {
        thread_times.push_back(timing.total_thread_time.count() * 1000.0);
        points_processed.push_back(timing.data_points_processed);
    }
    
    double min_time = *std::min_element(thread_times.begin(), thread_times.end());
    double max_time = *std::max_element(thread_times.begin(), thread_times.end());
    double avg_time = std::accumulate(thread_times.begin(), thread_times.end(), 0.0) / thread_times.size();
    
    double load_imbalance = ((max_time - min_time) / avg_time) * 100.0;
    
    std::cout << "  Time imbalance:               " << std::fixed << std::setprecision(1) 
              << std::setw(9) << load_imbalance << " %" << std::endl;
    
    if (load_imbalance < 5.0) {
        std::cout << "  ✅ Excellent load balancing" << std::endl;
    } else if (load_imbalance < 15.0) {
        std::cout << "  ✅ Good load balancing" << std::endl;
    } else {
        std::cout << "  ⚠️  Load imbalance detected" << std::endl;
    }
}

void AMXInnerProductBF16PtrMTEnhanced::analyze_memory_efficiency() const
{
    if (thread_timings.empty()) return;

    std::cout << "\n💾 Memory Efficiency Analysis:" << std::endl;
    
    // Calculate average time spent in different memory operations
    double avg_data_formatting = 0.0;
    double avg_tile_loading = 0.0;
    double avg_merging = 0.0;
    double avg_total = 0.0;
    
    for (const auto& timing : thread_timings) {
        avg_data_formatting += timing.data_formatting_time.count() * 1000.0;
        avg_tile_loading += timing.tile_loading_time.count() * 1000.0;
        avg_merging += timing.result_merging_time.count() * 1000.0;
        avg_total += timing.total_thread_time.count() * 1000.0;
    }
    
    size_t num_threads = thread_timings.size();
    avg_data_formatting /= num_threads;
    avg_tile_loading /= num_threads;
    avg_merging /= num_threads;
    avg_total /= num_threads;
    
    double memory_percentage = ((avg_data_formatting + avg_tile_loading + avg_merging) / avg_total) * 100.0;
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Memory operations:            " << std::setw(9) << memory_percentage << " % of total" << std::endl;
    
    if (memory_percentage > 60.0) {
        std::cout << "  ⚠️  Memory-bound workload" << std::endl;
    } else {
        std::cout << "  ✅ Compute-bound workload" << std::endl;
    }
}

void AMXInnerProductBF16PtrMTEnhanced::analyze_threading_efficiency() const
{
    if (thread_timings.empty()) return;

    std::cout << "\n⚡ Threading Efficiency Analysis:" << std::endl;
    
    // Calculate actual computation vs total time
    double total_computation_time = 0.0;
    double total_thread_time = 0.0;
    
    for (const auto& timing : thread_timings) {
        total_computation_time += timing.actual_computation_time.count() * 1000.0;
        total_thread_time += timing.total_thread_time.count() * 1000.0;
    }
    
    double computation_percentage = (total_computation_time / total_thread_time) * 100.0;
    
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Computation efficiency:       " << std::setw(9) << computation_percentage << " %" << std::endl;
    
    if (computation_percentage > 80.0) {
        std::cout << "  ✅ Excellent computation efficiency" << std::endl;
    } else if (computation_percentage > 60.0) {
        std::cout << "  ✅ Good computation efficiency" << std::endl;
    } else {
        std::cout << "  ⚠️  Low computation efficiency" << std::endl;
    }
}

// Timing getter implementations
double AMXInnerProductBF16PtrMTEnhanced::get_total_compute_time_ms() const
{
    return total_compute_time.count() * 1000.0;
}

double AMXInnerProductBF16PtrMTEnhanced::get_thread_spawn_time_ms() const
{
    return thread_spawn_time.count() * 1000.0;
}

double AMXInnerProductBF16PtrMTEnhanced::get_thread_join_time_ms() const
{
    return thread_join_time.count() * 1000.0;
}

double AMXInnerProductBF16PtrMTEnhanced::get_memory_allocation_time_ms() const
{
    return memory_allocation_time.count() * 1000.0;
}

double AMXInnerProductBF16PtrMTEnhanced::get_initialization_time_ms() const
{
    return initialization_time.count() * 1000.0;
}

double AMXInnerProductBF16PtrMTEnhanced::get_avg_thread_time_ms() const
{
    if (thread_timings.empty()) return 0.0;
    
    double total = 0.0;
    for (const auto& timing : thread_timings) {
        total += timing.total_thread_time.count() * 1000.0;
    }
    return total / thread_timings.size();
}

double AMXInnerProductBF16PtrMTEnhanced::get_avg_actual_computation_time_ms() const
{
    if (thread_timings.empty()) return 0.0;
    
    double total = 0.0;
    for (const auto& timing : thread_timings) {
        total += timing.actual_computation_time.count() * 1000.0;
    }
    return total / thread_timings.size();
}

// Helper function implementations
bfloat16_t AMXInnerProductBF16PtrMTEnhanced::float_to_bfloat16_enhanced(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = 0x00007FFF + ((bits >> 16) & 1);
    return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}

void AMXInnerProductBF16PtrMTEnhanced::float_to_bfloat16(const float* input_buffer, bfloat16_t* output_buffer, size_t count)
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
        output_buffer[i] = float_to_bfloat16_enhanced(input_buffer[i]);
    }
}
