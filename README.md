# AMX Inner Product Acceleration

A high-performance implementation of inner product (dot product) computation using Intel AMX (Advanced Matrix Extensions) instructions with comprehensive performance analysis and comparison against standard implementations.

## 🚀 Features

### Core Implementations
- **Scalar Reference**: Standard floating-point implementation for baseline comparison
- **Single-threaded AMX**: BF16-optimized AMX implementation with detailed timing
- **Multi-threaded AMX Enhanced**: Advanced multi-threaded AMX with comprehensive performance analysis
- **HNSWLIB Integration**: Performance comparison with the popular HNSWLIB library

### Advanced Performance Analysis
- **Detailed Timing Breakdown**: Per-component timing analysis (tile loading, computation, merging)
- **Threading Efficiency Analysis**: Load balancing, memory efficiency, and bottleneck identification
- **Accuracy Validation**: Comprehensive accuracy comparison between implementations
- **Throughput Metrics**: GFLOPS calculations and performance scaling analysis

### Key Optimizations
- **Memory-optimized Chunking**: Efficient data formatting for AMX tile operations
- **Prefetching Strategies**: Advanced memory prefetching for improved cache utilization
- **SIMD Vectorization**: AVX-512 optimized BF16 conversion routines
- **Load Balancing**: Dynamic work distribution across threads

## 📋 Requirements

### Hardware Requirements
- **Intel CPU with AMX support** (4th Gen Xeon Scalable or newer)
- **Minimum 32 GB RAM** (for large-scale testing)
- **AVX-512 and BF16 instruction support**

### Software Requirements
- **GCC 11+** with AMX instruction support
- **OpenMP** for multi-threading
- **Apache Arrow/Parquet** libraries for data loading
- **pkg-config** for dependency management

### Optional Dependencies
- **HNSWLIB** (auto-downloaded) for performance comparison
- **Large embedding dataset** for comprehensive testing

## 🛠️ Installation

### Setup
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y \
    build-essential libomp-dev libarrow-dev libparquet-dev pkg-config git

# Clone repository
git clone <repository-url>
cd amx-inner-product

# Setup dependencies and build
make setup
make all
```

## 🏃‍♂️ Running

### Basic Testing
```bash
# Run comprehensive AMX analysis
make test-enhanced

# Run large-scale performance tests, compare AMX vs HNSWLIB performance
make test-large
```

### Example Code
```cpp
#include "AMXInnerProductBF16PtrMTEnhanced.h"

// Initialize AMX calculator with 8 threads
AMXInnerProductBF16PtrMTEnhanced amx_calc(8);
if (!amx_calc.initialize()) {
    throw std::runtime_error("AMX initialization failed");
}

// Compute inner products
size_t result = amx_calc.compute_inner_products(
    data_points,     // BF16 input vectors
    data_count,      // Number of vectors
    centroids,       // BF16 centroids
    centroid_count,  // Number of centroids
    dimension,       // Vector dimension (must be multiple of 64)
    distances        // Output array
);

// Analyze performance
amx_calc.print_comprehensive_timing_stats();
amx_calc.print_per_thread_breakdown();
```

## 🔧 Configuration

### AMX Constraints
The implementation requires specific data alignment for optimal performance:
- **Vector dimension**: Must be divisible by 32
- **Number of centroids**: Must be divisible by 16  
- **Data batch size**: Must be divisible by 16 x (number of threads used)

### Tuning Parameters
```cpp
// In AMXCommon.h
#define MAX_SIZE 16        // AMX tile size
#define MAX_COLS 32        // AMX tile width
#define STRIDE 64          // Memory stride

// Threading configuration
const size_t optimal_threads = std::thread::hardware_concurrency();
```

## 📈 Detailed Performance Analysis

The enhanced implementation provides comprehensive performance metrics:

### Timing Breakdown
- **Total compute time**: End-to-end execution time
- **Thread spawn/join overhead**: Multi-threading setup costs
- **Memory allocation**: Data structure initialization
- **Tile loading**: AMX register loading time
- **Actual computation**: Pure AMX instruction execution
- **Result merging**: Output data assembly

### Threading Analysis
- **Load balancing**: Work distribution across threads
- **Memory efficiency**: Cache utilization and bandwidth analysis
- **Threading overhead**: Synchronization and coordination costs
- **Bottleneck identification**: Performance limiting factors

### Example Output
```
=== COMPREHENSIVE MULTITHREADED AMX TIMING ANALYSIS ===
📊 Main Process Timing:
  Total compute time:           2,847.234 ms
  ├─ Initialization:               12.345 ms
  ├─ Memory allocation:            45.678 ms  
  ├─ Thread spawning:               5.432 ms
  ├─ Thread joining:                3.210 ms
  └─ Actual parallel work:      2,780.569 ms (avg)

🧵 Per-Thread Timing Statistics:
  Number of threads:                    8
  Min thread time:              2,765.123 ms
  Max thread time:              2,795.876 ms
  Load imbalance:                    1.1 %

📈 Performance Counters:
  Total tile loads:             1,234,567
  Total AMX operations:           456,789
  AMX ops per millisecond:          164.3
```

## 📚 Documentation

### Implementation Details
- **AMX Tile Configuration**: 16x32 BF16 tiles with float32 accumulation (for 16 rows x 64 bytes)
- **Memory Layout**: Row-major storage with cache-optimized chunking
- **Prefetching Strategy**: Software prefetching with configurable distance
- **Error Handling**: Comprehensive validation and graceful degradation

### API Reference
See header files for detailed API documentation:
- `AMXInnerProductBF16PtrMTEnhanced.h` - Main enhanced implementation
- `AMXInnerProductBF16Ptr.h` - Single-threaded version
- `AMXCommon.h` - Shared constants and types

---

**Performance Note**: This implementation is optimized for Intel AMX-capable processors. Performance on other architectures will fall back to scalar implementations. For optimal results, ensure your hardware supports AMX instructions and your dataset meets the alignment requirements specified above.
