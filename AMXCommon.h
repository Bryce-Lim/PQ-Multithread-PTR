#ifndef AMX_COMMON_H
#define AMX_COMMON_H

/**
 * @file AMXCommon.h
 * @brief Common definitions and constants for Intel AMX implementations
 * 
 * This header provides shared constants, data types, and structures used across
 * all AMX-based inner product implementations. Contains hardware-specific 
 * configurations optimized for Intel Advanced Matrix Extensions (AMX) instruction set.
 * 
 * Hardware Requirements:
 * - Intel CPU with AMX support (4th Gen Xeon Scalable or newer)
 * - AVX-512 and BF16 instruction support
 * - Minimum 32GB RAM for large-scale computations
 */

#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>

// ==================== AMX Hardware Constants ====================

/**
 * @brief AMX tile dimensions and configuration
 * 
 * These constants are determined by Intel AMX hardware specifications
 * and should not be modified unless targeting different hardware.
 */

/** @brief Maximum tile height (rows) supported by AMX hardware */
#define MAX_SIZE 16

/** @brief Maximum tile width (columns) in BF16 elements */
#define MAX_COLS 32

/** @brief Memory stride for optimal tile loading (in bytes) */
#define STRIDE 64

// ==================== System Call Constants ====================

/**
 * @brief Linux system call constants for AMX permission management
 * 
 * These constants are used with arch_prctl() system calls to enable
 * AMX functionality in user space applications.
 */

/** @brief Get extended state component permissions */
#define ARCH_GET_XCOMP_PERM 0x1022

/** @brief Request extended state component permissions */
#define ARCH_REQ_XCOMP_PERM 0x1023

/** @brief Extended state feature bit for tile configuration */
#define XFEATURE_XTILECFG 17

/** @brief Extended state feature bit for tile data */
#define XFEATURE_XTILEDATA 18

// ==================== Data Type Definitions ====================

/**
 * @brief Brain Floating Point 16-bit type
 * 
 * BF16 format provides optimal performance on AMX hardware while maintaining
 * reasonable numerical precision for machine learning workloads. Uses 1 sign bit,
 * 8 exponent bits, and 7 mantissa bits (same exponent range as FP32).
 */
typedef uint16_t bfloat16_t;

// ==================== AMX Tile Configuration ====================

/**
 * @brief AMX tile configuration data structure
 * 
 * This structure defines the layout and properties of AMX tiles used for
 * matrix operations. Must be properly initialized before any tile operations.
 * 
 * Memory Layout:
 * - palette_id: Identifies the tile configuration set (typically 1)
 * - start_row: Starting row for tile operations (typically 0)
 * - colsb[16]: Column widths in bytes for each of 8 tiles
 * - rows[16]: Row counts for each of 8 tiles
 * 
 * Typical Configuration:
 * - Tile 0: 16×16 float32 accumulator (16 rows × 64 bytes)
 * - Tiles 1-3: 16×32 bfloat16 operands (16 rows × 64 bytes)
 * 
 * @note This structure must be aligned and zero-initialized before use
 * @note Configuration is per-thread and must be set up in each worker thread
 */
typedef struct __tile_config {
    /** @brief Palette identifier (1 for standard AMX operations) */
    uint8_t palette_id;
    
    /** @brief Starting row index (typically 0) */
    uint8_t start_row;
    
    /** @brief Reserved bytes for future extensions */
    uint8_t reserved_0[14];
    
    /** @brief Column widths in bytes for each tile [0-7] */
    uint16_t colsb[16];
    
    /** @brief Row counts for each tile [0-7] */
    uint8_t rows[16];
} __tilecfg;

// ==================== Performance Optimization Hints ====================

/**
 * @brief Recommended data alignment constraints for optimal AMX performance
 * 
 * These constraints ensure efficient memory access patterns and maximal
 * utilization of AMX tile operations. Violating these constraints will
 * result in suboptimal performance or runtime errors.
 */

/** @brief Vector dimension alignment requirement (multiple of 32) */
#define AMX_DIMENSION_ALIGNMENT 32

/** @brief Centroid count alignment requirement (multiple of 16) */
#define AMX_CENTROID_ALIGNMENT 16

/** @brief Data batch size alignment requirement (multiple of 16) */
#define AMX_DATA_ALIGNMENT 16

/** @brief Memory alignment for optimal cache performance (64-byte aligned) */
#define AMX_MEMORY_ALIGNMENT 64

// ==================== Utility Macros ====================

/**
 * @brief Compile-time validation macros for AMX constraints
 * 
 * These macros help catch configuration errors at compile time rather
 * than runtime, improving development efficiency and reliability.
 */

/** @brief Validate dimension meets AMX alignment requirements */
#define AMX_VALIDATE_DIMENSION(dim) \
    static_assert((dim) % AMX_DIMENSION_ALIGNMENT == 0, "Dimension must be multiple of 32")

/** @brief Validate centroid count meets AMX alignment requirements */
#define AMX_VALIDATE_CENTROIDS(count) \
    static_assert((count) % AMX_CENTROID_ALIGNMENT == 0, "Centroid count must be multiple of 16")

/** @brief Validate data count meets AMX alignment requirements */
#define AMX_VALIDATE_DATA_COUNT(count) \
    static_assert((count) % AMX_DATA_ALIGNMENT == 0, "Data count must be multiple of 16")

// ==================== Error Codes ====================

/**
 * @brief AMX-specific error codes for consistent error handling
 */
typedef enum {
    AMX_SUCCESS = 0,              /**< Operation completed successfully */
    AMX_ERROR_NOT_INITIALIZED,    /**< AMX subsystem not initialized */
    AMX_ERROR_INVALID_DIMENSION,  /**< Invalid dimension size */
    AMX_ERROR_INVALID_COUNT,      /**< Invalid data or centroid count */
    AMX_ERROR_NULL_POINTER,       /**< Null pointer passed to function */
    AMX_ERROR_HARDWARE_SUPPORT,   /**< AMX not supported on this hardware */
    AMX_ERROR_MEMORY_ALLOCATION,  /**< Memory allocation failed */
    AMX_ERROR_ALIGNMENT           /**< Data alignment requirements not met */
} amx_error_t;

// ==================== Function Attributes ====================

/**
 * @brief Function attributes for performance optimization
 */

/** @brief Mark functions as inline for performance-critical paths */
#define AMX_INLINE inline __attribute__((always_inline))

/** @brief Mark functions as hot for optimization prioritization */
#define AMX_HOT __attribute__((hot))

/** @brief Mark functions as pure (no side effects) for optimization */
#define AMX_PURE __attribute__((pure))

/** @brief Mark functions as const (no side effects, same output for same input) */
#define AMX_CONST __attribute__((const))

// ==================== Debugging Support ====================

#ifdef DEBUG
/** @brief Debug assertion macro (enabled only in debug builds) */
#define AMX_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "AMX Assertion failed: %s at %s:%d\n", \
                    (message), __FILE__, __LINE__); \
            abort(); \
        } \
    } while(0)
#else
#define AMX_ASSERT(condition, message) ((void)0)
#endif

/** @brief Debug trace macro for performance analysis */
#ifdef AMX_TRACE
#define AMX_TRACE_ENTER(func_name) \
    fprintf(stderr, "AMX: Entering %s\n", (func_name))
#define AMX_TRACE_EXIT(func_name) \
    fprintf(stderr, "AMX: Exiting %s\n", (func_name))
#else
#define AMX_TRACE_ENTER(func_name) ((void)0)
#define AMX_TRACE_EXIT(func_name) ((void)0)
#endif

// ==================== Version Information ====================

/** @brief Minimum required GCC version for AMX support */
#define AMX_MIN_GCC_VERSION_MAJOR 11
#define AMX_MIN_GCC_VERSION_MINOR 0

// ==================== Compatibility Checks ====================

/**
 * @brief Compile-time compatibility validation
 */
#if defined(__GNUC__) && !defined(__clang__)
    #if (__GNUC__ < AMX_MIN_GCC_VERSION_MAJOR) || \
        (__GNUC__ == AMX_MIN_GCC_VERSION_MAJOR && __GNUC_MINOR__ < AMX_MIN_GCC_VERSION_MINOR)
        #error "AMX implementation requires GCC 11.0 or later for proper AMX instruction support"
    #endif
#endif

#ifndef __AMX_TILE__
    #warning "AMX tile instructions may not be available. Compile with -mamx-tile for full support."
#endif

#ifndef __AMX_BF16__
    #warning "AMX BF16 instructions may not be available. Compile with -mamx-bf16 for optimal performance."
#endif

#endif // AMX_COMMON_H
