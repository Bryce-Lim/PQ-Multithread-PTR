// AMXCommon.h - Common definitions for AMX implementations
#ifndef AMX_COMMON_H
#define AMX_COMMON_H

#include <immintrin.h>
#include <stdint.h>
#include <stdbool.h>

// AMX constants
#define MAX_SIZE 16
#define MAX_COLS 32
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

// Define bfloat16 type
typedef uint16_t bfloat16_t;

// Define tile config data structure (only once)
typedef struct __tile_config
{
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

#endif // AMX_COMMON_H
