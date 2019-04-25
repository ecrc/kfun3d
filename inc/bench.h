#ifndef __FUN3D_INC_BENCH_H
#define __FUN3D_INC_BENCH_H

#include <time.h>
#include <stdint.h>

typedef struct bench {
  double time_omp;
  clock_t time_clock;
  struct timespec time_posix;
  uint64_t cycles;
} BENCH;

BENCH
rdbench();

void
fun3d_close_log();

void
fun3d_init_log();

#define KERNEL_FLUX     0
#define KERNEL_TIMESTEP 1
#define KERNEL_FILL     2
#define KERNEL_SETUP    3
#define KERNEL_CORE     4
#define FUN3D           5
#define KERNEL_BLAS     6
#define KERNEL_SPTRSV   7
#define KERNEL_NUMILU   8
#define KERNEL_FORCES   9

void
fun3d_log(const BENCH, const uint32_t);

#endif /* __FUN3D_INC_BENCH_H */