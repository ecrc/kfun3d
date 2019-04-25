#include <omp.h>
#include <time.h>
#include <inttypes.h>
#include <stdint.h>
#ifndef __INTEL_COMPILER
#ifndef __llvm__
#ifndef __clang__
#include <x86intrin.h> /* ONLY GCC */
#endif /* INTEL */
#endif /* LLVM */
#endif /* CLANG */
#include "bench.h"
#include "allocator.h"
#include "utils.h"

static void
acubench(const BENCH read_bench, BENCH *bench)
{
  bench->time_omp += read_bench.time_omp;
  bench->time_clock += read_bench.time_clock;
  bench->cycles += read_bench.cycles;
  {
    bench->time_posix.tv_sec += read_bench.time_posix.tv_sec;
    bench->time_posix.tv_nsec += read_bench.time_posix.tv_nsec;
  }
}

static BENCH
zerobench()
{
  BENCH bench;

  bench.time_omp = 0.f;
  bench.time_clock = 0.f;
  bench.cycles = 0;
  {
    bench.time_posix.tv_sec = 0.f;
    bench.time_posix.tv_nsec = 0.f;
  }

  return(bench);
}

static void
prtbench(const BENCH bench, const char *msg)
{
  fun3d_printf(0, "[%s]\n", msg);
  fun3d_printf(1, "CYCLES: %"PRIu64"\n", bench.cycles);
  fun3d_printf(1, "CLOCK TICKS: %lf\n", (double) ((double) (bench.time_clock) / CLOCKS_PER_SEC));
  fun3d_printf(1, "OMP WALL-CLOCK: %lf\n", (double) bench.time_omp);

  double posix_time = (double) bench.time_posix.tv_nsec / 1000000000.f;
  posix_time += (double) bench.time_posix.tv_sec;
  fun3d_printf(1, "POSIX WALL-CLOCK: %lf\n", posix_time);
}

static BENCH
clcbench(const BENCH start_bench, const BENCH end_bench)
{
  BENCH bench;

  bench.time_omp = end_bench.time_omp - start_bench.time_omp;
  bench.time_clock = end_bench.time_clock - start_bench.time_clock;
  bench.cycles = end_bench.cycles - start_bench.cycles;
  {
    bench.time_posix.tv_sec = (end_bench.time_posix.tv_sec - start_bench.time_posix.tv_sec);
    bench.time_posix.tv_nsec = (end_bench.time_posix.tv_nsec - start_bench.time_posix.tv_nsec);
  }

  return(bench);
}

typedef struct benchtable {
  BENCH flux;
  //BENCH timestep;
  //BENCH fill;
  BENCH setup;
  BENCH core;
  BENCH fun3d;
  BENCH blas;
  BENCH sptrsv;
  BENCH numilu;
  //BENCH forces;
} BENCHTABLE;

static BENCHTABLE *benchtable;

BENCH
rdbench()
{
  BENCH bench;

  bench.time_omp = omp_get_wtime();
  bench.time_clock = clock();
  bench.cycles = __rdtsc();
  clock_gettime(CLOCK_MONOTONIC, &bench.time_posix);

  return(bench);
}

void
fun3d_init_log()
{
  benchtable = (BENCHTABLE *) fun3d_malloc(1, sizeof(BENCHTABLE));

  benchtable->flux = zerobench();
  //benchtable->timestep = zerobench();
  //benchtable->fill = zerobench();
  benchtable->setup = zerobench();
  benchtable->core = zerobench();
  benchtable->fun3d = zerobench();
  benchtable->blas = zerobench();
  benchtable->sptrsv = zerobench();
  benchtable->numilu = zerobench();
  //benchtable->forces = zerobench();
}

void
fun3d_close_log()
{
  prtbench(benchtable->flux, "FLUX");
  //prtbench(benchtable->timestep, "TIMESTEP");
  //prtbench(benchtable->fill, "FILL");
  prtbench(benchtable->setup, "SETUP STAGE");
  prtbench(benchtable->core, "ALL KERNELS");
  prtbench(benchtable->fun3d, "FUN3D");
  prtbench(benchtable->blas, "BLAS");
  prtbench(benchtable->sptrsv, "SpTRSV");
  prtbench(benchtable->numilu, "ILU");
  //prtbench(benchtable->forces, "FORCES");

  fun3d_free(benchtable);
}

void
fun3d_log(const BENCH start_bench, const uint32_t k)
{
  const BENCH b = clcbench(start_bench, rdbench());

  switch(k)
  {
    case(KERNEL_FLUX):
      acubench(b, &benchtable->flux);
    break;
    //case(KERNEL_TIMESTEP):
    //  acubench(b, &benchtable->timestep);
    //break;
    //case(KERNEL_FILL):
    //  acubench(b, &benchtable->fill);
    //break;
    case(KERNEL_SETUP):
      acubench(b, &benchtable->setup);
    break;
    case(KERNEL_CORE):
      acubench(b, &benchtable->core);
    break;
    case(FUN3D):
      acubench(b, &benchtable->fun3d);
    break;
    case(KERNEL_BLAS):
      acubench(b, &benchtable->blas);
    break;
    case(KERNEL_SPTRSV):
      acubench(b, &benchtable->sptrsv);
    break;
    case(KERNEL_NUMILU):
      acubench(b, &benchtable->numilu);
    break;
    //case(KERNEL_FORCES):
    //  acubench(b, &benchtable->forces);
    //break;
  }
}