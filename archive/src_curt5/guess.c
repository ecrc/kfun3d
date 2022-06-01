
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <ktime.h>
#include <geometry.h>
#ifdef __USE_HW_COUNTER
#include <perf.h>
#include <kperf.h>
#endif
#include <phy.h>

inline void
iguess(struct igtbl *restrict ig)
{
#ifdef __USE_HW_COUNTER
  const struct fd fd = ig->perf_counters->fd;
  
  struct counters start;
  perf_read(fd, &start);

  const uint64_t icycle = __rdtsc();
#endif

  struct ktime ktime;
  setktime(&ktime);

  const size_t sz = ig->sz;
  const size_t bsz = ig->bsz;

  struct ivals *restrict iv = ig->iv;
  double *restrict q0 = ig->q0;
  double *restrict q1 = ig->q1;

  double conv = ALPHA / (180.f / M_PI);

  iv->p = 1.f;        /* Pressure */
  iv->u = cos(conv);  /* Velocity */
  iv->v = sin(conv);  /* Velocity */
  iv->w = 0.f;        /* Velocity */

  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < sz; i++)
  {
#ifdef __USE_COMPRESSIBLE_FLOW

#else
    q0[i * bsz + 0] = iv->p;
    q0[i * bsz + 1] = iv->u;
    q0[i * bsz + 2] = iv->v;
    q0[i * bsz + 3] = iv->w;

    q1[i * bsz + 0] = iv->p;
    q1[i * bsz + 1] = iv->u;
    q1[i * bsz + 2] = iv->v;
    q1[i * bsz + 3] = iv->w;
#endif    
  }

  compute_time(&ktime, &ig->t->iguess);

#ifdef __USE_HW_COUNTER
  const uint64_t cycle = __rdtsc() - icycle;

  struct counters end;
  perf_read(fd, &end);

  struct tot tot;
  perf_calc(start, end, &tot);

  ig->perf_counters->ctrs->setup.cycles += cycle;
  ig->perf_counters->ctrs->setup.tot.imcR += tot.imcR;
  ig->perf_counters->ctrs->setup.tot.imcW += tot.imcW;
  ig->perf_counters->ctrs->setup.tot.edcR += tot.edcR;
  ig->perf_counters->ctrs->setup.tot.edcW += tot.edcW;
#endif
}
