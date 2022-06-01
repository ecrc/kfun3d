
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include "inc/ktime.h"
#include "inc/geometry.h"
#include "inc/ker/phy.h"

inline void
iguess(struct igtbl *restrict ig)
{
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
}
