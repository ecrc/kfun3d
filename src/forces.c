#include <stdint.h>
#include <omp.h>
#include "geometry.h"
#include "core_kernel.h"
#include "bench.h"
#include "phy.h"

static void
_KRN_ComputeForces(
  const size_t snfc,
  const uint32_t bsz,
  const uint32_t *snfic,
  const uint32_t *fptrn0,
  const uint32_t *fptrn1,
  const uint32_t *fptrn2,
  const double* xyzx0,
  const double* xyzx1,
  const double* xyzx2,
  const double *q,
  double forces[])
{
  double lift = 0.f;
  double drag = 0.f;
  double momn = 0.f;

  uint32_t i;
  for(i = 0; i < snfc; i++)
  {
    uint32_t if0 = snfic[i];
    uint32_t if1 = snfic[i+1];

    uint32_t j;
#pragma omp parallel for reduction(+: lift, drag, momn)
    for(j = if0; j < if1; j++)
    {
      uint32_t n0 = fptrn0[j];
      uint32_t n1 = fptrn1[j];
      uint32_t n2 = fptrn2[j];

      double x0 = xyzx0[n0];
      double y0 = xyzx1[n0];
      double z0 = xyzx2[n0];
                                   
      double x1 = xyzx0[n1];
      double y1 = xyzx1[n1];
      double z1 = xyzx2[n1];
                                   
      double x2 = xyzx0[n2];
      double y2 = xyzx1[n2];
      double z2 = xyzx2[n2];

      /* Delta coordinates in all directions */
                 
      double ax = x1 - x0;
      double ay = y1 - y0;
      double az = z1 - z0;

      double bx = x2 - x0;
      double by = y2 - y0;
      double bz = z2 - z0;

      /*
        Norm points outward, away from grid interior.
        Norm magnitude is area of surface triangle.
      */
      
      double xnorm = ay * bz;
      xnorm -= az * by;
      xnorm = -0.5f * xnorm;
      
      double ynorm =  ax * bz;
      ynorm -= az * bx;
      ynorm = 0.5f * ynorm;
      
      /* Pressure values store at every face node */

      double p0 = q[bsz * n0];
      double p1 = q[bsz * n1];
      double p2 = q[bsz * n2];

      double press = (p0 + p1 + p2) / 3.f;
      double cp = 2.f * (press - 1.f);

      double dcx = cp * xnorm;
      double dcy = cp * ynorm;

      double xmid = x0 + x1 + x2;
      double ymid = y0 + y1 + y2;

      lift = lift - dcx * V + dcy * U;
      drag = drag + dcx * U + dcy * V;
      momn = momn + (xmid - 0.25f) * dcy - ymid * dcx;
    }
  }

  forces[0] = lift;
  forces[1] = drag;
  forces[2] = momn;
}

void
ComputeForces(const GEOMETRY *g, double forces[])
{
  //BENCH start_bench = rdbench();

  _KRN_ComputeForces(
  g->t->sz,
  g->c->b,
  g->t->i,
  g->b->fc->fptr->n0,
  g->b->fc->fptr->n1,
  g->b->fc->fptr->n2,
  g->n->xyz->x0,
  g->n->xyz->x1,
  g->n->xyz->x2,
  g->q->q,
  forces);

  //fun3d_log(start_bench, KERNEL_FORCES);
}