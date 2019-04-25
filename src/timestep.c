#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include "geometry.h"
#include "bench.h"
#include "phy.h"
#include "core_kernel.h"

/*
  Calculate a time step for each cell
  Note that this routine assumes conservative variables

  Local time stepping, loop over faces and calculate time step as:
  cdt = V / (sum(|u.n| + c.area)
  This is time step for CFL=1
  Late it will be multiplied by CFL
*/

static void
_KRN_ComputeTimeStep(
  const size_t nnodes,
  const size_t nsnodes,
  const size_t nfnodes,
  const uint32_t bsz,
  const uint32_t *nsptr,
  const uint32_t *nfptr,
  const double *s_xyz0,
  const double *s_xyz1,
  const double *s_xyz2,
  const double *f_xyz0,
  const double *f_xyz1,
  const double *f_xyz2,
  const uint32_t *ie,
  const uint32_t *part,
  const uint32_t *n0,
  const uint32_t *n1,
  const double *x0,
  const double *x1,
  const double *x2,
  const double *x3,
  const double cfl,
  const double *q,
	double *cdt)
{
  memset(cdt, 0, nnodes * sizeof(double));

#pragma omp parallel
  {
    uint32_t i;

    const uint32_t t = (uint32_t) omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    for(i = ie0; i < ie1; i++)
    {
      const double xn = x0[i];
      const double yn = x1[i];
      const double zn = x2[i];
      const double ln = x3[i];

      const double xnorm = xn * ln;
      const double ynorm = yn * ln;
      const double znorm = zn * ln;

      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const uint32_t idx0 = (unsigned int) bsz * node0;
      const uint32_t idx1 = (unsigned int) bsz * node1;

      /* Get average values on face */
      const double u = 0.5f * (q[idx0 + 1] + q[idx1 + 1]); // u
      const double v = 0.5f * (q[idx0 + 2] + q[idx1 + 2]); // v
      const double w = 0.5f * (q[idx0 + 3] + q[idx1 + 3]); // w

      const double ubar = xn * u + yn * v + zn * w;

      const double c = sqrt(ubar * ubar + B);

      double term = u * xnorm;
      term += v * ynorm;
      term += w * znorm;
      term = fabs(term) + c * ln;

      cdt[node0] = (part[node0] == t) ? cdt[node0] + term : cdt[node0];
      cdt[node1] = (part[node1] == t) ? cdt[node1] + term : cdt[node1];
    }

#pragma omp barrier

#pragma omp for
    for(i = 0; i < nsnodes; i++)
    {
      const uint32_t n = nsptr[i];

      const double xn = s_xyz0[i];
      const double yn = s_xyz1[i];
      const double zn = s_xyz2[i];

      const double ln = sqrt(xn * xn + yn * yn + zn * zn);

      const double u = q[bsz * n + 1];
      const double v = q[bsz * n + 2];
      const double w = q[bsz * n + 3];

      const double ubar = u * xn + v * yn + w * zn;

      const double ubar_ = ubar / ln;

      const double c = sqrt(ubar_ * ubar_ + B);

      const double Vn = fabs(ubar) + c * ln;

      cdt[n] += Vn;
    }

#pragma omp barrier

#pragma omp for
    for(i = 0; i < nfnodes; i++)
    {
      const uint32_t n = nfptr[i];

      const double xn = f_xyz0[i];
      const double yn = f_xyz1[i];
      const double zn = f_xyz2[i];

      const double ln = sqrt(xn * xn + yn * yn + zn * zn);

      const double u = q[bsz * n + 1];
      const double v = q[bsz * n + 2];
      const double w = q[bsz * n + 3];

      const double ubar = u * xn + v * yn + w * zn;

      const double ubar_ = ubar / ln;

      const double c = sqrt(ubar_ * ubar_ + B);

      const double Vn = fabs(ubar) + c * ln;

      cdt[n] += Vn;
    }

#pragma omp barrier

#pragma omp for  
    for(i = 0; i < nnodes; i++) cdt[i] /= cfl;
  }
}

void
ComputeTimeStep(const double cfl, GEOMETRY *g)
{
  BENCH start_bench = rdbench();

  _KRN_ComputeTimeStep(
    g->n->sz,
    g->b->s->sz,
    g->b->f->sz,
    g->c->b,
    g->b->s->nptr,
    g->b->f->nptr,
    g->b->s->xyz->x0,
    g->b->s->xyz->x1,
    g->b->s->xyz->x2,
    g->b->f->xyz->x0,
    g->b->f->xyz->x1,
    g->b->f->xyz->x2,
    g->s->i,
    g->n->part,
    g->e->eptr->n0,
    g->e->eptr->n1,
    g->e->xyzn->x0,
    g->e->xyzn->x1,
    g->e->xyzn->x2,
    g->e->xyzn->x3,
    cfl,
    g->q->q,
    g->n->cdt
  );

  fun3d_log(start_bench, KERNEL_FLUX);
}