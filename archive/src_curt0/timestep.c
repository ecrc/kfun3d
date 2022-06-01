
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include "inc/ktime.h"
#include "inc/geometry.h"
#include "inc/ker/phy.h"

/*
  Calculate a time step for each cell
  Note that this routine assumes conservative variables

  Local time stepping, loop over faces and calculate time step as:
  cdt = V / (sum(|u.n| + c.area)
  This is time step for CFL=1
  Late it will be multiplied by CFL
*/

void
compute_deltat2(struct delta *restrict delta)
{ 
  struct ktime ktime;
  setktime(&ktime);

  const size_t nnodes = delta->nnodes;
  const size_t nsnodes = delta->nsnodes;
  const size_t nfnodes = delta->nfnodes;
  const size_t bsz = delta->bsz;

  const uint32_t *restrict nsptr = delta->nsptr;
  const uint32_t *restrict nfptr = delta->nfptr;

  const double *restrict s_xyz0 = delta->s_xyz0;
  const double *restrict s_xyz1 = delta->s_xyz1;
  const double *restrict s_xyz2 = delta->s_xyz2;

  const double *restrict f_xyz0 = delta->f_xyz0;
  const double *restrict f_xyz1 = delta->f_xyz1;
  const double *restrict f_xyz2 = delta->f_xyz2;

  const uint32_t *restrict ie = delta->ie;
  const uint32_t *restrict part = delta->part;
  const uint32_t *restrict n0 = delta->n0;
  const uint32_t *restrict n1 = delta->n1;
  
  const double *restrict area = delta->area;
  const double *restrict q = delta->q;
  const double *restrict x0 = delta->x0;
  const double *restrict x1 = delta->x1;
  const double *restrict x2 = delta->x2;
  const double *restrict x3 = delta->x3;
  
	double *restrict cdt = delta->cdt;

  memset(cdt, 0, nnodes * sizeof(double));

#pragma omp parallel
  {
    const uint32_t t = omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    uint32_t i;

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

      const uint32_t idx0 = bsz * node0;
      const uint32_t idx1 = bsz * node1;

      /* Get average values on face */
      const double u = 0.5f * (q[idx0 + 1] + q[idx1 + 1]); // u
      const double v = 0.5f * (q[idx0 + 2] + q[idx1 + 2]); // v
      const double w = 0.5f * (q[idx0 + 3] + q[idx1 + 3]); // w

      const double ubar = xn * u + yn * v + zn * w;

      const double c = sqrt(ubar * ubar + BETA);

      double term = u * xnorm;
      term += v * ynorm;
      term += w * znorm;
      term = fabs(term) + c * ln;

      cdt[node0] = (part[node0] == t) ? cdt[node0] + term : cdt[node0];
      cdt[node1] = (part[node1] == t) ? cdt[node1] + term : cdt[node1];
    }
  }

  /*
    Now loop over boundaries and close the contours
  */

  uint32_t i;

#pragma omp parallel for
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

    const double c = sqrt(ubar_ * ubar_ + BETA);

    const double Vn = fabs(ubar) + c * ln;

    cdt[n] += Vn;
  }

#pragma omp parallel for
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

    const double c = sqrt(ubar_ * ubar_ + BETA);

    const double Vn = fabs(ubar) + c * ln;

    cdt[n] += Vn;
  }

#pragma omp parallel for  
  for(i = 0; i < nnodes; i++) cdt[i] = area[i] / cdt[i];

  compute_time(&ktime, delta->t);
}
