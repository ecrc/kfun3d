
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "inc/ktime.h"
#include "inc/geometry.h"
#include "inc/ker/phy.h"

/*
  Calculates the residual
*/
void
compute_grad(struct grad *restrict grad)
{
  struct ktime ktime;
  setktime(&ktime);

  const size_t bsz = grad->bsz;
  const size_t dofs = grad->dofs;

  const uint32_t *restrict ie = grad->ie;
  const uint32_t *restrict part = grad->part;
  const uint32_t *restrict n0 = grad->n0;
  const uint32_t *restrict n1 = grad->n1;

  const double *restrict q = grad->q;

  const double *restrict w0termsx = grad->w0termsx;
  const double *restrict w0termsy = grad->w0termsy;
  const double *restrict w0termsz = grad->w0termsz;

  const double *restrict w1termsx = grad->w1termsx;
  const double *restrict w1termsy = grad->w1termsy;
  const double *restrict w1termsz = grad->w1termsz;

  double *restrict gradx0 = grad->gradx0;
  double *restrict gradx1 = grad->gradx1;
  double *restrict gradx2 = grad->gradx2;

  memset(gradx0, 0, dofs * sizeof(double));
  memset(gradx1, 0, dofs * sizeof(double));
  memset(gradx2, 0, dofs * sizeof(double));

  __assume_aligned(gradx0, 64);
  __assume_aligned(gradx1, 64);
  __assume_aligned(gradx2, 64);

  /*
    Calculates the gradients at the nodes using weighted least squares
    This solves using Gram-Schmidt
  */

#pragma omp parallel
  {
    const uint32_t t = omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    uint32_t i;

    for(i = ie0; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const uint32_t idx0 = bsz * node0;
      const uint32_t idx1 = bsz * node1;

      double dq;

      double termx;
      double termy;
      double termz;

      if(part[node0] == t)
      {
        termx = w0termsx[i];
        termy = w0termsy[i];
        termz = w0termsz[i];
                
        dq = q[idx1 + 0] - q[idx0 + 0];

        gradx0[idx0 + 0] += termx * dq;
        gradx1[idx0 + 0] += termy * dq;
        gradx2[idx0 + 0] += termz * dq;

        dq = q[idx1 + 1] - q[idx0 + 1];

        gradx0[idx0 + 1] += termx * dq;
        gradx1[idx0 + 1] += termy * dq;
        gradx2[idx0 + 1] += termz * dq;

        dq = q[idx1 + 2] - q[idx0 + 2];

        gradx0[idx0 + 2] += termx * dq;
        gradx1[idx0 + 2] += termy * dq;
        gradx2[idx0 + 2] += termz * dq;

        dq = q[idx1 + 3] - q[idx0 + 3];

        gradx0[idx0 + 3] += termx * dq;
        gradx1[idx0 + 3] += termy * dq;
        gradx2[idx0 + 3] += termz * dq; 
      }

      if(part[node1] == t)
      {
        termx = w1termsx[i];
        termy = w1termsy[i];
        termz = w1termsz[i];

        dq = q[idx0 + 0] - q[idx1 + 0];

        gradx0[idx1 + 0] += termx * dq;
        gradx1[idx1 + 0] += termy * dq;
        gradx2[idx1 + 0] += termz * dq;

        dq = q[idx0 + 1] - q[idx1 + 1];

        gradx0[idx1 + 1] += termx * dq;
        gradx1[idx1 + 1] += termy * dq;
        gradx2[idx1 + 1] += termz * dq;

        dq = q[idx0 + 2] - q[idx1 + 2];

        gradx0[idx1 + 2] += termx * dq;
        gradx1[idx1 + 2] += termy * dq;
        gradx2[idx1 + 2] += termz * dq;

        dq = q[idx0 + 3] - q[idx1 + 3];

        gradx0[idx1 + 3] += termx * dq;
        gradx1[idx1 + 3] += termy * dq;
        gradx2[idx1 + 3] += termz * dq;
      } 
    }
  }

  compute_time(&ktime, grad->t);
}
