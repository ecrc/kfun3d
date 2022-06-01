
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <immintrin.h>
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

  const double *restrict w0termsx = grad->w0termsx;
  const double *restrict w0termsy = grad->w0termsy;
  const double *restrict w0termsz = grad->w0termsz;

  const double *restrict w1termsx = grad->w1termsx;
  const double *restrict w1termsy = grad->w1termsy;
  const double *restrict w1termsz = grad->w1termsz;

  const uint32_t *restrict ie = grad->ie;
  const uint32_t *restrict part = grad->part;
  const uint32_t *restrict n0 = grad->n0;
  const uint32_t *restrict n1 = grad->n1;

  const double *restrict q = grad->q;

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

    const uint32_t lim = ie1 - ((ie1-ie0) % 8);

    const __m256i _bsz = _mm256_set1_epi32(bsz);

    const __m256i _shift1 = _mm256_set1_epi32(1);
    const __m256i _shift2 = _mm256_set1_epi32(2);
    const __m256i _shift3 = _mm256_set1_epi32(3);

    const __m512i _ng = _mm512_set1_epi32(-1);
    const __m512i _t = _mm512_set1_epi32(t);
    const __m512d _und = _mm512_undefined_pd();

    uint32_t i;

    for(i = ie0; i < lim; i+=8)
    {
      const __m256i _n0 = _mm256_load_si256((__m256i const *) &n0[i]);
      const __m256i _n1 = _mm256_load_si256((__m256i const *) &n1[i]);

      /* Compute the indices for the Q and the Grad vectors */

      const __m256i _idx0 = _mm256_mullo_epi32(_bsz, _n0);
      const __m256i _idx1 = _mm256_mullo_epi32(_bsz, _n1);

      const __m256i _idx01 = _mm256_add_epi32(_idx0, _shift1);
      const __m256i _idx11 = _mm256_add_epi32(_idx1, _shift1);

      const __m256i _idx02 = _mm256_add_epi32(_idx0, _shift2);
      const __m256i _idx12 = _mm256_add_epi32(_idx1, _shift2);

      const __m256i _idx03 = _mm256_add_epi32(_idx0, _shift3);
      const __m256i _idx13 = _mm256_add_epi32(_idx1, _shift3);

      /* Gather the Q vector */

      /* Pressure */
      const __m512d _q00 = _mm512_i32gather_pd(_idx0, &q[0], 8);
      const __m512d _q10 = _mm512_i32gather_pd(_idx1, &q[0], 8);

      /* Velocity u */
      const __m512d _q01 = _mm512_i32gather_pd(_idx01, &q[0], 8);
      const __m512d _q11 = _mm512_i32gather_pd(_idx11, &q[0], 8);

      /* Velocity v */
      const __m512d _q02 = _mm512_i32gather_pd(_idx02, &q[0], 8);
      const __m512d _q12 = _mm512_i32gather_pd(_idx12, &q[0], 8);

      /* Velocity w */
      const __m512d _q03 = _mm512_i32gather_pd(_idx03, &q[0], 8);
      const __m512d _q13 = _mm512_i32gather_pd(_idx13, &q[0], 8);

      /* Update */

      __m512i _node, _part;
      __m512d _dq0, _dq1, _dq2, _dq3, _tx, _ty, _tz;
      __mmask _next;

      _node = _mm512_castsi256_si512(_n0);
      _part = _mm512_i32gather_epi32(_node, &part[0], 4);
      _next = _mm512_cmpeq_epi32_mask(_part, _t);

      _tx = _mm512_load_pd((void const *) &w0termsx[i]);
      _ty = _mm512_load_pd((void const *) &w0termsy[i]);
      _tz = _mm512_load_pd((void const *) &w0termsz[i]);

      /* compute the gradient terms of node 0*/

      _dq0 = _mm512_maskz_sub_pd(_next, _q10, _q00);
      _dq1 = _mm512_maskz_sub_pd(_next, _q11, _q01);
      _dq2 = _mm512_maskz_sub_pd(_next, _q12, _q02);
      _dq3 = _mm512_maskz_sub_pd(_next, _q13, _q03);
      
      /* conflict detection instructions with multiple node update */

      /* node 0 contributions */

      do {
        __m512i _cd, _bnext;
        __m512d _v, _d;
        __mmask _crt;

        _cd = _mm512_mask_conflict_epi32(_ng, _next, _node);
        _bnext = _mm512_broadcastmw_epi32(_next);
        _crt = _mm512_mask_testn_epi32_mask(_next, _cd, _bnext);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx0, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq0, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx0, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx0, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq0, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx0, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx0, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq0, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx0, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx01, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq1, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx01, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx01, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq1, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx01, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx01, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq1, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx01, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx02, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq2, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx02, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx02, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq2, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx02, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx02, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq2, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx02, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx03, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq3, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx03, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx03, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq3, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx03, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx03, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq3, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx03, _d, 8);

        _next = _mm512_kxor(_next, _crt);

      } while(_next);

      _node = _mm512_castsi256_si512(_n1);
      _part = _mm512_i32gather_epi32(_node, &part[0], 4);
      _next = _mm512_cmpeq_epi32_mask(_part, _t);

      /* compute the gradient terms of node 0*/

      _tx = _mm512_load_pd((void const *) &w1termsx[i]);
      _ty = _mm512_load_pd((void const *) &w1termsy[i]);
      _tz = _mm512_load_pd((void const *) &w1termsz[i]);

      _dq0 = _mm512_maskz_sub_pd(_next, _q00, _q10);
      _dq1 = _mm512_maskz_sub_pd(_next, _q01, _q11);
      _dq2 = _mm512_maskz_sub_pd(_next, _q02, _q12);
      _dq3 = _mm512_maskz_sub_pd(_next, _q03, _q13);
      
      /* conflict detection instructions with multiple node update */

      /* node 1 contributions */

      do {
        __m512i _cd, _bnext;
        __m512d _v, _d;
        __mmask _crt;

        _cd = _mm512_mask_conflict_epi32(_ng, _next, _node);
        _bnext = _mm512_broadcastmw_epi32(_next);
        _crt = _mm512_mask_testn_epi32_mask(_next, _cd, _bnext);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx1, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq0, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx1, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx1, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq0, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx1, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx1, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq0, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx1, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx11, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq1, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx11, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx11, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq1, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx11, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx11, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq1, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx11, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx12, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq2, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx12, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx12, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq2, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx12, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx12, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq2, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx12, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx13, &gradx0[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tx, _dq3, _v);
        _mm512_mask_i32scatter_pd(&gradx0[0], _crt, _idx13, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx13, &gradx1[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _ty, _dq3, _v);
        _mm512_mask_i32scatter_pd(&gradx1[0], _crt, _idx13, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx13, &gradx2[0], 8);
        _d = _mm512_maskz_fmadd_pd(_crt, _tz, _dq3, _v);
        _mm512_mask_i32scatter_pd(&gradx2[0], _crt, _idx13, _d, 8);

        _next = _mm512_kxor(_next, _crt);

      } while(_next);
    }

    for(i = lim; i < ie1; i++)
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
