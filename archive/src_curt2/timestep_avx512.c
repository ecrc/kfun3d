
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <immintrin.h>
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

    const uint32_t lim = ie1 - ((ie1-ie0) % 8);

    const __m256i _bsz = _mm256_set1_epi32(bsz);

    const __m256i _shift1 = _mm256_set1_epi32(1);
    const __m256i _shift2 = _mm256_set1_epi32(2);
    const __m256i _shift3 = _mm256_set1_epi32(3);

    const __m512i _ng = _mm512_set1_epi32(-1);
    const __m512i _t = _mm512_set1_epi32(t);
    const __m512d _und = _mm512_undefined_pd();

    const __m512d _half = _mm512_set1_pd(0.5);
    const __m512d _beta = _mm512_set1_pd(BETA);

    uint32_t i;
    
    for(i = ie0; i < lim; i+=8)
    {
      const __m512d _x0 = _mm512_load_pd(x0+i);
      const __m512d _x1 = _mm512_load_pd(x1+i);
      const __m512d _x2 = _mm512_load_pd(x2+i);
      const __m512d _x3 = _mm512_load_pd(x3+i);

      const __m512d _xn = _mm512_mul_pd(_x0, _x3);
      const __m512d _yn = _mm512_mul_pd(_x1, _x3);
      const __m512d _zn = _mm512_mul_pd(_x2, _x3);

      const __m256i _n0 = _mm256_loadu_si256((__m256i *) &n0[i]);
      const __m256i _n1 = _mm256_loadu_si256((__m256i *) &n1[i]);

      const __m256i _idx0 = _mm256_mullo_epi32(_bsz, _n0);
      const __m256i _idx1 = _mm256_mullo_epi32(_bsz, _n1);

      const __m256i _idx01 = _mm256_add_epi32(_idx0, _shift1);
      const __m256i _idx11 = _mm256_add_epi32(_idx1, _shift1);

      const __m256i _idx02 = _mm256_add_epi32(_idx0, _shift2);
      const __m256i _idx12 = _mm256_add_epi32(_idx1, _shift2);

      const __m256i _idx03 = _mm256_add_epi32(_idx0, _shift3);
      const __m256i _idx13 = _mm256_add_epi32(_idx1, _shift3);

      const __m512d _q01 = _mm512_i32gather_pd(_idx01, &q[0], 8);
      const __m512d _q11 = _mm512_i32gather_pd(_idx11, &q[0], 8);

      const __m512d _q02 = _mm512_i32gather_pd(_idx02, &q[0], 8);
      const __m512d _q12 = _mm512_i32gather_pd(_idx12, &q[0], 8);

      const __m512d _q03 = _mm512_i32gather_pd(_idx03, &q[0], 8);
      const __m512d _q13 = _mm512_i32gather_pd(_idx13, &q[0], 8);

      __m512d _u = _mm512_add_pd(_q01, _q11);
      _u = _mm512_mul_pd(_u, _half);

      __m512d _v = _mm512_add_pd(_q02, _q12);
      _v = _mm512_mul_pd(_v, _half);

      __m512d _w = _mm512_add_pd(_q03, _q13);
      _w = _mm512_mul_pd(_w, _half);

      __m512d _ubar = _mm512_mul_pd(_x0, _u);
      _ubar = _mm512_fmadd_pd(_x1, _v, _ubar);
      _ubar = _mm512_fmadd_pd(_x2, _w, _ubar);

      __m512d _c = _mm512_fmadd_pd(_ubar, _ubar, _beta);

#ifdef __USE_AVX512_RCP
      _c = _mm512_rcp28_pd(_mm512_rsqrt28_pd(_c));
#else
      _c = _mm512_sqrt_pd(_c);
#endif

      __m512d _term = _mm512_mul_pd(_xn, _u);
      _term = _mm512_fmadd_pd(_yn, _v, _term);
      _term = _mm512_fmadd_pd(_zn, _w, _term);
      _term = _mm512_abs_pd(_term);
      _term = _mm512_fmadd_pd(_c, _x3, _term);

      __m512i _node, _part;
      __mmask _next;
    
      _node = _mm512_castsi256_si512(_n0);
      _part = _mm512_i32gather_epi32(_node, &part[0], 4);
      _next = _mm512_cmpeq_epi32_mask(_part, _t);

      do {
        __m512i _cd = _mm512_mask_conflict_epi32(_ng, _next, _node);
        __m512i _bnext = _mm512_broadcastmw_epi32(_next);
        __mmask _crt = _mm512_mask_testn_epi32_mask(_next, _cd, _bnext);

        __m512d _v = _mm512_mask_i32gather_pd(_und, _crt, _n0, &cdt[0], 8);
        __m512d _d = _mm512_mask_add_pd(_v, _crt, _v, _term);

        _mm512_mask_i32scatter_pd(&cdt[0], _crt, _n0, _d, 8);

        _next = _mm512_kxor(_next, _crt);

      } while(_next);
    
      _node = _mm512_castsi256_si512(_n1);
      _part = _mm512_i32gather_epi32(_node, &part[0], 4);
      _next = _mm512_cmpeq_epi32_mask(_part, _t);

      do {
        __m512i _cd = _mm512_mask_conflict_epi32(_ng, _next, _node);
        __m512i _bnext = _mm512_broadcastmw_epi32(_next);
        __mmask _crt = _mm512_mask_testn_epi32_mask(_next, _cd, _bnext);

        __m512d _v = _mm512_mask_i32gather_pd(_und, _crt, _n1, &cdt[0], 8);
        __m512d _d = _mm512_mask_add_pd(_v, _crt, _v, _term);

        _mm512_mask_i32scatter_pd(&cdt[0], _crt, _n1, _d, 8);

        _next = _mm512_kxor(_next, _crt);

      } while(_next);
    }

    for(i = lim; i < ie1; i++)
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

      double ubar = xn * u;
      ubar += yn * v;
      ubar += zn * w;

      const double c = sqrt(ubar * ubar + BETA);

      double term = u * xnorm;
      term += v * ynorm;
      term += w * znorm;
      term = fabs(term) + c * ln;

      cdt[node0] = (part[node0] == t) ? cdt[node0] + term : cdt[node0];
      cdt[node1] = (part[node1] == t) ? cdt[node1] + term : cdt[node1];
    }
  }

  uint32_t i;

  /*
    Now loop over boundaries and close the contours
  */

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
