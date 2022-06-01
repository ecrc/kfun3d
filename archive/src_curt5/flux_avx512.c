
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <mathimf.h>
#include <immintrin.h>
#include <ktime.h>
#include <geometry.h>
#ifdef __USE_HW_COUNTER
#include <perf.h>
#include <kperf.h>
#endif
#include <phy.h>

#define MAG0  (0.5 / 3)
#define MAG1  (-MAG0)

/*
  Calculates the residual
*/

void
compute_residual(struct residual *restrict res)
{

#ifdef __USE_HW_COUNTER
  const struct fd fd = res->perf_counters->fd;

  struct counters start;
  perf_read(fd, &start);

  const uint64_t icycle = __rdtsc();
#endif

  struct ktime ktime;
  setktime(&ktime);

  const size_t bsz = res->bsz;
  const size_t nfnodes = res->nfnodes;
  const size_t dofs = res->dofs;
  const uint32_t snfc = res->snfc;

  const double pressure = res->pressure;
  const double velocity_u = res->velocity_u;
  const double velocity_v = res->velocity_v;
  const double velocity_w = res->velocity_w;

  const double *restrict f_xyz0 = res->f_xyz0;
  const double *restrict f_xyz1 = res->f_xyz1;
  const double *restrict f_xyz2 = res->f_xyz2;

  const double *restrict xyz0 = res->xyz0;
  const double *restrict xyz1 = res->xyz1;
  const double *restrict xyz2 = res->xyz2;

  const uint32_t *restrict ie = res->ie;
  const uint32_t *restrict part = res->part;
  const uint32_t *restrict snfic = res->snfic;
  const uint32_t *restrict n0 = res->n0;
  const uint32_t *restrict n1 = res->n1;
  const uint32_t *restrict nfptr = res->nfptr;
  const uint32_t *restrict sn0 = res->sn0;
  const uint32_t *restrict sn1 = res->sn1;
  const uint32_t *restrict sn2 = res->sn2;

  const double *restrict x0 = res->x0;
  const double *restrict x1 = res->x1;
  const double *restrict x2 = res->x2;
  const double *restrict x3 = res->x3;
  const double *restrict q = res->q;

  const double *restrict w0termsx = res->w0termsx;
  const double *restrict w0termsy = res->w0termsy;
  const double *restrict w0termsz = res->w0termsz;

  const double *restrict w1termsx = res->w1termsx;
  const double *restrict w1termsy = res->w1termsy;
  const double *restrict w1termsz = res->w1termsz;

  double *restrict gradx0 = res->gradx0;
  double *restrict gradx1 = res->gradx1;
  double *restrict gradx2 = res->gradx2;

  memset(gradx0, 0, dofs * sizeof(double));
  memset(gradx1, 0, dofs * sizeof(double));
  memset(gradx2, 0, dofs * sizeof(double));

  double *restrict r = res->r;

  memset(r, 0, dofs * sizeof(double));

  __assume_aligned(r, 64);

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

/*
  Calculates the fluxes on the face and performs the flux balance
*/

  /* AVX512 Registers */

  const __m512d _zero = _mm512_set1_pd(0);

  const __m512d _pos1 = _mm512_set1_pd(1.0);
  const __m512d _pos2 = _mm512_set1_pd(2.0);
  const __m512d _half = _mm512_set1_pd(0.5);
  const __m512d _nhalf = _mm512_set1_pd(-0.5);
  const __m512d _nu95 = _mm512_set1_pd(0.95);
  const __m512d _beta = _mm512_set1_pd(BETA);

#ifdef __USE_SKX
  const __m512d _rbeta = _mm512_rcp14_pd(_beta);
#else
  const __m512d _rbeta = _mm512_rcp28_pd(_beta);
#endif

  const __m256i _bsz = _mm256_set1_epi32(bsz);

  const __m256i _shift1 = _mm256_set1_epi32(1);
  const __m256i _shift2 = _mm256_set1_epi32(2);
  const __m256i _shift3 = _mm256_set1_epi32(3);

  const __m512i _ng = _mm512_set1_epi32(-1);
  const __m512d _und = _mm512_undefined_pd(); 

#pragma omp parallel
  {
    const uint32_t t = omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    const uint32_t lim = ie1 - ((ie1-ie0) % 8);

    const __m512i _t = _mm512_set1_epi32(t);

    uint32_t i;

    for(i = ie0; i < lim; i+=8)
    {
      const __m512d _xn = _mm512_load_pd((void const *) &x0[i]);
      const __m512d _yn = _mm512_load_pd((void const *) &x1[i]);
      const __m512d _zn = _mm512_load_pd((void const *) &x2[i]);
      const __m512d _ln = _mm512_load_pd((void const *) &x3[i]);

      /*
        Now lets get our other 2 vectors
        For first vector, use {1,0,0} and subtract off the component
        in the direction of the face normal. If the inner product of
        {1,0,0} is close to unity, use {0,1,0}
      */

      const __m512d _fdot = _mm512_abs_pd(_xn);

      __mmask _k0;
      __m512d _dot, _X1, _Y1, _Z1;

      _k0 = _mm512_cmp_pd_mask(_fdot, _nu95, _CMP_LT_OS);

      _X1 = _mm512_mask_fnmadd_pd(_xn, _k0, _xn, _pos1);
      _Y1 = _mm512_mask_fnmadd_pd(_yn, _k0, _xn, _zero);
      _Z1 = _mm512_mask_fnmadd_pd(_zn, _k0, _xn, _zero);
      
      _k0 = _mm512_cmp_pd_mask(_fdot, _nu95, _CMP_GE_OS);

      _X1 = _mm512_mask_fnmadd_pd(_X1, _k0, _yn, _zero);
      _Y1 = _mm512_mask_fnmadd_pd(_Y1, _k0, _yn, _pos1);
      _Z1 = _mm512_mask_fnmadd_pd(_Z1, _k0, _yn, _zero);

      /*
        Normalize the first vector
      */

      __m512d _size;

      _size = _mm512_mul_pd(_X1, _X1);
      _size = _mm512_fmadd_pd(_Y1, _Y1, _size);
      _size = _mm512_fmadd_pd(_Z1, _Z1, _size);

#ifdef __USE_SKX
      _size = _mm512_rsqrt14_pd(_size);
#else
      _size = _mm512_rsqrt28_pd(_size);
#endif

      _X1 = _mm512_mul_pd(_X1, _size);
      _Y1 = _mm512_mul_pd(_Y1, _size);
      _Z1 = _mm512_mul_pd(_Z1, _size);

      const __m256i _n0 = _mm256_load_si256((__m256i const *) &n0[i]);
      const __m256i _n1 = _mm256_load_si256((__m256i const *) &n1[i]);

      const __m512d _x00 = _mm512_i32gather_pd(_n0, &xyz0[0], 8);
      const __m512d _x01 = _mm512_i32gather_pd(_n0, &xyz1[0], 8);
      const __m512d _x02 = _mm512_i32gather_pd(_n0, &xyz2[0], 8);

      const __m512d _x10 = _mm512_i32gather_pd(_n1, &xyz0[0], 8);
      const __m512d _x11 = _mm512_i32gather_pd(_n1, &xyz1[0], 8);
      const __m512d _x12 = _mm512_i32gather_pd(_n1, &xyz2[0], 8);

      const __m512d _xmean = _mm512_mul_pd(_half, _mm512_add_pd(_x00, _x10));
      const __m512d _ymean = _mm512_mul_pd(_half, _mm512_add_pd(_x01, _x11));
      const __m512d _zmean = _mm512_mul_pd(_half, _mm512_add_pd(_x02, _x12));

      /*
        Take cross-product of normal and V1 to get V2
      */
      const __m512d _X2 = _mm512_fmsub_pd(_yn, _Z1, _mm512_mul_pd(_zn, _Y1));
      const __m512d _Y2 = _mm512_fmsub_pd(_zn, _X1, _mm512_mul_pd(_xn, _Z1));
      const __m512d _Z2 = _mm512_fmsub_pd(_xn, _Y1, _mm512_mul_pd(_yn, _X1));

      /*
        Compute the stride indices
      */

      const __m256i _idx0 = _mm256_mullo_epi32(_bsz, _n0);
      const __m256i _idx1 = _mm256_mullo_epi32(_bsz, _n1);

      const __m256i _idx01 = _mm256_add_epi32(_idx0, _shift1);
      const __m256i _idx11 = _mm256_add_epi32(_idx1, _shift1);

      const __m256i _idx02 = _mm256_add_epi32(_idx0, _shift2);
      const __m256i _idx12 = _mm256_add_epi32(_idx1, _shift2);

      const __m256i _idx03 = _mm256_add_epi32(_idx0, _shift3);
      const __m256i _idx13 = _mm256_add_epi32(_idx1, _shift3);


      /*
        Get variables on "left" and "right" side of face
      */

      __m512d _q;
      __m512d _ubarL, _ubarR;
      __m512d _rx, _ry, _rz;
      __m512d _g0, _g1, _g2;
      __m512d _pL, _uL, _vL, _wL;
      __m512d _pR, _uR, _vR, _wR;

      /* Left */

      _rx = _mm512_sub_pd(_xmean, _x00);
      _ry = _mm512_sub_pd(_ymean, _x01);
      _rz = _mm512_sub_pd(_zmean, _x02);

      /* Pressure */

      _g0 = _mm512_i32gather_pd(_idx0, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx0, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx0, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx0, &q[0], 8);

      _pL = _mm512_fmadd_pd(_g0, _rx, _q);
      _pL = _mm512_fmadd_pd(_g1, _ry, _pL);
      _pL = _mm512_fmadd_pd(_g2, _rz, _pL);

      /* Velocity u */

      _g0 = _mm512_i32gather_pd(_idx01, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx01, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx01, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx01, &q[0], 8);

      _uL = _mm512_fmadd_pd(_g0, _rx, _q);
      _uL = _mm512_fmadd_pd(_g1, _ry, _uL);
      _uL = _mm512_fmadd_pd(_g2, _rz, _uL);

      /* Velocity v */

      _g0 = _mm512_i32gather_pd(_idx02, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx02, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx02, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx02, &q[0], 8);

      _vL = _mm512_fmadd_pd(_g0, _rx, _q);
      _vL = _mm512_fmadd_pd(_g1, _ry, _vL);
      _vL = _mm512_fmadd_pd(_g2, _rz, _vL);

      /* Velocity w */

      _g0 = _mm512_i32gather_pd(_idx03, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx03, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx03, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx03, &q[0], 8);

      _wL = _mm512_fmadd_pd(_g0, _rx, _q);
      _wL = _mm512_fmadd_pd(_g1, _ry, _wL);
      _wL = _mm512_fmadd_pd(_g2, _rz, _wL);


      _ubarL = _mm512_mul_pd(_xn, _uL);
      _ubarL = _mm512_fmadd_pd(_yn, _vL, _ubarL);
      _ubarL = _mm512_fmadd_pd(_zn, _wL, _ubarL);


      /* Right */

      _rx = _mm512_sub_pd(_xmean, _x10);
      _ry = _mm512_sub_pd(_ymean, _x11);
      _rz = _mm512_sub_pd(_zmean, _x12);

      /* Pressure */

      _g0 = _mm512_i32gather_pd(_idx1, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx1, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx1, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx1, &q[0], 8);

      _pR = _mm512_fmadd_pd(_g0, _rx, _q);
      _pR = _mm512_fmadd_pd(_g1, _ry, _pR);
      _pR = _mm512_fmadd_pd(_g2, _rz, _pR);

      /* Velocity u */

      _g0 = _mm512_i32gather_pd(_idx11, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx11, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx11, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx11, &q[0], 8);

      _uR = _mm512_fmadd_pd(_g0, _rx, _q);
      _uR = _mm512_fmadd_pd(_g1, _ry, _uR);
      _uR = _mm512_fmadd_pd(_g2, _rz, _uR);

      /* Velocity v */

      _g0 = _mm512_i32gather_pd(_idx12, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx12, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx12, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx12, &q[0], 8);

      _vR = _mm512_fmadd_pd(_g0, _rx, _q);
      _vR = _mm512_fmadd_pd(_g1, _ry, _vR);
      _vR = _mm512_fmadd_pd(_g2, _rz, _vR);

      /* Velocity w */

      _g0 = _mm512_i32gather_pd(_idx13, &gradx0[0], 8);
      _g1 = _mm512_i32gather_pd(_idx13, &gradx1[0], 8);
      _g2 = _mm512_i32gather_pd(_idx13, &gradx2[0], 8);

      _q = _mm512_i32gather_pd(_idx13, &q[0], 8);

      _wR = _mm512_fmadd_pd(_g0, _rx, _q);
      _wR = _mm512_fmadd_pd(_g1, _ry, _wR);
      _wR = _mm512_fmadd_pd(_g2, _rz, _wR);


      _ubarR = _mm512_mul_pd(_xn, _uR);
      _ubarR = _mm512_fmadd_pd(_yn, _vR, _ubarR);
      _ubarR = _mm512_fmadd_pd(_zn, _wR, _ubarR);


      const __m512d _dp = _mm512_sub_pd(_pR, _pL);
      const __m512d _du = _mm512_sub_pd(_uR, _uL);
      const __m512d _dv = _mm512_sub_pd(_vR, _vL);
      const __m512d _dw = _mm512_sub_pd(_wR, _wL);


      /* Compute averages for velocity variables only */

      const __m512d _u = _mm512_mul_pd(_half, _mm512_add_pd(_uL, _uR));
      const __m512d _v = _mm512_mul_pd(_half, _mm512_add_pd(_vL, _vR));
      const __m512d _w = _mm512_mul_pd(_half, _mm512_add_pd(_wL, _wR));

      __m512d _ubar;
      _ubar = _mm512_mul_pd(_xn, _u);
      _ubar = _mm512_fmadd_pd(_yn, _v, _ubar);
      _ubar = _mm512_fmadd_pd(_zn, _w, _ubar);

      /* Compute Phi's */
  
      __m512d _phi1;
      _phi1 = _mm512_mul_pd(_xn, _beta);
      _phi1 = _mm512_fmadd_pd(_u, _ubar, _phi1);

      __m512d _phi2;
      _phi2 = _mm512_mul_pd(_yn, _beta);
      _phi2 = _mm512_fmadd_pd(_v, _ubar, _phi2);

      __m512d _phi3;
      _phi3 = _mm512_mul_pd(_zn, _beta);
      _phi3 = _mm512_fmadd_pd(_w, _ubar, _phi3);

      __m512d _phi4;
      _phi4 = _mm512_mul_pd(_Z2, _phi2);
      _phi4 = _mm512_fmsub_pd(_Y2, _phi3, _phi4);

      __m512d _phi5;
      _phi5 = _mm512_mul_pd(_X2, _phi3);
      _phi5 = _mm512_fmsub_pd(_Z2, _phi1, _phi5);

      __m512d _phi6;
      _phi6 = _mm512_mul_pd(_Y2, _phi1);
      _phi6 = _mm512_fmsub_pd(_X2, _phi2, _phi6);

      __m512d _phi7;
      _phi7 = _mm512_mul_pd(_Y1, _phi3);
      _phi7 = _mm512_fmsub_pd(_Z1, _phi2, _phi7);

      __m512d _phi8;
      _phi8 = _mm512_mul_pd(_Z1, _phi1);
      _phi8 = _mm512_fmsub_pd(_X1, _phi3, _phi8);

      __m512d _phi9;
      _phi9 = _mm512_mul_pd(_X1, _phi2);
      _phi9 = _mm512_fmsub_pd(_Y1, _phi1, _phi9);

      /*
        Compute eigenvalues, eigenvectors, and strengths
      */

      const __m512d _c2 = _mm512_fmadd_pd(_ubar, _ubar, _beta);

#ifdef __USE_SKX
      const __m512d _c = _mm512_mul_pd(_mm512_rsqrt14_pd(_c2), _c2);
      const __m512d _c2r = _mm512_rcp14_pd(_c2);
#else
      const __m512d _c = _mm512_mul_pd(_mm512_rsqrt28_pd(_c2), _c2);
      const __m512d _c2r = _mm512_rcp28_pd(_c2);
#endif

      const __m512d _bac = _mm512_add_pd(_ubar, _c);
      const __m512d _bsc = _mm512_sub_pd(_ubar, _c);

      /*
        Components of T(inverse)
      */

      __m512d _ti11;
      _ti11 = _mm512_mul_pd(_u, _phi4);
      _ti11 = _mm512_fmadd_pd(_v, _phi5, _ti11);
      _ti11 = _mm512_fmadd_pd(_w, _phi6, _ti11);

      _ti11 = _mm512_fnmadd_pd(_ti11, _rbeta, _zero);

      __m512d _ti21;
      _ti21 = _mm512_mul_pd(_u, _phi7);
      _ti21 = _mm512_fmadd_pd(_v, _phi8, _ti21);
      _ti21 = _mm512_fmadd_pd(_w, _phi9, _ti21);

      _ti21 = _mm512_fnmadd_pd(_ti21, _rbeta, _zero);

      __m512d _ti31;
      _ti31 = _mm512_mul_pd(_half, _mm512_sub_pd(_c, _ubar));
      _ti31 = _mm512_mul_pd(_ti31, _rbeta);

      __m512d _ti41;
      _ti41 = _mm512_mul_pd(_nhalf, _bac);
      _ti41 = _mm512_mul_pd(_ti41, _rbeta);

      /*
        jumps (T(inverse) * dq)
      */

      __m512d _dv1;
      _dv1 = _mm512_mul_pd(_ti11, _dp);
      _dv1 = _mm512_fmadd_pd(_phi4, _du, _dv1);
      _dv1 = _mm512_fmadd_pd(_phi5, _dv, _dv1);
      _dv1 = _mm512_fmadd_pd(_phi6, _dw, _dv1);
      _dv1 = _mm512_mul_pd(_dv1, _c2r);

      __m512d _dv2;
      _dv2 = _mm512_mul_pd(_ti21, _dp);
      _dv2 = _mm512_fmadd_pd(_phi7, _du, _dv2);
      _dv2 = _mm512_fmadd_pd(_phi8, _dv, _dv2);
      _dv2 = _mm512_fmadd_pd(_phi9, _dw, _dv2);
      _dv2 = _mm512_mul_pd(_dv2, _c2r);

      __m512d _dv34;
      _dv34 = _mm512_mul_pd(_xn, _du);
      _dv34 = _mm512_fmadd_pd(_yn, _dv, _dv34);
      _dv34 = _mm512_fmadd_pd(_zn, _dw, _dv34);
   
      __m512d _dv3;
      _dv3 = _mm512_fmadd_pd(_mm512_mul_pd(_pos2, _ti31), _dp, _dv34);
      _dv3 = _mm512_mul_pd(_dv3, _mm512_mul_pd(_half, _c2r));

      __m512d _dv4;
      _dv4 = _mm512_fmadd_pd(_mm512_mul_pd(_pos2, _ti41), _dp, _dv34);
      _dv4 = _mm512_mul_pd(_dv4, _mm512_mul_pd(_half, _c2r));

      /*
        Now get elements of T
      */

      const __m512d _r13 = _mm512_mul_pd(_c, _beta);

      __m512d _r23;
      _r23 = _mm512_mul_pd(_u, _bac);
      _r23 = _mm512_fmadd_pd(_xn, _beta, _r23);

      __m512d _r33;
      _r33 = _mm512_mul_pd(_v, _bac);
      _r33 = _mm512_fmadd_pd(_yn, _beta, _r33);

      __m512d _r43;
      _r43 = _mm512_mul_pd(_w, _bac);
      _r43 = _mm512_fmadd_pd(_zn, _beta, _r43);

      const __m512d _r14 = _mm512_fnmadd_pd(_c, _beta, _zero);

      __m512d _r24;
      _r24 = _mm512_mul_pd(_u, _bsc);
      _r24 = _mm512_fmadd_pd(_xn, _beta, _r24);

      __m512d _r34;
      _r34 = _mm512_mul_pd(_v, _bsc);
      _r34 = _mm512_fmadd_pd(_yn, _beta, _r34);

      __m512d _r44;
      _r44 = _mm512_mul_pd(_w, _bsc);
      _r44 = _mm512_fmadd_pd(_zn, _beta, _r44);

      /*
        Calculate T* |lambda| * T(inverse)
      */

      const __m512d _eig1 = _mm512_abs_pd(_ubar);
      const __m512d _eig2 = _mm512_abs_pd(_bac);
      const __m512d _eig3 = _mm512_abs_pd(_bsc);

      __m512d _t1;
      _t1 = _mm512_mul_pd(_mm512_mul_pd(_eig2, _r13), _dv3);
      _t1 = _mm512_fmadd_pd(_mm512_mul_pd(_eig3, _r14), _dv4, _t1);

      __m512d _t2;
      _t2 = _mm512_mul_pd(_mm512_mul_pd(_eig1, _X1), _dv1);
      _t2 = _mm512_fmadd_pd(_mm512_mul_pd(_eig1, _X2), _dv2, _t2);
      _t2 = _mm512_fmadd_pd(_mm512_mul_pd(_eig2, _r23), _dv3, _t2);
      _t2 = _mm512_fmadd_pd(_mm512_mul_pd(_eig3, _r24), _dv4, _t2);

      __m512d _t3;
      _t3 = _mm512_mul_pd(_mm512_mul_pd(_eig1, _Y1), _dv1);
      _t3 = _mm512_fmadd_pd(_mm512_mul_pd(_eig1, _Y2), _dv2, _t3);
      _t3 = _mm512_fmadd_pd(_mm512_mul_pd(_eig2, _r33), _dv3, _t3);
      _t3 = _mm512_fmadd_pd(_mm512_mul_pd(_eig3, _r34), _dv4, _t3);

      __m512d _t4;
      _t4 = _mm512_mul_pd(_mm512_mul_pd(_eig1, _Z1), _dv1);
      _t4 = _mm512_fmadd_pd(_mm512_mul_pd(_eig1, _Z2), _dv2, _t4);
      _t4 = _mm512_fmadd_pd(_mm512_mul_pd(_eig2, _r43), _dv3, _t4);
      _t4 = _mm512_fmadd_pd(_mm512_mul_pd(_eig3, _r44), _dv4, _t4);


      /*
        Modify to calculate .5(fl +fr) from nodes
        instead of extrapolated ones
      */

      /* Left Side */

      __m512d _fluxp1;
      _fluxp1 = _mm512_mul_pd(_mm512_mul_pd(_ln, _beta), _ubarL);

      __m512d _fluxp2;
      _fluxp2 = _mm512_mul_pd(_uL, _ubarL);
      _fluxp2 = _mm512_fmadd_pd(_xn, _pL, _fluxp2);
      _fluxp2 = _mm512_mul_pd(_ln, _fluxp2);

      __m512d _fluxp3;
      _fluxp3 = _mm512_mul_pd(_vL, _ubarL);
      _fluxp3 = _mm512_fmadd_pd(_yn, _pL, _fluxp3);
      _fluxp3 = _mm512_mul_pd(_ln, _fluxp3);

      __m512d _fluxp4;
      _fluxp4 = _mm512_mul_pd(_wL, _ubarL);
      _fluxp4 = _mm512_fmadd_pd(_zn, _pL, _fluxp4);
      _fluxp4 = _mm512_mul_pd(_ln, _fluxp4);

      /* Right Side */

      __m512d _fluxm1;
      _fluxm1 = _mm512_mul_pd(_mm512_mul_pd(_ln, _beta), _ubarR);

      __m512d _fluxm2;
      _fluxm2 = _mm512_mul_pd(_uR, _ubarR);
      _fluxm2 = _mm512_fmadd_pd(_xn, _pR, _fluxm2);
      _fluxm2 = _mm512_mul_pd(_ln, _fluxm2);

      __m512d _fluxm3;
      _fluxm3 = _mm512_mul_pd(_vR, _ubarR);
      _fluxm3 = _mm512_fmadd_pd(_yn, _pR, _fluxm3);
      _fluxm3 = _mm512_mul_pd(_ln, _fluxm3);

      __m512d _fluxm4;
      _fluxm4 = _mm512_mul_pd(_wR, _ubarR);
      _fluxm4 = _mm512_fmadd_pd(_zn, _pR, _fluxm4);
      _fluxm4 = _mm512_mul_pd(_ln, _fluxm4);

      __m512d _res1;
      _res1 = _mm512_fnmadd_pd(_ln, _t1, _mm512_add_pd(_fluxm1, _fluxp1));

      __m512d _res2;
      _res2 = _mm512_fnmadd_pd(_ln, _t2, _mm512_add_pd(_fluxm2, _fluxp2));

      __m512d _res3;
      _res3 = _mm512_fnmadd_pd(_ln, _t3, _mm512_add_pd(_fluxm3, _fluxp3));

      __m512d _res4;
      _res4 = _mm512_fnmadd_pd(_ln, _t4, _mm512_add_pd(_fluxm4, _fluxp4));


      /* Update the residual */

      __m512i _node, _part;
      __mmask _next;

      _node = _mm512_castsi256_si512(_n0);
      _part = _mm512_i32gather_epi32(_node, &part[0], 4);
      _next = _mm512_cmpeq_epi32_mask(_part, _t);

      /* Conflict detection instructions with multiple node update */

      /* Node 0 Contributions */

      do {
        __m512i _cd, _bnext;
        __m512d _v, _d;
        __mmask _crt;

        _cd = _mm512_mask_conflict_epi32(_ng, _next, _node);
        _bnext = _mm512_broadcastmw_epi32(_next);
        _crt = _mm512_mask_testn_epi32_mask(_next, _cd, _bnext);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx0, &r[0], 8);
        _d = _mm512_mask_fmadd_pd(_res1, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx0, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx01, &r[0], 8);
        _d = _mm512_mask_fmadd_pd(_res2, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx01, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx02, &r[0], 8);
        _d = _mm512_mask_fmadd_pd(_res3, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx02, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx03, &r[0], 8);
        _d = _mm512_mask_fmadd_pd(_res4, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx03, _d, 8);

        _next = _mm512_kxor(_next, _crt);

      } while(_next);

      _node = _mm512_castsi256_si512(_n1);
      _part = _mm512_i32gather_epi32(_node, &part[0], 4);
      _next = _mm512_cmpeq_epi32_mask(_part, _t);

      /* Node 1 Contributions */

      do {
        __m512i _cd, _bnext; 
        __m512d _v, _d;
        __mmask _crt;

        _cd = _mm512_mask_conflict_epi32(_ng, _next, _node);
        _bnext = _mm512_broadcastmw_epi32(_next);
        _crt = _mm512_mask_testn_epi32_mask(_next, _cd, _bnext);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx1, &r[0], 8);
        _d = _mm512_mask_fnmadd_pd(_res1, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx1, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx11, &r[0], 8);
        _d = _mm512_mask_fnmadd_pd(_res2, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx11, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx12, &r[0], 8);
        _d = _mm512_mask_fnmadd_pd(_res3, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx12, _d, 8);

        _v = _mm512_mask_i32gather_pd(_und, _crt, _idx13, &r[0], 8);
        _d = _mm512_mask_fnmadd_pd(_res4, _crt, _half, _v);
        _mm512_mask_i32scatter_pd(&r[0], _crt, _idx13, _d, 8);

        _next = _mm512_kxor(_next, _crt);

      } while(_next);
    }

    /* Remainder loop */

    for(i = lim; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const double xn = x0[i];
      const double yn = x1[i];
      const double zn = x2[i];
      const double ln = x3[i];

      const double xmean = 0.5f * (xyz0[node0] + xyz0[node1]);
      const double ymean = 0.5f * (xyz1[node0] + xyz1[node1]);
      const double zmean = 0.5f * (xyz2[node0] + xyz2[node1]);

      /*
        Now lets get our other 2 vectors
        For first vector, use {1,0,0} and subtract off the component
        in the direction of the face normal. If the inner product of
        {1,0,0} is close to unity, use {0,1,0}
      */

      double X1 = (fabs(xn) < 0.95) ? (1 - xn * xn) : (- yn * xn);
      double Y1 = (fabs(xn) < 0.95) ? (- xn * yn) : (1 - yn * yn);
      double Z1 = (fabs(xn) < 0.95) ? (- xn * zn) : (- yn * zn);

      /*
        Normalize the first vector
      */

      double size = X1 * X1;
      size += Y1 * Y1;
      size += Z1 * Z1;
      size = sqrt(size);

      X1 /= size;
      Y1 /= size;
      Z1 /= size;

      /*
        Take cross-product of normal and V1 to get V2
      */

      const double X2 = yn * Z1 - zn * Y1;
      const double Y2 = zn * X1 - xn * Z1;
      const double Z2 = xn * Y1 - yn * X1;

      /*
        Get variables on "left" and "right" side of face
      */
      double rx = xmean - xyz0[node0];
      double ry = ymean - xyz1[node0];
      double rz = zmean - xyz2[node0];

      const uint32_t idx0 = bsz * node0;
      const uint32_t idx1 = bsz * node1;

      // Pressure

      double pL = q[idx0 + 0] + gradx0[idx0 + 0] * rx;
      pL += gradx1[idx0 + 0] * ry;
      pL += gradx2[idx0 + 0] * rz;

      // Velocity u

      double uL = q[idx0 + 1] + gradx0[idx0 + 1] * rx;
      uL += gradx1[idx0 + 1] * ry;
      uL += gradx2[idx0 + 1] * rz;

      // Velocity v

      double vL = q[idx0 + 2] + gradx0[idx0 + 2] * rx;
      vL += gradx1[idx0 + 2] * ry;
      vL += gradx2[idx0 + 2] * rz;

      // Velocity w

      double wL = q[idx0 + 3] + gradx0[idx0 + 3] * rx;
      wL += gradx1[idx0 + 3] * ry;
      wL += gradx2[idx0 + 3] * rz;

      double ubarL = xn * uL;
      ubarL += yn * vL;
      ubarL += zn * wL;

      rx = xmean - xyz0[node1];
      ry = ymean - xyz1[node1];
      rz = zmean - xyz2[node1];
      // Pressure

      double pR = q[idx1 + 0] + gradx0[idx1 + 0] * rx;
      pR += gradx1[idx1 + 0] * ry;
      pR += gradx2[idx1 + 0] * rz;

      // Velocity u

      double uR = q[idx1 + 1] + gradx0[idx1 + 1] * rx;
      uR += gradx1[idx1 + 1] * ry;
      uR += gradx2[idx1 + 1] * rz;

      // Velocity v

      double vR = q[idx1 + 2] + gradx0[idx1 + 2] * rx;
      vR += gradx1[idx1 + 2] * ry;
      vR += gradx2[idx1 + 2] * rz;

      // Velocity w

      double wR = q[idx1 + 3] + gradx0[idx1 + 3] * rx;
      wR += gradx1[idx1 + 3] * ry;
      wR += gradx2[idx1 + 3] * rz;

      double ubarR = xn * uR;
      ubarR += yn * vR;
      ubarR += zn * wR;

      /* Compute averages */

      const double u = 0.5f * (uL + uR);
      const double v = 0.5f * (vL + vR);
      const double w = 0.5f * (wL + wR);

      double ubar = xn * u;
      ubar += yn * v;
      ubar += zn * w;

      double phi1 = xn * BETA;
      phi1 += u * ubar;

      double phi2 = yn * BETA;
      phi2 += v * ubar;

      double phi3 = zn * BETA;
      phi3 += w * ubar;

      double phi4 = Y2 * phi3;
      phi4 -= Z2 * phi2;

      double phi5 = Z2 * phi1;
      phi5 -= X2 * phi3;

      double phi6 = X2 * phi2;
      phi6 -= Y2 * phi1;

      double phi7 = Z1 * phi2;
      phi7 -= Y1 * phi3;

      double phi8 = X1 * phi3;
      phi8 -= Z1 * phi1;

      double phi9 = Y1 * phi1;
      phi9 -= X1 * phi2;

      double c2 = ubar * ubar + BETA;
      double c = sqrt(c2);

      /*
        Now compute eigenvalues, eigenvectors, and strengths
      */

      const double uac = ubar + c;
      const double usc = ubar - c;

      const double eig1 = fabs(ubar);
      const double eig2 = fabs(uac);
      const double eig3 = fabs(usc);

      const double dp = pR - pL;
      const double du = uR - uL;
      const double dv = vR - vL;
      const double dw = wR - wL;

      /*
        Components of T(inverse)
      */
      double ti11 = u * phi4;
      ti11 += v * phi5;
      ti11 += w * phi6;
      ti11 = -ti11 / BETA;

      double ti21 = u * phi7;
      ti21 += v * phi8;
      ti21 += w * phi9;
      ti21 = -ti21 / BETA;

      double ti31 = 0.5f * (c - ubar);
      ti31 /= BETA;

      double ti41 = -0.5f * uac;
      ti41 /= BETA;

      /*
        jumps (T(inverse) * dq)
      */
      double dv1 = ti11 * dp;
      dv1 += phi4 * du;
      dv1 += phi5 * dv;
      dv1 += phi6 * dw;
      dv1 /= c2;

      double dv2 = ti21 * dp;
      dv2 += phi7 * du;
      dv2 += phi8 * dv;
      dv2 += phi9 * dw;
      dv2 /= c2;

      double dv3 = 2.f * ti31 * dp;
      dv3 += xn * du;
      dv3 += yn * dv;
      dv3 += zn * dw;
      dv3 *= 0.5f / c2;

      double dv4 = 2.f * ti41 * dp;
      dv4 += xn * du;
      dv4 += yn * dv;
      dv4 += zn * dw;
      dv4 *= 0.5f / c2;

      /*
        Now get elements of T
      */

      const double r13 = c * BETA;

      const double r23 = u * uac + xn * BETA;
      const double r33 = v * uac + yn * BETA;
      const double r43 = w * uac + zn * BETA;

      const double r14 = -c * BETA;

      const double r24 = u * usc + xn * BETA;
      const double r34 = v * usc + yn * BETA;
      const double r44 = w * usc + zn * BETA;

      /*
        Calculate T* |lambda| * T(inverse)
      */

      double t1 = eig2 * r13 * dv3 + eig3 * r14 * dv4;

      double t2 = eig1 * X1 * dv1 + eig1 * X2 * dv2;
      t2 += eig2 * r23 * dv3 + eig3 * r24 * dv4;

      double t3 = eig1 * Y1 * dv1 + eig1 * Y2 * dv2;
      t3 += eig2 * r33 * dv3 + eig3 * r34 * dv4;

      double t4 = eig1 * Z1 * dv1 + eig1 * Z2 * dv2;
      t4 += eig2 * r43 * dv3 + eig3 * r44 * dv4;

      /*
        Modify to calculate .5(fl +fr) from nodes
        instead of extrapolated ones
      */

      const double fluxp1 = ln * BETA * ubarL;
      const double fluxp2 = ln * (uL * ubarL + xn * pL);
      const double fluxp3 = ln * (vL * ubarL + yn * pL);
      const double fluxp4 = ln * (wL * ubarL + zn * pL);

      /*
        Now the right side
      */

      const double fluxm1 = ln * BETA * ubarR;
      const double fluxm2 = ln * (uR * ubarR + xn * pR);
      const double fluxm3 = ln * (vR * ubarR + yn * pR);
      const double fluxm4 = ln * (wR * ubarR + zn * pR);

      const double res1 = 0.5f * (fluxp1 + fluxm1 - ln * t1);
      const double res2 = 0.5f * (fluxp2 + fluxm2 - ln * t2);
      const double res3 = 0.5f * (fluxp3 + fluxm3 - ln * t3);
      const double res4 = 0.5f * (fluxp4 + fluxm4 - ln * t4);
      
      r[idx0 + 0] = (part[node0] == t) ? (r[idx0 + 0] + res1) : r[idx0 + 0];
      r[idx0 + 1] = (part[node0] == t) ? (r[idx0 + 1] + res2) : r[idx0 + 1];
      r[idx0 + 2] = (part[node0] == t) ? (r[idx0 + 2] + res3) : r[idx0 + 2];
      r[idx0 + 3] = (part[node0] == t) ? (r[idx0 + 3] + res4) : r[idx0 + 3];
      
      r[idx1 + 0] = (part[node1] == t) ? (r[idx1 + 0] - res1) : r[idx1 + 0];
      r[idx1 + 1] = (part[node1] == t) ? (r[idx1 + 1] - res2) : r[idx1 + 1];
      r[idx1 + 2] = (part[node1] == t) ? (r[idx1 + 2] - res3) : r[idx1 + 2];
      r[idx1 + 3] = (part[node1] == t) ? (r[idx1 + 3] - res4) : r[idx1 + 3];
    }
  }

  uint32_t i;

  for(i = 0; i < snfc; i++)
  {
    const uint32_t if0 = snfic[i];
    const uint32_t if1 = snfic[i+1];

    uint32_t j;

#pragma omp parallel for
    for(j = if0; j < if1; j++)
    {
      const uint32_t node0 = sn0[j];
      const uint32_t node1 = sn1[j];
      const uint32_t node2 = sn2[j];

      const double p1 = q[bsz * node0];
      const double p2 = q[bsz * node1];
      const double p3 = q[bsz * node2];

      const double ax = xyz0[node1] - xyz0[node0];
      const double ay = xyz1[node1] - xyz1[node0];
      const double az = xyz2[node1] - xyz2[node0];

      const double bx = xyz0[node2] - xyz0[node0];
      const double by = xyz1[node2] - xyz1[node0];
      const double bz = xyz2[node2] - xyz2[node0];

      /*
        Normal points away from grid interior.
        Magnitude is 1/3 area of surface triangle.
      */

      double xn = ay * bz;
      xn -= az * by;
      xn *= MAG1;

      double yn = ax * bz;
      yn -= az * bx;
      yn *= MAG0;

      double zn = ax * by;
      zn -= ay * bx;
      zn *= MAG1;

      double pa = 0.125f * (p2 + p3);
      pa += 0.75f * p1;

      double pb = 0.125f * (p3 + p1);
      pb += 0.75f * p2;

      double pc = 0.125f * (p1 + p2);
      pc += 0.75f * p3;

      uint32_t idx;

      idx = bsz * node0;

      r[idx + 1] += xn * pa;
      r[idx + 2] += yn * pa;
      r[idx + 3] += zn * pa;

      idx = bsz * node1;

      r[idx + 1] += xn * pb;
      r[idx + 2] += yn * pb;
      r[idx + 3] += zn * pb;

      idx = bsz * node2;

      r[idx + 1] += xn * pc;
      r[idx + 2] += yn * pc;
      r[idx + 3] += zn * pc;
    }
  }

  /* Do the free boundaries */
#pragma omp parallel for
  for(i = 0; i < nfnodes; i++)
  {
    uint32_t n = nfptr[i];

    /*
      Get normal and "other" 2 vectors. Remember that fxn,fyn and fzn
      has the magnitude of the face contained in it.
    */

    double xn = f_xyz0[i];
    double yn = f_xyz1[i];
    double zn = f_xyz2[i];

    double area = xn * xn;
    area += yn * yn;
    area += zn * zn;
    area = sqrt(area);

    xn /= area;
    yn /= area;
    zn /= area;

    /*
      Now lets get our other 2 vectors
      For first vector, use {1,0,0} and subtract off the component
      in the direction of the face normal. If the inner product of
      {1,0,0} is close to unity, use {0,1,0}
    */

    double X1, Y1, Z1;

    double dot = xn;
    if(fabs(dot) < 0.95f)
    {
      X1 = 1.f - dot * xn;
      Y1 = -dot * yn;
      Z1 = -dot * zn;
    }
    else
    {
      dot = yn;
      X1  = -dot * xn;
      Y1  = 1.f - dot * yn;
      Z1  = -dot * zn;
    }

    /*
      Normalize the first vector (V1)
    */

    double size = X1 * X1;
    size += Y1 * Y1;
    size += Z1 * Z1;
    size = sqrt(size);

    X1 /= size;
    Y1 /= size;
    Z1 /= size;

    /*
      Take cross-product of normal with V1 to get V2
    */
    double X2 = yn * Z1;
    X2 -= zn * Y1;

    double Y2 = zn * X1;
    Y2 -= xn * Z1;

    double Z2 = xn * Y1;
    Z2 -= yn * X1;

    /*
      Calculate elements of T and T(inverse) evaluated at free-stream
    */
    double ubar0 = xn * velocity_u;
    ubar0 += yn * velocity_v;
    ubar0 += zn * velocity_w;

    double c20 = ubar0 * ubar0 + BETA;
    double c0 = sqrt(c20);

    double phi1 = xn * BETA;
    phi1 += velocity_u * ubar0;

    double phi2 = yn * BETA;
    phi2 += velocity_v * ubar0;

    double phi3 = zn * BETA;
    phi3 += velocity_w * ubar0;

    double phi4 = Y2 * phi3;
    phi4 -= Z2 * phi2;

    double phi5 = Z2 * phi1;
    phi5 -= X2 * phi3;

    double phi6 = X2 * phi2;
    phi6 -= Y2 * phi1;

    double phi7 = Z1 * phi2;
    phi7 -= Y1 * phi3;

    double phi8 = X1 * phi3;
    phi8 -= Z1 * phi1;

    double phi9 = Y1 * phi1;
    phi9 -= X1 * phi2;

    double t13 = c0 * BETA;

    double t23 = velocity_u * (ubar0 + c0);
    t23 += xn * BETA;

    double t33 = velocity_v * (ubar0 + c0);
    t33 += yn * BETA;

    double t43 = velocity_w * (ubar0 + c0);
    t43 += zn * BETA;

    double t14 = -c0 * BETA;

    double t24 = velocity_u * (ubar0 - c0);
    t24 += xn * BETA;

    double t34 = velocity_v * (ubar0 - c0);
    t34 += yn * BETA;

    double t44 = velocity_w * (ubar0 - c0);
    t44 += zn * BETA;

    double ti11 = velocity_u * phi4;
    ti11 += velocity_v * phi5;
    ti11 += velocity_w * phi6;
    ti11 = -ti11/BETA;

    double ti21 = velocity_u * phi7;
    ti21 += velocity_v * phi8;
    ti21 += velocity_w * phi9;
    ti21 = -ti21/BETA;

    double ti31 = 0.5f * (c0 - ubar0);
    ti31 /= BETA;

    double ti41 = -0.5f * (c0 + ubar0);
    ti41 /= BETA;

    /*
      Now, get the variables on the "inside"
    */
    double pi = q[bsz * n + 0];
    double ui = q[bsz * n + 1];
    double vi = q[bsz * n + 2];
    double wi = q[bsz * n + 3];

    double un = xn * ui;
    un += yn * vi;
    un += zn * wi;
    /*
      If ubar is negative, take the reference condition from outside
    */
    double pr, ur, vr, wr;

    if(un > 0.f)
    {
      pr = pi;
      ur = ui;
      vr = vi;
      wr = wi;
    }
    else
    {
      pr = pressure;
      ur = velocity_u;
      vr = velocity_v;
      wr = velocity_w;
    }

    /*
      Set rhs
    */

    double rhs1 = ti11 * pr;
    rhs1 += phi4 * ur;
    rhs1 += phi5 * vr;
    rhs1 += phi6 * wr;
    rhs1 /= c20;

    double rhs2 = ti21 * pr;
    rhs2 += phi7 * ur;
    rhs2 += phi8 * vr;
    rhs2 += phi9 * wr;
    rhs2 /= c20;

    double rhs3 = 2.f * ti31 * pi;
    rhs3 += xn * ui;
    rhs3 += yn * vi;
    rhs3 += zn * wi;
    rhs3 = 0.5f * rhs3 / c20;

    double rhs4 =  2.f * ti41 * pressure;
    rhs4 += xn * velocity_u;
    rhs4 += yn * velocity_v;
    rhs4 += zn * velocity_w;
    rhs4 = 0.5f * rhs4 / c20;

    /*
      Now do matrix multiplication to get values on boundary
    */
    double pb = t13 * rhs3;
    pb += t14 * rhs4;

    double ub = X1 * rhs1;
    ub += X2 * rhs2;
    ub += t23 * rhs3;
    ub += t24 * rhs4;

    double vb = Y1 * rhs1;
    vb += Y2 * rhs2;
    vb += t33 * rhs3;
    vb += t34 * rhs4;

    double wb = Z1 * rhs1;
    wb += Z2 * rhs2;
    wb += t43 * rhs3;
    wb += t44 * rhs4;

    double ubar = xn * ub;
    ubar += yn * vb;
    ubar += zn * wb;

    uint32_t idx = bsz * n;

    r[idx + 0] += area * BETA * ubar;
    r[idx + 1] += area * (ub * ubar + xn * pb);
    r[idx + 2] += area * (vb * ubar + yn * pb);
    r[idx + 3] += area * (wb * ubar + zn * pb);
  }

  compute_time(&ktime, res->t);

#ifdef __USE_HW_COUNTER
  const uint64_t cycle = __rdtsc() - icycle;

  struct counters end;
  perf_read(fd, &end);

  struct tot tot;
  perf_calc(start, end, &tot);

  res->perf_counters->ctrs->flux.cycles += cycle;
  res->perf_counters->ctrs->flux.tot.imcR += tot.imcR;
  res->perf_counters->ctrs->flux.tot.imcW += tot.imcW;
  res->perf_counters->ctrs->flux.tot.edcR += tot.edcR;
  res->perf_counters->ctrs->flux.tot.edcW += tot.edcW;
#endif

}