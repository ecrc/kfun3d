#ifdef __cplusplus
extern "C" {
#endif
extern void CXX_ILU(void *ptr);
#ifdef __cplusplus
}
#endif

#if 0
#include <string.h>
#include "geometry.h"
#include "core_kernel.h"
#include "bench.h"

static const unsigned int bsz2 = 16;
static const unsigned int bsz = 4;

inline static void axpy1(const double a, const double *A, double *B)
{
  for(unsigned int i = 0; i < bsz2; i++)
  {
    /* AXPY */
    const double ax = a * A[i];
    const double axpy = ax + B[i];

    /* Update the vector component */
    B[i] = axpy;
  }
}

inline static unsigned int bidx2(const unsigned int index)
{
  return(index * bsz2);
}

inline static void dgemm(const double A[], const double B[], double C[])
{
  for(unsigned int i = 0; i < bsz; i++)
  {
    for(unsigned int j = 0; j < bsz; j++)
    {
      const unsigned int ij = i * bsz + j;

      C[ij] = 0.f;
      
      for(unsigned int k = 0; k < bsz; k++)
      {
        const unsigned int kj = k * bsz + j;
        const unsigned int ik = i * bsz + k;

        C[ij] += A[ik] * B[kj];
      }
    }
  }
}

inline static void inv(double *src)
{
  double dst[16];

  /* Compute adjoint: */
  dst[0] =
      + src[ 5] * src[10] * src[15]
      - src[ 5] * src[11] * src[14]
      - src[ 9] * src[ 6] * src[15]
      + src[ 9] * src[ 7] * src[14]
      + src[13] * src[ 6] * src[11]
      - src[13] * src[ 7] * src[10];

  dst[1] =
      - src[ 1] * src[10] * src[15]
      + src[ 1] * src[11] * src[14]
      + src[ 9] * src[ 2] * src[15]
      - src[ 9] * src[ 3] * src[14]
      - src[13] * src[ 2] * src[11]
      + src[13] * src[ 3] * src[10];

  dst[2] =
      + src[ 1] * src[ 6] * src[15]
      - src[ 1] * src[ 7] * src[14]
      - src[ 5] * src[ 2] * src[15]
      + src[ 5] * src[ 3] * src[14]
      + src[13] * src[ 2] * src[ 7]
      - src[13] * src[ 3] * src[ 6];

  dst[3] =
      - src[ 1] * src[ 6] * src[11]
      + src[ 1] * src[ 7] * src[10]
      + src[ 5] * src[ 2] * src[11]
      - src[ 5] * src[ 3] * src[10]
      - src[ 9] * src[ 2] * src[ 7]
      + src[ 9] * src[ 3] * src[ 6];

  dst[4] =
      - src[ 4] * src[10] * src[15]
      + src[ 4] * src[11] * src[14]
      + src[ 8] * src[ 6] * src[15]
      - src[ 8] * src[ 7] * src[14]
      - src[12] * src[ 6] * src[11]
      + src[12] * src[ 7] * src[10];

  dst[5] =
      + src[ 0] * src[10] * src[15]
      - src[ 0] * src[11] * src[14]
      - src[ 8] * src[ 2] * src[15]
      + src[ 8] * src[ 3] * src[14]
      + src[12] * src[ 2] * src[11]
      - src[12] * src[ 3] * src[10];

  dst[6] =
      - src[ 0] * src[ 6] * src[15]
      + src[ 0] * src[ 7] * src[14]
      + src[ 4] * src[ 2] * src[15]
      - src[ 4] * src[ 3] * src[14]
      - src[12] * src[ 2] * src[ 7]
      + src[12] * src[ 3] * src[ 6];

  dst[7] =
      + src[ 0] * src[ 6] * src[11]
      - src[ 0] * src[ 7] * src[10]
      - src[ 4] * src[ 2] * src[11]
      + src[ 4] * src[ 3] * src[10]
      + src[ 8] * src[ 2] * src[ 7]
      - src[ 8] * src[ 3] * src[ 6];

  dst[8] =
      + src[ 4] * src[ 9] * src[15]
      - src[ 4] * src[11] * src[13]
      - src[ 8] * src[ 5] * src[15]
      + src[ 8] * src[ 7] * src[13]
      + src[12] * src[ 5] * src[11]
      - src[12] * src[ 7] * src[ 9];

  dst[9] =
      - src[ 0] * src[ 9] * src[15]
      + src[ 0] * src[11] * src[13]
      + src[ 8] * src[ 1] * src[15]
      - src[ 8] * src[ 3] * src[13]
      - src[12] * src[ 1] * src[11]
      + src[12] * src[ 3] * src[ 9];

  dst[10] =
      + src[ 0] * src[ 5] * src[15]
      - src[ 0] * src[ 7] * src[13]
      - src[ 4] * src[ 1] * src[15]
      + src[ 4] * src[ 3] * src[13]
      + src[12] * src[ 1] * src[ 7]
      - src[12] * src[ 3] * src[ 5];

  dst[11] =
      - src[ 0] * src[ 5] * src[11]
      + src[ 0] * src[ 7] * src[ 9]
      + src[ 4] * src[ 1] * src[11]
      - src[ 4] * src[ 3] * src[ 9]
      - src[ 8] * src[ 1] * src[ 7]
      + src[ 8] * src[ 3] * src[ 5];

  dst[12] =
      - src[ 4] * src[ 9] * src[14]
      + src[ 4] * src[10] * src[13]
      + src[ 8] * src[ 5] * src[14]
      - src[ 8] * src[ 6] * src[13]
      - src[12] * src[ 5] * src[10]
      + src[12] * src[ 6] * src[ 9];

  dst[13] =
      + src[ 0] * src[ 9] * src[14]
      - src[ 0] * src[10] * src[13]
      - src[ 8] * src[ 1] * src[14]
      + src[ 8] * src[ 2] * src[13]
      + src[12] * src[ 1] * src[10]
      - src[12] * src[ 2] * src[ 9];

  dst[14] =
      - src[ 0] * src[ 5] * src[14]
      + src[ 0] * src[ 6] * src[13]
      + src[ 4] * src[ 1] * src[14]
      - src[ 4] * src[ 2] * src[13]
      - src[12] * src[ 1] * src[ 6]
      + src[12] * src[ 2] * src[ 5];

  dst[15] =
      + src[ 0] * src[ 5] * src[10]
      - src[ 0] * src[ 6] * src[ 9]
      - src[ 4] * src[ 1] * src[10]
      + src[ 4] * src[ 2] * src[ 9]
      + src[ 8] * src[ 1] * src[ 6]
      - src[ 8] * src[ 2] * src[ 5];

  /* Compute determinant: */

  double det = + src[0] * dst[0] + src[1] * dst[4] + src[2] * dst[8] + src[3] * dst[12];
  
  /* Multiply adjoint with reciprocal of determinant: */
  det = 1.0f / det;

  dst[ 0] *= det;
  dst[ 1] *= det;
  dst[ 2] *= det;
  dst[ 3] *= det;
  dst[ 4] *= det;
  dst[ 5] *= det;
  dst[ 6] *= det;
  dst[ 7] *= det;
  dst[ 8] *= det;
  dst[ 9] *= det;
  dst[10] *= det;
  dst[11] *= det;
  dst[12] *= det;
  dst[13] *= det;
  dst[14] *= det;
  dst[15] *= det;

  memcpy(src, dst, 16 * sizeof(double));
}

void
_KRN_ComputeILU(
  const size_t nrows,
  const uint32_t bs2,
  const uint32_t *ai,
  const uint32_t *aj,
  const uint32_t *adiag,
  uint32_t *iw,
  double *aa)
{
  memset(iw, 0, nrows * sizeof(unsigned int));

  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int j0 = ai[i];
    const unsigned int j1 = ai[i+1];

    for(unsigned int j = j0; j < j1; j++) iw[aj[j]] = j;

    /* ILU factorization IKJ version k-loop */
    for(unsigned int j = j0; j < j1; j++)
    {
      const unsigned int k = aj[j];

      if(k < i)
      {
        double T[bs2];
        dgemm(aa + bidx2(j), aa + bidx2(adiag[k]), T);
        memcpy(aa + bidx2(j), T, bs2 * sizeof(double));

        for(unsigned int jj = adiag[k] + 1; jj < ai[k+1]; jj++)
        {
          const unsigned int jw = iw[aj[jj]];

          if(jw != 0)
          {
            double TT[bs2];
            dgemm(T, aa + bidx2(jj), TT);
            axpy1(-1.f, TT, aa + bidx2(jw));
          }
        }
      }
      else break;
    }

    inv(aa + bidx2(adiag[i]));

    for(unsigned int j = j0; j < j1; j++) iw[aj[j]] = 0;
  }
}
#endif
#include "geometry.h"
#include "core_kernel.h"
#include "bench.h"
void
ComputeNumericalILU(GEOMETRY *g)
{
  BENCH start_bench = rdbench();

  CXX_ILU(g->matrix);
/*
  _KRN_ComputeILU(
  g->n->sz,
  g->c->b2,
  g->c->mat->i,
  g->c->mat->j,
  g->c->mat->d,
  g->c->mat->w,
  g->c->mat->a);
*/
  fun3d_log(start_bench, KERNEL_NUMILU);
}