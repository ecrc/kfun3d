#ifdef __cplusplus
extern "C" {
#endif
extern void CXX_SpTRSV(const double *b, double *x, void *ptr);
#ifdef __cplusplus
}
#endif
#if 0
#include <string.h>
#include "stdint.h"
#include "geometry.h"
#include "core_kernel.h"
#include "bench.h"

inline static unsigned int bidx2(const unsigned int index)
{
  return(index * 16);
}
inline static unsigned int bidx(const unsigned int index)
{
  return(index * 4);
}

inline static void dgemv(const double A[], const double B[], double C[])
{
  for(unsigned int i = 0; i < 4; i++)
  {
    C[i] = 0.f;
    for(unsigned int j = 0; j < 4; j++) C[i] += A[i * 4 + j] * B[j];
  }
}
inline static void axpy(const double a, const double *x, double *y)
{
  for(unsigned int i = 0; i < 4; i++)
  {
    /* AXPY */
    const double ax = a * x[i];
    const double axpy = ax + y[i];

    /* Update the vector component */
    y[i] = axpy;
  }
}

static void
_KRN_ComputeSparseTriangularSolve(
  const size_t nrows,
  const unsigned int bs,
  const unsigned int *ai,
  const unsigned int *aj,
  const double *aa,
  const unsigned int *adiag,
  const double *b,
  double *x)
{
  for(unsigned int i = 0; i < nrows; i++)
  {
    memcpy(x + bidx(i), b + bidx(i), 4 * sizeof(double));

    for(unsigned int j = ai[i]; j < adiag[i]; j++)
    {
      double t[bs];
      dgemv(aa + bidx2(j), x + bidx(aj[j]), t);
      axpy(-1.f, t, x + bidx(i));
    }
  }

  for(int i = (int)nrows-1; i >= 0; i--)
  {
    for(unsigned int j = adiag[i] + 1; j < ai[i+1]; j++)
    {
      double t[bs];
      dgemv(aa + bidx2(j), x + bidx(aj[j]), t);
      axpy(-1.f, t, x + bidx((unsigned int)i));
    }

    double t[bs];
    dgemv(aa + bidx2(adiag[i]), x + bidx((unsigned int) i), t);
    memcpy(x + bidx((unsigned int) i), t, bs * sizeof(double));
  }
}
#endif
#include "geometry.h"
#include "core_kernel.h"
#include "bench.h"
void
ComputeSparseTriangularSolve(const GEOMETRY *g, const double *b, double *x)
{
  BENCH start_bench = rdbench();

  CXX_SpTRSV(b, x, g->matrix);
/*
  _KRN_ComputeSparseTriangularSolve(
    g->n->sz,
    g->c->b,
    g->c->mat->i,
    g->c->mat->j,
    g->c->mat->a,
    g->c->mat->d,
    b,
    x
  );
*/
  fun3d_log(start_bench, KERNEL_SPTRSV);
}