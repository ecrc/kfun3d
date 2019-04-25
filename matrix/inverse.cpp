#include <omp.h>
#include <string.h>
#include "matrix.h"

#ifdef DEBUG

#include <assert.h>

void fun3d::Matrix::inv(double *src)
{
  double dst[bs2];

  if(bs == 2)
  {    
    /* Compute adjoint: */
    dst[0] = + src[3];
    dst[1] = - src[1];
    dst[2] = - src[2];
    dst[3] = + src[0];

    /* Compute determinant: */
    double det = src[0] * dst[0] + src[1] * dst[2];
    assert(det != 0);

    /* Multiply adjoint with reciprocal of determinant: */
    det = 1.0f / det;

    dst[0] *= det;
    dst[1] *= det;
    dst[2] *= det;
    dst[3] *= det;
  }
  else
  {
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
    assert(det != 0);
    
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
  }

  memcpy(src, dst, bs2 * sizeof(double));
}

#else

void fun3d::Matrix::inv(double *src)
{
  double dst[bs2];

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

  memcpy(src, dst, bs2 * sizeof(double));
}

#endif

void fun3d::Matrix::inv(const double *src, double *dst)
{
  memcpy(dst, src, bs2 * sizeof(double));

  inv(dst);
}

void fun3d::Matrix::inv(const MATRIX *A)
{
#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++) inv(A->a + bidx2(A->d[i]));
}

void fun3d::Matrix::inv()
{
  inv(A);
}