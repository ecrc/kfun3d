#include <omp.h>
#include <string.h>
#include "matrix.h"

void fun3d::Matrix::sptrsv_YousefSaad(const double *b, double *x)
{
  for(unsigned int i = 0; i < nrows; i++)
  {
    memcpy(x + bidx(i), b + bidx(i), bs * sizeof(double));

    for(unsigned int j = A->i[i]; j < A->d[i]; j++)
    {
      double t[bs];
      dgemv(A->a + bidx2(j), x + bidx(A->j[j]), t);
      axpy(-1.f, t, x + bidx(i));
    }
  }

  for(int i = (int)nrows-1; i >= 0; i--)
  {
    for(unsigned int j = A->d[i] + 1; j < A->i[i+1]; j++)
    {
      double t[bs];
      dgemv(A->a + bidx2(j), x + bidx(A->j[j]), t);
      axpy(-1.f, t, x + bidx((unsigned int)i));
    }

    double t[bs];
    dgemv(A->a + bidx2(A->d[i]), x + bidx((unsigned int) i), t);
    memcpy(x + bidx((unsigned int) i), t, bs * sizeof(double));
  }
}

void fun3d::Matrix::sptrsv_LevelScheduling(const double *b, double *x)
{
  for(unsigned int l = 0; l < TL->n; l++)
  {
#pragma omp parallel for
    for(unsigned int i = TL->i[l]; i < TL->i[l+1]; i++)
    {
      unsigned int row = TL->j[i];

      memcpy(x + bidx(row), b + bidx(row), bs * sizeof(double));

      for(unsigned int j = A->i[row]; j < A->d[row]; j++)
      {
        double t[bs];
        dgemv(A->a + bidx2(j), x + bidx(A->j[j]), t);
        axpy(-1.f, t, x + bidx(row));
      }
    }
  }

  for(unsigned int l = 0; l < TU->n; l++)
  {
#pragma omp parallel for
    for(unsigned int i = TU->i[l]; i < TU->i[l+1]; i++)
    {
      unsigned int row = TU->j[i];

      for(unsigned int j = A->d[row] + 1; j < A->i[row+1]; j++)
      {
        double t[bs];
        dgemv(A->a + bidx2(j), x + bidx(A->j[j]), t);
        axpy(-1.f, t, x + bidx((unsigned int)row));
      }

      double t[bs];
      dgemv(A->a + bidx2(A->d[row]), x + bidx((unsigned int) row), t);
      memcpy(x + bidx((unsigned int) row), t, bs * sizeof(double));
    }
  }
}

void fun3d::Matrix::sptrsv_Jacobi(const double *b, double *x)
{
  unsigned int sweep;

  /* L */
  memcpy(this->x, b, nrows * bs * sizeof(double));

  sweep = 0;
  do {
#pragma omp parallel for
    for(unsigned int i = 0; i < nrows; i++)
    {
      const unsigned int jstart = L->i[i];
      const unsigned int jend = L->d[i];

      double sum[bs];
      memset(sum, 0, bs * sizeof(double));

      for(unsigned int j = jstart; j < jend; j++)
      {
        double temp[bs];
        dgemv(L->a + bidx2(j), this->x + bidx(L->j[j]), temp);
        axpy(1.f, temp, sum);
      }
      axpy(-1.f, sum, b + bidx(i), c + bidx(i));
    }
    memcpy(this->x, c, nrows * bs * sizeof(double));
  } while(++sweep < jacobi_sweeps);

  /* U */
#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
    dgemv(U->a + bidx2(U->d[i]), this->x + bidx(i), x + bidx(i));

  sweep = 0;
  do {
#pragma omp parallel for
    for(unsigned int i = 0; i < nrows; i++)
    {
      const unsigned int jstart = U->d[i]+1;
      const unsigned int jend = U->i[i+1];

      double sum[bs];
      memset(sum, 0, bs * sizeof(double));

      for(unsigned int j = jstart; j < jend; j++)
      {
        double temp[bs];
        dgemv(U->a + bidx2(j), x + bidx(U->j[j]), temp);
        axpy(1.f, temp, sum);
      }
      double buf[bs];
      axpy(-1.f, sum, this->x + bidx(i), buf);
      dgemv(U->a + bidx2(U->d[i]), buf, c + bidx(i));
    }
    memcpy(x, c, nrows * bs * sizeof(double));
  } while(++sweep < jacobi_sweeps);
}