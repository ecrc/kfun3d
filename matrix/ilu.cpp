#include <omp.h>
#include <assert.h>
#include <string.h>
#include "matrix.h"
#include "memory.h"

void fun3d::Matrix::ilu_YousefSaad()
{
  memset(iw, 0, nrows * sizeof(unsigned int));

  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int j0 = A->i[i];
    const unsigned int j1 = A->i[i+1];

    for(unsigned int j = j0; j < j1; j++) iw[A->j[j]] = j;

    for(unsigned int j = j0; j < j1; j++)
    {
      const unsigned int k = A->j[j];

      if(k < i)
      {
        double T[bs2];
        dgemm(A->a + bidx2(j), A->a + bidx2(A->d[k]), T);
        memcpy(A->a + bidx2(j), T, bs2 * sizeof(double));

        for(unsigned int jj = A->d[k] + 1; jj < A->i[k+1]; jj++)
        {
          const unsigned int jw = iw[A->j[jj]];

          if(jw != 0)
          {
            double TT[bs2];
            dgemm(T, A->a + bidx2(jj), TT);
            axpy2(-1.f, TT, A->a + bidx2(jw));
          }
        }
      }
      else break;
    }

    inv(A->a + bidx2(A->d[i]));

    for(unsigned int j = j0; j < j1; j++) iw[A->j[j]] = 0;
  }
}

void fun3d::Matrix::ilu_Edmond()
{
  unsigned int sweep = 0;
  do {
#pragma omp parallel for
    for(unsigned int i = 0; i < nrows; i++)
    {
      const unsigned int jstart = A->i[i];
      const unsigned int jend = A->i[i+1];

      for(unsigned int j = jstart; j < jend; j++)
      {
        const unsigned int k = A->j[j];

        unsigned int il = L->i[i];
        unsigned int iu = Ut->i[k];

        double a[bs2];
        memcpy(a, A->a + bidx2(j), bs2 * sizeof(double));
        double d[bs2];
        memset(d, 0, bs2 * sizeof(double));

        while(il < L->i[i+1] && iu < Ut->i[k+1])
        {
          const unsigned int jl = L->j[il];
          const unsigned int ju = Ut->j[iu];

          if(jl == ju)
          {
            dgemm(L->a + bidx2(il), Ut->a + bidx2(iu), d);
            axpy2(-1.f, d, a);
          }
          
          if(jl <= ju) il++;
          if(jl >= ju) iu++;
        }

        axpy2(1.f, d, a);

        if(i > k)
        {
          double T[bs2];
          inv(Ut->a + bidx2(Ut->i[k+1]-1), T);
          dgemm(a, T, L->a + bidx2(il-1));
        }
        else memcpy(Ut->a + bidx2(iu-1), a, bs2 * sizeof(double));
      }
    }
  } while(++sweep < ilu_sweeps);

#if 0

#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = Ut->i[i];
    const unsigned int jend  = Ut->i[i+1];

    for(unsigned int j = jstart; j < jend; j++)
      memcpy(U->a + bidx2(U_index[j]), Ut->a + bidx2(j), bs2 * sizeof(double));

    inv(U->a + bidx2(U->d[i]));
  }
  
#else

#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = A->i[i];
    const unsigned int jend = A->i[i+1];

    unsigned int idxL = L->i[i];

    for(unsigned int j = jstart; j < jend; j++)
    {
      if(A->j[j] < i)
      {
        memcpy(A->a + bidx2(j), L->a + bidx2(idxL++), bs2 * sizeof(double));
      }
      if(A->j[j] >= i)
      {
        memcpy(A->a + bidx2(j), Ut->a + bidx2(Ut_index[j]), bs2 * sizeof(double));
      }
    }

    inv(A->a + bidx2(A->d[i]));
  }
#endif
}