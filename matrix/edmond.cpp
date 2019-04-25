#include <algorithm>
#include <string.h>
#include <omp.h>
#include "memory.h"
#include "matrix.h"

/* 
  Extract lower (strict L with ones in the diagonal)
  and upper triangular (U with the diagonal)
*/
void fun3d::Matrix::split()
{
  /* Count L and U elements */
#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = A->i[i];
    const unsigned int jend = A->i[i+1];

    for(unsigned int j = jstart; j < jend; j++)
    {
      /* Equality to include the diagonal element */
      if(A->j[j] <= i) L->i[i]++;
      if(A->j[j] >= i) U->i[i]++;
    }
  }

  scan(L->i);
  scan(U->i);

  L->j = fun3d::malloc<unsigned int>(nnz(L));
  U->j = fun3d::malloc<unsigned int>(nnz(U));

  L->a = fun3d::calloc<double>(nnz(L) * bs2);
  U->a = fun3d::calloc<double>(nnz(U) * bs2);

#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = A->i[i];
    const unsigned int jend = A->i[i+1];

    unsigned int idxL = L->i[i];
    unsigned int idxU = U->i[i];

    for(unsigned int j = jstart; j < jend; j++)
    {
      if(A->j[j] <= i)
      {
        L->j[idxL] = A->j[j];

        if(A->j[j] == i)
        {
          L->d[i] = idxL;

          (L->a + bidx2(idxL))[0] = 1.f;
          (L->a + bidx2(idxL))[1] = 0.f;
          (L->a + bidx2(idxL))[2] = 0.f;
          (L->a + bidx2(idxL))[3] = 0.f;
          (L->a + bidx2(idxL))[4] = 0.f;
          (L->a + bidx2(idxL))[5] = 1.f;
          (L->a + bidx2(idxL))[6] = 0.f;
          (L->a + bidx2(idxL))[7] = 0.f;
          (L->a + bidx2(idxL))[8] = 0.f;
          (L->a + bidx2(idxL))[9] = 0.f;
          (L->a + bidx2(idxL))[10] = 1.f;
          (L->a + bidx2(idxL))[11] = 0.f;
          (L->a + bidx2(idxL))[12] = 0.f;
          (L->a + bidx2(idxL))[13] = 0.f;
          (L->a + bidx2(idxL))[14] = 0.f;
          (L->a + bidx2(idxL))[15] = 1.f;
        }

        idxL++;
      }
      if(A->j[j] >= i)
      {
        U->j[idxU] = A->j[j];
        if(A->j[j] == i) U->d[i] = idxU;
        idxU++;
      }
    }
  }
}

void fun3d::Matrix::csr2csc()
{
  unsigned int *offsets = fun3d::calloc<unsigned int>(nnz(U));

  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = U->i[i];
    const unsigned int jend  = U->i[i + 1];
  
    for(unsigned int j = jstart; j < jend; j++) //Ut->i[U->j[j]]++;
      offsets[j] = __sync_fetch_and_add(&(Ut->i[U->j[j]]), 1);
  }

  scan(Ut->i);

  Ut->j = fun3d::malloc<unsigned int>(nnz(Ut));
  Ut->a = fun3d::malloc<double>(nnz(Ut) * bs2);

#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = U->i[i];
    const unsigned int jend  = U->i[i+1];

    for(unsigned int j = jstart; j < jend; j++)
    {
      const unsigned int idx = Ut->i[U->j[j]] + offsets[j];
      Ut->j[idx] = i;
      if(j == jstart) Ut->d[i] = idx;
    }
  }

#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = Ut->i[i];
    const unsigned int jend  = Ut->i[i+1];

    std::sort(Ut->j + jstart, Ut->j + jend);
  }

  fun3d::free(offsets);
}

void fun3d::Matrix::setUtIndex()
{
  memset(iw, 0, nrows * sizeof(unsigned int));

  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = A->i[i];
    const unsigned int jend  = A->i[i+1];

    for(unsigned int j = jstart; j < jend; j++)
    {
      if(A->j[j] >= i)
      {
        const unsigned int idx = Ut->i[A->j[j]] + iw[A->j[j]]++;
        Ut_index[j] = idx;
      } 
    }
  }
}

void fun3d::Matrix::setUIndex()
{
  memset(iw, 0, nrows * sizeof(unsigned int));

  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = Ut->i[i];
    const unsigned int jend  = Ut->i[i+1];

    for(unsigned int j = jstart; j < jend; j++)
    {
      const unsigned int idx = U->i[Ut->j[j]] + iw[Ut->j[j]]++;
      U_index[j] = idx;
    }
  }
}

void fun3d::Matrix::csr2csc(MATRIX *CSC)
{
#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = A->i[i];
    const unsigned int jend  = A->i[i+1];

    for(unsigned int j = jstart; j < jend; j++)
    {
      if(A->j[j] >= i) memcpy(CSC->a + bidx2(Ut_index[j]), A->a + bidx2(j), bs2 * sizeof(double));
    }
  }
}