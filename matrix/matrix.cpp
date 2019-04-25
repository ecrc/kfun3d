#include <stdio.h>
#include <string.h>
#include <metis.h>
#include <stddef.h>
#include <omp.h>
#include "matrix.h"
#include "memory.h"

fun3d::Matrix::Matrix(const size_t nrows, const unsigned int bs, const unsigned int nthreads, const unsigned int ilu_sweeps, const unsigned int jacobi_sweeps)
{
  this->nrows = nrows;
  this->bs = bs;
  this->bs2 = bs * bs;
  this->iw = fun3d::calloc<unsigned int>(nrows);
  A = fun3d::malloc<MATRIX>();
  A->i = fun3d::calloc<unsigned int>((nrows + 1)); /* Row pointers */
  A->d = fun3d::calloc<unsigned int>((nrows)); /* Diagonal pointers */

  L = fun3d::malloc<MATRIX>();
  L->i = fun3d::calloc<unsigned int>(nrows + 1);
  L->d = fun3d::calloc<unsigned int>(nrows);
  U = fun3d::malloc<MATRIX>();
  U->i = fun3d::calloc<unsigned int>(nrows + 1);
  U->d = fun3d::calloc<unsigned int>(nrows);
  Ut = fun3d::malloc<MATRIX>();
  Ut->i = fun3d::calloc<unsigned int>(nrows + 1);
  Ut->d = fun3d::calloc<unsigned int>(nrows);

  this->ilu_sweeps = ilu_sweeps;
  this->jacobi_sweeps = jacobi_sweeps;
  
  c = fun3d::calloc<double>(nrows * bs);
  x = fun3d::calloc<double>(nrows * bs);

  omp_set_num_threads(nthreads);
}

fun3d::Matrix::~Matrix()
{
  fun3d::free(A->i);
  fun3d::free(A->j);
  fun3d::free(A->a);
  fun3d::free(A->d);
  fun3d::free(A);

  fun3d::free(L->i);
  fun3d::free(L->j);
  fun3d::free(L->a);
  fun3d::free(L->d);
  fun3d::free(L);

  fun3d::free(U->i);
  fun3d::free(U->j);
  fun3d::free(U->a);
  fun3d::free(U->d);
  fun3d::free(U);

  fun3d::free(Ut->i);
  fun3d::free(Ut->j);
  fun3d::free(Ut->a);
  fun3d::free(Ut->d);
  fun3d::free(Ut);

  fun3d::free(iw);
  fun3d::free(Ut_index);
  fun3d::free(U_index);

  fun3d::free(TL->j);
  fun3d::free(TL->i);
  fun3d::free(TL);
  fun3d::free(TU->j);
  fun3d::free(TU->i);
  fun3d::free(TU);

  fun3d::free(c);
  fun3d::free(x);
}

void fun3d::Matrix::mesh2csr(const size_t nedges, const unsigned int *n0, const unsigned int *n1)
{
  degree(nedges, n0, n1);
  for(unsigned int i = 0; i <= nrows; i++) A->i[i]++;
  scan(A->i);
  /* Add one to avoid METIS Segfault */
  A->j = fun3d::calloc<unsigned int>(nnz() + 1);
  columns(nedges, n0, n1);
  rcm();
  A->a = fun3d::calloc<double>(nnz() * this->bs2);
  /* Extract L and U */
  split();
  csr2csc();
  Ut_index = fun3d::calloc<unsigned int>(nnz());
  setUtIndex();
  U_index = fun3d::calloc<unsigned int>(nnz(Ut));
  setUIndex();
  task_graph();
}

size_t fun3d::Matrix::csr2parts(const unsigned int nparts, unsigned int *parts)
{
  size_t ncuts = 0;

  if(nparts > 1)
  {
    /* Number of balancing constraints that METIS needs to partitioning
     * the graph. The minimum is 1, which means you want to equidistribute 
     * the workload between the nodes as much as possible. 
     * So, one is the default.
     *
     * For more information, refer to METIS user manual
     * */
    int32_t ncon = 1;
    
    A->i[nrows]++; // Just for METIS

    METIS_PartGraphKway(
      (int32_t *) &nrows,
      &ncon,
      (int32_t *) A->i,
      (int32_t *) A->j,
      NULL,
      NULL,
      NULL,
      (int32_t *) &nparts,
      NULL,
      NULL,
      NULL,
      (int32_t *) &ncuts,
      (int32_t *) parts);

    A->i[nrows]--; // Just for METIS
  }

  return(ncuts);
}

void fun3d::Matrix::fill(const unsigned int row, const unsigned int col, const double a, const double v[])
{
  for(unsigned int i = A->i[row]; i < A->i[row+1]; i++)
  {
    if(A->j[i] == col)
    {
      for(unsigned int j = 0; j < bs2; j++) A->a[bs2 * i + j] += a * v[j];

      if(A->j[i] < row) memcpy(L->a + bidx2(L->i[row]), A->a + bidx2(i), bs2 * sizeof(double));
      if(A->j[i] >= row) memcpy(Ut->a + bidx2(Ut_index[i]), A->a + bidx2(i), bs2 * sizeof(double));

      break;
    }
  }
}

void fun3d::Matrix::fill(const unsigned int row, const unsigned int col, const double v[])
{
  for(unsigned int i = A->i[row]; i < A->i[row+1]; i++)
  {
    if(A->j[i] == col)
    {
      for(unsigned int j = 0; j < bs2; j++) A->a[bs2 * i + j] += v[j];

      if(A->j[i] < row) memcpy(L->a + bidx2(L->i[row]), A->a + bidx2(i), bs2 * sizeof(double));
      if(A->j[i] >= row) memcpy(Ut->a + bidx2(Ut_index[i]), A->a + bidx2(i), bs2 * sizeof(double));

      break;
    }
  }
}

void fun3d::Matrix::fill(const unsigned int row, const unsigned int col, const double v)
{
  for(unsigned int i = A->i[row]; i < A->i[row+1]; i++)
  {
    if(A->j[i] == col)
    {
      A->a[bs2 * i + 0]  += v;
      A->a[bs2 * i + 5]  += v;
      A->a[bs2 * i + 10] += v;
      A->a[bs2 * i + 15] += v;

      if(A->j[i] < row) memcpy(L->a + bidx2(L->i[row]), A->a + bidx2(i), bs2 * sizeof(double));
      if(A->j[i] >= row) memcpy(Ut->a + bidx2(Ut_index[i]), A->a + bidx2(i), bs2 * sizeof(double));

      break;
    }
  }
}

void fun3d::Matrix::fill(const unsigned int row, const unsigned int col, const unsigned int n, const double v[])
{
  for(unsigned int i = A->i[row]; i < A->i[row+1]; i++)
  {
    if(A->j[i] == col)
    {
      for(unsigned int j = 1; j <= n; j++) A->a[bs2 * i + (j * n + j)] += v[j-1];

      if(A->j[i] < row) memcpy(L->a + bidx2(L->i[row]), A->a + bidx2(i), bs2 * sizeof(double));
      if(A->j[i] >= row) memcpy(Ut->a + bidx2(Ut_index[i]), A->a + bidx2(i), bs2 * sizeof(double));

      break;
    }
  }
}

void fun3d::Matrix::fill()
{
  memset(A->a, 0, nnz() * bs2 * sizeof(double));
}

void fun3d::Matrix::example()
{
  for(unsigned int i = 0; i < nnz(); i++)
    for(unsigned int j = 0; j < bs2; j++) A->a[idx2(i, j)] = -1.f * (double)(idx2(i, j+1));

  for(unsigned i = 1; i <= bs2; i++)
  {
    (A->a + bidx2(A->d[1]))[i-1] = 4 * (double)i * -1;
    (A->a + bidx2(A->d[5]))[i-1] = 2 * (double)i * -1;
    (A->a + bidx2(A->d[6]))[i-1] = 3 * (double)i * -1;
  }
}

void fun3d::Matrix::print(FILE *stream)
{
  print(stream, A);
}