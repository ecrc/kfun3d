#include <algorithm>
#include <string.h>
#include <omp.h>
#include "memory.h"
#include "matrix.h"

void fun3d::Matrix::degree(const size_t nnedges, const unsigned int *n0, const unsigned int *n1)
{
  for(unsigned int i = 0; i < nnedges; i++)
  {
    A->i[n0[i]]++;
    A->i[n1[i]]++;
  }
}

void fun3d::Matrix::scan(unsigned int *array)
{
  unsigned int sum = 0;
  for(unsigned int i = 0; i <= nrows; i++)
  {
    const unsigned int t = array[i];
    array[i] = sum;
    sum += t;
  }
}

void fun3d::Matrix::columns(const size_t nedges, const unsigned int *n0, const unsigned int *n1)
{
  for(unsigned int i = 0; i < nrows; i++)
  {
    A->j[A->i[i]] = i;
    iw[i] = 1;
  }

  for(unsigned int i = 0; i < nedges; i++)
  {
    unsigned int index;

    index = A->i[n0[i]] + iw[n0[i]];
    iw[n0[i]]++;
    A->j[index] = n1[i];

    index = A->i[n1[i]] + iw[n1[i]];
    iw[n1[i]]++;
    A->j[index] = n0[i];
  }
}

void fun3d::Matrix::diagonal(const unsigned int start, const unsigned int end, const unsigned int row)
{
  for(unsigned int i = start; i < end; i++)
  {
    if(A->j[i] == row)
    {
      A->d[row] = i;
      break;
    }
  }
}

void fun3d::Matrix::rcm()
{
#pragma omp parallel for
  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = A->i[i];
    const unsigned int jend = A->i[i+1];

    std::sort(A->j + jstart, A->j + jend);

    /* Mark the diagonal */
    diagonal(jstart, jend, i);
  }
}

unsigned int fun3d::Matrix::bidx2(const unsigned int index) const
{
  return(index * bs2);
}

unsigned int fun3d::Matrix::bidx(const unsigned int index) const
{
  return(index * bs);
}

unsigned int fun3d::Matrix::idx2(const unsigned int i, const unsigned int j) const
{
  return(bidx2(i) + j);
}

unsigned int fun3d::Matrix::idx(const unsigned int i, const unsigned int j) const
{
  return(bidx(i) + j);
}

void fun3d::Matrix::dgemv(const double A[], const double x[], double y[])
{
  for(unsigned int i = 0; i < bs; i++)
  {
    y[i] = 0.f;
    for(unsigned int j = 0; j < bs; j++) y[i] += A[idx(i, j)] * x[j];
  }
}

void fun3d::Matrix::dgemm(const double A[], const double B[], double C[])
{
  for(unsigned int i = 0; i < bs; i++)
  {
    for(unsigned int j = 0; j < bs; j++)
    {
      C[idx(i, j)] = 0.f;
      
      for(unsigned int k = 0; k < bs; k++) C[idx(i, j)] += A[idx(i, k)] * B[idx(k, j)];
    }
  }
}

void fun3d::Matrix::axpy(const double a, const double *A, double *B)
{
  for(unsigned int i = 0; i < bs; i++)
  {
    /* AXPY */
    const double ax = a * A[i];
    const double axpy = ax + B[i];

    /* Update the vector component */
    B[i] = axpy;
  }
}

void fun3d::Matrix::axpy(const double a, const double *A, const double *B, double *C)
{
  for(unsigned int i = 0; i < bs; i++)
  {
    /* AXPY */
    const double ax = a * A[i];
    const double axpy = ax + B[i];

    /* Update the vector component */
    C[i] = axpy;
  }
}

void fun3d::Matrix::axpy2(const double a, const double *A, double *B)
{
  for(unsigned int i = 0; i < bs2; i++)
  {
    /* AXPY */
    const double ax = a * A[i];
    const double axpy = ax + B[i];

    /* Update the vector component */
    B[i] = axpy;
  }
}

size_t fun3d::Matrix::nnz(const MATRIX *matrix) const
{
  return((size_t) matrix->i[nrows]);
}

size_t fun3d::Matrix::nnz() const
{
  return(nnz(A));
}

void fun3d::Matrix::print(FILE *stream, const MATRIX *C)
{
  memset(iw, 0, nrows * sizeof(unsigned int));

  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = C->i[i];
    const unsigned int jend = C->i[i+1];

    for(unsigned int j = jstart; j < jend; j++) iw[C->j[j]] = 1;

    for(unsigned int i = 0; i < nrows; i++) fprintf(stream, "%d ", iw[i]);

    fprintf(stream, "\n");

    memset(iw, 0, nrows * sizeof(unsigned int));
  }

  fprintf(stream, "\n\n");
  for(unsigned int i = 0; i < nnz(C); i++)
  {
    fprintf(stream, "[%d]\n", i);
    for(unsigned int j = 0; j < bs2; j++) fprintf(stream, "\t[%d]: %lf\n", idx2(i,j), C->a[idx2(i,j)]);
  }
  fprintf(stream, "\n\n");
}

void fun3d::Matrix::task_graph()
{
  unsigned int *row = fun3d::calloc<unsigned int>(nrows);

  TL = fun3d::malloc<TASK>();
  TL->i = fun3d::calloc<unsigned int>(nrows);
  TL->n = 1;

  for(unsigned int i = 0; i < nrows; i++)
  {
    const unsigned int jstart = A->i[i];
    const unsigned int jend = A->d[i];

    unsigned int depth = 1;

    for(unsigned int j = jstart; j < jend; j++)
    {
      depth = std::max(depth, row[A->j[j]] + 1);
    }

    TL->n = std::max(TL->n, depth);
    row[i] = depth;
    ++TL->i[depth];
  }

  unsigned int *top = fun3d::calloc<unsigned int>(nrows);
  TL->j = fun3d::malloc<unsigned int>(nrows);

  for(unsigned int i = 1; i < TL->n + 1; ++i )
  {
    TL->i[i] += TL->i[i-1];
    top[i] = TL->i[i];
  }

  for(unsigned int i = 0; i < nrows; ++i )
  {
    TL->j[top[row[i]-1]] = i;
    ++top[row[i]-1];
  }

  memset(row, 0, nrows * sizeof(unsigned int));

  TU = fun3d::malloc<TASK>();
  TU->i = fun3d::calloc<unsigned int>(nrows);
  TU->n = 1;

  for(int i = (int)nrows-1; i >= 0; i--)
  {
    const unsigned int jstart = A->d[(int)i]+1;
    const unsigned int jend = A->i[(int)i+1];

    unsigned int depth = 1;

    for(unsigned int j = jstart; j < jend; j++)
    {
      depth = std::max(depth, row[A->j[j]]+1);
    }

    TU->n = std::max(TU->n, depth);
    row[i] = depth;
    ++TU->i[depth];
  }

  memset(top, 0, nrows * sizeof(unsigned int));
  TU->j = fun3d::malloc<unsigned int>(nrows);

  for(unsigned int i = 1; i < TU->n+1; i++)
  {
    TU->i[i] += TU->i[i-1];
    top[i] = TU->i[i];
  }

  for(unsigned int i = 0; i < nrows; i++)
  {
    TU->j[top[row[i]-1]] = i;
    ++top[row[i]-1];
  }

  fun3d::free(top);
  fun3d::free(row);
}