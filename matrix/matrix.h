#ifndef __FUN3D_INC_MATRIX_H
#define __FUN3D_INC_MATRIX_H

#include <stdio.h>
#include <stddef.h>

namespace fun3d {
  class Matrix {
    typedef struct matrix_t {
      unsigned int *i;
      unsigned int *j;
      unsigned int *d;
      double *a;
    } MATRIX;
    typedef struct task_t {
      unsigned int n;
      unsigned int *j;
      unsigned int *i;
    } TASK;
  private:
    MATRIX *A;

    MATRIX *L;
    MATRIX *U;
    MATRIX *Ut;

    double *c;
    double *x;

    size_t nrows;
    unsigned int bs;
    unsigned int bs2;
    unsigned int *iw; /* Working row */
    unsigned int ilu_sweeps;
    unsigned int jacobi_sweeps;

    unsigned int *Ut_index;
    unsigned int *U_index;

    TASK *TU;
    TASK *TL;

    /* mesh2csr */
    void degree(const size_t, const unsigned int *, const unsigned int *);
    void scan(unsigned int *);
    void columns(const size_t, const unsigned int *, const unsigned int *);
    void diagonal(const unsigned int, const unsigned int, const unsigned int);
    void rcm();
    /* Matrix factorization and matrix solve */
    unsigned int bidx2(const unsigned int) const;
    unsigned int bidx(const unsigned int) const;
    unsigned int idx2(const unsigned int, const unsigned int) const;
    unsigned int idx(const unsigned int, const unsigned int) const;
    void dgemv(const double *, const double *, double *);
    void dgemm(const double *, const double *, double *);
    void axpy(const double, const double *, double *);
    void axpy(const double, const double *, const double *, double *);
    void axpy2(const double, const double *, double *);    
    void inv();
    void inv(double *);
    void inv(const double *, double *);
    void inv(const MATRIX *);
    bool chk0pivot(double *);
    size_t nnz(const MATRIX *) const;
    size_t nnz() const;
    void csr2csc();
    void csr2csc(MATRIX *);
    //void csr2csc(const MATRIX *, MATRIX *);
    void setUtIndex();
    void setUIndex();
    void task_graph();
    void split();
    //void merge();
    void print(FILE *, const MATRIX *);
  public:
    Matrix(const size_t, const unsigned int, const unsigned int, const unsigned int, const unsigned int);
    ~Matrix();
    void mesh2csr(const size_t, const unsigned int*, const unsigned int*);
    size_t csr2parts(const unsigned int, unsigned int *);
    void fill(const unsigned int, const unsigned int, const double, const double *);
    void fill(const unsigned int, const unsigned int, const double *);
    void fill(const unsigned int, const unsigned int, const double);
    void fill(const unsigned int, const unsigned int, const unsigned int, const double *);
    void fill();
    void example();
    void print(FILE *);
    void ilu_YousefSaad();
    void ilu_Edmond();
    void sptrsv_YousefSaad(const double *, double *);
    void sptrsv_LevelScheduling(const double *, double *);
    void sptrsv_Jacobi(const double *, double *);
  }; /* class matrix */
}; /* namespace fun3d */

#endif /* __FUN3D_INC_MATRIX_H */