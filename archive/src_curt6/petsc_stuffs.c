
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "kernel.h"
#include "petsc_stuffs.h"

#define GMRES_RTL_THRESHOLD 0.01
#define GMRES_RST_THRESHOLD 30
#define GMRES_ITS_THRESHOLD 10000

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

double *VTEMP;
double *VMOPT;

GMRES *
CreateGMRESTable(const size_t sz)
{
  /* 
    Allocate array to hold pointers to user vectors.
    Note that we need 4 + max_k + 1
    (since we need it+1 vectors, and it <= max_k)
    VEC_OFFSET + 2 + max_k;
  */
  const size_t number_of_basis = GMRES_RST_THRESHOLD + 2 + 2;

  int hs = (GMRES_RST_THRESHOLD + 1) * (GMRES_RST_THRESHOLD + 1);
  int hh = (GMRES_RST_THRESHOLD + 2) * (GMRES_RST_THRESHOLD + 1);
  int rs = (GMRES_RST_THRESHOLD + 2);
  int cc = (GMRES_RST_THRESHOLD + 1);

  GMRES *gmres = (GMRES *) malloc(sizeof(GMRES));

  gmres->hh_origin = (double *) calloc(hh, sizeof(double));
  gmres->hs_origin = (double *) calloc(hs, sizeof(double));
  gmres->rs_origin = (double *) calloc(rs, sizeof(double));
  gmres->cc_origin = (double *) calloc(cc, sizeof(double));
  gmres->ss_origin = (double *) calloc(cc, sizeof(double));

  gmres->orthogwork = (double *) malloc((GMRES_RST_THRESHOLD + 2) * sizeof(double));

  double **arr = (double **) malloc(number_of_basis * sizeof(double *));

  int i;
  for(i = 0; i < number_of_basis; i++) arr[i] = (double *) calloc(sz, sizeof(double));

  gmres->vecs = arr;

  VTEMP = VEC_TEMP;
  VMOPT = VEC_TEMP_MATOP;

  return gmres;
}


void
ComputeResidual(const double *, double *, void *);

double
Compute2ndNorm(const size_t sz, const double *v)
{
  double norm = 0;

  unsigned int i;
  for(i = 0; i < sz; i++) norm += v[i] * v[i];

  return(sqrt(norm));
}

void
UpdateHessenbergMatrix(
  const int it,
 // const int hapend,
  double *res,
  double *cc,
  double *ss,
  double *hh,
  double *grs)
{
  double tt;

  /*
    Apply all the previously computed
    plane rotations to the new column
    of the Hessenberg matrix
  */
  int i;
  for(i = 0; i < it; i++)
  {
    tt  = *hh;
    *hh = *cc * tt + *ss * *(hh+1);
    hh++;
    *hh = *cc++ * *hh - (*ss++ * tt);
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system
     2) the new column of the Hessenberg matrix
    thus obtaining the updated value of the residual
  */
  //if(!hapend)
  //{
    tt = sqrt(*hh * *hh + *(hh+1) * *(hh+1));
    if (tt == 0.0)
    {
      /* ksp->reason = KSP_DIVERGED_NULL; -2 */
      /* return(0); */
    }
    *cc        = *hh / tt;
    *ss        = *(hh+1) / tt;
    grs[it+1] = -(*ss * grs[it]);
    grs[it]   = *cc * grs[it];
    *hh        = *cc * *hh + *ss * *(hh+1);
    *res       = fabs(grs[it+1]);
  //}
  //else {
    /* happy breakdown: HH(it+1, it) = 0, therfore we don't need to apply
            another rotation matrix (so RH doesn't change).  The new residual is
            always the new sine term times the residual from last time (GRS(it)),
            but now the new sine rotation would be zero...so the residual should
            be zero...so we will multiply "zero" by the last residual.  This might
            not be exactly what we want to do here -could just return "zero". */

//    *res = 0.0;
  //}
}


void
ComputeSparseTriangularSolve(
  const int n,
  const int bs,
  const int bs2,
  const int *ai,
  const int *aj,
  const double *aa,
  const int *adiag,
  const double *b,
  double *x)
{
  const double *v;
  const int *vi;
  double s1, s2, s3, s4, x1, x2, x3, x4;
  int idx  = 0, nz;

  /* forward solve the lower triangular */
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];x[3] = b[3+idx];

  int i;

  for(i = 1; i < n; i++)
  {
    v   = aa + bs2 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];

    int k;
    for(k = 0; k < nz; k++)
    {
      int jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];x4 =x[3+jdx];
      s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3 + v[12]*x4;
      s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3 + v[13]*x4;
      s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
      s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;

      v +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
  }

  /* backward solve the upper triangular */
  for(i = n-1; i >= 0; i--)
  {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    int idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];s3 = x[2+idt];s4 = x[3+idt];

    int k;
    for(k = 0; k < nz; k++)
    {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];x4 = x[3+idx];
      s1 -= v[0]*x1 + v[4]*x2 + v[8]*x3 + v[12]*x4;
      s2 -= v[1]*x1 + v[5]*x2 + v[9]*x3 + v[13]*x4;
      s3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
      s4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;

      v +=  bs2;
    }

    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[4]*s2 + v[8]*s3 + v[12]*s4;
    x[1+idt] = v[1]*s1 + v[5]*s2 + v[9]*s3 + v[13]*s4;;
    x[2+idt] = v[2]*s1 + v[6]*s2 + v[10]*s3 + v[14]*s4;
    x[3+idt] = v[3]*s1 + v[7]*s2 + v[11]*s3 + v[15]*s4;
  }
}



/*
  BlockMult4x4: A = A * B with size bs=4

  Input Parameters:
+  A,B - square bs by bs arrays stored in column major order
-  W   - bs*bs work arrary

  Output Parameter:
.  A = A * B
*/

#define BlockMult4x4(A, B, W) \
  { \
    memcpy((void *) W, (const void *) A, (size_t) (16 * sizeof(double))); \
    A[0]  =  W[0]*B[0]  + W[4]*B[1]  + W[8]*B[2]   + W[12]*B[3]; \
    A[1]  =  W[1]*B[0]  + W[5]*B[1]  + W[9]*B[2]   + W[13]*B[3]; \
    A[2]  =  W[2]*B[0]  + W[6]*B[1]  + W[10]*B[2]  + W[14]*B[3]; \
    A[3]  =  W[3]*B[0]  + W[7]*B[1]  + W[11]*B[2]  + W[15]*B[3]; \
    A[4]  =  W[0]*B[4]  + W[4]*B[5]  + W[8]*B[6]   + W[12]*B[7]; \
    A[5]  =  W[1]*B[4]  + W[5]*B[5]  + W[9]*B[6]   + W[13]*B[7]; \
    A[6]  =  W[2]*B[4]  + W[6]*B[5]  + W[10]*B[6]  + W[14]*B[7]; \
    A[7]  =  W[3]*B[4]  + W[7]*B[5]  + W[11]*B[6]  + W[15]*B[7]; \
    A[8]  =  W[0]*B[8]  + W[4]*B[9]  + W[8]*B[10]  + W[12]*B[11]; \
    A[9]  =  W[1]*B[8]  + W[5]*B[9]  + W[9]*B[10]  + W[13]*B[11]; \
    A[10] = W[2]*B[8]  + W[6]*B[9]  + W[10]*B[10] + W[14]*B[11]; \
    A[11] = W[3]*B[8]  + W[7]*B[9]  + W[11]*B[10] + W[15]*B[11]; \
    A[12] = W[0]*B[12] + W[4]*B[13] + W[8]*B[14]  + W[12]*B[15]; \
    A[13] = W[1]*B[12] + W[5]*B[13] + W[9]*B[14]  + W[13]*B[15]; \
    A[14] = W[2]*B[12] + W[6]*B[13] + W[10]*B[14] + W[14]*B[15]; \
    A[15] = W[3]*B[12] + W[7]*B[13] + W[11]*B[14] + W[15]*B[15]; \
  }

/*
  BlockSubANDMult4x4: A = A - B * C with size bs=4

  Input Parameters:
+  A,B,C - square bs by bs arrays stored in column major order

  Output Parameter:
.  A = A - B*C
*/

#define BlockSubANDMult4x4(A, B, C) \
  { \
    A[0]  -=  B[0]*C[0]  + B[4]*C[1]  + B[8]*C[2]   + B[12]*C[3]; \
    A[1]  -=  B[1]*C[0]  + B[5]*C[1]  + B[9]*C[2]   + B[13]*C[3]; \
    A[2]  -=  B[2]*C[0]  + B[6]*C[1]  + B[10]*C[2]  + B[14]*C[3]; \
    A[3]  -=  B[3]*C[0]  + B[7]*C[1]  + B[11]*C[2]  + B[15]*C[3]; \
    A[4]  -=  B[0]*C[4]  + B[4]*C[5]  + B[8]*C[6]   + B[12]*C[7]; \
    A[5]  -=  B[1]*C[4]  + B[5]*C[5]  + B[9]*C[6]   + B[13]*C[7]; \
    A[6]  -=  B[2]*C[4]  + B[6]*C[5]  + B[10]*C[6]  + B[14]*C[7]; \
    A[7]  -=  B[3]*C[4]  + B[7]*C[5]  + B[11]*C[6]  + B[15]*C[7]; \
    A[8]  -=  B[0]*C[8]  + B[4]*C[9]  + B[8]*C[10]  + B[12]*C[11]; \
    A[9]  -=  B[1]*C[8]  + B[5]*C[9]  + B[9]*C[10]  + B[13]*C[11]; \
    A[10] -= B[2]*C[8]  + B[6]*C[9]  + B[10]*C[10] + B[14]*C[11]; \
    A[11] -= B[3]*C[8]  + B[7]*C[9]  + B[11]*C[10] + B[15]*C[11]; \
    A[12] -= B[0]*C[12] + B[4]*C[13] + B[8]*C[14]  + B[12]*C[15]; \
    A[13] -= B[1]*C[12] + B[5]*C[13] + B[9]*C[14]  + B[13]*C[15]; \
    A[14] -= B[2]*C[12] + B[6]*C[13] + B[10]*C[14] + B[14]*C[15]; \
    A[15] -= B[3]*C[12] + B[7]*C[13] + B[11]*C[14] + B[15]*C[15]; \
  }

void
Invert4x4Block(double *a)
{
  int i__2,i__3,kp1,j,l,ll,i,ipvt[4],kb,k3;
  int k4,j3;
  double *aa, *ax, *ay, work[16], stmp;
  double tmp, max;

  /* Parameter adjustments */
  a -= 5;

  int k;
  for(k = 1; k <= 3; k++)
  {
    kp1 = k + 1;
    k3 = 4 * k;
    k4 = k3 + k;

    /* find l = pivot index */
    i__2 = 5 - k;
    aa   = &a[k4];
    max  = fabs(aa[0]);
    l    = 1;
    for (ll=1; ll<i__2; ll++)
    {
      tmp = fabs(aa[ll]);
      if (tmp > max) { max = tmp; l = ll+1;}
    }

    l        += k - 1;
    ipvt[k-1] = l;

    if (a[l + k3] == 0.0)
    {
      /*
      Detect a zero pivot
      */
    }

    /* interchange if necessary */
    if (l != k) {
      stmp      = a[l + k3];
      a[l + k3] = a[k4];
      a[k4]     = stmp;
    }

    /* compute multipliers */
    stmp = -1. / a[k4];
    i__2 = 4 - k;
    aa   = &a[1 + k4];
    for (ll=0; ll<i__2; ll++) aa[ll] *= stmp;

    /* row elimination with column indexing */
    ax = &a[k4+1];
    for (j = kp1; j <= 4; ++j)
    {
      j3   = 4*j;
      stmp = a[l + j3];
      if (l != k)
      {
      a[l + j3] = a[k + j3];
      a[k + j3] = stmp;
      }

      i__3 = 4 - k;
      ay   = &a[1+k+j3];
      for (ll=0; ll<i__3; ll++) ay[ll] += stmp*ax[ll];
    }
  
  }

  ipvt[3] = 4;
  if(a[20] == 0.0)
  {
  /* Detect a zero pivot */
  }

  /* Now form the inverse */
  /* compute inverse(u) */
  for(k = 1; k <= 4; k++)
  {
    k3 = 4 * k;
    k4 = k3 + k;
    a[k4] = 1.0 / a[k4];
    stmp = -a[k4];
    i__2 = k - 1;
    aa = &a[k3 + 1];

    for(ll = 0; ll < i__2; ll++) aa[ll] *= stmp;

    kp1 = k + 1;

    if(4 < kp1) continue;

    ax = aa;
    for(j = kp1; j <= 4; j++)
    {
      j3 = 4 * j;
      stmp = a[k + j3];
      a[k + j3] = 0.0;
      ay = &a[j3 + 1];
      for (ll=0; ll<k; ll++) ay[ll] += stmp*ax[ll];
    }
  }

  /* form inverse(u)*inverse(l) */
  for (kb = 1; kb <= 3; ++kb)
  {
    k   = 4 - kb;
    k3  = 4*k;
    kp1 = k + 1;
    aa  = a + k3;

    for (i = kp1; i <= 4; ++i)
    {
      work[i-1] = aa[i];
      aa[i]     = 0.0;
    }

    for (j = kp1; j <= 4; ++j)
    {
      stmp   = work[j-1];
      ax     = &a[4*j + 1];
      ay     = &a[k3 + 1];
      ay[0] += stmp*ax[0];
      ay[1] += stmp*ax[1];
      ay[2] += stmp*ax[2];
      ay[3] += stmp*ax[3];
    }
    l = ipvt[k-1];
    if (l != k)
    {
      ax   = &a[k3 + 1];
      ay   = &a[4*l + 1];
      stmp = ax[0]; ax[0] = ay[0]; ay[0] = stmp;
      stmp = ax[1]; ax[1] = ay[1]; ay[1] = stmp;
      stmp = ax[2]; ax[2] = ay[2]; ay[2] = stmp;
      stmp = ax[3]; ax[3] = ay[3]; ay[3] = stmp;
    }
  }
}

void
ComputeNumericalILU(
  const int n,
  const int bs2,
  const int *ai,
  const int *aj,
  const int *bi,
  const int *bj,
  const int *bdiag,
  double *aa,
  double *ba)
{
  double *mwork = (double*) malloc((size_t) (bs2 * sizeof(double)));
  double *rtmp = (double*) malloc((size_t) (bs2 * n * sizeof(double)));

  memset((void *) rtmp, 0, (size_t) (bs2 * n * sizeof(double)));

  int nz, nzL, row;
  const int *ajtmp, *bjtmp, *pj;
  double *pc, *v, *pv;

  int i;
  for(i = 0; i < n; i++)
  {
    /* zero rtmp */

    /* L part */
    nz = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];

    int j;

    for(j = 0; j < nz; j++)
    {
      memset((void *) (rtmp + bs2 * bjtmp[j]), 0, (size_t) (bs2 * sizeof(double)));
    }

    /* U part */
    nz = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;

    for(j = 0; j < nz; j++)
    {
      memset((void *) (rtmp + bs2 * bjtmp[j]), 0, (size_t) (bs2 * sizeof(double)));
    }

    /* load in initial (unfactored row) */
    nz = ai[i+1] - ai[i];
    ajtmp = aj + ai[i];
    v = aa + bs2*ai[i];

    for(j = 0; j < nz; j++)
    {
      memcpy((void *) (rtmp + bs2 * ajtmp[j]), (const void *) (v + bs2 * j), (size_t) (bs2 * sizeof(double)));
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL = bi[i+1] - bi[i];

    int k;
    for(k = 0; k < nzL; k++)
    {
      row = bjtmp[k];
      pc = rtmp + bs2 * row;

      int flag;
      for(flag = 0, j = 0; j < bs2; j++)
      {
        if(pc[j] != 0.0)
        {
          flag = 1;
          break;
        }
      }

      if(flag)
      {
        pv = ba + bs2*bdiag[row];

        BlockMult4x4(pc, pv, mwork);

        /* begining of U(row,:) */
        pj = bj + bdiag[row+1]+1;
        pv = ba + bs2*(bdiag[row+1]+1);

        /* num of entries inU(row,:), excluding diag */
        nz = bdiag[row] - bdiag[row+1] - 1;

        for(j = 0; j < nz; j++)
        {
          v = rtmp + bs2 * pj[j];

          BlockSubANDMult4x4(v, pc, pv);

          pv += bs2;
        }
      }
    }

    /* finished row so stick it into ba */
    /* L part */
    pv = ba + bs2*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];

    for(j = 0; j < nz; j++)
    {
      memcpy((void *) (pv + bs2 * j), (const void *) (rtmp + bs2 * pj[j]), (size_t) (bs2 * sizeof(double)));
    }

    /* Mark diagonal and invert diagonal for simplier triangular solves */
    pv = ba + bs2*bdiag[i];
    pj = bj + bdiag[i];

    memcpy((void *) pv, (const void *) (rtmp + bs2 * pj[0]), (size_t) (bs2 * sizeof(double)));

    Invert4x4Block(pv);

    /* U part */
    pv = ba + bs2*(bdiag[i+1]+1);
    pj = bj + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;

    for(j = 0; j < nz; j++)
    {
      memcpy((void *) (pv + bs2 * j), (const void *) (rtmp + bs2 * pj[j]), (size_t) (bs2 * sizeof(double)));
    }
  }

  free(rtmp);
  free(mwork);
}

void
ComputeSymbolicILU(
//  const int nnz,
  const int n,
  const int m,
  const int bs2,
  const int *ai,
  int *aj,
//  int *bbs2,
//  int *bmbs,
//  int *bnbs,
//  int *bnz,
//  int *bmaxnz,
  int **bii,
  int **bjj,
  double **baa,
//  int **aadiag,
  int **bbdiag)
{
//  *bbs2 = bs2;
//  *bmbs = n;
//  *bnbs = m;
//  *bnz = nnz;
//  *bmaxnz = nnz;

  int *adiag = (int *) malloc((size_t) (n * sizeof(int)));
  int *bdiag = (int *) malloc((size_t) ((n + 1) * sizeof(int)));

  int i, nz, bi_temp;

  for(i = 0; i < n; i++)
  {
    adiag[i] = ai[i + 1];

    int j;
    for(j = ai[i]; j < ai[i+1]; j++)
    {
      if(aj[j] == i)
      {
        bdiag[i] = adiag[i] = j;
        break;
      }
    }
  }

  double *ba = (double *) malloc((size_t) ((bs2 * ai[n] + 1) * sizeof(double)));
  memset((void *) ba, 0, (size_t) (bs2 * ai[n] * sizeof(double)));

  int *bi = (int *) malloc((size_t) ((n + 1) * sizeof(int)));

  int *bj = (int *) malloc((size_t) ((ai[n] + 1) * sizeof(int)));

  *bii = bi;
  *bjj = bj;

  /* L part */

  bi[0] = 0;
  for(i=0; i<n; i++)
  {
    nz = adiag[i] - ai[i];
    bi[i+1] = bi[i] + nz;
    
    int *ajj = aj + ai[i];

    int j;
    for(j = 0; j < nz; j++)
    {
      *bj = ajj[j]; bj++;
    }
  }

  /* U part */

  bi_temp  = bi[n];
  bdiag[n] = bi[n]-1;

  for(i = n-1; i >= 0; i--)
  {
    nz      = ai[i+1] - adiag[i] - 1;
    bi_temp = bi_temp + nz + 1;
    
    int *ajj = aj + adiag[i] + 1;

    int j;
    for( j = 0; j < nz; j++)
    {
      *bj = ajj[j]; bj++;
    }

    /* diag[i] */
    *bj      = i; bj++;
    bdiag[i] = bi_temp - 1;
  }

  free(adiag);

  *bbdiag = bdiag;

  *baa = ba;
}


Jacobian *
CreateJacobianTable(const size_t sz)
{
  Jacobian *j = (Jacobian *) calloc(1, sizeof(Jacobian));

  j->q_norm = 0.f;
  j->q = (double *) malloc(sz * sizeof(double));
  j->w = (double *) malloc(sz * sizeof(double));

  return j;
}

void
BuildJacobianMatrix(const size_t sz, const double *q, Jacobian *j)
{
  void *destination = (void *) j->q;
  const void *source = (const void *) q;
  size_t num = (size_t) (sz * sizeof(double));

  memcpy(destination, source, num);

  double q_norm = Compute2ndNorm(sz, j->q);
  j->q_norm = sqrt(1.f + q_norm);
}

void
ComputeAXPY(const size_t sz, const double a, const double *x, double *y)
{
  int i;
  for(i = 0; i < sz; i++)
  {
    /* AXPY */
    const double ax = a * x[i];
    const double axpy = ax + y[i];

    /* Update the vector component */
    y[i] = axpy;
  }
}
#if 0 /* Will be Enabled with CXX ( overloading ) */
void
ComputeAXPY(const size_t sz, const double a, const double *x, const double *y, double *yy)
{
  for(int i = 0; i < sz; i++)
  {
    /* AXPY */
    const double ax = a * x[i];
    const double axpy = ax + y[i];

    /* Update the vector component */
    yy[i] = axpy;
  }
}

void
ComputeAXPY(const size_t sz, const double a, const double s, const double *x, double *y)
{
  for(int i = 0; i < sz; i++)
  {
    /* AXPY */
    const double ax = a * x[i];
    const double axpy = ax + y[i];

    /* Scale */
    const double scale_axpy = axpy * s;

    /* Update the vector component */
    y[i] = scale_axpy;
  }
}
#endif

void
ComputeJacobianVectorProduct(Jacobian *hctx, const size_t sz, const double *r, const double *x, double *y, void *ctxx)
{
  double *ww = hctx->w;
  double *qq = hctx->q;

  double x_norm = Compute2ndNorm(sz, x);

  if(x_norm == 0.0)
  {
    memset(y, 0, sz * sizeof(double));
    return;
  }

  /*
    Compute differencing parameter
    The default matrix-free matrix-vector product routine computes
      r'(u)*x = [r(u+h*x) - r(u)]/h where
      h = PETSC_SQRT_MACHINE_EPSILON*u'x/||x||^2 if  |u'x| > umin*||x||_{1}
        = PETSC_SQRT_MACHINE_EPSILON*umin*sign(u'x)*||x||_{1}/||x||^2 else

  */
  const double epsilon = 1.490116119384766e-08;
  double h = epsilon * hctx->q_norm / x_norm;

  /*
    w = u + ha
  */
  int i;
  for(i = 0; i < sz; i++) ww[i] = qq[i] + h * x[i];

  ComputeResidual(ww, y, ctxx);

  /* AXPY and Scale */
  /* VecAXPY(y, -1.f, r); */
  /* VecScale(y, (1.f / h)); */
  for(i = 0; i < sz; i++)
  {
    /* AXPY */
    const double alpha = -1.f;
    const double ax = alpha * r[i];
    const double axpy = ax + y[i];

    /* Scale */
    const double scale = 1.f / h;
    const double scale_axpy = axpy * scale;

    /* Update the vector component */
    y[i] = scale_axpy;
  }
}

double
Normalize(const size_t sz, double *x)
{
  double norm = Compute2ndNorm(sz, x);
  
  if(norm == 0.0)
  {
    /*
    Vector of zero norm can not be normalized; Returning only the zero norm
    */
  }
  else if (norm != 1.0)
  {
    int i;
    for(i = 0; i < sz; i++) x[i] *= (1.f / norm);
  }

  return norm;
}

double
ComputeDotProduct(const size_t sz, const double *x, const double *y)
{
  double dot = 0.f;

  int i;
  for(i = 0; i < sz; i++) dot += x[i] * y[i];

  return dot;
}

/* PETSc KSPGMRESClassicalGramSchmidtOrthogonalization */
void
Orthogonalize(
  GMRES *gmres,
  int it,
  const size_t sz)
{
  int j;
  double *hh,*hes,*lhh;
  double hnrm, wnrm;

  lhh = gmres->orthogwork;

  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for (j=0; j<=it; j++)
   {
    hh[j]  = 0.0;
    hes[j] = 0.0;
  }

  /*
     This is really a matrix-vector product, with the matrix stored
     as pointer to rows
  */
  // VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh); /* <v,vnew> */

  double **y = &(VEC_VV(0));
  double *a1 = VEC_VV(it+1);

  int mm = 0;
  for(mm = 0; mm < (it+1); mm++)
  {
    double *a2 = y[mm];

    // VecDot(VEC_VV(it+1), a2, &lhh[mm]);
    // VecDot(a1, a2, &lhh[mm]);

    lhh[mm] = ComputeDotProduct(sz, a1, a2);
  }


  for (j=0; j<=it; j++)
  {
    lhh[j] = -lhh[j];
  }

  /*
    This is really a matrix vector product:
    [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it+1].
  */

  int v;
  for(v = 0; v < (it+1); v++)
  {
    double *xxx = y[v];
    // VecGetArray(y[v], &xxx);
    /* DO AXPY on each vector */
    ComputeAXPY(sz, lhh[v], xxx, a1);
  }
  
  // VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0));
  /* note lhh[j] is -<v,vnew> , hence the subtraction */

  for(j=0; j <= it; j++)
  {
    hh[j]  -= lhh[j];     /* hh += <v,vnew> */
    hes[j] -= lhh[j];     /* hes += <v,vnew> */
  }
}


unsigned int
ComputeGMRESCycle(
  int *kspitt,
  double *r,
  GMRES *gmres,
  void * ctx,
  Jacobian *jacobian,
  unsigned int *isConverged,
  double *ttol)
{
  struct ctx *restrict c = (struct ctx *) ctx;
  int kspits = *kspitt;

  double res_norm,res, tt;
  int       it     = 0, max_k = GMRES_RST_THRESHOLD;//gmres->max_k;
  double *kkk = VEC_VV(0);
  res = Normalize(c->g->c->sz, kkk);
	gmres->rs_origin[0] = res;

  if(kspits == 0) /* The first iteration */
    *ttol = GMRES_RTL_THRESHOLD * res;

  if(res <= *ttol) *isConverged = 1;

  while(!(*isConverged) && it < GMRES_RST_THRESHOLD && kspits < GMRES_ITS_THRESHOLD)
  {
    const double *xx = VEC_VV(it);
    ComputeJacobianVectorProduct(jacobian, c->g->c->sz, r, xx, VMOPT, ctx);

    double *x = VEC_VV(1+it);

    ComputeSparseTriangularSolve(
      c->g->n->sz,//a->mbs,
      c->g->c->bsz,//A->rmap->bs,
      c->g->c->bsz2,//a->bs2,
      c->ilu->ia,//a->i,
      c->ilu->ja,//a->j,
      c->ilu->aa,//a->a,
      c->ilu->diag,//a->diag,
      VMOPT,
      x);

    /* update hessenberg matrix and do Gram-Schmidt */
    Orthogonalize(gmres,it, c->g->c->sz);

    tt = Normalize(c->g->c->sz, x);
    /* save the magnitude */
    *HH(it+1,it)  = tt;
    *HES(it+1,it) = tt;

    UpdateHessenbergMatrix(it, &res, gmres->cc_origin, gmres->ss_origin, HH(0,it), gmres->rs_origin);

    it++;
    kspits++;

    if(res <= *ttol) *isConverged = 1;
  }

  *kspitt = kspits;

  return(it);
}


void
BuildGMRESSolution(double *nrs, const size_t sz, double *x, GMRES *gmres, int it)
{
  double tt;
  int ii,k,j;
  /* Solve for solution vector that minimizes the residual */

  if(*HH(it, it) != 0.f)
  {
    nrs[it] = *GRS(it) / *HH(it,it);
  }

  for(ii = 1; ii <= it; ii++)
  {
    k  = it - ii;
    tt = *GRS(k);

    for(j = k+1; j <= it; j++) tt = tt - *HH(k,j) * nrs[j];

    nrs[k] = tt / *HH(k,k); 
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */

  memset(VTEMP, 0, sz * sizeof(double));

  double **xx = &VEC_VV(0);

  /* VecMAXPY(VEC_TEMP,it+1,nrs,&VEC_VV(0)); */
  int v;
  for(v = 0; v < (it+1); v++)
  {
    double *xxx = xx[v];

    /* DO AXPY on each vector */
    // VecAXPY(VEC_TEMP, nrs[v], xx[v]);
    ComputeAXPY(sz, nrs[v], xxx, VTEMP);
  }

  /* AXPY */
  /* VecAXPY(vdest, 1.0, VEC_TEMP); */
  ComputeAXPY(sz, 1.f, VTEMP, x);
}

unsigned int
ComputeGMRES(GMRES *gmres, double *r, double *x, void *ctx, Jacobian *jacobian)
{
  struct ctx *c = (struct ctx *) ctx;

  ComputeNumericalILU(
    c->g->n->sz,
    c->g->c->bsz2,
    (int *) c->g->c->ia,
    (int *) c->g->c->ja,
    c->ilu->ia,
    c->ilu->ja,
    c->ilu->diag,
    c->g->c->aa,
    c->ilu->aa);

  memset(x, 0, c->g->c->sz * sizeof(double));

  int kspits = 0;
  int itcount = 0;
  int guess_zero = 1;
  unsigned int isConverged = 0;
  double ttol = 0.f;

  while(!isConverged)
  {
    /*
      Compute the initial guess
        C * (b - A * x)
        [left preconditioning preconditioned residual]
        A x = b
        if C is a preconditioner such that CA x = Cb
        the true residual b - Ax
    */
    if(!guess_zero)
    {
      ComputeJacobianVectorProduct(jacobian, c->g->c->sz, r, x, VTEMP, ctx);

      memcpy(VMOPT, r, c->g->c->sz * sizeof(double));

      ComputeAXPY(c->g->c->sz, -1.f, VTEMP, VMOPT);
  
      double *x = VEC_VV(0);

      ComputeSparseTriangularSolve(
        c->g->n->sz,
        c->g->c->bsz,
        c->g->c->bsz2,
        c->ilu->ia,
        c->ilu->ja,
        c->ilu->aa,
        c->ilu->diag,
        VMOPT,
        x);
    }
    else
    {
      double *x = VEC_VV(0);
      memcpy(VMOPT, r, c->g->c->sz * sizeof(double));

      ComputeSparseTriangularSolve(
        c->g->n->sz,
        c->g->c->bsz,
        c->g->c->bsz2,
        c->ilu->ia,
        c->ilu->ja,
        c->ilu->aa,
        c->ilu->diag,
        r,
        x);
    }

    unsigned int its =
    ComputeGMRESCycle(&kspits, r, gmres, ctx, jacobian, &isConverged, &ttol);

    // printf(ANSI_COLOR_RED "\tLinear iterations in a Cycle %d" ANSI_COLOR_RED "\n", its);

    /*
      Down here we have to solve for the "best" coefficients of the Krylov
      columns, add the solution values together, and possibly unwind the
      preconditioning from the solution
    */
    /* Form the solution (or the solution so far) */
    BuildGMRESSolution(GRS(0), c->g->c->sz, x, gmres, its-1);

    if(its == GMRES_RST_THRESHOLD)
    {
      /* Complete a GMRES cycle */
      // printf(ANSI_COLOR_GREEN "\tComplete A cycle -- %d Vectors" ANSI_COLOR_RESET "\n", GMRES_RST_THRESHOLD);
    }

    itcount += its;
    if(itcount >= GMRES_ITS_THRESHOLD)
    {
      /* Diverges due to exceeding the maximum number of iterations */
      break;
    }

    guess_zero = 0;
  }

  return(kspits);
}

void
ComputeNewton(GMRES *gmres, void * ctx, double *r, double *x, Jacobian *J)
{
  struct ctx *restrict c = (struct ctx *) ctx;

  ComputeResidual(c->q, r, ctx);

  double r_norm = Compute2ndNorm(c->g->c->sz, r);

  if(isinf(r_norm) || isnan(r_norm))
  {
    /* Divergence */
  }

  FillPreconditionerMatrix(ctx);

  BuildJacobianMatrix(c->g->c->sz, c->q, J);

  unsigned int its = ComputeGMRES(gmres, r, x, ctx, J);

  // printf(ANSI_COLOR_CYAN "\tLinear iterations in a nonlinear time step %d" ANSI_COLOR_RESET "\n", its);

  /* Compute a (scaled) negative update in the line search routine:
       q <- q - lambda(-1.f) * x
     and evaluate r = function(q) (depends on the line search).
  */

  /* update */

  /* AXPY */
  /* VecAXPY(q, lambda, x); */
  ComputeAXPY(c->g->c->sz, -1.f, x, c->q);

  double q_norm = Compute2ndNorm(c->g->c->sz, c->q);
  double x_norm = Compute2ndNorm(c->g->c->sz, x);

  r_norm = Compute2ndNorm(c->g->c->sz, r);

  if(isinf(r_norm) || isnan(r_norm))
  {
    /* Divergence */
  }
}


/*
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
  =======================================================================================================
*/

#if 0


/*
  Begin PETSc Involvement!
  Before, No PETSc At all
*/



#include <petsc/private/vecimpl.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>
#include <petsc/private/pcimpl.h>
#include <../src/mat/impls/mffd/mffdimpl.h>
#include <petsc/private/matimpl.h>
#include <../src/snes/impls/ls/lsimpl.h>
#include <petsc/private/linesearchimpl.h>
#include <petsc/private/snesimpl.h>
#include <petscdmshell.h>
#include <petscdraw.h>
#include <petscds.h>
#include <petscdmadaptor.h>
#include <petscconvest.h>
#include <petsc/private/kspimpl.h>
//#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>

#include <petscdmshell.h>
#include <petscmat.h>
#include <petsc/private/dmimpl.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#include <../src/mat/impls/baij/seq/baij.h>
#include <petscblaslapack.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petsc/private/kernels/blockmatmult.h>

//PETSC_EXTERN PetscErrorCode PetscKernel_A_gets_inverse_A_4(MatScalar*,PetscReal,PetscBool,PetscBool*);

#include <petsc/private/matimpl.h>
#include <../src/mat/impls/mffd/mffdimpl.h>
#include <../src/mat/utils/freespace.h>




//extern PetscErrorCode MatDuplicateNoCreate_SeqBAIJ(Mat, Mat, MatDuplicateOption, PetscBool);
//extern PetscErrorCode MatSeqBAIJSetNumericFactorization(Mat, PetscBool);



//int
//KSPSolve1(GMRES *, double *, double *, void *, Jacobian *);










#if 0
/*
   This routine allocates more work vectors, starting from VEC_VV(it).
 */
PetscErrorCode KSPGMRESGetNewVectors1(GMRES *gmres, int it, size_t sz)//Vec x)
{
//  KSP_GMRES      *gmres = (KSP_GMRES*)ksp->data;
  int nwork = gmres->nwork_alloc, k, nalloc;

  nalloc = 10;//PetscMin(ksp->max_it,gmres->delta_allocate);


  /* Adjust the number to allocate to make sure that we don't exceed the
    number of available slots */
  if (it + VEC_OFFSET + nalloc >= 34) {
    nalloc = 34/*gmres->vecs_allocated*/ - it - VEC_OFFSET;
  }
  if (!nalloc) return(0);

  gmres->vv_allocated += nalloc;

  //KSPCreateVecs1(ksp,nalloc,&gmres->user_work[nwork],0,NULL);


//   VecDuplicateVecs(x, 10, &gmres->user_work[nwork]);

  Vec **V = &gmres->user_work[nwork];

  //PetscMalloc1(10,V);
  *V = (Vec *) malloc(10 * sizeof(Vec));

  int i;
    for (i=0; i<10; i++) {

      VecCreateSeq(PETSC_COMM_SELF, sz, *V+i);


//VecDuplicate(x,*V+i);

}




//  int ierr = PetscLogObjectParents(ksp,nalloc,gmres->user_work[nwork]);

//  gmres->mwork_alloc[nwork] = nalloc;
  for (k=0; k<nalloc; k++) {
    gmres->vecs[it+VEC_OFFSET+k] = gmres->user_work[nwork][k];
  }
  gmres->nwork_alloc++;


  return(0);
}
#endif



#if 0
extern PetscErrorCode KSPCreate_GMRES(KSP);
//extern PetscErrorCode PCCreate_ILU(PC);

PetscErrorCode  SNESSetFromOptions1(KSP * iksp)//, Mat Pmat)
{
  Mat Pmat;
  PetscBool      flg,pcset,persist,set;
  PetscInt       i,indx,lag,grids;
  const char     *convtests[] = {"default","skip"};
  SNESKSPEW      *kctx        = NULL;
  char           type[256], monfilename[PETSC_MAX_PATH_LEN];
  PCSide         pcside;
  const char     *optionsprefix;

  SNESRegisterAll();

  KSP ksp;

  KSPInitializePackage();

  PetscHeaderCreate(ksp,KSP_CLASSID,"KSP","Krylov Method","KSP",MPI_COMM_SELF,KSPDestroy,KSPView);

  void *ctx;

  ksp->max_it  = 10000;
  ksp->pc_side = ksp->pc_side_set = PC_SIDE_DEFAULT;
  ksp->rtol    = 1.e-5;
  ksp->abstol  = 1.e-50;
  ksp->divtol  = 1.e4;

  ksp->chknorm        = -1;
  ksp->normtype       = ksp->normtype_set = KSP_NORM_DEFAULT;
  ksp->rnorm          = 0.0;
  ksp->its            = 0;
  ksp->guess_zero     = PETSC_TRUE;
  ksp->calc_sings     = PETSC_FALSE;
  ksp->res_hist       = NULL;
  ksp->res_hist_alloc = NULL;
  ksp->res_hist_len   = 0;
  ksp->res_hist_max   = 0;
  ksp->res_hist_reset = PETSC_TRUE;
  ksp->numbermonitors = 0;

  KSPConvergedDefaultCreate(&ctx);
  KSPSetConvergenceTest(ksp,KSPConvergedDefault,ctx,KSPConvergedDefaultDestroy);
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;

  ksp->vec_sol    = 0;
  ksp->vec_rhs    = 0;
  ksp->pc         = 0;
  ksp->data       = 0;
  ksp->nwork      = 0;
  ksp->work       = 0;
  ksp->reason     = KSP_CONVERGED_ITERATING;
  ksp->setupstage = KSP_SETUP_NEW;

  PetscMemzero(ksp->normsupporttable,sizeof(ksp->normsupporttable));
  ksp->pc_side  = ksp->pc_side_set;
  ksp->normtype = ksp->normtype_set;

  PCInitializePackage();

  PetscHeaderCreate(ksp->pc,PC_CLASSID,"PC","Preconditioner","PC",MPI_COMM_SELF,PCDestroy,PCView);

  ksp->pc->mat                  = 0;
  ksp->pc->pmat                 = 0;
  ksp->pc->setupcalled          = 0;
  ksp->pc->setfromoptionscalled = 0;
  ksp->pc->data                 = 0;
  ksp->pc->diagonalscale        = PETSC_FALSE;
  ksp->pc->diagonalscaleleft    = 0;
  ksp->pc->diagonalscaleright   = 0;

  ksp->pc->modifysubmatrices  = 0;
  ksp->pc->modifysubmatricesP = 0;

  PetscObjectIncrementTabLevel((PetscObject)ksp->pc,(PetscObject)ksp,0);
  PetscLogObjectParent((PetscObject)ksp,(PetscObject)ksp->pc);

  ksp->pc->matnonzerostate = -1;
  ksp->pc->matstate        = -1;

  ksp->reason = KSP_CONVERGED_ITERATING;

  ksp->normtype = KSP_NORM_PRECONDITIONED;
  ksp->pc_side = PC_LEFT;

  //PetscObjectReference((PetscObject)Amat);
  //PetscObjectReference((PetscObject)Pmat);

//  ksp->pc->mat  = Amat;
  //ksp->pc->pmat = Pmat;

  PCRegisterAll(); 

  PetscObjectOptionsBegin((PetscObject)ksp->pc);

  PetscFunctionListDestroy(&((PetscObject)ksp->pc)->qlist);
    
  PetscMemzero(ksp->pc->ops,sizeof(struct _PCOps));
  ksp->pc->modifysubmatrices  = 0;
  ksp->pc->modifysubmatricesP = 0;

  ksp->pc->setupcalled = 0;

  PetscObjectChangeTypeName((PetscObject)ksp->pc,PCILU);
  //PCCreate_ILU(ksp->pc);

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)ksp->pc);
  PetscOptionsEnd();

  ksp->pc->setfromoptionscalled++;
 
  KSPRegisterAll();
  PetscObjectOptionsBegin((PetscObject)ksp);

  /* Reinitialize function pointers in KSPOps structure */
  PetscMemzero(ksp->ops,sizeof(struct _KSPOps));
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;

  PetscMemzero(ksp->normsupporttable,sizeof(ksp->normsupporttable));
  ksp->pc_side  = ksp->pc_side_set;
  ksp->normtype = ksp->normtype_set;

  ksp->pc->erroriffailure = ksp->errorifnotconverged;

  /* Call the KSPCreate_XXX routine for this particular Krylov solver */
  ksp->setupstage = KSP_SETUP_NEW;
  PetscObjectChangeTypeName((PetscObject)ksp,KSPGMRES);
  KSPCreate_GMRES(ksp);

  ksp->rtol = 0.01; /*Relative decrease in residual norm*/

//  ksp->pc->mat  = Amat;
  //ksp->pc->pmat = Pmat;

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)ksp);
  PetscOptionsEnd();

  *iksp = ksp;

  return(0);
}
#endif

#endif
