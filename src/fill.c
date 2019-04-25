#ifdef __cplusplus
extern "C" {
#endif
extern void CXX_Fill_alpha(const unsigned int row, const unsigned int col, const double a, const double v[], void *ptr);
extern void CXX_Fill(const unsigned int row, const unsigned int col, const double v[], void *ptr);
extern void CXX_Fill_boundary(const unsigned int row, const unsigned int col, const unsigned int n, const double v[], void *ptr);
extern void CXX_Fill_diagonal(const unsigned int row, const unsigned int col, const double v, void *ptr);
extern void CXX_Fill_Reset(void *ptr);
#ifdef __cplusplus
}
#endif

#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include "geometry.h"
#include "bench.h"
#include "phy.h"
#include "core_kernel.h"

//static void fill_alpha(
//  const unsigned int row,
//  const unsigned int col,
//  const double a,
//  const double v[],
//  const unsigned int bsz2,
//  const unsigned int *ia,
//  const unsigned int *ja,
//  double *aa)
//{
//  uint32_t i;
//  for(i = ia[row]; i < ia[row+1]; i++)
//  {
//    if(ja[i] == col)
//    {
//      uint32_t j;
//      for(j = 0; j < bsz2; j++) aa[bsz2 * i + j] += a * v[j];
//
//      break;
//    }
//  }
//}
//
//static void fill(
//  const unsigned int row,
//  const unsigned int col,
//  const double v[],
//  const unsigned int bsz2,
//  const unsigned int *ia,
//  const unsigned int *ja,
//  double *aa)
//{
//  uint32_t i;
//  for(i = ia[row]; i < ia[row+1]; i++)
//  {
//    if(ja[i] == col)
//    {
//      uint32_t j;
//      for(j = 0; j < bsz2; j++) aa[bsz2 * i + j] += v[j];
//
//      break;
//    }
//  }
//}
//
//static void fill_boundary(
//  const unsigned int row,
//  const unsigned int col,
//  const double v[],
//  const unsigned int n,
//  const unsigned int bsz2,
//  const unsigned int *ia,
//  const unsigned int *ja,
//  double *aa)
//{
//  uint32_t i;
//  for(i = ia[row]; i < ia[row+1]; i++)
//  {
//    if(ja[i] == col)
//    {
//      uint32_t j;
//      for(j = 1; j <= n; j++) aa[bsz2 * i + (j * n + j)] += v[j-1];
//
//      break;
//    }
//  }
//}
//
//static void fill_diagonal(
//  const unsigned int row,
//  const unsigned int col,
//  const double v,
//  const unsigned int bsz2,
//  const unsigned int *ia,
//  const unsigned int *ja,
//  double *aa)
//{
//  uint32_t i;
//  for(i = ia[row]; i < ia[row+1]; i++)
//  {
//    if(ja[i] == col)
//    {
//      aa[bsz2 * i + 0]  += v;
//      aa[bsz2 * i + 5]  += v;
//      aa[bsz2 * i + 10] += v;
//      aa[bsz2 * i + 15] += v;
//
//      break;
//    }
//  }
//}

static void
_KRN_ComputeA(
  const size_t nnodes,
  const size_t nsnodes,
  const size_t nfnodes,
  const uint32_t bsz,
  const uint32_t *nsptr,
  const uint32_t *nfptr,
  const double *s_xyz0,
  const double *s_xyz1,
  const double *s_xyz2,
  const double *f_xyz0,
  const double *f_xyz1,
  const double *f_xyz2,
  const uint32_t *ie,
  const uint32_t *part,
  const uint32_t *n0,
  const uint32_t *n1,
  const double *x0,
  const double *x1,
  const double *x2,
  const double *x3,
  const double *q,
  const double *cdt,
  void *matrix)//,
  //const size_t nnz,
  //const uint32_t bsz2,
  //const uint32_t *ia,
  //const uint32_t *ja,
  //double *aa)
{
  //memset(aa, 0, nnz * sizeof(double));
  
  CXX_Fill_Reset(matrix);

#pragma omp parallel
  {
    uint32_t i;

#pragma omp for
    for(i = 0; i < nnodes; i++)
    {
      // Store in the diagonal of the block
      //fill_diagonal(i, i, cdt[i], bsz2, ia, ja, aa);
      
      CXX_Fill_diagonal(i, i, cdt[i], matrix);

    }

#pragma omp barrier

    uint32_t t = (uint32_t) omp_get_thread_num();

    uint32_t ie0 = ie[t];
    uint32_t ie1 = ie[t+1];

    for(i = ie0; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const double xn = x0[i];
      const double yn = x1[i];
      const double zn = x2[i];
      const double ln = x3[i];

      /*
        Now lets get our other 2 vectors
        For first vector, use {1,0,0} and subtract off the component
        in the direction of the face normal. If the inner product of
        {1,0,0} is close to unity, use {0,1,0}
      */

      double dot = xn;
      double X1, Y1, Z1;

      if(fabs(dot) < 0.95f)
      {
        X1 = 1.f - dot * xn;
        Y1 = - dot * yn;
        Z1 = - dot * zn;
      }
      else
      {
        dot = yn;
        X1  = - dot * xn;
        Y1  = 1.f - dot * yn;
        Z1  = - dot * zn;
      }

      /* Normalize the first vector */

      double size = X1 * X1;
      size += Y1 * Y1;
      size += Z1 * Z1;
      size = sqrt(size);

      X1 /= size;
      Y1 /= size;
      Z1 /= size;

      /* Take cross-product of normal and V1 to get V2 */

      double X2 = yn * Z1;
      X2 -= zn * Y1;

      double Y2 = zn * X1;
      Y2 -= xn * Z1;

      double Z2 = xn * Y1;
      Z2 -= yn * X1;

      /* Variables on left */

      // Velocity u
      double uL = q[bsz * node0 + 1];

      // Velocity v
      double vL = q[bsz * node0 + 2];

      // Velocity w
      double wL = q[bsz * node0 + 3];


      double ubarL = xn * uL;
      ubarL += yn * vL;
      ubarL += zn * wL;

      /* Variables on right */

      // Velocity u
      double uR = q[bsz * node1 + 1];

      // Velocity v
      double vR = q[bsz * node1 + 2];

      // Velocity w
      double wR = q[bsz * node1 + 3];

      double ubarR = xn * uR;
      ubarR += yn * vR;
      ubarR += zn * wR;

      /*
        Now compute eigenvalues and |A| from averaged variables
        Avergage variables
      */

      double u = 0.5f * (uL + uR);
      double v = 0.5f * (vL + vR);
      double w = 0.5f * (wL + wR);

      double ubar = xn * u;
      ubar += yn * v;
      ubar += zn * w;

      double c2 = ubar * ubar + B;
      double c = sqrt(c2);

      /* Put in the eigenvalue smoothing stuff */

      double eig1  = ln * fabs(ubar);
      double eig2  = ln * fabs(ubar);
      double eig3  = ln * fabs(ubar + c);
      double eig4  = ln * fabs(ubar - c);

      double phi1 = xn * B;
      phi1 += u * ubar;

      double phi2 = yn * B;
      phi2 += v * ubar;

      double phi3 = zn * B;
      phi3 += w * ubar;

      double phi4 = Y2 * phi3;
      phi4 -= Z2 * phi2;

      double phi5 = Z2 * phi1;
      phi5 -= X2 * phi3;

      double phi6 = X2 * phi2;
      phi6 -= Y2 * phi1;

      double phi7 = Z1 * phi2;
      phi7 -= Y1 * phi3;

      double phi8 = X1 * phi3;
      phi8 -= Z1 * phi1;

      double phi9 = Y1 * phi1;
      phi9 -= X1 * phi2;

      /* Components of T(inverse) (call this y) */

      double c2inv = 1.f / c2;

      double y11 = u * phi4;
      y11 += v * phi5;
      y11 += w * phi6;
      y11 = -c2inv * y11 / B;

      double y21 = u * phi7;
      y21 += v * phi8;
      y21 += w * phi9;
      y21 = -c2inv * y21 / B;

      double y31 = c2inv * (c - ubar);
      y31 = 0.5f * y31 / B;

      double y41 = c2inv * (c + ubar);
      y41 = -0.5f * y41 / B;

      double y12 = c2inv * phi4;
      double y22 = c2inv * phi7;
      double y32 = c2inv * 0.5f * xn;
      double y42 = c2inv * 0.5f * xn;

      double y13 = c2inv * phi5;
      double y23 = c2inv * phi8;
      double y33 = c2inv * 0.5f * yn;
      double y43 = c2inv * 0.5f * yn;

      double y14 = c2inv * phi6;
      double y24 = c2inv * phi9;
      double y34 = c2inv * 0.5f * zn;
      double y44 = c2inv * 0.5f * zn;

      /* Now get elements of T */

      double t13 = c * B;

      double t23 = u * (ubar + c);
      t23 += xn * B;

      double t33 = v * (ubar + c);
      t33 += yn * B;

      double t43 = w * (ubar + c);
      t43 += zn * B;

      double t14 = -c * B;

      double t24 = u * (ubar - c);
      t24 += xn * B;

      double t34 = v * (ubar - c);
      t34 += yn * B;

      double t44 = w * (ubar - c);
      t44 += zn * B;

      /* Compute T * |lambda| * T(inv) */

      double a11 = eig3 * t13 * y31;
      a11 += eig4 * t14 * y41;

      double a12 = eig3 * t13 * y32;
      a12 += eig4 * t14 * y42;

      double a13 = eig3 * t13 * y33;
      a13 += eig4 * t14 * y43;

      double a14 = eig3 * t13 * y34;
      a14 += eig4 * t14 * y44;

      double a21 = eig1 * X1 * y11;
      a21 += eig2 * X2 * y21;
      a21 += eig3 * t23 * y31;
      a21 += eig4 * t24 * y41;

      double a22 = eig1 * X1 * y12;
      a22 += eig2 * X2 * y22;
      a22 += eig3 * t23 * y32;
      a22 += eig4 * t24 * y42;

      double a23 = eig1 * X1 * y13;
      a23 += eig2 * X2 * y23;
      a23 += eig3 * t23 * y33;
      a23 += eig4 * t24 * y43;

      double a24 = eig1 * X1 * y14;
      a24 += eig2 * X2 * y24;
      a24 += eig3 * t23 * y34;
      a24 += eig4 * t24 * y44;

      double a31 = eig1 * Y1 * y11;
      a31 += eig2 * Y2 * y21;
      a31 += eig3 * t33 * y31;
      a31 += eig4 * t34 * y41;

      double a32 = eig1 * Y1 * y12;
      a32 += eig2 * Y2 * y22;
      a32 += eig3 * t33 * y32;
      a32 += eig4 * t34 * y42;

      double a33 = eig1 * Y1 * y13;
      a33 += eig2 * Y2 * y23;
      a33 += eig3 * t33 * y33;
      a33 += eig4 * t34 * y43;

      double a34 = eig1 * Y1* y14;
      a34 += eig2 * Y2 * y24;
      a34 += eig3 * t33 * y34;
      a34 += eig4 * t34 * y44;

      double a41 = eig1 * Z1 * y11;
      a41 += eig2 * Z2 * y21;
      a41 += eig3 * t43 * y31;
      a41 += eig4 * t44 * y41;

      double a42 = eig1 * Z1 * y12;
      a42 += eig2 * Z2 * y22;
      a42 += eig3 * t43 * y32;
      a42 += eig4 * t44 * y42;

      double a43 = eig1 * Z1 * y13;
      a43 += eig2 * Z2 * y23;
      a43 += eig3 * t43 * y33;
      a43 += eig4 * t44 * y43;

      double a44 = eig1 * Z1 * y14;
      a44 += eig2 * Z2 * y24;
      a44 += eig3 * t43 * y34;
      a44 += eig4 * t44 * y44;

      /* Regular Jacobians on left: Form 0.5 * (A + |A|) */

      double lb = ln * B;
      double lx = ln * xn;
      double ly = ln * yn;
      double lz = ln * zn;

      /* Regular Jaobians on left */

      double v0[16];

      v0[0] = 0.5f * a11;
      v0[4] = 0.5f * (lx + a21);
      v0[8] = 0.5f * (ly + a31);
      v0[12] = 0.5f * (lz + a41);

      v0[1] = 0.5f * ((lb * xn) + a12);
      v0[5] = 0.5f * ((ln * (ubarL + xn * uL)) + a22);
      v0[9] = 0.5f * ((lx * vL) + a32);
      v0[13] = 0.5f * ((lx * wL) + a42);

      v0[2] = 0.5f * ((lb * yn) + a13);
      v0[6] = 0.5f * ((ly * uL) + a23);
      v0[10] = 0.5f * ((ln * (ubarL + yn * vL)) + a33);
      v0[14] = 0.5f * ((ly * wL) + a43);

      v0[3] = 0.5f * ((lb * zn) + a14);
      v0[7] = 0.5f * ((lz * uL) + a24);
      v0[11] = 0.5f * ((lz * vL) + a34);
      v0[15] = 0.5f * ((ln * (ubarL + zn * wL)) + a44);

      /* Regular Jaobians on right */

      double v1[16];

      v1[0] = 0.5f * -a11;
      v1[4] = 0.5f * (lx - a21);
      v1[8] = 0.5f * (ly - a31);
      v1[12] = 0.5f * (lz - a41);

      v1[1] = 0.5f * ((lb * xn) - a12);
      v1[5] = 0.5f * ((ln * (ubarR + xn * uR)) - a22);
      v1[9] = 0.5f * ((lx * vR) - a32);
      v1[13] = 0.5f * ((lx * wR) - a42);

      v1[2]   = 0.5f * ((lb * yn) - a13);
      v1[6]   = 0.5f * ((ly * uR) - a23);
      v1[10]  = 0.5f * ((ln * (ubarR + yn * vR)) - a33);
      v1[14]  = 0.5f * ((ly * wR) - a43);

      v1[3] = 0.5f * ((lb * zn) - a14);
      v1[7] = 0.5f * ((lz * uR) - a24);
      v1[11] = 0.5f * ((lz * vR) - a34);
      v1[15] = 0.5f * ((ln * (ubarR + zn * wR)) - a44);

      if(part[node0] == t)
      {
        //fill(node0, node0, v0, bsz2, ia, ja, aa);
        //fill(node0, node1, v1, bsz2, ia, ja, aa);

        CXX_Fill(node0, node0, v0, matrix);
        CXX_Fill(node0, node1, v1, matrix);
      }
      if(part[node1] == t)
      {
        //fill_alpha(node1, node0, -1.f, v0, bsz2, ia, ja, aa);
        //fill_alpha(node1, node1, -1.f, v1, bsz2, ia, ja, aa);

        CXX_Fill_alpha(node1, node0, -1.f, v0, matrix);
        CXX_Fill_alpha(node1, node1, -1.f, v1, matrix);
      }
    }

#pragma omp barrier

#pragma omp for
    for(i = 0; i < nsnodes; i++)
    {
      const double v[] = {s_xyz0[i], s_xyz1[i], s_xyz2[i]};
      //fill_boundary(nsptr[i], nsptr[i], v, 3, bsz2, ia, ja, aa);

      CXX_Fill_boundary(nsptr[i], nsptr[i], 3, v, matrix);
    }

#pragma omp barrier

#pragma omp for
    for(i = 0; i < nfnodes; i++)
    {
      uint32_t n = nfptr[i];

      double xn = f_xyz0[i];
      double yn = f_xyz1[i];
      double zn = f_xyz2[i];

      double ln = sqrt(xn * xn + yn * yn + zn * zn);

      xn /= ln;
      yn /= ln;
      zn /= ln;

      /* 9 FLOPS */

      /*
        Now lets get our other 2 vectors
        For first vector, use {1,0,0} and subtract off the component
        in the direction of the face normal. If the inner product of
        {1,0,0} is close to unity, use {0,1,0}
      */

      double dot = xn;
      double X1, Y1, Z1;
      if(fabs(dot) < 0.95f)
      {
        X1 = 1.f - dot * xn;
        Y1 = - dot * yn;
        Z1 = - dot * zn;
      }
      else
      {
        dot = yn;
        X1 = - dot * xn;
        Y1 = 1.f - dot * yn;
        Z1 = - dot * zn;
      }

      /* 6 FLOPS */

      /* Normalize the first vector (V1) */

      double size = sqrt(X1 * X1 + Y1 * Y1 + Z1 * Z1);
      X1 /= size;
      Y1 /= size;
      Z1 /= size;

      /* 9 FLOPS */

      /* Take cross-product of normal with V1 to get V2 */

      double X2 = yn * Z1 - zn * Y1;
      double Y2 = zn * X1 - xn * Z1;
      double Z2 = xn * Y1 - yn * X1;

      /* 9 FLOPS */

      /* Calculate elements of T and T(inverse)
          evaluated at freestream */

      double ubar0 = xn * U;
      ubar0 += yn * V;
      ubar0 += zn * W;

      double c20 = ubar0 * ubar0 + B;
      double c0 = sqrt(c20);

      double phi1 = xn * B;
      phi1 += U * ubar0;

      double phi2 = yn * B;
      phi2 += V * ubar0;

      double phi3 = zn * B;
      phi3 += W * ubar0;

      double phi4 = Y2 * phi3;
      phi4 -= Z2 * phi2;

      double phi5 = Z2 * phi1;
      phi5 -= X2 * phi3;

      double phi6 = X2 * phi2;
      phi6 -= Y2 * phi1;

      double phi7 = Z1 * phi2;
      phi7 -= Y1 * phi3;

      double phi8 = X1 * phi3;
      phi8 -= Z1 * phi1;

      double phi9 = Y1 * phi1;
      phi9 -= X1 * phi2;

      /* 9 * 3 + 8 FLOPS */

      double t13 = c0 * B;

      double t23 = U * (ubar0 + c0);
      t23 += xn * B;

      double t33 = V * (ubar0 + c0);
      t33 += yn * B;

      double t43 = W * (ubar0 + c0);
      t43 += zn * B;

      double t14 = -c0 * B;

      double t24 = U * (ubar0 - c0);
      t24 += xn * B;

      double t34 = V * (ubar0 - c0);
      t34 += yn * B;

      double t44 = W * (ubar0 - c0);
      t44 += zn * B;

      double ti11 = U * phi4;
      ti11 += V * phi5;
      ti11 += W * phi6;
      ti11 = -ti11 / B / c20;

      double ti21 = U * phi7;
      ti21 += V * phi8;
      ti21 += W * phi9;
      ti21 = -ti21 / B / c20;

      double ti31 = (c0 - ubar0) / (2.f * B * c20);

      double ti41 = -(c0 + ubar0) / (2.f * B * c20);

      double ti12 = phi4 / c20;
      double ti22 = phi7 / c20;
      double ti32 = 0.5f * xn / c20;
      double ti42 = 0.5f * xn / c20;

      double ti13 = phi5 / c20;
      double ti23 = phi8 / c20;
      double ti33 = 0.5f * yn / c20;
      double ti43 = 0.5f * yn / c20;

      double ti14 = phi6 / c20;
      double ti24 = phi9 / c20;
      double ti34 = 0.5f * zn / c20;
      double ti44 = 0.5f * zn / c20;

      /* 27 + 16 + 9 + 6 + 6 + 6 FLOPS */

      /* Now, get the variables on the "inside" */

      double pi = q[bsz * n + 0];
      double ui = q[bsz * n + 1];
      double vi = q[bsz * n + 2];
      double wi = q[bsz * n + 3];

      double un = xn * ui;
      un += yn * vi;
      un += zn * wi;

      /* 5 FLOPS */

      /* If ubar is negative, take the reference
        condition from outside */

      double pr, prp, ur, uru, vr, vrv, wr, wrw;

      if(un > 0.f)
      {
        pr = pi;
        prp = 1.f;
        ur = ui;
        uru = 1.f;
        vr = vi;
        vrv = 1.f;
        wr = wi;
        wrw = 1.f;
      }
      else
      {
        pr = P;
        prp = 0.f;
        ur = U;
        uru = 0.f;
        vr = V;
        vrv = 0.f;
        wr = W;
        wrw = 0.f;
      }

      /* Set rhs */

      double rhs1 = ti11 * pr;
      rhs1 += ti12 * ur;
      rhs1 += ti13 * vr;
      rhs1 += ti14 * wr;

      double rhs1p = ti11 * prp;
      double rhs1u = ti12 * uru;
      double rhs1v = ti13 * vrv;
      double rhs1w = ti14 * wrw;

      double rhs2 = ti21 * pr;
      rhs2 += ti22 * ur;
      rhs2 += ti23 * vr;
      rhs2 += ti24 * wr;

      double rhs2p = ti21 * prp;
      double rhs2u = ti22 * uru;
      double rhs2v = ti23 * vrv;
      double rhs2w = ti24 * wrw;

      double rhs3 = ti31 * pi;
      rhs3 += ti32 * ui;
      rhs3 += ti33 * vi;
      rhs3 += ti34 * wi;

      double rhs4 = ti41 * P;
      rhs4 += ti42 * U;
      rhs4 += ti43 * V;
      rhs4 += ti44 * W;

      /* 12 + 24 FLOPS */

      /* Now do matrix multiplication to get values on boundary */

      double pb = t13 * rhs3;
      pb += t14 * rhs4;

      double pbp = t13 * ti31;
      double pbu = t13 * ti32;
      double pbv = t13 * ti33;
      double pbw = t13 * ti34;

      double ub = X1 * rhs1;
      ub += X2 * rhs2;
      ub += t23 * rhs3;
      ub += t24 * rhs4;

      double ubp = X1 * rhs1p;
      ubp += X2 * rhs2p;
      ubp += t23 * ti31;

      double ubu = X1 * rhs1u;
      ubu += X2 * rhs2u;
      ubu += t23 * ti32;

      double ubv = X1 * rhs1v;
      ubv += X2 * rhs2v;
      ubv += t23 * ti33;

      double ubw = X1 * rhs1w;
      ubw += X2 * rhs2w;
      ubw += t23 * ti34;

      double vb = Y1 * rhs1;
      vb += Y2 * rhs2;
      vb += t33 * rhs3;
      vb += t34 * rhs4;

      double vbp = Y1 * rhs1p;
      vbp += Y2 * rhs2p;
      vbp += t33 * ti31;

      double vbu = Y1 * rhs1u;
      vbu += Y2 * rhs2u;
      vbu += t33 * ti32;

      double vbv = Y1 * rhs1v;
      vbv += Y2 * rhs2v;
      vbv += t33 * ti33;

      double vbw = Y1 * rhs1w;
      vbw += Y2 * rhs2w;
      vbw += t33 * ti34;

      double wb = Z1 * rhs1;
      wb += Z2 * rhs2;
      wb += t43 * rhs3;
      wb += t44 * rhs4;

      double wbp = Z1 * rhs1p;
      wbp += Z2 * rhs2p;
      wbp += t43 * ti31;

      double wbu = Z1 * rhs1u;
      wbu += Z2 * rhs2u;
      wbu += t43 * ti32;

      double wbv = Z1 * rhs1v;
      wbv += Z2 * rhs2v;
      wbv += t43 * ti33;

      double wbw = Z1 * rhs1w;
      wbw += Z2 * rhs2w;
      wbw += t43 * ti34;

      /* 5 * 15 + 6 + 5 + 2 FLOPS */

      double unb = xn * ub;
      unb += yn * vb;
      unb += zn * wb;

      double unbp = xn * ubp;
      unbp += yn * vbp;
      unbp += zn * wbp;

      double unbu = xn * ubu;
      unbu += yn * vbu;
      unbu += zn * wbu;

      double unbv = xn * ubv;
      unbv += yn * vbv;
      unbv += zn * wbv;

      double unbw = xn * ubw;
      unbw += yn * vbw;
      unbw += zn * wbw;

      /* 5 * 5 FLOPS */

      /* Now add contribution to lhs */

      double v[16];

      v[0] = ln * B * unbp;
      v[4] = ln * (ub * unbp + unb * ubp + xn * pbp);
      v[8] = ln * (vb * unbp + unb * vbp + yn * pbp);
      v[12] = ln * (wb * unbp + unb * wbp + zn * pbp);

      v[1] = ln * B * unbu;
      v[5] = ln * (ub * unbu + unb * ubu + xn * pbu);
      v[9] = ln * (vb * unbu + unb * vbu + yn * pbu);
      v[13] = ln * (wb * unbu + unb * wbu + zn * pbu);

      v[2]  = ln * B * unbv;
      v[6]  = ln * (ub * unbv + unb * ubv + xn * pbv);
      v[10] = ln * (vb * unbv + unb * vbv + yn * pbv);
      v[14] = ln * (wb * unbv + unb * wbv + zn * pbv);

      v[3] = ln * B * unbw;
      v[7] = ln * (ub * unbw + unb * ubw + xn * pbw);
      v[11] = ln * (vb * unbw + unb * vbw + yn * pbw);
      v[15] = ln * (wb * unbw + unb * wbw + zn * pbw);

      //fill(n, n, v, bsz2, ia, ja, aa);

      CXX_Fill(n, n, v, matrix);
    }
  }
}

void
ComputeA(GEOMETRY *g)
{
  BENCH start_bench = rdbench();

  _KRN_ComputeA(
    g->n->sz,
    g->b->s->sz,
    g->b->f->sz,
    g->c->b,
    g->b->s->nptr,
    g->b->f->nptr,
    g->b->s->xyz->x0,
    g->b->s->xyz->x1,
    g->b->s->xyz->x2,
    g->b->f->xyz->x0,
    g->b->f->xyz->x1,
    g->b->f->xyz->x2,
    g->s->i,
    g->n->part,
    g->e->eptr->n0,
    g->e->eptr->n1,
    g->e->xyzn->x0,
    g->e->xyzn->x1,
    g->e->xyzn->x2,
    g->e->xyzn->x3,
    g->q->q,
    g->n->cdt,
    g->matrix);//,
    //g->c->mat->i[g->n->sz] * g->c->b2,
    //g->c->b2,
    //g->c->mat->i,
    //g->c->mat->j,
    //g->c->mat->a);

  fun3d_log(start_bench, KERNEL_FLUX);
}