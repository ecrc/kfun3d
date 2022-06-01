
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include "inc/ktime.h"
#include "inc/geometry.h"
#include "inc/ker/phy.h"

#define MAG0  (0.5 / 3)
#define MAG1  (-MAG0)

/*
  Calculates the residual
*/

void
compute_flux(struct flux *restrict flux)
{
  struct ktime ktime;
  setktime(&ktime);

  const size_t bsz = flux->bsz;
  const size_t nfnodes = flux->nfnodes;
  const size_t dofs = flux->dofs;
  const uint32_t snfc = flux->snfc;

  const double pressure = flux->pressure;
  const double velocity_u = flux->velocity_u;
  const double velocity_v = flux->velocity_v;
  const double velocity_w = flux->velocity_w;

  const double *restrict f_xyz0 = flux->f_xyz0;
  const double *restrict f_xyz1 = flux->f_xyz1;
  const double *restrict f_xyz2 = flux->f_xyz2;

  const double *restrict xyz0 = flux->xyz0;
  const double *restrict xyz1 = flux->xyz1;
  const double *restrict xyz2 = flux->xyz2;

  const uint32_t *restrict ie = flux->ie;
  const uint32_t *restrict part = flux->part;
  const uint32_t *restrict snfic = flux->snfic;
  const uint32_t *restrict n0 = flux->n0;
  const uint32_t *restrict n1 = flux->n1;
  const uint32_t *restrict nfptr = flux->nfptr;
  const uint32_t *restrict sn0 = flux->sn0;
  const uint32_t *restrict sn1 = flux->sn1;
  const uint32_t *restrict sn2 = flux->sn2;

  const double *restrict x0 = flux->x0;
  const double *restrict x1 = flux->x1;
  const double *restrict x2 = flux->x2;
  const double *restrict x3 = flux->x3;
  const double *restrict q = flux->q;

  const double *restrict gradx0 = flux->gradx0;
  const double *restrict gradx1 = flux->gradx1;
  const double *restrict gradx2 = flux->gradx2;

  double *restrict r = flux->r;

  memset(r, 0, dofs * sizeof(double));

  __assume_aligned(r, 64);

/*
  Calculates the fluxes on the face and performs the flux balance
*/

#pragma omp parallel
  {
    uint32_t t = omp_get_thread_num();

    uint32_t ie0 = ie[t];
    uint32_t ie1 = ie[t+1];

    uint32_t i;
    for(i = ie0; i < ie1; i++)
    {
      uint32_t node0 = n0[i];
      uint32_t node1 = n1[i];

      double xn = x0[i];
      double yn = x1[i];
      double zn = x2[i];
      double ln = x3[i];

      double xmean = 0.5f * (xyz0[node0] + xyz0[node1]);
      double ymean = 0.5f * (xyz1[node0] + xyz1[node1]);
      double zmean = 0.5f * (xyz2[node0] + xyz2[node1]);

      /*
        Now lets get our other 2 vectors
        For first vector, use {1,0,0} and subtract off the component
        in the direction of the face normal. If the inner product of
        {1,0,0} is close to unity, use {0,1,0}
      */
      double X1, Y1, Z1;
      double dot = xn;
      if(fabs(dot) < 0.95f)
      {
        X1 = 1.f - dot * xn;
        Y1 = -dot * yn;
        Z1 = -dot * zn;
      }
      else
      {
        dot = yn;
        X1 = -dot * xn;
        Y1 = 1.f - dot * yn;
        Z1 = -dot * zn;
      }

      /*
        Normalize the first vector
      */
      double size = X1 * X1;
      size += Y1 * Y1;
      size += Z1 * Z1;
      size = sqrt(size);

      X1 /= size;
      Y1 /= size;
      Z1 /= size;

      /*
        Take cross-product of normal and V1 to get V2
      */
      double X2 = yn * Z1;
      X2 -= zn * Y1;

      double Y2 = zn * X1;
      Y2 -= xn * Z1;

      double Z2 = xn * Y1;
      Z2 -= yn * X1;

      /*
        Get variables on "left" and "right" side of face
      */
      double rx = xmean - xyz0[node0];
      double ry = ymean - xyz1[node0];
      double rz = zmean - xyz2[node0];

      uint32_t idx0 = bsz * node0;
      uint32_t idx1 = bsz * node1;

      // Pressure

      double pL = q[idx0 + 0] + gradx0[idx0 + 0] * rx;
      pL += gradx1[idx0 + 0] * ry;
      pL += gradx2[idx0 + 0] * rz;

      // Velocity u

      double uL = q[idx0 + 1] + gradx0[idx0 + 1] * rx;
      uL += gradx1[idx0 + 1] * ry;
      uL += gradx2[idx0 + 1] * rz;

      // Velocity v

      double vL = q[idx0 + 2] + gradx0[idx0 + 2] * rx;
      vL += gradx1[idx0 + 2] * ry;
      vL += gradx2[idx0 + 2] * rz;

      // Velocity w

      double wL = q[idx0 + 3] + gradx0[idx0 + 3] * rx;
      wL += gradx1[idx0 + 3] * ry;
      wL += gradx2[idx0 + 3] * rz;

      double ubarL = xn * uL;
      ubarL += yn * vL;
      ubarL += zn * wL;

      rx = xmean - xyz0[node1];
      ry = ymean - xyz1[node1];
      rz = zmean - xyz2[node1];
      // Pressure

      double pR = q[idx1 + 0] + gradx0[idx1 + 0] * rx;
      pR += gradx1[idx1 + 0] * ry;
      pR += gradx2[idx1 + 0] * rz;

      // Velocity u

      double uR = q[idx1 + 1] + gradx0[idx1 + 1] * rx;
      uR += gradx1[idx1 + 1] * ry;
      uR += gradx2[idx1 + 1] * rz;

      // Velocity v

      double vR = q[idx1 + 2] + gradx0[idx1 + 2] * rx;
      vR += gradx1[idx1 + 2] * ry;
      vR += gradx2[idx1 + 2] * rz;

      // Velocity w

      double wR = q[idx1 + 3] + gradx0[idx1 + 3] * rx;
      wR += gradx1[idx1 + 3] * ry;
      wR += gradx2[idx1 + 3] * rz;

      double ubarR = xn * uR;
      ubarR += yn * vR;
      ubarR += zn * wR;

      /* Compute averages */

      //double p = 0.5f * (pL + pR);
      double u = 0.5f * (uL + uR);
      double v = 0.5f * (vL + vR);
      double w = 0.5f * (wL + wR);

      double ubar = xn * u;
      ubar += yn * v;
      ubar += zn * w;

      double phi1 = xn * BETA;
      phi1 += u * ubar;

      double phi2 = yn * BETA;
      phi2 += v * ubar;

      double phi3 = zn * BETA;
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

      double c2 = ubar * ubar + BETA;
      double c = sqrt(c2);

      /*
        Now compute eigenvalues, eigenvectors, and strengths
      */
      double eig1 = fabs(ubar);
      double eig2 = fabs(ubar);
      double eig3 = fabs(ubar + c);
      double eig4 = fabs(ubar - c);

      double dp = pR - pL;
      double du = uR - uL;
      double dv = vR - vL;
      double dw = wR - wL;

      /*
        Components of T(inverse)
      */
      double ti11 = u * phi4;
      ti11 += v * phi5;
      ti11 += w * phi6;
      ti11 = -ti11 / BETA;

      double ti21 = u * phi7;
      ti21 += v * phi8;
      ti21 += w * phi9;
      ti21 = -ti21 / BETA;

      double ti31 = 0.5f * (c - ubar);
      ti31 /= BETA;

      double ti41 = -0.5f * (c + ubar);
      ti41 /= BETA;

      /*
        jumps (T(inverse) * dq)
      */
      double dv1 = ti11 * dp;
      dv1 += phi4 * du;
      dv1 += phi5 * dv;
      dv1 += phi6 * dw;
      dv1 /= c2;

      double dv2 = ti21 * dp;
      dv2 += phi7 * du;
      dv2 += phi8 * dv;
      dv2 += phi9 * dw;
      dv2 /= c2;

      double dv3 = 2.f * ti31 * dp;
      dv3 += xn * du;
      dv3 += yn * dv;
      dv3 += zn * dw;
      dv3 *= 0.5f / c2;

      double dv4 = 2.f * ti41 * dp;
      dv4 += xn * du;
      dv4 += yn * dv;
      dv4 += zn * dw;
      dv4 *= 0.5f / c2;

      /*
        Now get elements of T
      */

      double r13 = c * BETA;

      double r23 = u * (ubar + c);
      r23 += xn * BETA;

      double r33 = v * (ubar + c);
      r33 += yn * BETA;

      double r43 = w * (ubar + c);
      r43 += zn * BETA;

      double r14 = -c * BETA;

      double r24 = u * (ubar - c);
      r24 += xn * BETA;

      double r34 = v * (ubar - c);
      r34 += yn * BETA;

      double r44 = w * (ubar - c);
      r44 += zn * BETA;

      /*
        Calculate T* |lambda| * T(inverse)
      */
      double t1 = eig3 * r13 * dv3 + eig4 * r14 * dv4;

      double t2 = eig1 * X1 * dv1 + eig2 * X2 * dv2;
      t2 += eig3 * r23 * dv3 + eig4 * r24 * dv4;

      double t3 = eig1 * Y1 * dv1 + eig2 * Y2 * dv2;
      t3 += eig3 * r33 * dv3 + eig4 * r34 * dv4;

      double t4 = eig1 * Z1 * dv1 + eig2 * Z2 * dv2;
      t4 += eig3 * r43 * dv3 + eig4 * r44 * dv4;

      /*
        Modify to calculate .5(fl +fr) from nodes
        instead of extrapolated ones
      */
      double fluxp1 = ln * BETA * ubarL;
      double fluxp2 = ln * (uL * ubarL + xn * pL);
      double fluxp3 = ln * (vL * ubarL + yn * pL);
      double fluxp4 = ln * (wL * ubarL + zn * pL);

      /*
        Now the right side
      */
      double fluxm1 = ln * BETA * ubarR;
      double fluxm2 = ln * (uR * ubarR + xn * pR);
      double fluxm3 = ln * (vR * ubarR + yn * pR);
      double fluxm4 = ln * (wR * ubarR + zn * pR);

      double res1 = 0.5f * (fluxp1 + fluxm1 - ln * t1);
      double res2 = 0.5f * (fluxp2 + fluxm2 - ln * t2);
      double res3 = 0.5f * (fluxp3 + fluxm3 - ln * t3);
      double res4 = 0.5f * (fluxp4 + fluxm4 - ln * t4);
      
      r[idx0 + 0] = (part[node0] == t) ? (r[idx0 + 0] + res1) : r[idx0 + 0];
      r[idx0 + 1] = (part[node0] == t) ? (r[idx0 + 1] + res2) : r[idx0 + 1];
      r[idx0 + 2] = (part[node0] == t) ? (r[idx0 + 2] + res3) : r[idx0 + 2];
      r[idx0 + 3] = (part[node0] == t) ? (r[idx0 + 3] + res4) : r[idx0 + 3];
      
      r[idx1 + 0] = (part[node1] == t) ? (r[idx1 + 0] - res1) : r[idx1 + 0];
      r[idx1 + 1] = (part[node1] == t) ? (r[idx1 + 1] - res2) : r[idx1 + 1];
      r[idx1 + 2] = (part[node1] == t) ? (r[idx1 + 2] - res3) : r[idx1 + 2];
      r[idx1 + 3] = (part[node1] == t) ? (r[idx1 + 3] - res4) : r[idx1 + 3];
    }
  }

  uint32_t i;
  for(i = 0; i < snfc; i++)
  {
    uint32_t if0 = snfic[i];
    uint32_t if1 = snfic[i+1];

    uint32_t j;
#pragma omp parallel for
    for(j = if0; j < if1; j++)
    {
      uint32_t node0 = sn0[j];
      uint32_t node1 = sn1[j];
      uint32_t node2 = sn2[j];

      double p1 = q[bsz * node0];
      double p2 = q[bsz * node1];
      double p3 = q[bsz * node2];
             
      double ax = xyz0[node1] - xyz0[node0];
      double ay = xyz1[node1] - xyz1[node0];
      double az = xyz2[node1] - xyz2[node0];

      double bx = xyz0[node2] - xyz0[node0];
      double by = xyz1[node2] - xyz1[node0];
      double bz = xyz2[node2] - xyz2[node0];

      /*
        Normal points away from grid interior.
        Magnitude is 1/3 area of surface triangle.
      */

      double xn = ay * bz;
      xn -= az * by;
      xn *= MAG1;

      double yn = ax * bz;
      yn -= az * bx;
      yn *= MAG0;

      double zn = ax * by;
      zn -= ay * bx;
      zn *= MAG1;

      double pa = 0.125f * (p2 + p3);
      pa += 0.75f * p1;

      double pb = 0.125f * (p3 + p1);
      pb += 0.75f * p2;

      double pc = 0.125f * (p1 + p2);
      pc += 0.75f * p3;

      uint32_t idx;

      idx = bsz * node0;

      r[idx + 1] += xn * pa;
      r[idx + 2] += yn * pa;
      r[idx + 3] += zn * pa;

      idx = bsz * node1;

      r[idx + 1] += xn * pb;
      r[idx + 2] += yn * pb;
      r[idx + 3] += zn * pb;

      idx = bsz * node2;

      r[idx + 1] += xn * pc;
      r[idx + 2] += yn * pc;
      r[idx + 3] += zn * pc;
    }
  }

  /* Do the free boundaries */
#pragma omp parallel for
  for(i = 0; i < nfnodes; i++)
  {
    uint32_t n = nfptr[i];

    /*
      Get normal and "other" 2 vectors. Remember that fxn,fyn and fzn
      has the magnitude of the face contained in it.
    */

    double xn = f_xyz0[i];
    double yn = f_xyz1[i];
    double zn = f_xyz2[i];

    double area = xn * xn;
    area += yn * yn;
    area += zn * zn;
    area = sqrt(area);

    xn /= area;
    yn /= area;
    zn /= area;

    /*
      Now lets get our other 2 vectors
      For first vector, use {1,0,0} and subtract off the component
      in the direction of the face normal. If the inner product of
      {1,0,0} is close to unity, use {0,1,0}
    */

    double X1, Y1, Z1;

    double dot = xn;
    if(fabs(dot) < 0.95f)
    {
      X1 = 1.f - dot * xn;
      Y1 = -dot * yn;
      Z1 = -dot * zn;
    }
    else
    {
      dot = yn;
      X1  = -dot * xn;
      Y1  = 1.f - dot * yn;
      Z1  = -dot * zn;
    }

    /*
      Normalize the first vector (V1)
    */

    double size = X1 * X1;
    size += Y1 * Y1;
    size += Z1 * Z1;
    size = sqrt(size);

    X1 /= size;
    Y1 /= size;
    Z1 /= size;

    /*
      Take cross-product of normal with V1 to get V2
    */
    double X2 = yn * Z1;
    X2 -= zn * Y1;

    double Y2 = zn * X1;
    Y2 -= xn * Z1;

    double Z2 = xn * Y1;
    Z2 -= yn * X1;

    /*
      Calculate elements of T and T(inverse) evaluated at free-stream
    */
    double ubar0 = xn * velocity_u;
    ubar0 += yn * velocity_v;
    ubar0 += zn * velocity_w;

    double c20 = ubar0 * ubar0 + BETA;
    double c0 = sqrt(c20);

    double phi1 = xn * BETA;
    phi1 += velocity_u * ubar0;

    double phi2 = yn * BETA;
    phi2 += velocity_v * ubar0;

    double phi3 = zn * BETA;
    phi3 += velocity_w * ubar0;

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

    double t13 = c0 * BETA;

    double t23 = velocity_u * (ubar0 + c0);
    t23 += xn * BETA;

    double t33 = velocity_v * (ubar0 + c0);
    t33 += yn * BETA;

    double t43 = velocity_w * (ubar0 + c0);
    t43 += zn * BETA;

    double t14 = -c0 * BETA;

    double t24 = velocity_u * (ubar0 - c0);
    t24 += xn * BETA;

    double t34 = velocity_v * (ubar0 - c0);
    t34 += yn * BETA;

    double t44 = velocity_w * (ubar0 - c0);
    t44 += zn * BETA;

    double ti11 = velocity_u * phi4;
    ti11 += velocity_v * phi5;
    ti11 += velocity_w * phi6;
    ti11 = -ti11/BETA;

    double ti21 = velocity_u * phi7;
    ti21 += velocity_v * phi8;
    ti21 += velocity_w * phi9;
    ti21 = -ti21/BETA;

    double ti31 = 0.5f * (c0 - ubar0);
    ti31 /= BETA;

    double ti41 = -0.5f * (c0 + ubar0);
    ti41 /= BETA;

    /*
      Now, get the variables on the "inside"
    */
    double pi = q[bsz * n + 0];
    double ui = q[bsz * n + 1];
    double vi = q[bsz * n + 2];
    double wi = q[bsz * n + 3];

    double un = xn * ui;
    un += yn * vi;
    un += zn * wi;
    /*
      If ubar is negative, take the reference condition from outside
    */
    double pr, ur, vr, wr;

    if(un > 0.f)
    {
      pr = pi;
      ur = ui;
      vr = vi;
      wr = wi;
    }
    else
    {
      pr = pressure;
      ur = velocity_u;
      vr = velocity_v;
      wr = velocity_w;
    }

    /*
      Set rhs
    */

    double rhs1 = ti11 * pr;
    rhs1 += phi4 * ur;
    rhs1 += phi5 * vr;
    rhs1 += phi6 * wr;
    rhs1 /= c20;

    double rhs2 = ti21 * pr;
    rhs2 += phi7 * ur;
    rhs2 += phi8 * vr;
    rhs2 += phi9 * wr;
    rhs2 /= c20;

    double rhs3 = 2.f * ti31 * pi;
    rhs3 += xn * ui;
    rhs3 += yn * vi;
    rhs3 += zn * wi;
    rhs3 = 0.5f * rhs3 / c20;

    double rhs4 =  2.f * ti41 * pressure;
    rhs4 += xn * velocity_u;
    rhs4 += yn * velocity_v;
    rhs4 += zn * velocity_w;
    rhs4 = 0.5f * rhs4 / c20;

    /*
      Now do matrix multiplication to get values on boundary
    */
    double pb = t13 * rhs3;
    pb += t14 * rhs4;

    double ub = X1 * rhs1;
    ub += X2 * rhs2;
    ub += t23 * rhs3;
    ub += t24 * rhs4;

    double vb = Y1 * rhs1;
    vb += Y2 * rhs2;
    vb += t33 * rhs3;
    vb += t34 * rhs4;

    double wb = Z1 * rhs1;
    wb += Z2 * rhs2;
    wb += t43 * rhs3;
    wb += t44 * rhs4;

    double ubar = xn * ub;
    ubar += yn * vb;
    ubar += zn * wb;

    uint32_t idx = bsz * n;

    r[idx + 0] += area * BETA * ubar;
    r[idx + 1] += area * (ub * ubar + xn * pb);
    r[idx + 2] += area * (vb * ubar + yn * pb);
    r[idx + 3] += area * (wb * ubar + zn * pb);
  }

  compute_time(&ktime, flux->t);
}
