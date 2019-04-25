#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include "geometry.h"
#include "bench.h"
#include "phy.h"
#include "core_kernel.h"

#define MAG0 (0.5 / 3)
#define MAG1 (-MAG0)

static void
 _KRN_ComputeFlux(
  const size_t nfnodes,
  const uint32_t bsz,
  const uint32_t *nfptr,
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
  const size_t dofs,
  const size_t snfc,
  const uint32_t *snfic,
  const double *xyz0,
  const double *xyz1,
  const double *xyz2,
  const uint32_t *sn0,
  const uint32_t *sn1,
  const uint32_t *sn2,
  const double *w0termsx,
  const double *w0termsy,
  const double *w0termsz,
  const double *w1termsx,
  const double *w1termsy,
  const double *w1termsz,
  double *gradx0,
  double *gradx1,
  double *gradx2,
  double *r)
{
  memset(gradx0, 0, dofs * sizeof(double));
  memset(gradx1, 0, dofs * sizeof(double));
  memset(gradx2, 0, dofs * sizeof(double));
  memset(r, 0, dofs * sizeof(double));
 /*
    Calculates the gradients at the nodes using weighted least squares
    This solves using Gram-Schmidt
  */

#pragma omp parallel
  {
    const uint32_t t = (unsigned int) omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    uint32_t i;

    for(i = ie0; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const uint32_t idx0 = (unsigned int) bsz * node0;
      const uint32_t idx1 = (unsigned int) bsz * node1;

      double dq;

      double termx;
      double termy;
      double termz;

      if(part[node0] == t)
      {
        termx = w0termsx[i];
        termy = w0termsy[i];
        termz = w0termsz[i];
                
        dq = q[idx1 + 0] - q[idx0 + 0];

        gradx0[idx0 + 0] += termx * dq;
        gradx1[idx0 + 0] += termy * dq;
        gradx2[idx0 + 0] += termz * dq;

        dq = q[idx1 + 1] - q[idx0 + 1];

        gradx0[idx0 + 1] += termx * dq;
        gradx1[idx0 + 1] += termy * dq;
        gradx2[idx0 + 1] += termz * dq;

        dq = q[idx1 + 2] - q[idx0 + 2];

        gradx0[idx0 + 2] += termx * dq;
        gradx1[idx0 + 2] += termy * dq;
        gradx2[idx0 + 2] += termz * dq;

        dq = q[idx1 + 3] - q[idx0 + 3];

        gradx0[idx0 + 3] += termx * dq;
        gradx1[idx0 + 3] += termy * dq;
        gradx2[idx0 + 3] += termz * dq; 
      }

      if(part[node1] == t)
      {
        termx = w1termsx[i];
        termy = w1termsy[i];
        termz = w1termsz[i];

        dq = q[idx0 + 0] - q[idx1 + 0];

        gradx0[idx1 + 0] += termx * dq;
        gradx1[idx1 + 0] += termy * dq;
        gradx2[idx1 + 0] += termz * dq;

        dq = q[idx0 + 1] - q[idx1 + 1];

        gradx0[idx1 + 1] += termx * dq;
        gradx1[idx1 + 1] += termy * dq;
        gradx2[idx1 + 1] += termz * dq;

        dq = q[idx0 + 2] - q[idx1 + 2];

        gradx0[idx1 + 2] += termx * dq;
        gradx1[idx1 + 2] += termy * dq;
        gradx2[idx1 + 2] += termz * dq;

        dq = q[idx0 + 3] - q[idx1 + 3];

        gradx0[idx1 + 3] += termx * dq;
        gradx1[idx1 + 3] += termy * dq;
        gradx2[idx1 + 3] += termz * dq;
      } 
    }
  }

/*
  Calculates the fluxes on the face and performs the flux balance
*/

#pragma omp parallel
  {
    uint32_t t = (unsigned int) omp_get_thread_num();

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

      uint32_t idx0 = (unsigned int) bsz * node0;
      uint32_t idx1 = (unsigned int) bsz * node1;

      // P

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
      // P

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

      double c2 = ubar * ubar + B;
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
      ti11 = -ti11 / B;

      double ti21 = u * phi7;
      ti21 += v * phi8;
      ti21 += w * phi9;
      ti21 = -ti21 / B;

      double ti31 = 0.5f * (c - ubar);
      ti31 /= B;

      double ti41 = -0.5f * (c + ubar);
      ti41 /= B;

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

      double r13 = c * B;

      double r23 = u * (ubar + c);
      r23 += xn * B;

      double r33 = v * (ubar + c);
      r33 += yn * B;

      double r43 = w * (ubar + c);
      r43 += zn * B;

      double r14 = -c * B;

      double r24 = u * (ubar - c);
      r24 += xn * B;

      double r34 = v * (ubar - c);
      r34 += yn * B;

      double r44 = w * (ubar - c);
      r44 += zn * B;

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
      double fluxp1 = ln * B * ubarL;
      double fluxp2 = ln * (uL * ubarL + xn * pL);
      double fluxp3 = ln * (vL * ubarL + yn * pL);
      double fluxp4 = ln * (wL * ubarL + zn * pL);

      /*
        Now the right side
      */
      double fluxm1 = ln * B * ubarR;
      double fluxm2 = ln * (uR * ubarR + xn * pR);
      double fluxm3 = ln * (vR * ubarR + yn * pR);
      double fluxm4 = ln * (wR * ubarR + zn * pR);

      double res1 = 0.5f * (fluxp1 + fluxm1 - ln * t1);
      double res2 = 0.5f * (fluxp2 + fluxm2 - ln * t2);
      double res3 = 0.5f * (fluxp3 + fluxm3 - ln * t3);
      double res4 = 0.5f * (fluxp4 + fluxm4 - ln * t4);

      if(part[node0] == t)
      {
        r[idx0 + 0] = r[idx0 + 0] + res1;
        r[idx0 + 1] = r[idx0 + 1] + res2;
        r[idx0 + 2] = r[idx0 + 2] + res3;
        r[idx0 + 3] = r[idx0 + 3] + res4;
      }
      if(part[node1] == t)
      {
        r[idx1 + 0] = r[idx1 + 0] - res1;
        r[idx1 + 1] = r[idx1 + 1] - res2;
        r[idx1 + 2] = r[idx1 + 2] - res3;
        r[idx1 + 3] = r[idx1 + 3] - res4;
      }
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
    ti11 = -ti11/B;

    double ti21 = U * phi7;
    ti21 += V * phi8;
    ti21 += W * phi9;
    ti21 = -ti21/B;

    double ti31 = 0.5f * (c0 - ubar0);
    ti31 /= B;

    double ti41 = -0.5f * (c0 + ubar0);
    ti41 /= B;

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
      pr = P;
      ur = U;
      vr = V;
      wr = W;
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

    double rhs4 =  2.f * ti41 * P;
    rhs4 += xn * U;
    rhs4 += yn * V;
    rhs4 += zn * W;
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

    uint32_t idx = (unsigned int) bsz * n;

    r[idx + 0] += area * B * ubar;
    r[idx + 1] += area * (ub * ubar + xn * pb);
    r[idx + 2] += area * (vb * ubar + yn * pb);
    r[idx + 3] += area * (wb * ubar + zn * pb);
  }
}

void
ComputeFlux(const GEOMETRY *g, const double *q, GRADIENT *grad, double *r)
{
  BENCH start_bench = rdbench();

  _KRN_ComputeFlux(
    g->b->f->sz,
    g->c->b,
    g->b->f->nptr,
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
    q,
    g->c->sz,
    g->t->sz,
    g->t->i,
    g->n->xyz->x0,
    g->n->xyz->x1,
    g->n->xyz->x2,
    g->b->fc->fptr->n0,
    g->b->fc->fptr->n1,
    g->b->fc->fptr->n2,
    g->e->w->w0->x0,
    g->e->w->w0->x1,
    g->e->w->w0->x2,
    g->e->w->w1->x0,
    g->e->w->w1->x1,
    g->e->w->w1->x2,
    grad->x0,
    grad->x1,
    grad->x2,
    r
  );

  fun3d_log(start_bench, KERNEL_FLUX);
}
