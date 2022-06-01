
/* Fill the nonzero term of the A matrix */

#include <petscmat.h>
#include <petscvec.h>
#include <petscsnes.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include <omp.h>
#include "inc/ktime.h"
#include "inc/geometry.h"
#include "inc/ker/phy.h"
#include "inc/ker/kernel.h"

int
fill_mat(struct fill *restrict fill)
{
  struct ktime ktime;
  setktime(&ktime);

  const double *restrict q = fill->q;
  const struct geometry *restrict g = fill->g;
  const struct ivals *restrict iv = fill->iv;
  const struct ts *restrict ts = fill->ts;
  Mat A = fill->A;

  int ierr;

  uint32_t i;

  Mat_SeqBAIJ *restrict a = (Mat_SeqBAIJ *) A->data;

  size_t sz = (a->bs2 * a->i[a->mbs]);

  __assume_aligned(a->a, 64);
  memset(a->a, 0, sz * sizeof(double));

  size_t nnodes = g->n->sz;
  size_t bsz = g->c->bsz;

  double cfl = ts->cfl;

  double *restrict area = g->n->area;
  double *restrict cdt = ts->cdt;
  /*
    Loop over the nodes to compute the local indices of each row
    and column to insert the values using PETSc routine
  */
#pragma omp parallel for
  for(i = 0; i < nnodes; i++)
  {
    double tmp = area[i] / (cfl * cdt[i]);

    uint32_t j;
    for(j = 0; j < bsz; j++)
    {
      uint32_t idx = j + bsz * i;
      /*
        Inserts or adds values into certain locations of a matrix,
        using a local ordering of the nodes
      */
      MatSetValues(A, 1, (const int *) &idx, 1, (const int *) &idx,
                  (const double *) &tmp, ADD_VALUES);
    }
  }

  struct edge *restrict eptr = g->e->eptr;
  struct xyzn *restrict xyzn = g->e->xyzn;

  uint32_t *restrict ie = g->s->ie;
  uint32_t *restrict part = g->s->part;

#pragma omp parallel
  {
    uint32_t t = omp_get_thread_num();

    uint32_t ie0 = ie[t];
    uint32_t ie1 = ie[t+1];

    uint32_t i;
    for(i = ie0; i < ie1; i++)
    {
      uint32_t n0 = eptr->n0[i];
      uint32_t n1 = eptr->n1[i];

      double xn = xyzn->x0[i];
      double yn = xyzn->x1[i];
      double zn = xyzn->x2[i];
      double ln = xyzn->x3[i];

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
      double uL = q[bsz * n0 + 1];

      // Velocity v
      double vL = q[bsz * n0 + 2];

      // Velocity w
      double wL = q[bsz * n0 + 3];


      double ubarL = xn * uL;
      ubarL += yn * vL;
      ubarL += zn * wL;

      /* Variables on right */

      // Velocity u
      double uR = q[bsz * n1 + 1];

      // Velocity v
      double vR = q[bsz * n1 + 2];

      // Velocity w
      double wR = q[bsz * n1 + 3];

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

      double c2 = ubar * ubar + BETA;
      double c = sqrt(c2);

      /* Put in the eigenvalue smoothing stuff */

      double eig1  = ln * fabs(ubar);
      double eig2  = ln * fabs(ubar);
      double eig3  = ln * fabs(ubar + c);
      double eig4  = ln * fabs(ubar - c);

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

      /* Components of T(inverse) (call this y) */

      double c2inv = 1.f / c2;

      double y11 = u * phi4;
      y11 += v * phi5;
      y11 += w * phi6;
      y11 = -c2inv * y11 / BETA;

      double y21 = u * phi7;
      y21 += v * phi8;
      y21 += w * phi9;
      y21 = -c2inv * y21 / BETA;

      double y31 = c2inv * (c - ubar);
      y31 = 0.5f * y31 / BETA;

      double y41 = c2inv * (c + ubar);
      y41 = -0.5f * y41 / BETA;

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

      double t13 = c * BETA;

      double t23 = u * (ubar + c);
      t23 += xn * BETA;

      double t33 = v * (ubar + c);
      t33 += yn * BETA;

      double t43 = w * (ubar + c);
      t43 += zn * BETA;

      double t14 = -c * BETA;

      double t24 = u * (ubar - c);
      t24 += xn * BETA;

      double t34 = v * (ubar - c);
      t34 += yn * BETA;

      double t44 = w * (ubar - c);
      t44 += zn * BETA;

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

      double lb = ln * BETA;
      double lx = ln * xn;
      double ly = ln * yn;
      double lz = ln * zn;

      double val[4][8];

      val[0][0] = 0.5f * a11;
      val[0][1] = 0.5f * ((lb * xn) + a12);
      val[0][2] = 0.5f * ((lb * yn) + a13);
      val[0][3] = 0.5f * ((lb * zn) + a14);
                                                           
      val[1][0] = 0.5f * (lx + a21);
      val[1][1] = 0.5f * ((ln * (ubarL + xn * uL)) + a22);
      val[1][2] = 0.5f * ((ly * uL) + a23);
      val[1][3] = 0.5f * ((lz * uL) + a24);
                                                           
      val[2][0] = 0.5f * (ly + a31);
      val[2][1] = 0.5f * ((lx * vL) + a32);
      val[2][2] = 0.5f * ((ln * (ubarL + yn * vL)) + a33);
      val[2][3] = 0.5f * ((lz * vL) + a34);
                                                           
      val[3][0] = 0.5f * (lz + a41);
      val[3][1] = 0.5f * ((lx * wL) + a42);
      val[3][2] = 0.5f * ((ly * wL) + a43);
      val[3][3] = 0.5f * ((ln * (ubarL + zn * wL)) + a44);

      /* Regular Jaobians on right */

      val[0][4] = 0.5f * -a11;
      val[0][5] = 0.5f * ((lb * xn) - a12);
      val[0][6] = 0.5f * ((lb * yn) - a13);
      val[0][7] = 0.5f * ((lb * zn) - a14);
                                                           
      val[1][4] = 0.5f * (lx - a21);
      val[1][5] = 0.5f * ((ln * (ubarR + xn * uR)) - a22);
      val[1][6] = 0.5f * ((ly * uR) - a23);
      val[1][7] = 0.5f * ((lz * uR) - a24);
                                                           
      val[2][4] = 0.5f * (ly - a31);
      val[2][5] = 0.5f * ((lx * vR) - a32);
      val[2][6] = 0.5f * ((ln * (ubarR + yn * vR)) - a33);
      val[2][7] = 0.5f * ((lz * vR) - a34);
                                                           
      val[3][4] = 0.5f * (lz - a41);
      val[3][5] = 0.5f * ((lx * wR) - a42);
      val[3][6] = 0.5f * ((ly * wR) - a43);
      val[3][7] = 0.5f * ((ln * (ubarR + zn * wR)) - a44);

      uint32_t idxn[2];

      idxn[0] = n0;
      idxn[1] = n1;

      if(part[n0] == t)
      {
        MatSetValuesBlocked(A, 1, (const int *) &n0, 2, (const int *) idxn,
                            (const double *) val, ADD_VALUES);
      }
      if(part[n1] == t)
      {
        /*
          Exchange elements in place
        */
        uint32_t j;
        for(j = 0; j < bsz; j++)
        {
          uint32_t k;
          for(k = 0; k < 8; k++) val[j][k] = -val[j][k];
        }

        MatSetValuesBlocked(A, 1, (const int *) &n1, 2, (const int *) idxn,
                            (const double *) val, ADD_VALUES);
      }
    }
  }

  size_t nsnodes = g->b->s->n->sz;
  uint32_t *restrict nsptr = g->b->s->n->nptr;
  struct xyz *restrict s_xyz = g->b->s->n->xyz;

  /* Solid boundary points */
#pragma omp parallel for
  for(i = 0; i < nsnodes; i++)
  {
    uint32_t n = nsptr[i];

    double v[3];

    v[0] = s_xyz->x0[i];
    v[1] = s_xyz->x1[i];
    v[2] = s_xyz->x2[i];

    uint32_t idxm[3];
    idxm[0] = bsz * n + 1;
    idxm[1] = bsz * n + 2;
    idxm[2] = bsz * n + 3;

    uint32_t idxn = bsz * n;

    MatSetValues(A, 3, (const int *) idxm, 1, (const int *) &idxn,
                (const double *) v, ADD_VALUES);
  }

  size_t nfnodes = g->b->f->n->sz;

  uint32_t *restrict nfptr = g->b->f->n->nptr;

  struct xyz *restrict f_xyz = g->b->f->n->xyz;

  /* Free boundary points */
#pragma omp parallel for
  for(i = 0; i < nfnodes; i++)
  {
    uint32_t n = nfptr[i];

    double xn = f_xyz->x0[i];
    double yn = f_xyz->x1[i];
    double zn = f_xyz->x2[i];

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

    double ubar0 = xn * iv->u;
    ubar0 += yn * iv->v;
    ubar0 += zn * iv->w;

    double c20 = ubar0 * ubar0 + BETA;
    double c0 = sqrt(c20);

    double phi1 = xn * BETA;
    phi1 += iv->u * ubar0;

    double phi2 = yn * BETA;
    phi2 += iv->v * ubar0;

    double phi3 = zn * BETA;
    phi3 += iv->w * ubar0;

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

    double t13 = c0 * BETA;

    double t23 = iv->u * (ubar0 + c0);
    t23 += xn * BETA;

    double t33 = iv->v * (ubar0 + c0);
    t33 += yn * BETA;

    double t43 = iv->w * (ubar0 + c0);
    t43 += zn * BETA;

    double t14 = -c0 * BETA;

    double t24 = iv->u * (ubar0 - c0);
    t24 += xn * BETA;

    double t34 = iv->v * (ubar0 - c0);
    t34 += yn * BETA;

    double t44 = iv->w * (ubar0 - c0);
    t44 += zn * BETA;

    double ti11 = iv->u * phi4;
    ti11 += iv->v * phi5;
    ti11 += iv->w * phi6;
    ti11 = -ti11 / BETA / c20;

    double ti21 = iv->u * phi7;
    ti21 += iv->v * phi8;
    ti21 += iv->w * phi9;
    ti21 = -ti21 / BETA / c20;

    double ti31 = (c0 - ubar0) / (2.f * BETA * c20);

    double ti41 = -(c0 + ubar0) / (2.f * BETA * c20);

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
      pr = iv->p;
      prp = 0.f;
      ur = iv->u;
      uru = 0.f;
      vr = iv->v;
      vrv = 0.f;
      wr = iv->w;
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

    double rhs4 = ti41 * iv->p;
    rhs4 += ti42 * iv->u;
    rhs4 += ti43 * iv->v;
    rhs4 += ti44 * iv->w;

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

    v[0] = ln * BETA * unbp;
    v[1] = ln * BETA * unbu;
    v[2] = ln * BETA * unbv;
    v[3] = ln * BETA * unbw;

    v[4] = ln * (ub * unbp + unb * ubp + xn * pbp);
    v[5] = ln * (ub * unbu + unb * ubu + xn * pbu);
    v[6] = ln * (ub * unbv + unb * ubv + xn * pbv);
    v[7] = ln * (ub * unbw + unb * ubw + xn * pbw);

    v[8]  = ln * (vb * unbp + unb * vbp + yn * pbp);
    v[9]  = ln * (vb * unbu + unb * vbu + yn * pbu);
    v[10] = ln * (vb * unbv + unb * vbv + yn * pbv);
    v[11] = ln * (vb * unbw + unb * vbw + yn * pbw);

    v[12] = ln * (wb * unbp + unb * wbp + zn * pbp);
    v[13] = ln * (wb * unbu + unb * wbu + zn * pbu);
    v[14] = ln * (wb * unbv + unb * wbv + zn * pbv);
    v[15] = ln * (wb * unbw + unb * wbw + zn * pbw);

    /* 6 * 12 + 8 FLOPS */

    MatSetValuesBlocked(A, 1, (const int *) &n, 1, (const int *) &n,
                        (const double *) v, ADD_VALUES);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  compute_time(&ktime, &fill->t->fill);

  return 0;
}
