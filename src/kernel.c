#include <omp.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "geometry.h"
#include "allocator.h"
#include "utils.h"
#include "core_kernel.h"

/* Jacobian-Vector Product */
#define MACHINE_EPSILON           1.490116119384766e-08 /* Machine epsilon */
/* GMRES Linear Solver */
#define GMRES_RELATIVE_TOLERANCE  0.01
#define GMRES_RESTART             30
#define GMRES_BASIS               31
#define GMRES_MAX_IT              10000
/* CFL Condition for Pseudo TimeStep */
#define CFL_INIT                  50
#define CLF_MAX                   100000
#define TS_MAX                    15
#define FNORM_MAX                 10000000

void
iJacVec(const GEOMETRY *g, const double *x, const double *r, const double norm, GRADIENT *grad, double *w, double *y)
{
  const double xnorm = Compute2ndNorm(g->c->sz, x);
  const double h = MACHINE_EPSILON * norm;

  ComputeNewAXPY(g->c->sz, (h/xnorm), x, g->q->q, w);
  ComputeFlux(g, w, grad, y);

  const double a = xnorm / h;

#pragma omp parallel for
  for(unsigned int i = 0; i < g->n->sz; i++)
  {
    const uint32_t idx =  g->c->b * i;

    y[idx + 0] = (y[idx + 0] - r[idx + 0] + g->n->cdt[i] * (w[idx + 0] - g->q->q[idx + 0])) * a;
    y[idx + 1] = (y[idx + 1] - r[idx + 1] + g->n->cdt[i] * (w[idx + 1] - g->q->q[idx + 1])) * a;
    y[idx + 2] = (y[idx + 2] - r[idx + 2] + g->n->cdt[i] * (w[idx + 2] - g->q->q[idx + 2])) * a;
    y[idx + 3] = (y[idx + 3] - r[idx + 3] + g->n->cdt[i] * (w[idx + 3] - g->q->q[idx + 3])) * a;
  }
}

void
Kernel(GEOMETRY *g)
{
  GRADIENT *grad = (GRADIENT *) fun3d_malloc(1, sizeof(GRADIENT));

  grad->x0 = (double *) fun3d_malloc(g->c->sz, sizeof(double));
  grad->x1 = (double *) fun3d_malloc(g->c->sz, sizeof(double));
  grad->x2 = (double *) fun3d_malloc(g->c->sz, sizeof(double));

  double *x = (double *) fun3d_malloc(g->c->sz, sizeof(double));
  double *r = (double *) fun3d_malloc(g->c->sz, sizeof(double));
  double *w = (double *) fun3d_malloc(g->c->sz, sizeof(double));
  double *t = (double *) fun3d_malloc(g->c->sz, sizeof(double));

  double *ghh = (double *) fun3d_calloc((size_t) (GMRES_BASIS * GMRES_BASIS), sizeof(double));

  double *vecs[GMRES_BASIS];
  unsigned int v;
  for(v = 0; v < GMRES_BASIS; v++) vecs[v] = (double *) fun3d_calloc(g->c->sz, sizeof(double));

  double forces[3];

  double cc[GMRES_BASIS];
  double ss[GMRES_BASIS];
  double rs[GMRES_BASIS];

  double fnorm_init = 0.f;
  double cfl = CFL_INIT;

  uint32_t iTS = 0;
  while(iTS < TS_MAX)
  {
    ComputeFlux(g, g->q->q, grad, r);

    double fnorm = Compute2ndNorm(g->c->sz, r);

    if(iTS == 0) fnorm_init = fnorm;
    else 
    {
      double cfl0 = 1.1 * CFL_INIT * fnorm_init;
      cfl0 /= fnorm;
      cfl = (cfl0 < CLF_MAX) ? cfl0 : CLF_MAX;
    }

    ComputeTimeStep(cfl, g);

    ComputeA(g);

    double qnorm = sqrt(1.f + Compute2ndNorm(g->c->sz, g->q->q));

    memset(x, 0, g->c->sz * sizeof(double));

    ComputeNumericalILU(g);
    ComputeSparseTriangularSolve(g, r, vecs[0]);

    double tol = 0.f;
    uint32_t iGMRES_total = 0;

    unsigned int flag = 0;

    while(iGMRES_total <= GMRES_MAX_IT)
    {
      if(iGMRES_total > 0)
      {
        iJacVec(g, x, r, qnorm, grad, w, t);
        ComputeNewAXPY(g->c->sz, -1.f, t, r, w);
        ComputeSparseTriangularSolve(g, w, vecs[0]);
      }

      rs[0] = Normalize(g->c->sz, vecs[0]);

      tol = (iGMRES_total == 0) ? GMRES_RELATIVE_TOLERANCE * rs[0] : tol;

      uint32_t iGMRES = 0;
      while(iGMRES < GMRES_RESTART)
      {
        iJacVec(g, vecs[iGMRES], r, qnorm, grad, w, t);
        ComputeSparseTriangularSolve(g, t, vecs[1+iGMRES]);

        /* update Hessenberg matrix and do unmodified Gram-Schmidt */
        double *hh  = (ghh + iGMRES * (GMRES_BASIS + 1));

        for(unsigned int v = 0; v < (iGMRES+1); v++)
        {
          double dot = 0.f;

#pragma omp parallel for reduction(+:dot)
          for(unsigned int i = 0; i < g->c->sz; i++) dot += vecs[iGMRES+1][i] * vecs[v][i];

          hh[v] = dot;

          ComputeAXPY(g->c->sz, -hh[v], vecs[v], vecs[iGMRES+1]);
        }

        hh[iGMRES+1] = Normalize(g->c->sz, vecs[1+iGMRES]);

        for(unsigned int i = 0; i < iGMRES; i++)
        {
          const double tt  = *hh;
          *hh = cc[i] * *hh + ss[i] * *(hh+1);
          hh++;
          *hh = cc[i] * *hh - ss[i] * tt;
        }

        const double tt = sqrt(*hh * *hh + *(hh+1) * *(hh+1));

        double cc_temp = *hh / tt;
        double ss_temp = *(hh+1) / tt;

        cc[iGMRES] = cc_temp;
        ss[iGMRES] = ss_temp;

        rs[iGMRES+1] = -(ss_temp * rs[iGMRES]);

        rs[iGMRES] = cc_temp * rs[iGMRES];

        *hh = cc_temp * *hh + ss_temp * *(hh+1);

        const double res = fabs(rs[iGMRES+1]);

        iGMRES++;

        if(res <= tol)
        {
          flag = 1;
          break; /* GMRES converged */
        }
      }

      iGMRES_total += iGMRES;

      rs[(iGMRES-1)] /= ghh[(iGMRES-1) * (GMRES_BASIS + 1) + (iGMRES-1)];

      for(unsigned int i = 1; i <= (iGMRES-1); i++)
      {
        uint32_t k = (iGMRES-1) - i;
        double tt = rs[k];

        for(unsigned int j = k+1; j <= (iGMRES-1); j++) tt -= ghh[j * (GMRES_BASIS + 1) + k] * rs[j];

        rs[k] = tt / ghh[k * (GMRES_BASIS + 1) + k];
      }

      for(unsigned int v = 0; v < iGMRES; v++) ComputeAXPY(g->c->sz, rs[v], vecs[v], x);

      if(iGMRES == GMRES_RESTART) fun3d_printf(5, ">> GMRES: COMPLETED ONE CYCLE\n");
      if(iGMRES_total == GMRES_MAX_IT) fun3d_printf(5, ">> GMRES: REACHED MAXIMUM LINEAR ITERATIONS\n");

      if(flag) break; /* GMRES converged */
    }

    fun3d_printf(5, ">> GMRES: TOTAL ITERATIONS %d\n", iGMRES_total);

    ComputeAXPY(g->c->sz, -1.f, x, g->q->q);

    ComputeForces(g, forces);

    const double clift = forces[0];
    const double cdrag = forces[1];
    const double cmomn = forces[2];

    printf("STEP: %d ", iTS);
    printf("CFL: %g ", cfl);
    printf("NORM: %g\n", fnorm);

    printf("LIFT: %g DRAG: %g MOMN: %g\n", clift, cdrag, cmomn);
    
    if(((uint32_t) round(fnorm_init / fnorm)) >= FNORM_MAX) break;

    iTS++;
  }

  fun3d_free(grad->x0);
  fun3d_free(grad->x1);
  fun3d_free(grad->x2);
  fun3d_free(grad);

  fun3d_free(ghh);

  for(v = 0; v < GMRES_BASIS; v++) fun3d_free(vecs[v]);

  fun3d_free(w);
  fun3d_free(x);
  fun3d_free(r);
  fun3d_free(t);
}