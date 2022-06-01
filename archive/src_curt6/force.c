
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include <ktime.h>
#include <geometry.h>
#ifdef __USE_HW_COUNTER
#include <perf.h>
#include <kperf.h>
#endif
#include <phy.h>

/* Calculates the forces (Drag FORCE, LIFT FORCE, and the momentum) */
void
compute_force(struct force *restrict f)
{
#ifdef __USE_HW_COUNTER
  const struct fd fd = f->perf_counters->fd;

  struct counters start;
  perf_read(fd, &start);

  const uint64_t icycle = __rdtsc();
#endif

  struct ktime ktime;
  setktime(&ktime);

  const struct geometry *restrict g = f->g;
  const struct ivals * iv = f->iv;
  const double *restrict q = f->q;

  double lift = 0.f;
  double drag = 0.f;
  double momn = 0.f;

  const uint32_t snfc = g->s->snfc;
  const uint32_t *restrict snfic = g->s->snfic;

  uint32_t i;
  for(i = 0; i < snfc; i++)
  {
    uint32_t if0 = snfic[i];
    uint32_t if1 = snfic[i+1];

    uint32_t j;
#pragma omp parallel for reduction(+: lift, drag, momn)
    for(j = if0; j < if1; j++)
    {
      uint32_t n0 = g->b->snfptr->n0[j];
      uint32_t n1 = g->b->snfptr->n1[j];
      uint32_t n2 = g->b->snfptr->n2[j];

      double x0 = g->n->xyz->x0[n0];
      double y0 = g->n->xyz->x1[n0];
      double z0 = g->n->xyz->x2[n0];
                                   
      double x1 = g->n->xyz->x0[n1];
      double y1 = g->n->xyz->x1[n1];
      double z1 = g->n->xyz->x2[n1];
                                   
      double x2 = g->n->xyz->x0[n2];
      double y2 = g->n->xyz->x1[n2];
      double z2 = g->n->xyz->x2[n2];

      /* Delta coordinates in all directions */
                 
      double ax = x1 - x0;
      double ay = y1 - y0;
      double az = z1 - z0;

      double bx = x2 - x0;
      double by = y2 - y0;
      double bz = z2 - z0;

      /*
        Norm points outward, away from grid interior.
        Norm magnitude is area of surface triangle.
      */
      
      double xnorm = ay * bz;
      xnorm -= az * by;
      xnorm = -0.5f * xnorm;
      
      double ynorm =  ax * bz;
      ynorm -= az * bx;
      ynorm = 0.5f * ynorm;
      
      /* Pressure values store at every face node */

      double p0 = q[g->c->bsz * n0];
      double p1 = q[g->c->bsz * n1];
      double p2 = q[g->c->bsz * n2];

      double press = (p0 + p1 + p2) / 3.f;
      double cp = 2.f * (press - 1.f);

      double dcx = cp * xnorm;
      double dcy = cp * ynorm;

      double xmid = x0 + x1 + x2;
      double ymid = y0 + y1 + y2;

      lift = lift - dcx * iv->v + dcy * iv->u;
      drag = drag + dcx * iv->u + dcy * iv->v;
      momn = momn + (xmid - 0.25f) * dcy - ymid * dcx;
    }
  }

  (* f->clift) = lift;
  (* f->cdrag) = drag;
  (* f->cmomn) = momn;

  compute_time(&ktime, &f->t->forces);

#ifdef __USE_HW_COUNTER
  const uint64_t cycle = __rdtsc() - icycle;

  struct counters end;
  perf_read(fd, &end);

  struct tot tot;
  perf_calc(start, end, &tot);

  f->perf_counters->ctrs->forces.cycles += cycle;
  f->perf_counters->ctrs->forces.tot.imcR += tot.imcR;
  f->perf_counters->ctrs->forces.tot.imcW += tot.imcW;
  f->perf_counters->ctrs->forces.tot.edcR += tot.edcR;
  f->perf_counters->ctrs->forces.tot.edcW += tot.edcW;
#endif
}
