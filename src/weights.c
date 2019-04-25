#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>
#include "geometry.h"
#include "allocator.h"
#include "mesh2geo.h"

static inline void
compute_terms(
  const struct geometry *g,
  const double *w,
  struct xyz *terms0,
  struct xyz *terms1)
{
  const uint32_t *ie = g->s->i;
  const uint32_t *part = g->n->part;

  const uint32_t *n0 = g->e->eptr->n0;
  const uint32_t *n1 = g->e->eptr->n1;

  const double *x0 = g->n->xyz->x0;
  const double *x1 = g->n->xyz->x1;
  const double *x2 = g->n->xyz->x2;

#pragma omp parallel
  {
    const uint32_t t = (uint32_t) omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    uint32_t i;

    for(i = ie0; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const double coordx0 = x0[node0];
      const double coordy0 = x1[node0];
      const double coordz0 = x2[node0];

      const double coordx1 = x0[node1];
      const double coordy1 = x1[node1];
      const double coordz1 = x2[node1];

      const double dx = coordx1 - coordx0;
      const double dy = coordy1 - coordy0;
      const double dz = coordz1 - coordz0;

      double c0;
      double c1;

      double termx;
      double termy;
      double termz;

      if(part[node0] == t)
      {
        c0 = - dx * w[node0 * 7 + 1] + dy;
        c1 = - dx * w[node0 * 7 + 2] + dz;
        c1 = - w[node0 * 7 + 4] * c0 + c1;

        termx = w[node0 * 7 + 3] * w[node0 * 7 + 1] * c0;
        termx = dx * w[node0 * 7 + 0] - termx;
        termx += w[node0 * 7 + 6] * c1;

        termy = w[node0 * 7 + 4] * w[node0 * 7 + 5] * c1;
        termy = w[node0 * 7 + 3] * c0 - termy;

        termz = w[node0 * 7 + 5] * c1;

        terms0->x0[i] = termx;
        terms0->x1[i] = termy;
        terms0->x2[i] = termz;
      }

      if(part[node1] == t)
      {
        c0 = dx * w[node1 * 7 + 1] - dy;
        c1 = dx * w[node1 * 7 + 2] - dz;
        c1 = - w[node1 * 7 + 4] * c0 + c1;

        termx = w[node1 * 7 + 3] * w[node1 * 7 + 1] * c0;
        termx = -dx * w[node1 * 7 + 0] - termx;
        termx += w[node1 * 7 + 6] * c1;

        termy = w[node1 * 7 + 4] * w[node1 * 7 + 5] * c1;
        termy = w[node1 * 7 + 3] * c0 - termy;

        termz = w[node1 * 7 + 5] * c1;

        terms1->x0[i] = termx;
        terms1->x1[i] = termy;
        terms1->x2[i] = termz;
      }
    }
  }
}

/* Do w22 */
static inline void
w2alloc(const struct geometry *g, double *w) 
{
  const uint32_t *ie = g->s->i;
  const uint32_t *part = g->n->part;

  const uint32_t *n0 = g->e->eptr->n0;
  const uint32_t *n1 = g->e->eptr->n1;

  const double *x0 = g->n->xyz->x0;
  const double *x1 = g->n->xyz->x1;
  const double *x2 = g->n->xyz->x2;

#pragma omp parallel
  {
    const uint32_t t = (uint32_t) omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];
    
    uint32_t i;
    for(i = ie0; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const double coordx0 = x0[node0];
      const double coordy0 = x1[node0];
      const double coordz0 = x2[node0];

      const double coordx1 = x0[node1];
      const double coordy1 = x1[node1];
      const double coordz1 = x2[node1];

      const double dx = coordx1 - coordx0;
      const double dy = coordy1 - coordy0;
      const double dz = coordz1 - coordz0;

      if(part[node0] == t)
      {
        const double d0   = w[node0 * 7 + 1] / w[node0 * 7 + 0];
        const double d0_  = dy - dx * d0;

        const double d1   = w[node0 * 7 + 2] / w[node0 * 7 + 0];
        const double d1_  = dz - dx * d1;

        const double d2   = w[node0 * 7 + 4] / w[node0 * 7 + 3];
        const double d2_  = d1_ - d2 * d0_;

        w[node0 * 7 + 5] += d2_ * d2_;
      }

      if(part[node1] == t)
      {
        const double d0   = w[node1 * 7 + 1] / w[node1 * 7 + 0];
        const double d0_  = -dy + dx * d0;

        const double d1   = w[node1 * 7 + 2] / w[node1 * 7 + 0];
        const double d1_  = -dz + dx * d1;

        const double d2   = w[node1 * 7 + 4] / w[node1 * 7 + 3];
        const double d2_  = d1_ - d2 * d0_;

        w[node1 * 7 + 5] += d2_ * d2_;
      }
    }
  }
}

/* Do w11 and w12 */
static inline void
w1alloc(const struct geometry *g, double *w) 
{
  const uint32_t *ie = g->s->i;
  const uint32_t *part = g->n->part;

  const uint32_t *n0 = g->e->eptr->n0;
  const uint32_t *n1 = g->e->eptr->n1;

  const double *x0 = g->n->xyz->x0;
  const double *x1 = g->n->xyz->x1;
  const double *x2 = g->n->xyz->x2;

#pragma omp parallel
  {
    const uint32_t t = (uint32_t) omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    uint32_t i;
    for(i = ie0; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const double coordx0 = x0[node0];
      const double coordy0 = x1[node0];
      const double coordz0 = x2[node0];

      const double coordx1 = x0[node1];
      const double coordy1 = x1[node1];
      const double coordz1 = x2[node1];

      /* Compute the difference of each coordinate component */

      const double dx = coordx1 - coordx0;
      const double dy = coordy1 - coordy0;
      const double dz = coordz1 - coordz0;

      if(part[node0] == t)
      {
        const double d  = w[node0 * 7 + 1] / w[node0 * 7 + 0];
        const double d_ = dy - dx * d;

        w[node0 * 7 + 3] += d_ * d_;
        w[node0 * 7 + 4] += d_ * dz;
      }

      if(part[node1] == t)
      {
        const double d  = w[node1 * 7 + 1] / w[node1 * 7 + 0];
        const double d_ = -dy + dx * d;

        w[node1 * 7 + 3] += d_ * d_;
        w[node1 * 7 + 4] -= d_ * dz;
      }
    }
  }
}

/*
  Compute w00, w01, and w02 in parallel
*/
static inline void
w0alloc(const struct geometry *g, double *w)
{
  const uint32_t *ie = g->s->i;
  const uint32_t *part = g->n->part;

  const uint32_t *n0 = g->e->eptr->n0;
  const uint32_t *n1 = g->e->eptr->n1;

  const double *x0 = g->n->xyz->x0;
  const double *x1 = g->n->xyz->x1;
  const double *x2 = g->n->xyz->x2;

#pragma omp parallel
  {
    const uint32_t t = (uint32_t) omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    uint32_t i;
    for(i = ie0; i < ie1; i++)
    {
      const uint32_t node0 = n0[i];
      const uint32_t node1 = n1[i];

      const double coordx0 = x0[node0];
      const double coordy0 = x1[node0];
      const double coordz0 = x2[node0];

      const double coordx1 = x0[node1];
      const double coordy1 = x1[node1];
      const double coordz1 = x2[node1];

      /* 
       * Write-back: Update endpoints
       * */
      if(part[node0] == t) // Do the left endpoint
      {
        const double res_x = coordx1 - coordx0;
        const double res_y = coordy1 - coordy0;
        const double res_z = coordz1 - coordz0;

        w[node0 * 7 + 0] += res_x * res_x;
        w[node0 * 7 + 1] += res_x * res_y;
        w[node0 * 7 + 2] += res_x * res_z;
      }

      if(part[node1] == t) // Do the right endpoint
      {
        const double res_x = coordx0 - coordx1;
        const double res_y = coordy0 - coordy1;
        const double res_z = coordz0 - coordz1;

        w[node1 * 7 + 0] += res_x * res_x;
        w[node1 * 7 + 1] += res_x * res_y;
        w[node1 * 7 + 2] += res_x * res_z;
      }
    }
  }
}

void
wmalloc(struct geometry *g)
{
  size_t nnodes = g->n->sz;

  double *w = (double *) fun3d_calloc(7 * nnodes, sizeof(double));
  
  uint32_t i;

  /* Do w00, w01, and w02 */
  w0alloc(g, w);

  /* Compute ||x|| (norm) and divide the other by
     the computed norm */
#pragma omp parallel for
  for(i = 0; i < nnodes; i++)
  {
    w[i * 7 + 0] = sqrt(w[i * 7 + 0]);
    w[i * 7 + 1] /= w[i * 7 + 0];
    w[i * 7 + 2] /= w[i * 7 + 0];
  }

  /* Do w11 and w12 */  
  w1alloc(g, w);

#pragma omp parallel for
  for(i = 0; i < nnodes; i++)
  {
    w[i * 7 + 3] = sqrt(w[i * 7 + 3]);
    w[i * 7 + 4] /= w[i * 7 + 3];
  }

  /* Do w22 */
  w2alloc(g, w);

  /* Update the magnitudes. Stuffs contributed by Dinesh 1998 */
#pragma omp parallel for
  for(i = 0; i < nnodes; i++)
  {
    w[i * 7 + 5] = sqrt(w[i * 7 + 5]);

    double sw00 = w[i * 7 + 0] * w[i * 7 + 0];
    double sw11 = w[i * 7 + 3] * w[i * 7 + 3];
    double sw22 = w[i * 7 + 5] * w[i * 7 + 5];

    double w00 = 1.f / sw00;
    double w11 = 1.f / sw11;
    double w22 = 1.f / sw22;

    double w01 = w[i * 7 + 1] / w[i * 7 + 0];
    double w02 = w[i * 7 + 2] / w[i * 7 + 0];
    double w12 = w[i * 7 + 4] / w[i * 7 + 3];

    double m0 = w[i * 7 + 1] * w[i * 7 + 4];
    m0 -= w[i * 7 + 2] * w[i * 7 + 3];

    double m1 = w[i * 7 + 0] * w[i * 7 + 3] * sw22;

    double w33 =  m0 / m1;

    w[i * 7 + 0] = w00;
    w[i * 7 + 3] = w11;
    w[i * 7 + 5] = w22;
    w[i * 7 + 1] = w01;
    w[i * 7 + 2] = w02;
    w[i * 7 + 4] = w12;
    w[i * 7 + 6] = w33;
  }

  size_t nedges = g->e->sz;

  struct xyz *terms0 = (struct xyz *) fun3d_malloc(1, sizeof(struct xyz));

  double *wtermsx0 = (double *) fun3d_calloc(nedges, sizeof(double));
  double *wtermsy0 = (double *) fun3d_calloc(nedges, sizeof(double));
  double *wtermsz0 = (double *) fun3d_calloc(nedges, sizeof(double));

  terms0->x0 = wtermsx0;
  terms0->x1 = wtermsy0;
  terms0->x2 = wtermsz0;

  struct xyz *terms1 = (struct xyz *) fun3d_malloc(1, sizeof(struct xyz));

  double *wtermsx1 = (double *) fun3d_calloc(nedges, sizeof(double));
  double *wtermsy1 = (double *) fun3d_calloc(nedges, sizeof(double));
  double *wtermsz1 = (double *) fun3d_calloc(nedges, sizeof(double));

  terms1->x0 = wtermsx1;
  terms1->x1 = wtermsy1;
  terms1->x2 = wtermsz1;

  compute_terms(g, w, terms0, terms1);

  fun3d_free(w);

  struct weights *weights = (struct weights *) fun3d_malloc(1, sizeof(struct weights));

  weights->w0 = terms0;
  weights->w1 = terms1;
  g->e->w = weights;
}