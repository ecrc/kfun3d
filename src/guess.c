#include <stdint.h>
#include <omp.h>
#include "geometry.h"
#include "mesh2geo.h"
#include "phy.h"

/* Set the initial guess */
void
iguess(struct geometry *g)
{
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < g->n->sz; i++)
  {
    g->q->q[i * g->c->b + 0] = P;
    g->q->q[i * g->c->b + 1] = U;
    g->q->q[i * g->c->b + 2] = V;
    g->q->q[i * g->c->b + 3] = W;
  }
}