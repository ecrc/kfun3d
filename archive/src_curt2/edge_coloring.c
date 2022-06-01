
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include "inc/allocator.h"
#include "inc/geometry.h"
#include "inc/msh/mesh.h"
#include "inc/msh/bit_map.h"

#define VEC_LEN 8

void
iecoloring(struct geometry *restrict g)
{
  const uint32_t *restrict ie = g->s->ie;

  const size_t nnodes = g->n->sz;
  const size_t nedges = g->e->sz;

  uint32_t *restrict n0 = g->e->eptr->n0;
  uint32_t *restrict n1 = g->e->eptr->n1;

  double *restrict x0 = g->e->xyzn->x0;
  double *restrict x1 = g->e->xyzn->x1;
  double *restrict x2 = g->e->xyzn->x2;
  double *restrict x3 = g->e->xyzn->x3;
 
  uint32_t *restrict nedges_per_color;
  kcalloc(nedges, sizeof(uint32_t), (void *) &nedges_per_color);

#pragma omp parallel
  {
    const uint32_t t = omp_get_thread_num();

    const uint32_t ie0 = ie[t];
    const uint32_t ie1 = ie[t+1];

    const size_t nsz_b = I2B(nnodes);

    uint8_t *restrict node_bmap;
    kcalloc(nsz_b, sizeof(uint8_t), (void *) &node_bmap);

    uint32_t k = ie0;
    uint32_t c = ie0;

    uint32_t i;
    for(i = ie0; i < ie1; i++)
    {
      uint32_t j;

      memset(node_bmap, 0, nsz_b * sizeof(uint8_t));

      for(j = i; j < ie1; j++)
      {
        uint32_t f0 = 0, f1 = 0;

        BGET(f0, node_bmap[IOFF(n0[j])], BOFF(n0[j]));
        BGET(f1, node_bmap[IOFF(n1[j])], BOFF(n1[j]));

        if(f0 | f1) continue;

        if(nedges_per_color[c] == VEC_LEN) break;

        BSET(node_bmap[IOFF(n0[j])], BOFF(n0[j]));
        BSET(node_bmap[IOFF(n1[j])], BOFF(n1[j]));

        const uint32_t _n0 = n0[j];
        const uint32_t _n1 = n1[j];

        const double _x0 = x0[j];
        const double _x1 = x1[j];
        const double _x2 = x2[j];
        const double _x3 = x3[j];

        n0[j] = n0[k];
        n1[j] = n1[k];

        x0[j] = x0[k];
        x1[j] = x1[k];
        x2[j] = x2[k];
        x3[j] = x3[k];


        n0[k] = _n0;
        n1[k] = _n1;

        x0[k] = _x0;
        x1[k] = _x1;
        x2[k] = _x2;
        x3[k] = _x3;

        k++;

        nedges_per_color[c]++;
      }

      c++; i = k;
    }

    kfree(node_bmap);
  }

  kfree(nedges_per_color);
}
