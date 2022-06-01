
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "inc/allocator.h"
#include "inc/geometry.h"
#include "inc/msh/mesh.h"
#include "inc/msh/bit_map.h"

void
ifcoloring(struct geometry *restrict g)
{
  const size_t nnodes = g->n->sz;
  const size_t nsfacets = g->b->s->f->sz;

  uint32_t *restrict n0 = g->b->snfptr->n0;
  uint32_t *restrict n1 = g->b->snfptr->n1;
  uint32_t *restrict n2 = g->b->snfptr->n2;

  uint32_t *restrict nfaces_per_color;
  kcalloc(nsfacets, sizeof(uint32_t), (void *) &nfaces_per_color);

  const size_t nsz_b = I2B(nnodes);

  uint8_t *restrict node_bmap;
  kcalloc(nsz_b, sizeof(uint8_t), (void *) &node_bmap);

  uint32_t k = 0;
  uint32_t c = 0;

  uint32_t i;
  for(i = 0; i < nsfacets;)
  {
    uint32_t j;

    memset(node_bmap, 0, nsz_b * sizeof(uint8_t));

    for(j = i; j < nsfacets; j++)
    {
      uint32_t f0 = 0, f1 = 0, f2 = 0;

      BGET(f0, node_bmap[IOFF(n0[j])], BOFF(n0[j]));
      BGET(f1, node_bmap[IOFF(n1[j])], BOFF(n1[j]));
      BGET(f2, node_bmap[IOFF(n2[j])], BOFF(n2[j]));

      if(f0 | f1 | f2) continue;

      BSET(node_bmap[IOFF(n0[j])], BOFF(n0[j]));
      BSET(node_bmap[IOFF(n1[j])], BOFF(n1[j]));
      BSET(node_bmap[IOFF(n2[j])], BOFF(n2[j]));

      uint32_t _n0 = n0[j];
      uint32_t _n1 = n1[j];
      uint32_t _n2 = n2[j];

      n0[j] = n0[k];
      n1[j] = n1[k];
      n2[j] = n2[k];

      n0[k] = _n0;
      n1[k] = _n1;
      n2[k] = _n2;

      k++;

      nfaces_per_color[c]++;
    }

    c++;
    i = k;
  }

  kfree(node_bmap);

  uint32_t *restrict ic;
  kmalloc((c + 1), sizeof(uint32_t), (void *) &ic);

  uint32_t sum = 0;

  for(i = 0; i < c; i++)
  {
    uint32_t temp = nfaces_per_color[i];
    ic[i] = sum;
    sum += temp;
  }
  ic[c] = sum;

  kfree(nfaces_per_color);

  {
    g->s->snfic = ic;
    g->s->snfc = c;
  }
}
