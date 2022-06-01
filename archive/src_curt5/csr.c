
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include "inc/allocator.h"
#include "inc/geometry.h"
#include "inc/msh/mesh.h"

/* c stdlib qsort comparable function */
static inline int 
comp(const void *restrict a, const void *restrict b)
{
    return (*((uint32_t *) a) - *((uint32_t *) b));
}

void
m2csr(struct geometry *restrict g)
{
  /* Row pointers */
  uint32_t *restrict ia;

#if defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW) 
  ia = calloc((g->n->sz + 1), sizeof(uint32_t));
#else
  kcalloc((g->n->sz+1), sizeof(uint32_t), (void *) &ia);
#endif

  uint32_t i;
  for(i = 0; i < g->e->sz; i++)
  {
    ia[g->e->eptr->n0[i]+1]++;
    ia[g->e->eptr->n1[i]+1]++;
  }

  ia[0] = 1;

  for(i = 1; i <= g->n->sz; i++)
  {
    ia[i] += ia[i-1];
    ia[i]++;
  }

  /* Adjust the IA array to Zero-index (c-style) */
  for(i = 0; i <= g->n->sz; i++) ia[i]--;

  uint32_t *restrict ja;

#if defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW) 
  ja = calloc(ia[g->n->sz], sizeof(uint32_t));
#else
  kmalloc(ia[g->n->sz], sizeof(uint32_t), (void *) &ja);
#endif

  double *aa = (double *) calloc(ia[g->n->sz] * 4 * 4, sizeof(double));

  /* A temp buffer used to keep tracking of each row elements */
  uint32_t *restrict buf;
  kmalloc(g->n->sz, sizeof(uint32_t), (void *) &buf);

  /* Column Index of the diagonal elements */
  for(i = 0; i < g->n->sz; i++)
  {
    ja[ia[i]] = i; // A diagonal element
    buf[i] = 1; // One element in this row has been added
  }

  /* Fill the rest of the array, ordered by RCM and using a 
   * modified version of Breadth-First Search traversing algorithm */
  for(i = 0; i < g->e->sz; i++)
  {
    uint32_t n0 = g->e->eptr->n0[i];
    uint32_t n1 = g->e->eptr->n1[i];

    /* Get the element index in the row 
     * The index is basically the row index plus the last element that
     * has been added in the row. */
    uint32_t indx = ia[n0] + buf[n0]; // Get the index
    buf[n0]++; // Column has been added (one more element in the row)
    ja[indx] = n1; // Store the node index in its corresponding index

    /* Do it for the other endpoint */
    indx = ia[n1] + buf[n1];
    buf[n1]++;
    ja[indx] = n0;
  }

  kfree(buf);

  // Number of nonzero block per row
//  uint32_t *restrict nnz;
//  kmalloc(g->n->sz, sizeof(uint32_t), (void *) &nnz);

  size_t nz_total = 0;


  /* Sort the each row of a ja array in an increasing order  
   * No we reorder them again to make sure the at each row 
   * we have the node ordered in increasing order plus based on 
   * their degree */
#pragma omp parallel for reduction(+:nz_total)
  for(i = 0; i < g->n->sz; i++)
  {
    uint32_t jstart = ia[i];
    uint32_t jend = ia[i+1];

    /* Qsort to sort the JA array */
    uint32_t * l = ja + jstart;  // Low address
    uint32_t * h = ja + jend;    // High address

    size_t sz = h - l;

    qsort(l, sz, sizeof(uint32_t), comp);

    uint32_t nz = 0;

    uint32_t j;
    for(j = jstart; j < jend; j++) nz++;

//    nnz[i] = nz;

    nz_total += nz;
  }

  g->c->aa = aa;

  g->c->ia = ia; // Starting row indices
  g->c->ja = ja; // Column indices
  g->c->nnz = nz_total; // Number of nonzero blocks

#ifdef __USE_COMPRESSIBLE_FLOW
  /* Compressible Euler flow */
  g->c->bsz = 5; // 5 unknowns per grid point
#else 
  /* Incompressible Euler flow */
  g->c->bsz = 4; // 4 unknowns per grid point
  g->c->bsz2 = 4*4; // 4 unknowns per grid point
#endif

    int *ilen = (int *) calloc(g->n->sz, sizeof(int));
    g->c->ailen = ilen;
  
  /* Number of the matrix rows | columns */
  g->c->sz = g->c->bsz * g->n->sz;  
}
