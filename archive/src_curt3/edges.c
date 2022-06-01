
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "inc/allocator.h"
#include "inc/geometry.h"
#include "inc/msh/mesh.h"
#include "inc/msh/fio.h"
#include "inc/msh/index.h"

/* Allocate the edges */
size_t
emalloc(char *restrict fbuf, struct etbl *restrict e)
{
  const size_t sz = e->sz; // Number of edges

  const size_t ndsz = sz * 2; // Number of edges' endpoints
  
  uint32_t *restrict buf0;
  kmalloc(ndsz, sizeof(uint32_t), (void *) &buf0);

  size_t bytes0 = ndsz * sizeof(uint32_t);

  struct wtbl w0;
  {
    w0.l   = fbuf;
    w0.h   = fbuf + bytes0;
    w0.t   = UINT;
    w0.sz  = ndsz;
  }
  walkfbuf(&w0, buf0);

  const size_t nrsz = sz * 4;

  double *restrict buf1;
  kmalloc(nrsz, sizeof(double), (void *) &buf1);

  size_t bytes1 = nrsz * sizeof(double);

  struct wtbl w1;
  {
    w1.l   = w0.h;
    w1.h   = w0.h + bytes1;
    w1.t   = DOUBLE;
    w1.sz  = nrsz;
  }
  walkfbuf(&w1, buf1);

  // Find the permutation array of a sorted sequence to reorder the
  // edges and their normals
  uint32_t *restrict p;
  kmalloc(sz, sizeof(uint32_t), (void *) &p);

  imain(sz, buf0, p);

  // Reorder the edge endpoints and their normals
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < sz; i++)
  {
    // Edge endpoints
    e->eptr->n0[i] = buf0[p[i]] - 1; // From Fortran to C
    e->eptr->n1[i] = buf0[p[i] + sz] - 1; // From Fortran to C
    
    // Unit normals of dual faces and area of the dual mesh face
    e->xyzn->x0[i] = buf1[p[i]];
    e->xyzn->x1[i] = buf1[p[i] + sz];
    e->xyzn->x2[i] = buf1[p[i] + sz + sz];
    e->xyzn->x3[i] = buf1[p[i] + sz + sz + sz];
  }

  kfree(buf0);
  kfree(buf1);
  kfree(p);

  return (bytes0 + bytes1);
}
