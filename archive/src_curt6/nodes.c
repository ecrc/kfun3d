
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

size_t 
nmalloc(char *restrict fbuf, struct ntbl *restrict n)
{
  size_t sz = n->sz * 4;

  double *restrict buf;
  kmalloc(sz, sizeof(double), (void *) &buf);

  size_t bytes = sz * sizeof(double);

  struct wtbl w;
  {
    w.l   = fbuf;
    w.h   = fbuf + bytes;
    w.t   = DOUBLE;
    w.sz  = sz;
  }
  walkfbuf(&w, buf);

  // Partitioned the data and arrange them
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < n->sz; i++)
  {
    n->xyz->x0[i] = buf[i];
    n->xyz->x1[i] = buf[i + n->sz];
    n->xyz->x2[i] = buf[i + n->sz + n->sz];

    n->area[i] = buf[i + n->sz + n->sz + n->sz];
  }

  kfree(buf);

  return bytes;
}
