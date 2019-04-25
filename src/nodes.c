#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void CXX_Walk_Double(char *l, const char *h, const size_t sz, double *b);
#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "allocator.h"
#include "geometry.h"
//#include "fio.h"
#include "mesh.h"

size_t 
nmalloc(char *fbuf, struct ntbl *n)
{
  size_t sz = n->sz * 4;

  double *buf = (double *) fun3d_malloc(sz, sizeof(double));

  size_t bytes = sz * sizeof(double);

  //struct wtbl w;
  //{
  //  w.l   = fbuf;
  //  w.h   = fbuf + bytes;
  //  w.t   = DOUBLE;
  //  w.sz  = sz;
  //}
  //walkfbuf(&w, buf);

  CXX_Walk_Double(fbuf, fbuf + bytes, sz, buf);

  // Partitioned the data and arrange them
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < n->sz; i++)
  {
    n->xyz->x0[i] = buf[i];
    n->xyz->x1[i] = buf[i + n->sz];
    n->xyz->x2[i] = buf[i + n->sz + n->sz];
    /* Ignore the area, deprecated in the newer version */
    /* n->area[i] = buf[i + n->sz + n->sz + n->sz]; */
  }

  fun3d_free(buf);

  n->cdt = (double *) fun3d_malloc(n->sz, sizeof(double));

  return bytes;
}