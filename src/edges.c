#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void CXX_Walk_Int(char *l, const char *h, const size_t sz, unsigned int *b);
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
#include "index.h"

/* Allocate the edges */
size_t
emalloc(char *fbuf, struct etbl *e)
{
  const size_t sz = e->sz;
  const size_t ndsz = sz * 2;
  
  uint32_t *buf0 = (uint32_t *) fun3d_malloc(ndsz, sizeof(uint32_t));

  size_t bytes0 = ndsz * sizeof(uint32_t);

  //struct wtbl w0;
  //{
  //  w0.l   = fbuf;
  //  w0.h   = fbuf + bytes0;
  //  w0.t   = UINT;
  //  w0.sz  = ndsz;
  //}
  //walkfbuf(&w0, buf0);

  CXX_Walk_Int(fbuf, fbuf + bytes0, ndsz, buf0);

  const size_t nrsz = sz * 4;

  double *buf1 = (double *) fun3d_malloc(nrsz, sizeof(double));

  size_t bytes1 = nrsz * sizeof(double);

  //struct wtbl w1;
  //{
  //  w1.l   = w0.h;
  //  w1.h   = w0.h + bytes1;
  //  w1.t   = DOUBLE;
  //  w1.sz  = nrsz;
  //}
  //walkfbuf(&w1, buf1);

  CXX_Walk_Double(fbuf + bytes0, fbuf + bytes0 + bytes1, nrsz, buf1);

  // Find the permutation array of a sorted sequence to reorder the
  // edges and their normals
  uint32_t *p = (uint32_t *) fun3d_malloc(sz, sizeof(uint32_t));

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

  fun3d_free(buf0);
  fun3d_free(buf1);
  fun3d_free(p);

  return (bytes0 + bytes1);
}