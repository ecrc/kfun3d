#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
extern void *CXX_Matrix_Init(const size_t nnodes, const size_t nedges, const unsigned int block_size, const unsigned int nthreads, const unsigned int *n0, const unsigned int *n1);
extern void CXX_Walk_Int(char *l, const char *h, const size_t sz, unsigned int *b);
#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <omp.h>
#include "allocator.h"
#include "geometry.h"
//#include "fio.h"
#include "mesh.h"
#include "mesh2geo.h"

void
Setup(const char * fpath, struct geometry *g)
{
  /* Open the mesh file for reading [rb: read a binary file] */
  FILE *fstream = fopen((char *) fpath, "rb");

  /* File doesn't exist */
  assert(fstream != NULL);

  /* Seek to the end of the mesh file to count the number of bytes 
   * the file has */
  assert((fseek(fstream, 0, SEEK_END)) == 0);

  size_t fsz = (size_t) ftell(fstream);
  
  rewind(fstream);

  char *fbuf = (char *) fun3d_malloc(fsz, sizeof(char));

  assert(fread(fbuf, sizeof(char), fsz, fstream) == fsz);

  /* Close the mesh file after reading  */
  assert((fclose(fstream)) == 0);

  /* Mesh parameters stack buffer */
  uint32_t p[13];
 
  ///* Walk through the file buffer and read the first 13 elements
  // * of the mesh file. These elements are the mesh parameters */
  //struct wtbl w;
  //{
  //  w.l   = fbuf;
  //  w.h   = fbuf + 52;
  //  w.t   = UINT;
  //  w.sz  = 13;
  //}
  //walkfbuf(&w, p);

  CXX_Walk_Int(fbuf, fbuf + 52, 13, p);

  size_t bytes = 52;

  struct etbl *e = (struct etbl *) fun3d_malloc(1, sizeof(struct etbl));
  struct edge *eptr = (struct edge *) fun3d_malloc(1, sizeof(struct edge));
  struct xyzn *xyzn = (struct xyzn *) fun3d_malloc(1, sizeof(struct xyzn));

  e->sz = p[2]; // Number of edges

  eptr->n0 = (uint32_t *) fun3d_malloc(e->sz, sizeof(uint32_t));
  eptr->n1 = (uint32_t *) fun3d_malloc(e->sz, sizeof(uint32_t));

  xyzn->x0 = (double *) fun3d_malloc(e->sz, sizeof(double));
  xyzn->x1 = (double *) fun3d_malloc(e->sz, sizeof(double));
  xyzn->x2 = (double *) fun3d_malloc(e->sz, sizeof(double));
  xyzn->x3 = (double *) fun3d_malloc(e->sz, sizeof(double));

  e->eptr = eptr;
  e->xyzn = xyzn;

  /* Read the edges from the mesh file */
  bytes += emalloc((fbuf + bytes), e);

  g->e = e;

  struct ntbl *n = (struct ntbl *) fun3d_malloc(1, sizeof(struct ntbl));

  n->sz = p[1]; // Number of nodes

  n->xyz = (struct xyz *) fun3d_malloc(1, sizeof(struct xyz));
  n->xyz->x0 = (double *) fun3d_malloc(n->sz, sizeof(double));
  n->xyz->x1 = (double *) fun3d_malloc(n->sz, sizeof(double));
  n->xyz->x2 = (double *) fun3d_malloc(n->sz, sizeof(double));

  /* Deprecated in the newer version */
  /* n->area = (double *) fun3d_malloc(n->sz, sizeof(double)); */

  /* Allocate the nodes and their coordinates */
  bytes += nmalloc((fbuf + bytes), n);

  g->n = n;

  struct btbl *b = (struct btbl *) fun3d_malloc(1, sizeof(struct btbl));

  bytes += bmalloc(p, (fbuf + bytes), b);

  g->b = b;

  assert(bytes == fsz);

  fun3d_free(fbuf);

  struct ctbl *c = (struct ctbl *) fun3d_malloc(1, sizeof(struct ctbl));

  g->c = c;

  // Generate IA and JA arrays of a sparse graph to construct
  // the operator matrix A
  m2csr(g);

  g->matrix = CXX_Matrix_Init(g->n->sz, g->e->sz, g->c->b, (unsigned int)omp_get_max_threads(), g->e->eptr->n0, g->e->eptr->n1);

  struct stbl *s = (struct stbl *) fun3d_malloc(1, sizeof(struct stbl));

  g->s = s;

  // Partition the domain into a subdomain using METIS
  // to divide the work across the OpenMP threads
  isubdomain(g);

  struct stbl *tt = (struct stbl *) fun3d_malloc(1, sizeof(struct stbl));
 
  g->t = tt;

  // Color the solid boundary facets to generate independent
  // subsets of facets within a specific color, which can be
  // effectively parallelized across multiple threads
  ifcoloring(g);

#ifdef BUCKET_SORT
  iecoloring(g);
#endif

  // Compute the weights for least square method
  wmalloc(g);

  struct qtbl *q = (struct qtbl *) fun3d_malloc(1, sizeof(struct qtbl));

  q->q = (double *) fun3d_malloc(g->c->sz, sizeof(double));

  g->q = q;

  iguess(g);
}