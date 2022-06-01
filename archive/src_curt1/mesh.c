
#include <stdio.h>
#include <stdint.h>
#include <allocator.h>
#include <geometry.h>
#include <mesh.h>
#include <fio.h>
#include <ktime.h>
#ifdef __USE_HW_COUNTER
#include <perf.h>
#include <kperf.h>
#endif
#include <main.h>

void
imesh(const uint8_t * fpath, struct geometry *restrict g)
{      
  /* Open the mesh file for reading [rb: read a binary file] */
  FILE * fstream = fopen((char *) fpath, "rb");

  /* File doesn't exist */
  if(fstream == NULL) panic("fopen");  

  /* Seek to the end of the mesh file to count the number of bytes 
   * the file has */
  if((fseek(fstream, 0, SEEK_END)) != 0) panic("fseek");

  size_t fsz = ftell(fstream);
  
  rewind(fstream);

  char *restrict fbuf;
  kmalloc(fsz, sizeof(char), (void *) &fbuf);

  if(fread(fbuf, sizeof(char), fsz, fstream) != fsz) panic("fread");

  /* Close the mesh file after reading  */
  if((fclose(fstream)) != 0) panic("fclose");

  /* Mesh parameters stack buffer */
  uint32_t p[13] __attribute__((aligned(MEMALIGN)));
 
  /* Walk through the file buffer and read the first 13 elements
   * of the mesh file. These elements are the mesh parameters */
  struct wtbl w;
  {
    w.l   = fbuf;
    w.h   = fbuf + 52;
    w.t   = UINT;
    w.sz  = 13;
  }
  walkfbuf(&w, p);

  size_t bytes = 52;

  struct etbl *restrict e;
  kmalloc(1, sizeof(struct etbl), (void *) &e);

  struct edge *restrict eptr;
  kmalloc(1, sizeof(struct edge), (void *) &eptr);

  struct xyzn *restrict xyzn;
  kmalloc(1, sizeof(struct xyzn), (void *) &xyzn);

  e->sz = p[2]; // Number of edges

  kmalloc(e->sz, sizeof(uint32_t), (void *) &eptr->n0);
  kmalloc(e->sz, sizeof(uint32_t), (void *) &eptr->n1);

  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x0);
  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x1);
  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x2);
  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x3);

  e->eptr = eptr;
  e->xyzn = xyzn;

  /* Read the edges from the mesh file */
  bytes += emalloc((fbuf + bytes), e);

  g->e = e;

  struct ntbl *restrict n;
  kmalloc(1, sizeof(struct ntbl), (void *) &n);

  n->sz = p[1]; // Number of nodes

  kmalloc(1, sizeof(struct xyz), (void *) &n->xyz);
  kmalloc(n->sz, sizeof(double), (void *) &n->xyz->x0);
  kmalloc(n->sz, sizeof(double), (void *) &n->xyz->x1);
  kmalloc(n->sz, sizeof(double), (void *) &n->xyz->x2);

  kmalloc(n->sz, sizeof(double), (void *) &n->area);

  /* Allocate the nodes and their coordinates */
  bytes += nmalloc((fbuf + bytes), n);

  g->n = n;

  struct btbl *restrict b;
  kmalloc(1, sizeof(struct btbl), (void *) &b);

  bytes += bmalloc(p, (fbuf + bytes), b);

  g->b = b;

  if(bytes != fsz) panic("bytes read do not match the total bytes");

  kfree(fbuf);

  uint32_t *restrict snfn0;
  kmalloc(g->b->s->f->sz, sizeof(uint32_t), (void *) &snfn0); 
  uint32_t *restrict snfn1;
  kmalloc(g->b->s->f->sz, sizeof(uint32_t), (void *) &snfn1);
  uint32_t *restrict snfn2;
  kmalloc(g->b->s->f->sz, sizeof(uint32_t), (void *) &snfn2);

  struct snfptr *restrict snfptr;
  kmalloc(1, sizeof(struct snfptr), (void *) &snfptr);

  uint32_t i;
  for(i = 0; i < g->b->s->f->sz; i++)
  {
    uint32_t n0 = g->b->s->n->nptr[g->b->s->f->fptr[i].f0];
    uint32_t n1 = g->b->s->n->nptr[g->b->s->f->fptr[i].f1];
    uint32_t n2 = g->b->s->n->nptr[g->b->s->f->fptr[i].f2];

    snfn0[i] = n0;
    snfn1[i] = n1;
    snfn2[i] = n2;
  }

  g->b->snfptr = snfptr;
  g->b->snfptr->n0 = snfn0;
  g->b->snfptr->n1 = snfn1;
  g->b->snfptr->n2 = snfn2;

  struct ctbl *restrict c;
  kmalloc(1, sizeof(struct ctbl), (void *) &c);

  g->c = c;

  // Generate IA and JA arrays of a sparse graph to construct
  // the operator matrix A
  m2csr(g);

  struct stbl *restrict s;
  kmalloc(1, sizeof(struct stbl), (void *) &s);

  g->s = s;

  // Partition the domain into a subdomain using METIS
  // to divide the work across the OpenMP threads
  isubdomain(g);

  // Color the solid boundary facets to generate independent
  // subsets of facets within a specific color, which can be
  // effectively parallelized across multiple threads
  ifcoloring(g);

#ifdef __USE_EDGE_COLORING
  iecoloring(g);
#endif

  // Compute the weights for least square method
  wmalloc(g);
}
