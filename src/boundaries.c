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
#include "boundaries.h"

static size_t
fbmalloc(char * fbuf, struct bface * f)
{
  size_t sz = f->sz * 4;

  uint32_t *buf = (uint32_t *) fun3d_malloc(sz, sizeof(uint32_t));

  size_t bytes = sz * sizeof(uint32_t);

  //struct wtbl w;
  //{
  //  w.l   = fbuf;
  //  w.h   = fbuf + bytes;
  //  w.t   = UINT;
  //  w.sz  = sz;
  //}
  //walkfbuf(&w, buf);

  CXX_Walk_Int(fbuf, fbuf + bytes, sz, buf);
  
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < f->sz; i++) 
  {
    f->fptr[i].f0 = buf[i] - 1;
    f->fptr[i].f1 = buf[i + f->sz] - 1;
    f->fptr[i].f2 = buf[i + f->sz + f->sz] - 1;
    f->fptr[i].f3 = buf[i + f->sz + f->sz + f->sz] - 1;
  }

  fun3d_free(buf);

  return bytes;
}

static size_t 
nbmalloc(char * fbuf, struct bnode * n)
{
  size_t bytes = n->sz * sizeof(uint32_t);

  //struct wtbl w0;
  //{
  //  w0.l   = fbuf;
  //  w0.h   = fbuf + bytes;
  //  w0.t   = UINT;
  //  w0.sz  = n->sz;
  //}
  //walkfbuf(&w0, n->nptr);

  CXX_Walk_Int(fbuf, fbuf + bytes, n->sz, n->nptr);

  size_t sz = n->sz * 3;

  size_t bytes_ = sz * sizeof(double);

  double *buf = (double *) fun3d_malloc(sz, sizeof(double));

  //struct wtbl w1;
  //{
  //  w1.l   = fbuf + bytes;
  //  w1.h   = w1.l + bytes_;
  //  w1.t   = DOUBLE;
  //  w1.sz  = sz;
  //}
  //walkfbuf(&w1, buf);
  
  CXX_Walk_Double(fbuf + bytes, fbuf + bytes + bytes_, sz, buf);

  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < n->sz; i++) 
  {
    n->nptr[i]--;

    n->xyz->x0[i] = buf[i];
    n->xyz->x1[i] = buf[i + n->sz];
    n->xyz->x2[i] = buf[i + n->sz + n->sz];
  }

  fun3d_free(buf);

  return (bytes + bytes_);
}

static size_t 
bmallocl(
  const size_t p1,
  const size_t p2,
  const size_t p3,
  char * fbuf,
  struct boundary * b)
{
  // Shift the file pointer to avoid reading the boundaries data,
  // which they are negligible in our core calculations 
  size_t bytes = p3 * 2 * sizeof(uint32_t); 

  struct bface *f = (struct bface *) fun3d_malloc(1, sizeof(struct bface));

  f->sz = p1;
  f->fptr = (struct facet *) fun3d_malloc(f->sz, sizeof(struct facet));

  bytes += fbmalloc((fbuf + bytes), f);

  b->f = f;

  struct bnode *n = (struct bnode *) fun3d_malloc(1, sizeof(struct bnode));

  n->sz = p2;
  n->nptr = (uint32_t *) fun3d_malloc(n->sz, sizeof(uint32_t));

  n->xyz = (struct xyz *) fun3d_malloc(1, sizeof(struct xyz));
  n->xyz->x0 = (double *) fun3d_malloc(n->sz, sizeof(double));
  n->xyz->x1 = (double *) fun3d_malloc(n->sz, sizeof(double));
  n->xyz->x2 = (double *) fun3d_malloc(n->sz, sizeof(double));

  bytes += nbmalloc((fbuf + bytes), n);

  b->n = n;

  return bytes;
}

size_t
bmalloc(const uint32_t *p, char *fbuf, struct btbl *b)
{
  struct boundary *s = (struct boundary *) fun3d_malloc(1, sizeof(struct boundary));
  struct boundary *f = (struct boundary *) fun3d_malloc(1, sizeof(struct boundary));

  size_t bytes = 0;
  bytes += bmallocl(p[6], p[9], p[3], fbuf, s);
  bytes += bmallocl(p[8], p[11], p[5], (fbuf + bytes), f);

  uint32_t *snfn0 = (uint32_t *) fun3d_malloc(s->f->sz, sizeof(uint32_t));
  uint32_t *snfn1 = (uint32_t *) fun3d_malloc(s->f->sz, sizeof(uint32_t));
  uint32_t *snfn2 = (uint32_t *) fun3d_malloc(s->f->sz, sizeof(uint32_t));

  uint32_t i;
  for(i = 0; i < s->f->sz; i++)
  {
    uint32_t n0 = s->n->nptr[s->f->fptr[i].f0];
    uint32_t n1 = s->n->nptr[s->f->fptr[i].f1];
    uint32_t n2 = s->n->nptr[s->f->fptr[i].f2];

    snfn0[i] = n0;
    snfn1[i] = n1;
    snfn2[i] = n2;
  }

  struct face *fptr = (struct face *) fun3d_malloc(1, sizeof(struct face));
  fptr->n0 = snfn0;
  fptr->n1 = snfn1;
  fptr->n2 = snfn2;

  struct bntbl *sn = (struct bntbl *) fun3d_malloc(1, sizeof(struct bntbl));
  struct bntbl *fn = (struct bntbl *) fun3d_malloc(1, sizeof(struct bntbl));
  struct bftbl *fc = (struct bftbl *) fun3d_malloc(1, sizeof(struct bftbl));

  sn->sz = s->n->sz;
  sn->nptr = s->n->nptr;
  sn->xyz = s->n->xyz;

  fn->sz = f->n->sz;
  fn->nptr = f->n->nptr;
  fn->xyz = f->n->xyz;

  fc->sz = s->f->sz;
  fc->fptr = fptr;

  b->s = sn;
  b->f = fn;
  b->fc = fc;

  fun3d_free(s->f->fptr);
  fun3d_free(s->f);
  fun3d_free(s->n);
  fun3d_free(s);

  fun3d_free(f->f->fptr);
  fun3d_free(f->f);
  fun3d_free(f->n);
  fun3d_free(f);

  return bytes;
}