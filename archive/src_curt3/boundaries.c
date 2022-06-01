
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

static size_t
fbmalloc(char *restrict fbuf, struct bface *restrict f)
{
  size_t sz = f->sz * 4;

  uint32_t *restrict buf;
  kmalloc(sz, sizeof(uint32_t), (void *) &buf);

  size_t bytes = sz * sizeof(uint32_t);

  struct wtbl w;
  {
    w.l   = fbuf;
    w.h   = fbuf + bytes;
    w.t   = UINT;
    w.sz  = sz;
  }
  walkfbuf(&w, buf);
  
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < f->sz; i++) 
  {
    f->fptr[i].f0 = buf[i] - 1;
    f->fptr[i].f1 = buf[i + f->sz] - 1;
    f->fptr[i].f2 = buf[i + f->sz + f->sz] - 1;
    f->fptr[i].f3 = buf[i + f->sz + f->sz + f->sz] - 1;
  }

  kfree(buf);

  return bytes;
}

static size_t 
nbmalloc(char *restrict fbuf, struct bnode *restrict n)
{
  size_t bytes = n->sz * sizeof(uint32_t);

  struct wtbl w0;
  {
    w0.l   = fbuf;
    w0.h   = fbuf + bytes;
    w0.t   = UINT;
    w0.sz  = n->sz;
  }
  walkfbuf(&w0, n->nptr);

  size_t sz = n->sz * 3;

  size_t bytes_ = sz * sizeof(double);

  double *restrict buf;
  kmalloc(sz, sizeof(double), (void *) &buf);

  struct wtbl w1;
  {
    w1.l   = fbuf + bytes;
    w1.h   = w1.l + bytes_;
    w1.t   = DOUBLE;
    w1.sz  = sz;
  }
  walkfbuf(&w1, buf);
  
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < n->sz; i++) 
  {
    n->nptr[i]--;

    n->xyz->x0[i] = buf[i];
    n->xyz->x1[i] = buf[i + n->sz];
    n->xyz->x2[i] = buf[i + n->sz + n->sz];
  }

  kfree(buf);

  return (bytes + bytes_);
}

static size_t 
bmallocl(const size_t p1, const size_t p2, const size_t p3,
        char *restrict fbuf, struct boundary *restrict b)
{
  // Shift the file pointer to avoid reading the boundaries data,
  // which they are negligible in our core calculations 
  size_t bytes = p3 * 2 * sizeof(uint32_t); 

  struct bface *restrict f;
  kmalloc(1, sizeof(struct bface), (void *) &f);

  f->sz = p1;

  kmalloc(f->sz, sizeof(struct facet), (void *) &f->fptr);

  bytes += fbmalloc((fbuf + bytes), f);

  b->f = f;

  struct bnode *restrict n;
  kmalloc(1, sizeof(struct bnode), (void *) &n);

  n->sz = p2;

  kmalloc(n->sz, sizeof(uint32_t), (void *) &n->nptr);

  kmalloc(1, sizeof(struct xyz), (void *) &n->xyz);
  kmalloc(n->sz, sizeof(double), (void *) &n->xyz->x0);
  kmalloc(n->sz, sizeof(double), (void *) &n->xyz->x1);
  kmalloc(n->sz, sizeof(double), (void *) &n->xyz->x2);

  bytes += nbmalloc((fbuf + bytes), n);

  b->n = n;

  return bytes;
}

size_t
bmalloc(const uint32_t *restrict p, char *restrict fbuf,
        struct btbl *restrict b)
{
  size_t bytes = 0;

  struct boundary *restrict s;
  kmalloc(1, sizeof(struct boundary), (void *) &s);

  bytes += bmallocl(p[6], p[9], p[3], fbuf, s);
  b->s = s;

  struct boundary *restrict f;
  kmalloc(1, sizeof(struct boundary), (void *) &f);

  bytes += bmallocl(p[8], p[11], p[5], (fbuf + bytes), f);
  b->f = f;

  return bytes;
}
