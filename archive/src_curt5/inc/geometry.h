
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __GEOMETRY_H
#define __GEOMETRY_H

struct xyzn {
  double * x0;
  double * x1;
  double * x2;
  double * x3;
};

struct xyz {
  double * x0;
  double * x1;
  double * x2;
};

struct weights {
  struct xyz * w0;
  struct xyz * w1;
};

struct edge {
  uint32_t * n0;
  uint32_t * n1;
};

struct etbl {
  size_t sz;
  struct edge * eptr;
  struct xyzn * xyzn;
  struct weights * w;
};

struct ntbl {
  size_t sz;
  struct xyz * xyz;
  double * area;
};

struct facet {
  uint32_t f0;
  uint32_t f1;
  uint32_t f2;
  uint32_t f3;
};

struct bnode {
  size_t sz;
  uint32_t * nptr;
  struct xyz * xyz;
};

struct bface {
  size_t sz;
  struct facet * fptr;
};

struct boundary {
  struct bface * f;
  struct bnode * n;
};

struct snfptr {
  uint32_t * n0;
  uint32_t * n1;
  uint32_t * n2;
};

struct btbl {
  struct boundary * s;
  struct boundary * f;
  struct snfptr * snfptr;
};

struct ctbl {
  size_t bsz2;
  size_t bsz;
  size_t sz;
  uint32_t * ia;
  uint32_t * ja;
  double *aa;
  uint32_t nnz;
  int *ailen;
};

struct stbl {
  uint32_t * part;
  uint32_t * ie;
  uint32_t * snfic;
  uint32_t snfc;
};

struct geometry {
  struct etbl *restrict e; // Edge table
  struct ntbl *restrict n; // Node table
  struct btbl *restrict b; // Boundary table
  struct ctbl *restrict c; // Sparse graph table
  struct stbl *restrict s; // Sub-domain table
};

#endif
