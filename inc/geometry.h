#ifndef __FUN3D_INC_GEOMETRY_H
#define __FUN3D_INC_GEOMETRY_H

struct xyzn {
  double *x0;
  double *x1;
  double *x2;
  double *x3;
};

struct xyz {
  double *x0;
  double *x1;
  double *x2;
};

typedef struct xyz GRADIENT;

struct edge {
  unsigned int * n0;
  unsigned int * n1;
};

struct face {
  unsigned int * n0;
  unsigned int * n1;
  unsigned int * n2;
};

struct weights {
  struct xyz * w0;
  struct xyz * w1;
};

struct etbl {
  unsigned long int sz;
  struct edge * eptr;
  struct xyzn * xyzn;
  struct weights * w;
};

struct ntbl {
  unsigned long int sz;
  struct xyz *xyz;
  unsigned int *part;
  double *cdt;
  /* double *area; */ /* Deprecated in the newer version */
};

struct stbl {
  unsigned long int sz;
  unsigned int *i;
};

struct bntbl {
  unsigned long int sz;
  unsigned int *nptr;
  struct xyz *xyz;
};

struct bftbl {
  unsigned long int sz;
  struct face *fptr;
};

struct btbl {
  struct bntbl *s;
  struct bntbl *f;
  struct bftbl *fc;
};

//struct mat {
//  double *a;
//  unsigned int *i;
//  unsigned int *j;
//  unsigned int *d;
//  unsigned int *w;
//};

struct ctbl {
  unsigned int b;
  unsigned int b2;
  unsigned long int sz;
  //struct mat *mat;
};

struct qtbl {
  double *q;
  /*
  double *p;
  double *v;
  double *u;
  double *w;
  */
};

typedef struct geometry {
  struct etbl *e; // Edge table
  struct ntbl *n; // Node table
  struct btbl *b; // Boundary table
  struct ctbl *c; // Sparse graph table
  struct stbl *s; // Sub-domain table: threading via METIS
  struct stbl *t; // Sub-domain table: threading via coloring
  struct qtbl *q; // Q Vector -- State Vector
  void *matrix;
} GEOMETRY;

#endif /* __FUN3D_INC_GEOMETRY_H */