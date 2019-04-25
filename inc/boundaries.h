#ifndef __FUN3D_INC_BOUNDARIES_H
#define __FUN3D_INC_BOUNDARIES_H

#include "geometry.h"

struct facet {
  unsigned int f0;
  unsigned int f1;
  unsigned int f2;
  unsigned int f3;
};

struct bnode {
  unsigned long int sz;
  unsigned int * nptr;
  struct xyz * xyz;
};

struct bface {
  unsigned long int sz;
  struct facet * fptr;
};

struct boundary {
  struct bface * f;
  struct bnode * n;
};

struct snfptr {
  unsigned int * n0;
  unsigned int * n1;
  unsigned int * n2;
};

#endif /* __FUN3D_INC_BOUNDARIES_H */