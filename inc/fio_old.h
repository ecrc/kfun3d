#ifndef __FUN3D_INC_FIO_H
#define __FUN3D_INC_FIO_H

#include <stddef.h>

#define LIMIT_D 100
#define LIMIT_I 250

enum dtype { UINT, DOUBLE };

struct wtbl {
  char * l;
  char * h;
  enum dtype t;
  size_t sz;
};

void
walkfbuf(const struct wtbl *, void *);

#endif /* __FUN3D_INC_FIO_H */