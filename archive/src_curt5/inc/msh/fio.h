
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __FIO_H
#define __FIO_H

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
walkfbuf(const struct wtbl *restrict, void *restrict);

#endif
