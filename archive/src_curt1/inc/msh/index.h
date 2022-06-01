
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __INDEX_H
#define __INDEX_H

/* Maximum search limit */
#define LIMIT 500
#define XCHG(i, j)  {uint32_t t = i; i = j; j = t;}

void
imain(const size_t, uint32_t *, uint32_t *);

#endif
