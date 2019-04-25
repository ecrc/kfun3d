#ifndef __FUN3D_INC_INDEX_H
#define __FUN3D_INC_INDEX_H

#include <stddef.h>

/* Maximum search limit */
#define LIMIT 500
#define XCHG(i, j)  {unsigned int t = i; i = j; j = t;}

void
imain(const size_t, unsigned int *, unsigned int *);

#endif /* __FUN3D_INC_INDEX_H */