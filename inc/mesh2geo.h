#ifndef __FUN3D_INC_MESH2GEO_H
#define __FUN3D_INC_MESH2GEO_H

#include "geometry.h"

void
m2csr(struct geometry *);

void
isymbolic(struct geometry *);

void
isubdomain(struct geometry *);

void
ifcoloring(struct geometry *);

void
wmalloc(struct geometry *);

#ifdef BUCKET_SORT
void
iecoloring(struct geometry *);
#endif

void
iguess(struct geometry *);

#endif  /* __FUN3D_INC_MESH2GEO_H */