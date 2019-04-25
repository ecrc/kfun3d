#ifndef __FUN3D_INC_MESH_H
#define __FUN3D_INC_MESH_H

#include "geometry.h"

size_t
emalloc(char *, struct etbl *);

size_t
nmalloc(char *, struct ntbl *);

size_t
bmalloc(const unsigned int *, char *, struct btbl *);

#endif  /* __FUN3D_INC_MESH_H */