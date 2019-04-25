#ifndef __FUN3D_INC_ALLOCATOR_H
#define __FUN3D_INC_ALLOCATOR_H

#include <stddef.h>

void *
fun3d_malloc(const size_t, const size_t);

void *
fun3d_calloc(const size_t, const size_t);

void
fun3d_free(void *);

#endif /* __FUN3D_INC_ALLOCATOR_H */