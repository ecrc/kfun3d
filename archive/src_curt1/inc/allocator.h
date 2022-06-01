
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __ALLOCATOR_H
#define __ALLOCATOR_H

#define MEMALIGN  64
#define PGSZ      HBW_PAGESIZE_4KB

inline void
panic(const char *);

void
kmalloc(const size_t, const size_t, void **);

void
kcalloc(const size_t, const size_t, void **);

void
kfree(void *);

#endif
