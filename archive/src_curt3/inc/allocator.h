
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __ALLOCATOR_H
#define __ALLOCATOR_H

#define MEMALIGN  64

#ifdef    __USE_PGSZ_2MB
#define PGSZ      HBW_PAGESIZE_2MB
#elif     __USE_PGSZ_1GB
#define PGSZ      HBW_PAGESIZE_1GB
#elif     __USE_PGSZ_1GBS
#define PGSZ      HBW_PAGESIZE_1GB_STRICT
#else
#define PGSZ      HBW_PAGESIZE_4KB
#endif


inline void
panic(const char *);

void
kmalloc(const size_t, const size_t, void **);

void
kcalloc(const size_t, const size_t, void **);

void
kfree(void *);

#endif
