#ifndef __FUN3D_INC_MEMORY_H
#define __FUN3D_INC_MEMORY_H

#include <assert.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>

#ifdef POSIX_HBW
#include <hbwmalloc.h>
#endif

#if 0
#define PAGE_SZ   HBW_PAGESIZE_2MB
#define PAGE_SZ   HBW_PAGESIZE_1GB
#define PAGE_SZ   HBW_PAGESIZE_1GB_STRICT
#else
#define PAGE_SZ   HBW_PAGESIZE_4KB
#endif

namespace fun3d
{
  const unsigned int MEMORY_ALIGNMENT = 64;

  template<typename T>
  T *malloc(const size_t num)
  {
    const size_t sz = sizeof(T) * num;
    T *memptr = NULL;
#ifdef POSIX_HBW
    assert(hbw_posix_memalign_psize(&memptr, (size_t) MEMORY_ALIGNMENT, sz, PAGE_SZ) == 0);
#else
    assert(posix_memalign((void**) &memptr, (size_t) MEMORY_ALIGNMENT, sz) == 0);
#endif
    return(memptr);
  }

  template<typename T>
  T *malloc()
  {
    return(malloc<T>(1));
  }

  template<typename T>
  T *calloc(const size_t num)
  {
    T *memptr = malloc<T>(num);
    const size_t sz = sizeof(T) * num;
    assert(memset(memptr, 0, sz) == memptr);
    return(memptr);
  }

  template<typename T>
  T *calloc()
  {
    return(calloc<T>(1));
  }

  inline static void free(void *ptr)
  {
#ifdef POSIX_HBW
    hbw_free(ptr);
#else
    std::free(ptr);
#endif
  }
};

#endif /* __FUN3D_INC_MEMORY_H */