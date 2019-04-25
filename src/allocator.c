#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "allocator.h"

#ifdef POSIX_HBW
#include <hbwmalloc.h>
#endif

// HBW_PAGESIZE_2MB
// HBW_PAGESIZE_1GB
// HBW_PAGESIZE_1GB_STRICT
// HBW_PAGESIZE_4KB

#define MEMORY_ALIGNMENT 64

void *
fun3d_malloc(const size_t num, const size_t size)
{
  const size_t sz = size * num;
  void *memptr = NULL;
#ifdef POSIX_HBW
  assert(hbw_posix_memalign_psize(&memptr, (size_t) MEMORY_ALIGNMENT, sz, HBW_PAGESIZE_4KB) == 0);
#else
  assert(posix_memalign(&memptr, (size_t) MEMORY_ALIGNMENT, sz) == 0);
#endif
  return(memptr);
}

void *
fun3d_calloc(const size_t num, const size_t size)
{
  const size_t sz = size * num;
  void *memptr = NULL;
#ifdef POSIX_HBW
  assert(hbw_posix_memalign_psize(&memptr, (size_t) MEMORY_ALIGNMENT, sz, HBW_PAGESIZE_4KB) == 0);
#else
  assert(posix_memalign(&memptr, (size_t) MEMORY_ALIGNMENT, sz) == 0);
#endif
  assert(memset(memptr, 0, sz) == memptr);
  return(memptr);
}

void
fun3d_free(void *ptr)
{
#ifdef POSIX_HBW
  hbw_free(ptr);
#else
  free(ptr);
#endif
}