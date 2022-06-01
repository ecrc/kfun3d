
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

/* *********
 * Kernel malloc function: This is a generic routine to allocate 
 * the memory block based on their type. The routine switches the 
 * allocation routine during the runtime, based upon a macro that 
 * specifies the requested allocation routine.
 * *********
 * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inc/allocator.h"

#if defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW)
#include <hbwmalloc.h>
#endif
/* Stop the execution sequence and print an error message in the 
 * standard error */
inline void
panic(const char * s)
{
  fprintf(stderr, "ERROR:  %s\n", s);
  exit(EXIT_FAILURE);
}

/* Internal implementation of malloc */
inline void
kmalloc(const size_t sz, const size_t nbytes, void **restrict buf)
{
  if(sz <= 0) panic("allocator.c: size must be > 0");

  if(nbytes <= 0)
    panic("allocator.c: number of bytes must be > 0");

#if defined(__USE_INTEL_ICC) && defined(__USE_INTEL_MALLOC)
  (* buf) = _mm_malloc(sz * nbytes, MEMALIGN);
#elif defined(__USE_POSIX) && defined(__USE_POSIX_MEMMEMALIGN)
  int err = posix_memalign(buf, MEMALIGN, sz * nbytes);

  if(err) panic("allocator.c: malloc error");
#elif defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW)
  int err = hbw_posix_memalign_psize(buf, MEMALIGN, sz * nbytes, PGSZ);

  if(err) panic("allocator.c: malloc error");
#else
  (* buf) = malloc(sz * nbytes);
#endif

  if((* buf) == NULL) panic("allocator.c: malloc error");
}

/* Internal implementation of calloc or malloc + memset */
inline void
kcalloc(const size_t sz, const size_t nbytes, void **restrict buf)
{
  if(sz <= 0) panic("allocator.c: size must be > 0");

  if(nbytes <= 0)
    panic("allocator.c: number of bytes must be > 0");

#if defined(__USE_INTEL_ICC) && defined(__USE_INTEL_MALLOC)
  (* buf) = _mm_malloc(sz * nbytes, MEMALIGN);

  if((* buf) == NULL) panic("allocator.c: calloc error");
  
  memset((* buf), 0, sz * nbytes);

#elif defined(__USE_POSIX) && defined(__USE_POSIX_MEMMEMALIGN)
  int err = posix_memalign(buf, MEMALIGN, sz * nbytes);

  if(err) panic("allocator.c: calloc error");
  if((* buf) == NULL) panic("allocator.c: calloc error");

  memset((* buf), 0, sz * nbytes);
#elif defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW)
  int err = hbw_posix_memalign_psize(buf, MEMALIGN, sz * nbytes, PGSZ);

  if(err) panic("allocator.c: calloc error");

  if((* buf) == NULL) panic("allocator.c: calloc error");

  memset((* buf), 0, sz * nbytes); 
#else
  (* buf) = calloc(sz, nbytes);

  if((* buf) == NULL) panic("allocator.c: calloc error");
#endif
}

/* Internal implementation of free */
inline void
kfree(void * buf)
{
#if defined(__USE_INTEL_ICC) && defined(__USE_INTEL_MALLOC)
  _mm_free(buf);
#elif defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW)
  hbw_free(buf);
#else
  free(buf);
#endif
}
