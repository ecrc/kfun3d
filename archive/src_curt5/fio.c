/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
  
  NOTE: walkfbuf is a recursive implementation to extract the data 
        parallelism using OpenMP
*/
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <byteswap.h>
#include <omp.h>
#include "inc/msh/fio.h"
#include "inc/allocator.h"

/* NON-USER FUNCTION
 *
 * Swap 64bit double precision 
 * Big-Endian 2 Little-Endian Scheme */
inline static double
__bswap_64d(double val)
{
  double val_;
  size_t sz   = sizeof(double);
  size_t sz_  = sz - 1;

  char * cval   = (char *) &val;
	char * cval_  = (char *) &val_;
  
  /*  Loop over the digits and flip them over:
      swap them from left-to-right,
      so that the most significant is transformed to a least significant
      Big-Endian to Little-Endian
  */
	uint32_t i;
  for(i = 0; i < sz; i++) cval_[i] = cval[sz_- i];

  for(i = 0; i < sz; i++) cval[i]	= cval_[i];

 	return val;
}

static void
walkd(char *restrict l, const char *restrict h, void *restrict data)
{
  double *restrict b = (double *) data;
  // loop over the fbuf and extract the data: each time shift 8-bytes
  for(; l < h; l += sizeof(double))
  {
    // Extract 8-bytes
    double k;
    memcpy(&k, l, sizeof(double));
    // big-endian to little-endian
    // bytesswap.h does not have a byteswap function for double
    // precision
    // data type. As such. we implement our own function.
    *(b++) =  __bswap_64d(k);
  }
}

static void
walkfbufd(char *restrict l, const char *restrict h,
          const size_t sz, char *restrict b)
{
  if(sz <= LIMIT_D) walkd(l, h, b);
  else
  {
    size_t sz_  = sz / 2;   // Middle size address
    size_t sz__ = sz - sz_; // Middle size address
    
    char *restrict m = l + (sz_ * sizeof(double));
    char *restrict k = b + (sz_ * sizeof(double));
    
#pragma omp task
    walkfbufd(l, m, sz_, b);
    walkfbufd(m, h, sz__, k);
#pragma omp taskwait
  }
}

static void
walki(char *restrict l, const char *restrict h, void *restrict data)
{
  uint32_t *restrict b = (uint32_t *) data;

  // Loop over the fbuf and extract the data: each time shift with the
  // size of unsigned int (4-bytes)
  for(; l < h; l += sizeof(uint32_t))
  {
    // Extract 4-bytes from the file buffer
    uint32_t k;
    memcpy(&k, l, sizeof(uint32_t));
    // Swap bytes from big-endian format into little-endian
    *(b++) =  __bswap_32(k); // This a GNU function: bytesswap.h
  }
}

static void
walkfbufi(char *restrict l, const char *restrict h,
          const size_t sz, char *restrict b)
{
  if(sz <= LIMIT_I) walki(l, h, b);
  else // Partition the data
  {
    size_t sz_  = sz / 2;   // The middle address size
    size_t sz__ = sz - sz_; // The size last address size
    
    char *restrict m = l + (sz_ * sizeof(uint32_t));
    char *restrict k = b + (sz_ * sizeof(uint32_t));

#pragma omp task
    walkfbufi(l, m, sz_, b);
    walkfbufi(m, h, sz__, k);
#pragma omp taskwait
  }
}

/* USER FUNCTION
 *
 * The main function that can be called by users to partition the buffer
 * so that we can walk through the file buffer in parallel with the 
 * help of OpenMP task-based parallelism
 *
 * Input:
 *  Low and high address
 *  Data type
 *  Buffer size
 *  Data buffer (for the results)
 * */
void
walkfbuf(const struct wtbl *restrict w, void *restrict b)
{
#pragma omp parallel
  {
#pragma omp single
    {
      switch(w->t)
      {
        default: panic("Unknown data type");
        case UINT: walkfbufi(w->l, w->h, w->sz, b);
        break;
        case DOUBLE: walkfbufd(w->l, w->h, w->sz, b);
        break;
      }
    }
  }
}
