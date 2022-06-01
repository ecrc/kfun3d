
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "inc/allocator.h"
#include "inc/msh/index.h"

/* Merge the two sorted lists using temporary array */
static void 
imerge_(uint32_t *restrict b, uint32_t *restrict l1,
        uint32_t *restrict h1, uint32_t *restrict l2,
        uint32_t *restrict h2, uint32_t *restrict l_)
{
  __assume_aligned((b), MEMALIGN);
  __assume_aligned((l1), MEMALIGN);
  __assume_aligned((h1), MEMALIGN);
  __assume_aligned((l2), MEMALIGN);
  __assume_aligned((h2), MEMALIGN);
  __assume_aligned((l_), MEMALIGN);

  for(;(l1 != h1) && (l2 != h2);) 
  {
    if(*(b + *(l2)) < *(b + *(l1)))
      memcpy(l_++, l2++, sizeof(uint32_t));
    else
      memcpy(l_++, l1++, sizeof(uint32_t));
  }

  /* Move the leftover data */
  memcpy(l_, l1, (h1 - l1) * sizeof(uint32_t));
  memcpy(l_, l2, (h2 - l2) * sizeof(uint32_t));
}

/* Sequential sort used to sort a small array list */
static void
srt(uint32_t *restrict b, uint32_t *restrict l, uint32_t *restrict h)
{
  __assume_aligned((b), MEMALIGN);
  __assume_aligned((l), MEMALIGN);
  __assume_aligned((h), MEMALIGN);

  uint32_t * i;
  for(i = l; i < h; i++)
  {
    uint32_t * elm = b + *(i);

    uint32_t  * j;
    for(j = (i+1); j < h; j++)
    {
      if(*(elm) > *(b + *(j)))
      {
        XCHG(*(i), *(j)); // Swap the elements
        elm = b + *(i); // Change the base index
      }
    }
  }
}

/* Index sort: Find the index of an array that gives a 
 * sorted array */
static void
isort(uint32_t *restrict b, uint32_t *restrict l,
      uint32_t *restrict h, uint32_t *restrict l_,
      uint8_t flg)
{
  __assume_aligned((b), MEMALIGN);
  __assume_aligned((l), MEMALIGN);
  __assume_aligned((h), MEMALIGN);
  __assume_aligned((l_), MEMALIGN);

  if((h - l) <= LIMIT) // Maximum sort limit
  {
    srt(b, l, h);
    if(!flg) memcpy(l_, l, (h - l) * sizeof(uint32_t));
  } 
  else
  {
    /* Use merge-sort to divide the array into subarrays */
    uint32_t * m  = l   + (h - l) / 2;
    uint32_t * m_ = l_  + (m - l);
    uint32_t * h_ = l_  + (h - l);

    /* Launch OpenMP task-based parallelism for each half of 
     * the array */
#pragma omp task
    isort(b, l, m, l_, !flg);
    isort(b, m, h, m_, !flg);
#pragma omp taskwait

    /* Merge the sorted sequences */
    if(flg) imerge_(b, l_, m_, m_, h_, l);
    else imerge_(b, l, m, m, h, l_);
  }
}

/* Initialize the find indexing function 
 * sz: size of the original array
 * b: the base array
 * l: the low address of the output permutation array */
void
imain(const size_t sz, uint32_t *restrict b, uint32_t *restrict d)
{
  /* initialize the output array 
   * Assume that the original list is already sorted */
  uint32_t i;
  for(i = 0; i < sz; i++) d[i] = i;

  /* A temporary buffer used for sorting */
  uint32_t *restrict l;
  kmalloc(sz, sizeof(uint32_t), (void *) &l);

  /* Launch the parallel region and start sorting based on an
   * index list */
#pragma omp parallel
  {
#pragma omp single
    {
      isort(b, d, (d + sz), l, 1); // Index sort
    }
  }
  
  kfree(l);
}
