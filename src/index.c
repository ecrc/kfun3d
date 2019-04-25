#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "allocator.h"
#include "index.h"

/* Merge the two sorted lists using temporary array */
static void 
imerge(
  uint32_t *b,
  uint32_t *l1,
  uint32_t *h1,
  uint32_t *l2,
  uint32_t *h2,
  uint32_t *l_)
{
  for(;(l1 != h1) && (l2 != h2);) 
  {
    if(*(b + *(l2)) < *(b + *(l1)))
      memcpy(l_++, l2++, sizeof(uint32_t));
    else
      memcpy(l_++, l1++, sizeof(uint32_t));
  }

  /* Move the leftover data */
  memcpy(l_, l1, (size_t) (h1 - l1) * sizeof(uint32_t));
  memcpy(l_, l2, (size_t) (h2 - l2) * sizeof(uint32_t));
}

/* Sequential sort used to sort a small array list */
static void
srt(uint32_t *b, uint32_t *l, uint32_t *h)
{
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
isort(
  uint32_t *b,
  uint32_t *l,
  uint32_t *h,
  uint32_t *l_,
  uint8_t flg)
{
  if((h - l) <= LIMIT) // Maximum sort limit
  {
    srt(b, l, h);
    if(!flg) memcpy(l_, l, (size_t) (h - l) * sizeof(uint32_t));
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
    isort(b, l, m, l_, (unsigned char) !flg);
    isort(b, m, h, m_, (unsigned char) !flg);
#pragma omp taskwait

    /* Merge the sorted sequences */
    if(flg) imerge(b, l_, m_, m_, h_, l);
    else imerge(b, l, m, m, h, l_);
  }
}

/* Initialize the find indexing function 
 * sz: size of the original array
 * b: the base array
 * l: the low address of the output permutation array */
void
imain(const size_t sz, uint32_t *b, uint32_t *d)
{
  /* initialize the output array 
   * Assume that the original list is already sorted */
  uint32_t i;
  for(i = 0; i < sz; i++) d[i] = i;

  /* A temporary buffer used for sorting */
  uint32_t *l = (uint32_t *) fun3d_malloc(sz, sizeof(uint32_t));

  /* Launch the parallel region and start sorting based on an
   * index list */
#pragma omp parallel
  {
#pragma omp single
    {
      isort(b, d, (d + sz), l, 1); // Index sort
    }
  }
  
  fun3d_free(l);
}