
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef unsigned int uint;

#define LIMIT 500
#define XCHG(i, j)  {uint t = i; i = j; j = t;}

static void 
imerge_(uint * b, uint * l1, uint * h1, uint * l2, uint * h2, uint * l_) 
{
  for(;(l1 != h1) && (l2 != h2);) 
  {
    if(*(b + *(l2)) < *(b + *(l1))) memcpy(l_++, l2++, sizeof(uint));
    else memcpy(l_++, l1++, sizeof(uint));
  }
  memcpy(l_, l1, (h1 - l1) * sizeof(uint));
  memcpy(l_, l2, (h2 - l2) * sizeof(uint));
}

static void
isort_(uint * b, uint * l, uint * h)
{
  uint * i;
  for(i = l; i < h; i++)
  {
    uint * elm = b + *(i);

    uint  * j;
    for(j = (i+1); j < h; j++)
    {
      if(*(elm) > *(b + *(j)))
      {
        XCHG(*(i), *(j));
        elm = b + *(i);
      }
    }
  }
}

static void
isort(uint * b, uint * l, uint * h, uint * l_, uint is_in_place) 
{
  if((h - l) <= LIMIT)
  {
    isort_(b, l, h);
    if(!is_in_place) memcpy(l_, l, (h - l) * sizeof(uint));
  } 
  else
  {
    uint* m  = l   + (h - l) / 2;
    uint* m_ = l_  + (m - l);
    uint* h_ = l_  + (h - l);
#pragma omp task
    isort(b, l, m, l_, !is_in_place);
    isort(b, m, h, m_, !is_in_place);
#pragma omp taskwait

    if(is_in_place) imerge_(b, l_, m_, m_, h_, l);
    else imerge_(b, l, m, m, h, l_);
  }
}


uint
imain(uint sz, uint * b, uint * l)
{
  uint i;
  for(i = 0; i < sz; i++) l[i] = i;

  uint * l_ = (uint *) _mm_malloc(sz * sizeof(uint), 64);

#pragma omp parallel
  {
#pragma omp single
    {
      isort(b, l, (l + sz), l_, 1);
    }
  }
  
  _mm_free(l_);

  return 1;
}
