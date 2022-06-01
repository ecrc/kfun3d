
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define THRESHOLD 8

static void
p_merge(unsigned int* base, unsigned int* l1, unsigned int* h1, 
        unsigned int* l2, unsigned int* h2, unsigned int* buf)
{ 
  unsigned int* i = l1;
  unsigned int* j = l2;
  unsigned int* k = buf;

  for(;(i != h1) && (j != h2);)
  {
    if(*(base+(*(j))) < *(base+(*(i)))) memcpy(k++, j++, sizeof(unsigned int));
    else memcpy(k++, i++, sizeof(unsigned int));
  }

  for(;i != h1;) memcpy(k++, i++, sizeof(unsigned int));

  for(;j != h2;) memcpy(k++, j++, sizeof(unsigned int));

  memcpy(l1, buf, ((h1-l1) + (h2-l2))  * sizeof(unsigned int));
}

static void
p_sort_(unsigned int* base, unsigned int* l, unsigned int* h)
{
  unsigned int t;

  unsigned int* i;
  for(i = l; i < h; i++)
  {  
    unsigned int* elm = base + (*(i));

    unsigned int* j;
    for(j = i+1; j < h; j++)
    {
      if(*elm > *(base + (*(j))))
      {
        unsigned int t = *(i);
        *(i)  = *(j); 
        *(j)  = t;

        elm   = base + (*(i));
      }
    }
  }
}

static void
p_sort(unsigned int* base, unsigned int* l, unsigned int* h, unsigned int* p_)
{

  if((h - l) <= THRESHOLD) p_sort_(base, l, h);
  else
  {
    unsigned int* m  = l  + (h - l) / 2;
    unsigned int* m_ = p_ + (m - l);

#pragma omp task
    p_sort(base, l, m, p_);
    p_sort(base, m, h, m_);
#pragma omp taskwait
 
    p_merge(base, l, m, m, h, p_);
  }
}

void
p_init(size_t sz, unsigned int* buf, unsigned int* p)
{
  unsigned int* p_;
  
  p_ = (unsigned int*) _mm_malloc(sz * sizeof(unsigned int), 64);

  unsigned int i;
  for(i = 0; i < sz; i++) p[i] = i;

  unsigned int* l  = p;
  unsigned int* h  = p + sz;

#pragma omp parallel
  {
#pragma omp single
    {
      p_sort(buf, l, h, p_);
    }
  }

  _mm_free(p_);
}
