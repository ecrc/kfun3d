
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAXSZ_P 32
#define MAXSZ_S 8

static void
p_merge_(unsigned int* base, unsigned int* l1, unsigned int* h1, 
        unsigned int* l2, unsigned int* h2)
{
  unsigned int* buf;
  
  buf = (unsigned int*) malloc(((h1-l1) + (h2-l2)) * sizeof(unsigned int));
 
  unsigned int* i = l1;
  unsigned int* j = l2;
  unsigned int* k = buf;

  for(;(i != h1) && (j != h2);)
  {
    if(*(base+(*(j))) < *(base+(*(i)))) 
      memmove(k++, j++, sizeof(unsigned int));
    else
      memmove(k++, i++, sizeof(unsigned int));
  }

  for(;i != h1;) memmove(k++, i++, sizeof(unsigned int));

  for(;j != h2;) memmove(k++, j++, sizeof(unsigned int));

  memmove(l1, buf, ((h1-l1) + (h2-l2))  * sizeof(unsigned int));

  free(buf);
}

static void
p_sort_ins(unsigned int* base, unsigned int* l, unsigned int* h)
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
p_sort_(unsigned int* base, unsigned int* l, unsigned int* h)
{
  if((h - l) <= MAXSZ_S) p_sort_ins(base, l, h);
  else
  {
    unsigned int* m  = l  + (h - l) / 2;

    p_sort_(base, l, m);
    p_sort_(base, m, h);
    
    p_merge_(base, l, m, m, h);
  }
}

static void
p_sort(unsigned int* base, unsigned int* l, unsigned int* h)
{

  if((h - l) <= MAXSZ_P) p_sort_(base, l, h);
  else
  {
    unsigned int* m  = l  + (h - l) / 2;

#pragma omp task
    p_sort(base, l, m);
    p_sort(base, m, h);
#pragma omp taskwait
 
    p_merge_(base, l, m, m, h);
  }
}

void
p_init(size_t sz, unsigned int* buf, unsigned int* p)
{
  unsigned int i;
  for(i = 0; i < sz; i++) p[i] = i;

  unsigned int* l  = p;
  unsigned int* h  = p + sz;

#pragma omp parallel
  {
#pragma omp single
    {
      p_sort(buf, l, h);
    }
  }
}
