#include <string.h>
#include <stdlib.h>
#include <byteswap.h>
#include <omp.h>

namespace fun3d
{
  const unsigned int THRESHOLD = 100;

  template<typename T>
  T __bswap(T val)
  {
    /* unsigned int */
    if(sizeof(T) == 4) return(__bswap_32((unsigned int)val));
    else
    {
      T val_;
      const size_t sz = sizeof(T);
      const size_t sz_ = sz - 1;

      char * cval   = (char *) &val;
      char * cval_  = (char *) &val_;
      
      for(unsigned int i = 0; i < sz; i++) cval_[i] = cval[sz_- i];
      for(unsigned int i = 0; i < sz; i++) cval[i] = cval_[i];

      return val;
    }
  }

  template<typename T>
  void walk(char *l, const char *h, T *b)
  {
    for(; l < h; l += sizeof(T))
    {
      T k;
      memcpy(&k, l, sizeof(T));

      *(b++) =  __bswap<T>(k);
    }
  }

  template<typename T>
  void walk(char *l, const char *h, const size_t sz, T *b)
  {
    if(sz <= THRESHOLD) walk<T>(l, h, b);
    else
    {
      const size_t sz_  = sz / 2;
      
      char *m = l + (sz_ * sizeof(T));
      char *k = (char *)b + (sz_ * sizeof(T));
    
#pragma omp task
      walk<T>(l, m, sz_, b);
      walk<T>(m, h, (sz-sz_), (T *)k);
#pragma omp taskwait
    }
  }

  template<typename T>
  void walkfbuf(char *l, const char *h, const size_t sz, T *b)
  {
#pragma omp parallel
    {
#pragma omp single
      {
        walk<T>(l, h, sz, b);
      }
    }
  }
};