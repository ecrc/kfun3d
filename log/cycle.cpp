#ifndef __INTEL_COMPILER
#ifndef __llvm__
#ifndef __clang__
#include <x86intrin.h> /* ONLY GCC */
#endif /* INTEL */
#endif /* LLVM */
#endif /* CLANG */
#include "cycle.h"
#include "memory.h"

fun3d::Cycle::Cycle()
{
  this->data = fun3d::calloc<cycle>();
  this->data->count = 0;
  this->data->total = 0;
}

fun3d::Cycle::~Cycle()
{
  fun3d::free(this->data);
}

void fun3d::Cycle::clear()
{
  this->data->count = 0;
  this->data->total = 0;
}


void fun3d::Cycle::reset()
{
  this->data->count = 0;
}

void fun3d::Cycle::set()
{
  this->data->count = __rdtsc();
}

void fun3d::Cycle::start()
{
  reset();
  set();
}

void fun3d::Cycle::stop()
{
  long long int count = this->data->count;
  set();
  
  this->data->count -= count;
  this->data->total += this->data->count;
}

unsigned long long int fun3d::Cycle::get()
{
  return(this->data->total);
}