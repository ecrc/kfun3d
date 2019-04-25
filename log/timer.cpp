#include <time.h>
#include "timer.h"
#include "memory.h"

fun3d::Timer::Timer()
{
  this->data = fun3d::calloc<timer>();
  this->data->count.tv_sec = 0;
  this->data->count.tv_nsec = 0;
  this->data->total.tv_sec = 0;
  this->data->total.tv_nsec = 0;
}

fun3d::Timer::~Timer()
{
  fun3d::free(this->data);
}

void fun3d::Timer::clear()
{
  this->data->count.tv_sec = 0;
  this->data->count.tv_nsec = 0;
  this->data->total.tv_sec = 0;
  this->data->total.tv_nsec = 0;
}

void fun3d::Timer::reset()
{
  this->data->count.tv_sec = 0;
  this->data->count.tv_nsec = 0;
}

void fun3d::Timer::set()
{
  clock_gettime(CLOCK_MONOTONIC, &this->data->count);
}

void fun3d::Timer::start()
{
  reset();
  set();
}

void fun3d::Timer::stop()
{
  struct timespec count = this->data->count;
  set();
  
  this->data->count.tv_sec -= count.tv_sec;
  this->data->count.tv_nsec -= count.tv_nsec;

  this->data->total.tv_sec += this->data->count.tv_sec;
  this->data->total.tv_nsec += this->data->count.tv_nsec;
}

double fun3d::Timer::get()
{
  return((double)(((double)this->data->total.tv_nsec / 1000000000.f) + (double)this->data->total.tv_sec));
}