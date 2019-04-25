#include <cmath>
#include <string>
#include <stdio.h>
#include <utility>
#include <new>
#include "log.h"
#include "memory.h"

fun3d::Log::Log(const unsigned int nstages, std::string *stages)
{
  for(unsigned int i = 0; i < nstages; i++) this->dictionary.insert(std::make_pair(stages[i], CreateLog()));
}

fun3d::Log::~Log()
{
  for(const auto &i : this->dictionary) DeleteLog((this->dictionary[(std::string)i.first]));

  this->dictionary.clear();
}

void fun3d::Log::DeleteLog(fun3d::Log::log *data)
{
  delete data->cycles;
  delete data->time;
  delete data->flops;
  delete data->instructions;

  fun3d::free(data);
}

fun3d::Log::log *fun3d::Log::CreateLog()
{
  log *data = fun3d::malloc<log>();

  data->cycles = new Cycle();
  data->time = new Timer();
  data->flops = new Perf(PERF_TYPE_RAW, 0x8010);
  /* Measures the total instruction count */
  data->instructions = new Perf(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
  return(data);
}

void fun3d::Log::start(const std::string stage)
{
  (this->dictionary[stage])->cycles->start();
  (this->dictionary[stage])->time->start();
  (this->dictionary[stage])->flops->start();
  (this->dictionary[stage])->instructions->start();
}

void fun3d::Log::stop(const std::string stage)
{
  (this->dictionary[stage])->cycles->stop();
  (this->dictionary[stage])->time->stop();
  (this->dictionary[stage])->flops->stop();
  (this->dictionary[stage])->instructions->stop();
}

void fun3d::Log::del(const std::string stage)
{
  DeleteLog((this->dictionary[stage]));

  this->dictionary.erase(stage);
}

void fun3d::Log::print(const std::string stage)
{
  printf("STAGE NAME: %s\n", stage.c_str());
  printf("  >> CPU CYCLES: %lld\n", (this->dictionary[stage])->cycles->get());
  printf("  >> TIME [SEC]: %lf\n", (this->dictionary[stage])->time->get());
  printf("  >> FLOPS: %lld\n", (this->dictionary[stage])->flops->get());

  double flop_sec = (double)((this->dictionary[stage])->flops->get()/(this->dictionary[stage])->time->get());
  if(std::isnan(flop_sec)) flop_sec = 0;

  printf("  >> FLOP/SEC: %lf\n", flop_sec);
  printf("  >> CPU INSTRUCTIONS: %lld\n", (this->dictionary[stage])->instructions->get());
}

void fun3d::Log::print()
{
  for(const auto &i : this->dictionary) this->print(i.first);
}

void fun3d::Log::reset(const std::string stage)
{
  (this->dictionary[stage])->cycles->clear();
  (this->dictionary[stage])->time->clear();
  (this->dictionary[stage])->flops->clear();
  (this->dictionary[stage])->instructions->clear();
}