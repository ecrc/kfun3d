#ifndef __FUN3D_INC_LOG_H
#define __FUN3D_INC_LOG_H

#include <string>
#include <unordered_map>
#include "perf.h"
#include "timer.h"
#include "cycle.h"

namespace fun3d
{
  class Log
  {
    typedef struct log_t {
      Cycle *cycles;
      Timer *time;
      Perf *flops;
      Perf *instructions;
    } log;
  private:
    std::unordered_map<std::string, fun3d::Log::log*> dictionary;   
    void DeleteLog(log *data);
    log *CreateLog();
  public:
    Log(const unsigned int, std::string *);
    ~Log();
    void start(const std::string);
    void stop(const std::string);
    void del(const std::string stage);
    void reset(const std::string stage);
    void print(const std::string);
    void print();
  }; /* class log */
}; /* namespace fun3d */

#endif /* __FUN3D_INC_LOG_H */