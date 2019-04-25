#ifndef __FUN3D_INC_PERF_H
#define __FUN3D_INC_PERF_H

#include <linux/perf_event.h> /* struct perf_event_attr */
#include <asm/types.h> /* __u32 and __u64 */
#include <sys/types.h> /* pid_t */

namespace fun3d
{
  class Perf
  {
    typedef struct perf_t {
      long fd;
      long long count;
      unsigned long long int total;
    } perf;
  private:
    perf *data;
    const pid_t PID = 0;
    const int CPU = -1;
    const int GROUP_FD = -1;
    const unsigned long FLAGS = 0;
    const __u32 SIZE = sizeof(perf_event_attr); /* Size of attribute structure */
    const __u64 DISABLED = 1; /* Off by default */
    const __u64 EXCLUDE_KERNEL = 1;/* Do not count the instruction the kernel executes */
    const __u64 EXCLUDE_HV = 1; /* Do not count the instruction the hypervisor executes */
    void enable();
    void disable();
    void reset();
    void set();
  public:
    Perf(const __u32, const __u64);
    ~Perf();
    void start();
    void stop();
    unsigned long long int get();
    void clear();
  }; /* class log */
}; /* namespace fun3d */

#endif /* __FUN3D_INC_PERF_H */