#include <asm/types.h> /* __u32 and __u64 */
#include <assert.h> /* assert() */
#include <unistd.h> /* syscall */
#include <asm/unistd.h> /* __NR_perf_event_open */
#include <unistd.h> /* read() */
#include <linux/perf_event.h> /* ERF_EVENT_IOC_* and struct perf_event_attr */
#include <sys/ioctl.h> /* ioctl() */
#include <stdio.h> /* fprintf() */
#include "perf.h"
#include "memory.h"

fun3d::Perf::Perf(const __u32 type /* Type of event */, const __u64 config  /* Type-specific configuration */)
{
  struct perf_event_attr *hw_event = fun3d::calloc<struct perf_event_attr>();
  hw_event->size = SIZE;
  hw_event->type = type;
  hw_event->config = config;
  hw_event->disabled = DISABLED;
  hw_event->exclude_kernel = EXCLUDE_KERNEL;
  hw_event->exclude_hv = EXCLUDE_HV;

  this->data = fun3d::calloc<perf>();
  this->data->count = 0;
  this->data->total = 0;
  this->data->fd = syscall(__NR_perf_event_open, hw_event, PID, CPU, GROUP_FD, FLAGS);
  if(this->data->fd == -1)
  {
    fprintf(stderr, ">> -1: means that you do not have permission to read the performance event file.\n");
    fprintf(stderr, ">> Change the value of: /proc/sys/kernel/perf_event_paranoid to be any value less than 1.\n");
    fprintf(stderr, ">> You need sudo permission in order to do so.\n");
    assert(this->data->fd != -1);  
  }
  
  fun3d::free(hw_event);
}

fun3d::Perf::~Perf()
{
  close(this->data->fd);
  fun3d::free(this->data);
}

void fun3d::Perf::clear()
{
  reset();
  this->data->count = 0;
  this->data->total = 0;
}

void fun3d::Perf::enable()
{
  ioctl(this->data->fd, PERF_EVENT_IOC_ENABLE, 0);
}

void fun3d::Perf::reset()
{
  ioctl(this->data->fd, PERF_EVENT_IOC_RESET, 0);
}

void fun3d::Perf::disable()
{
  ioctl(this->data->fd, PERF_EVENT_IOC_DISABLE, 0);
}

void fun3d::Perf::set()
{
  this->data->count = 0;
  read(this->data->fd, &this->data->count, sizeof(long long));
}

void fun3d::Perf::start()
{
  reset();
  enable();
}

void fun3d::Perf::stop()
{
  disable();
  set();
  this->data->total += this->data->count;
}

unsigned long long int fun3d::Perf::get()
{
  return(this->data->total);
}