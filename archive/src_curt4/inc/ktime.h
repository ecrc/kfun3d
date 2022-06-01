
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __KTIME_H
#define __KTIME_H

#include <sys/time.h>
#include <time.h>

struct ktime {
//  clock_t c_ctime;
  struct timeval w_ctime;
  double  w_otime;
#ifdef __USE_MAN_FLOPS_COUNTER
  uint64_t flops;
#endif
};

void 
setktime(struct ktime *);

void
printktime(const struct ktime, const char *);

struct kernel_time {
  struct ktime setup;
  struct ktime iguess;
  struct ktime tstep_contr;
  struct ktime flux;
  struct ktime grad;
  struct ktime deltat2;
  struct ktime forces;
  struct ktime fill;
  struct ktime kernel;
  struct ktime overall;
};

void
zero_kernel_time(struct kernel_time *restrict);

void
compute_time(const struct ktime *restrict, struct ktime *restrict);

void
print_kernel_time(const struct kernel_time);

#endif
