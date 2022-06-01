
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __KTIME_H
#define __KTIME_H

#include <sys/time.h>
#include <time.h>

struct ktime {
  clock_t c_ctime;
  struct timeval w_ctime;
  double  w_otime;
};

void 
setktime(struct ktime *);

void
printktime(const struct ktime, const char *);

#endif
