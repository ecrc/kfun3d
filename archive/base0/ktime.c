
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ktime.h"

/* A local private function that read the current time of the day;
 * the user time */
inline static struct timeval
read_ctime()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return t;
}

/* Set the struct time with different timing criteria given as follows */
void
setktime(struct ktime * t)
{
  t->c_ctime = clock();         // CPU time
  t->w_ctime = read_ctime();    // User time: C native function
  t->w_otime = omp_get_wtime(); // User time: OpenMP time
}

/* Print the time in three different format after do the required
 * calculations */
void
printktime(const struct ktime st, const char * msg)
{
  printf("\t+++++ Kernel Type: %s\n", msg);
  printf("\t\t+++++ Time Elapsed: \n");
 
  struct ktime et;
  setktime(&et);

  /* OpenMP time */
  double omp_t = et.w_otime - st.w_otime;
  printf("\t\t>>>> OMP Wall Time: %g\n", omp_t);

  /* Wall clock time  */
  struct timeval w_ct;
  timersub(&et.w_ctime, &st.w_ctime, &w_ct);
  double w_ct_ = w_ct.tv_sec * 1000.f + w_ct.tv_usec / 1000.f;
  printf("\t\t>>>> C Wall Time: %g\n", w_ct_ / 1000.f);

  /* CPU time */
  double c_ct = et.c_ctime - st.c_ctime;
  printf("\t\t>>>> CPU Time: %g\n", c_ct / CLOCKS_PER_SEC);
}
