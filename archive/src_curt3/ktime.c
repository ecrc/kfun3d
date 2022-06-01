
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "inc/ktime.h"

/* Set the struct time with different timing criteria given as follows */
inline void
setktime(struct ktime * t)
{
  struct timeval tm;
  gettimeofday(&tm, NULL);

  t->w_ctime = tm;
}

inline static double
get_time_sec(const struct timeval t)
{
  return ((t.tv_sec * 1000.f + t.tv_usec / 1000.f) / 1000.f);
}

/* Print the time in three different format after do the required
 * calculations */
inline void
printktime(const struct ktime st, const char * msg)
{
  struct ktime et;
  setktime(&et);

  struct timeval t;
  timersub(&et.w_ctime, &st.w_ctime, &t);
 
  double tim = get_time_sec(t);

  printf("%s Time: %g\n", msg, tim);
}

inline void
zero_kernel_time(struct kernel_time *restrict kt)
{
  timerclear(&kt->setup.w_ctime);
  timerclear(&kt->iguess.w_ctime);
  timerclear(&kt->tstep_contr.w_ctime);
  timerclear(&kt->flux.w_ctime);
  timerclear(&kt->grad.w_ctime);
#ifdef __USE_MAN_FLOPS_COUNTER
  kt->flux.flops = 0;
  kt->grad.flops = 0;
#endif
  timerclear(&kt->deltat2.w_ctime);
  timerclear(&kt->forces.w_ctime);
  timerclear(&kt->fill.w_ctime);
  timerclear(&kt->kernel.w_ctime);
  timerclear(&kt->overall.w_ctime);
}

inline void
print_kernel_time(const struct kernel_time kt)
{
  double tim = 0.f;

  double ptime = 0.f;

  printf("\n=================START BENCHMARKING=================\n\n");

  tim = get_time_sec(kt.setup.w_ctime);

  printf("Setup Phase Time: ");
  printf("%g\n", tim);

  ptime = tim;

  tim = get_time_sec(kt.iguess.w_ctime);

  printf("Compute Initial Guess: ");
  printf("%g\n", tim);

  tim = get_time_sec(kt.deltat2.w_ctime);
  tim += get_time_sec(kt.tstep_contr.w_ctime);

  ptime += tim;

  printf("Pseudo Time Stepping Time: "); 
  printf("%g\n", tim);

  tim = get_time_sec(kt.grad.w_ctime);

  ptime += tim;

  printf("Gradient Kernel Time: ");
  printf("%g\n", tim);

#ifdef __USE_MAN_FLOPS_COUNTER
  printf("Grad Flops: ");
  printf("%ld\n", kt.grad.flops);
  printf("Grad GFlop/Sec: ");
  printf("%g\n", (kt.grad.flops/tim) / 1e9);
#endif

  tim = get_time_sec(kt.flux.w_ctime);

  ptime += tim;

  printf("Flux Kernel Time: ");
  printf("%g\n", tim);

#ifdef __USE_MAN_FLOPS_COUNTER
  printf("Flux Flops: ");
  printf("%ld\n", kt.flux.flops);
  printf("Flux GFlop/Sec: ");
  printf("%g\n", (kt.flux.flops/tim) / 1e9);
#endif

  tim = get_time_sec(kt.forces.w_ctime);

  ptime += tim;

  printf("Forces Time: ");
  printf("%g\n", tim);

  tim = get_time_sec(kt.fill.w_ctime);

  ptime += tim;

  printf("Fill Jacobian Matrix Time: ");
  printf("%g\n", tim);

  printf("Phy Time: ");
  printf("%g\n", ptime);

  tim = get_time_sec(kt.kernel.w_ctime);

  ptime = tim - ptime;

  printf("Kernel Time: ");
  printf("%g\n", tim);

  printf("PETSc Time: ");
  printf("%g\n", ptime);

  tim = get_time_sec(kt.overall.w_ctime);

  printf("PETSc-FUN3D Time: ");
  printf("%g\n", tim);

  printf("\n==================END BENCHMAKRING==================\n\n");
}

inline void
compute_time(const struct ktime *restrict t0, struct ktime *restrict t1)
{
  struct ktime et;
  setktime(&et);

  struct timeval w_ct;
  timersub(&et.w_ctime, &t0->w_ctime, &w_ct);
  timeradd(&w_ct, &t1->w_ctime, &t1->w_ctime);
}
