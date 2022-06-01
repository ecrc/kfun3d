
#ifndef __KERNEL_H
#define __KERNEL_H

#define MAX_TS      15
#define FNORM_RATIO 1e+7
#define MAX_CFL     1e+5

/* Time step context */
struct ts {
  Vec r; // Residual vector for the FormFunction of PETSc
  Vec q; // State (Q) vector for the computed solution
  /*
    Courant–Friedrichs–Lewy stability restriction condition
  */
  double cfl;
  double cfl_init;
  /*
    Function 2nd norm
  */
  double fnorm; 
  double fnorm_init;

  double * cdt;
};

struct ctx {
  const struct geometry * g;
  const struct ivals * iv;
  struct xyz * grad;
  struct ts * ts;
  Vec q;
  struct kernel_time * t;
#ifdef __USE_HW_COUNTER
  struct perf_counters * perf_counters;
#endif
};

struct fill {
  const double * q;
  const struct geometry * g;
  const struct ivals * iv;
  const struct ts * ts;
  Mat A;
  struct kernel_time *restrict t;
#ifdef __USE_HW_COUNTER
  struct perf_counters * perf_counters;
#endif
};

int
fill_mat(struct fill *restrict);

int
ffunc(SNES, Vec, Vec, void *restrict);

int
jfunc(SNES, Vec, Mat, Mat, void *restrict);

int
update(SNES, struct ctx *restrict);

#endif
