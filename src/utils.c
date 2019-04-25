#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include "bench.h"

void
fun3d_printf(const uint32_t c, const char *format, ...)
{
  uint32_t val = 0;
  switch(c)
  {
    case 0: /* ANSI_COLOR_RED */
      val = 31;
    break;
    case 1: /* ANSI_COLOR_GREEN */
      val = 32;
    break;
    case 2: /* ANSI_COLOR_YELLOW */
      val = 33;
    break;
    case 3: /* ANSI_COLOR_BLUE */
      val = 34;
    break;
    case 4: /* ANSI_COLOR_MAGENTA */
      val = 35;
    break;
    case 5: /* ANSI_COLOR_CYAN */
      val = 36;
    break;
    default:
      val = 0;
    break;

  }
  char color[20];
  sprintf(color, "\x1b[%dm", val);

  va_list arg;

  va_start(arg, format);

  fprintf(stdout, "%s", color);
  vfprintf(stdout, format, arg);
  fprintf(stdout, "\x1b[0m");

  va_end(arg);
}

double
Compute2ndNorm(const size_t sz, const double *v)
{
  BENCH start_bench = rdbench();

  double norm = 0.f;

  uint32_t i;
#pragma omp parallel for reduction(+: norm)
  for(i = 0; i < sz; i++) norm += v[i] * v[i];

  fun3d_log(start_bench, KERNEL_BLAS);

  return(sqrt(norm));
}

void
ComputeAXPY(const size_t sz, const double a, const double *x, double *y)
{
  BENCH start_bench = rdbench();

  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < sz; i++)
  {
    /* AXPY */
    const double ax = a * x[i];
    const double axpy = ax + y[i];

    /* Update the vector component */
    y[i] = axpy;
  }

  fun3d_log(start_bench, KERNEL_BLAS);
}

void
ComputeNewAXPY(const size_t sz, const double a, const double *x, const double *y, double *w)
{
  BENCH start_bench = rdbench();

  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < sz; i++)
  {
    /* AXPY */
    const double ax = a * x[i];
    const double axpy = ax + y[i];

    /* Update the vector component */
    w[i] = axpy;
  }

  fun3d_log(start_bench, KERNEL_BLAS);
}

double
Normalize(const size_t sz, double *x)
{
  BENCH start_bench = rdbench();

  double norm = Compute2ndNorm(sz, x);
  
  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < sz; i++) x[i] *= (1.f / norm);

  fun3d_log(start_bench, KERNEL_BLAS);

  return norm;
}