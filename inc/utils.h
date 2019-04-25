#ifndef __FUN3D_INC_UTILS_H
#define __FUN3D_INC_UTILS_H

#include <stddef.h>
#include <stdint.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

void
fun3d_printf(const uint32_t, const char *, ...);

double
Compute2ndNorm(const size_t, const double *);

void
ComputeAXPY(const size_t, const double, const double *, double *);

void
ComputeNewAXPY(const size_t, const double, const double *, const double *, double *);

double
Normalize(const size_t, double *);

#endif /* __FUN3D_INC_UTILS_H */