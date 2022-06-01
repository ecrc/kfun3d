
#ifndef __KPERF_H
#define __KPERF_H

#ifdef  __USE_KNL_72
#define FREQ  1500240000
#elif   __USE_KNL_68
#define FREQ  1400000000 /* Needs to be validated */
#elif   __USE_KNL_64
#define FREQ  1293344000
#elif   __USE_SKY_58_DUAL_SOCKET
#define FREQ  2095092000
#else
#include <mkl.h>
#define FREQ  (uint64_t) (mkl_get_clocks_frequency() * 1e9)
#endif

struct ctr {
  uint64_t cycles;
  struct tot tot;
};

struct ctrs {
  struct ctr setup;
  struct ctr timestep;
  struct ctr flux;
  struct ctr grad;
  struct ctr jacobian;
  struct ctr forces;
};

struct perf_counters {
  struct fd fd;
  struct ctrs * ctrs;
};

void
create_perf_counters_tbl(struct ctrs *);

void
finalize_perf_counters(const struct ctrs);

#endif
