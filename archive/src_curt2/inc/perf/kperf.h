
#ifndef __KPERF_H
#define __KPERF_H

#define FREQ 1293344000

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
