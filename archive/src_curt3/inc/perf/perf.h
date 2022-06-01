
#ifndef __PERF_H
#define __PERF_H

#define IMCSZ 6 // Number of the memory controllers
#define EDCSZ 8 // Number of the EDC controllers

/* File descriptor struct table */

struct fd {
  int imcR[IMCSZ];
  int imcW[IMCSZ];
  int edcR[EDCSZ];
  int edcW[EDCSZ];
};

void
perf_init(struct fd *);

struct counters {
  uint64_t imcR[IMCSZ];
  uint64_t imcW[IMCSZ];
  uint64_t edcR[EDCSZ];
  uint64_t edcW[EDCSZ];
};

void
perf_read(const struct fd, struct counters *);

struct tot {
  double imcR;
  double imcW;
  double edcR;
  double edcW;
};

void
perf_calc(const struct counters, const struct counters, struct tot *);

#endif
