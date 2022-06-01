
#include <stdio.h>
#include <stdint.h>
#include <perf.h>
#include <kperf.h>

void
create_perf_counters_tbl(struct ctrs * ctrs)
{
  ctrs->setup.cycles = 0;
  ctrs->setup.tot.imcR = 0;
  ctrs->setup.tot.imcW = 0;
  ctrs->setup.tot.edcR = 0;
  ctrs->setup.tot.edcW = 0;

  ctrs->timestep.cycles = 0;
  ctrs->timestep.tot.imcR = 0;
  ctrs->timestep.tot.imcW = 0;
  ctrs->timestep.tot.edcR = 0;
  ctrs->timestep.tot.edcW = 0;

  ctrs->flux.cycles = 0;
  ctrs->flux.tot.imcR = 0;
  ctrs->flux.tot.imcW = 0;
  ctrs->flux.tot.edcR = 0;
  ctrs->flux.tot.edcW = 0;

  ctrs->grad.cycles = 0;
  ctrs->grad.tot.imcR = 0;
  ctrs->grad.tot.imcW = 0;
  ctrs->grad.tot.edcR = 0;
  ctrs->grad.tot.edcW = 0;

  ctrs->jacobian.cycles = 0;
  ctrs->jacobian.tot.imcR = 0;
  ctrs->jacobian.tot.imcW = 0;
  ctrs->jacobian.tot.edcR = 0;
  ctrs->jacobian.tot.edcW = 0;

  ctrs->forces.cycles = 0;
  ctrs->forces.tot.imcR = 0;
  ctrs->forces.tot.imcW = 0;
  ctrs->forces.tot.edcR = 0;
  ctrs->forces.tot.edcW = 0;
}

void
finalize_perf_counters(const struct ctrs ctrs)
{
  printf("\n=================== HARDWARE COUNTING ==================\n");
  printf("========================================================\n\n");

  double bwR;
  double bwW;
  double etime;

  etime = ctrs.setup.cycles / FREQ;

  printf("Setup Time: ");
  printf("%g Seconds\n", etime);

  bwR = 64 * ctrs.setup.tot.imcR / etime;
  bwW = 64 * ctrs.setup.tot.imcW / etime;

  printf("Setup DRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  bwR = 64 * ctrs.setup.tot.edcR / etime;
  bwW = 64 * ctrs.setup.tot.edcW / etime;

  printf("Setup MCDRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  etime = ctrs.forces.cycles / FREQ;

  printf("Forces Time: ");
  printf("%g Seconds\n", etime);

  bwR = 64 * ctrs.forces.tot.imcR / etime;
  bwW = 64 * ctrs.forces.tot.imcW / etime;

  printf("Forces DRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  bwR = 64 * ctrs.forces.tot.edcR / etime;
  bwW = 64 * ctrs.forces.tot.edcW / etime;

  printf("Forces MCDRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  etime = ctrs.jacobian.cycles / FREQ;

  printf("Jacobian Time: ");
  printf("%g Seconds\n", etime);

  bwR = 64 * ctrs.jacobian.tot.imcR / etime;
  bwW = 64 * ctrs.jacobian.tot.imcW / etime;

  printf("Jacobian DRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  bwR = 64 * ctrs.jacobian.tot.edcR / etime;
  bwW = 64 * ctrs.jacobian.tot.edcW / etime;

  printf("Jacobian MCDRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  etime = ctrs.timestep.cycles / FREQ;

  printf("Time Step Time: ");
  printf("%g Seconds\n", etime);

  bwR = 64 * ctrs.timestep.tot.imcR / etime;
  bwW = 64 * ctrs.timestep.tot.imcW / etime;

  printf("Time Step DRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  bwR = 64 * ctrs.timestep.tot.edcR / etime;
  bwW = 64 * ctrs.timestep.tot.edcW / etime;

  printf("Time Step MCDRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  etime = ctrs.grad.cycles / FREQ;

  printf("Gradient Time: ");
  printf("%g Seconds\n", etime);

  bwR = 64 * ctrs.grad.tot.imcR / etime;
  bwW = 64 * ctrs.grad.tot.imcW / etime;

  printf("Gradient DRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  bwR = 64 * ctrs.grad.tot.edcR / etime;
  bwW = 64 * ctrs.grad.tot.edcW / etime;

  printf("Gradient MCDRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  etime = ctrs.flux.cycles / FREQ;

  printf("Flux Time: ");
  printf("%g Seconds\n", etime);

  bwR = 64 * ctrs.flux.tot.imcR / etime;
  bwW = 64 * ctrs.flux.tot.imcW / etime;

  printf("Flux DRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  bwR = 64 * ctrs.flux.tot.edcR / etime;
  bwW = 64 * ctrs.flux.tot.edcW / etime;

  printf("Flux MCDRAM Bandwidth: ");
  printf("Read: %g GB/s, Write: %g GB/s\n", bwR / 1e9, bwW / 1e9);

  printf("========================================================\n");
}
