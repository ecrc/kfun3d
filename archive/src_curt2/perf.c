
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <syscall.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <perf.h>

static uint64_t
iperf_read(const int fd)
{
  uint64_t buf;
  ssize_t sz = read(fd, &buf, sizeof(buf));

  return buf;
}

void
perf_read(const struct fd fd, struct counters * counters)
{
  uint32_t i;

  /* MC RPQ and WPQ inserts: DRAM read and write requests */
  for(i = 0; i < IMCSZ; i++)
  {
    counters->imcR[i] = iperf_read(fd.imcR[i]);
    counters->imcW[i] = iperf_read(fd.imcW[i]);
  }
  /* EDC RPQ and WPQ inserts: MCDRAM read and write requests */
  for(i = 0; i < EDCSZ; i++)
  {
    counters->edcR[i] = iperf_read(fd.edcR[i]);
    counters->edcW[i] = iperf_read(fd.edcW[i]);
  }
}

static int
perf_event_open(struct perf_event_attr * attr,
                pid_t pid,
                int cpu, int group_fd,
                unsigned long flags)
{
  /*
    pid == -1 and cpu >= 0
    This measures all processes/threads on the specified CPU.
    This requires CAP_SYS_ADMIN capability or a
    /proc/sys/kernel/perf_event_paranoid value of less than 1.
  */

  return (syscall(__NR_perf_event_open, attr, pid, cpu,
                  group_fd, flags));
}

static int
iperf_init(const char * file_path, const int e, const int m)
{
  /* Read the Performance Monitor Unit (PMU) type from the file */

  FILE * pFile = fopen(file_path, "r");
  assert(pFile != NULL);

  int type;
  fscanf(pFile, "%d", &type);
  
  fclose(pFile);

  /* Use Linux Perf tool to monitor the performance */

  struct perf_event_attr attr = {};

  attr.size = sizeof(attr);
  attr.type = type;
  attr.config = e | (m << 8);

  int fd = perf_event_open(&attr, -1, 0, -1, 0);
  assert(fd != -1);

  return fd;
}

void
perf_init(struct fd * fd)
{
  uint32_t i;

  /* MC RPQ and WPQ inserts: DRAM read and write requests */
  for(i = 0; i < IMCSZ; i++)
  {
    char file_path[30];
    sprintf(file_path, "/sys/devices/uncore_imc_%d/type", i);
    
    /* RPQ Inserts */
    fd->imcR[i] = iperf_init(file_path, 0x1, 0x1);

    /* WPQ Inserts */
    fd->imcW[i] = iperf_init(file_path, 0x2, 0x1);
  }

  /* EDC RPQ and WPQ inserts: MCDRAM read and write requests */
  for(i = 0; i < EDCSZ; i++)
  {
    char file_path[35];
    sprintf(file_path, "/sys/devices/uncore_edc_eclk_%d/type", i);

    /* RPQ Inserts */
    fd->edcR[i] = iperf_init(file_path, 0x1, 0x1);
    
    /* WPQ Inserts */
    fd->edcW[i] = iperf_init(file_path, 0x2, 0x1);
  }
}

void
perf_calc(const struct counters start,
          const struct counters end,
          struct tot * tot)
{
  uint32_t i;

  tot->imcR = 0;
  tot->imcW = 0;
  tot->edcR = 0;
  tot->edcW = 0;

  /* MC RPQ and WPQ inserts: DRAM read and write requests */
  for(i = 0; i < IMCSZ; i++)
  {
    tot->imcR += end.imcR[i] - start.imcR[i];
    tot->imcW += end.imcW[i] - start.imcW[i];
  }

  for(i = 0; i < EDCSZ; i++)
  {
    tot->edcR += end.edcR[i] - start.edcR[i];
    tot->edcW += end.edcW[i] - start.edcW[i];
  }
}
