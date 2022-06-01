
#ifndef __MAIN_H
#define __MAIN_H

void
imesh(const uint8_t *, struct geometry *restrict);

int
#ifdef __USE_HW_COUNTER
ikernel(int, char **, const struct geometry *restrict,
        struct kernel_time *restrict,
        struct perf_counters *restrict);
#else
ikernel(int, char **, const struct geometry *restrict,
        struct kernel_time *restrict);
#endif

#endif
