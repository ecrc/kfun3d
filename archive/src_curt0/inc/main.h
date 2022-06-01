
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __MAIN_H
#define __MAIN_H

void
imesh(const uint8_t *, struct geometry *restrict);

int
ikernel(int, char **, const struct geometry *restrict,
        struct kernel_time *restrict);

#endif
