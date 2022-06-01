
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __MESH_H
#define __MESH_H

size_t
emalloc(char *restrict, struct etbl *restrict);

void
imain(const size_t, uint32_t *, uint32_t *);

size_t
nmalloc(char *restrict, struct ntbl *restrict);

size_t
bmalloc(const uint32_t *restrict, char *restrict,
        struct btbl *restrict);

void
m2csr(struct geometry *restrict);

void
isubdomain(struct geometry *restrict);

void
iecoloring(struct geometry *restrict);

void
ifcoloring(struct geometry *restrict);

void
wmalloc(struct geometry *restrict);

#endif
