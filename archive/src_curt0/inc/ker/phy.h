
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#ifndef __PHY_H
#define __PHY_H

#define ALPHA 3.00  // Angle of attack 
#define BETA  15.00 // Artificial compressibility

struct ivals {
  double p; // Pressure
  double u; // Velocity
  double v; // velocity
  double w; // Velocity
};

struct igtbl {
  size_t sz;
  size_t bsz;
  struct ivals * iv;
  double * q0;
  double * q1;
  struct kernel_time * t;
};

inline void
iguess(struct igtbl *restrict);

struct delta {
  size_t nnodes;
  size_t nsnodes;
  size_t nfnodes;
  const uint32_t *restrict nsptr;
  const uint32_t *restrict nfptr;
  const double *restrict s_xyz0;
  const double *restrict s_xyz1;
  const double *restrict s_xyz2;
  const double *restrict f_xyz0;
  const double *restrict f_xyz1;
  const double *restrict f_xyz2;
  const double *restrict area;
  const double *restrict q;
  const uint32_t *restrict ie;
  const uint32_t *restrict part;
  const uint32_t *restrict n0;
  const uint32_t *restrict n1;
  const double *restrict x0;
  const double *restrict x1;
  const double *restrict x2;
  const double *restrict x3;
  size_t bsz;
  double *restrict cdt;
  struct ktime *restrict t;
};

void
compute_deltat2(struct delta *restrict);

struct grad {
  size_t bsz;
  size_t dofs;
  const uint32_t *restrict ie;
  const uint32_t *restrict part;
  const uint32_t *restrict n0;
  const uint32_t *restrict n1;
  const double *restrict q;
  const double *restrict w0termsx;
  const double *restrict w0termsy;
  const double *restrict w0termsz;
  const double *restrict w1termsx;
  const double *restrict w1termsy;
  const double *restrict w1termsz;
  double *restrict gradx0;
  double *restrict gradx1;
  double *restrict gradx2;
  struct ktime *restrict t;
};

void
compute_grad(struct grad *restrict);

struct flux {
  size_t bsz;
  size_t nfnodes;
  size_t dofs;
  uint32_t snfc;
  double pressure;
  double velocity_u;
  double velocity_v;
  double velocity_w;
  const double *restrict f_xyz0;
  const double *restrict f_xyz1;
  const double *restrict f_xyz2;
  const double *restrict xyz0;
  const double *restrict xyz1;
  const double *restrict xyz2;
  const uint32_t *restrict ie;
  const uint32_t *restrict part;
  const uint32_t *restrict snfic;
  const uint32_t *restrict n0;
  const uint32_t *restrict n1;
  const uint32_t *restrict nfptr;
  const uint32_t *restrict sn0;
  const uint32_t *restrict sn1;
  const uint32_t *restrict sn2;
  const double *restrict x0; 
  const double *restrict x1; 
  const double *restrict x2; 
  const double *restrict x3; 
  const double *restrict q;
  double *restrict gradx0;
  double *restrict gradx1;
  double *restrict gradx2;
  double *restrict r;
  struct ktime *restrict t;
};

void
compute_flux(struct flux *restrict);

struct force {
  const struct geometry * g;
  const struct ivals * iv;
  const double *q;
  double * clift;
  double * cdrag;
  double * cmomn;
  struct kernel_time *restrict t;
};

void
compute_force(struct force *restrict);

#endif
