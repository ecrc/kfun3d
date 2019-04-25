#ifndef __FUN3D_INC_COREKERNEL_H
#define __FUN3D_INC_COREKERNEL_H

#include "geometry.h"

void
ComputeA(GEOMETRY *);

void
ComputeTimeStep(const double, GEOMETRY *);

void
ComputeFlux(const GEOMETRY *, const double *, GRADIENT *, double *);

void
ComputeForces(const GEOMETRY *, double *);

void
ComputeNumericalILU(GEOMETRY *);

void
ComputeSparseTriangularSolve(const GEOMETRY *, const double *, double *);

#endif /* __FUN3D_INC_KERNEL_H */