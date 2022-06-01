
static char help[] = "Incompressible Euler version of PETSc-FUN3D, which "
"is highly optimized for shared-memory multi- and many-core emerging "
"hardware. \nThis version does not support MPI, and it perfectly works "
"with native mode of Intel Xeon Phi mnay-core coprocessor.\n\n";
/*
  Maximum limits of failure
*/
#define MAXFAILS  10000

#define f77SUMGS  f77name(SUMGS,sumgs,sumgs_)
