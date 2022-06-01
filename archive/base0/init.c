
#include "defs.h"
#include "header.h"

#define f77INITCOMM f77name(INITCOMM, initcomm, initcomm_)

#undef  __FUNCT__
#define __FUNCT__ "init"
int
init(TstepCtx *tstepctx)
{
  int ierr;

  PetscFunctionBegin;
  /*
    Initialize the time step context
  */
  ierr = initTimeStepContext(tstepctx); CHKERRQ(ierr);
  
  nThreads    = 4;                      // Number of OpenMP threads
  gradTime    = 0.0;                    // Gradient calculation time
  fluxTime    = 0.0;

  // Number of OpenMP threads
  ierr = PetscOptionsGetInt(NULL, NULL, "-nThreads", &nThreads, NULL); 
  CHKERRQ(ierr);
/*  ierr = PetscOptionsGetInt(NULL, "-nThreads", &nThreads, NULL); 
  CHKERRQ(ierr);*/
  /*
    Specifies the number of threads used by default 
    in subsequent parallel sections
  */
  omp_set_num_threads(nThreads);
  /*
    Initialize Fortran common block
  */
  f77INITCOMM();

  PetscFunctionReturn(0);

}
/*
  Initialize the time step data structure context
*/
#undef  __FUNCT__
#define __FUNCT__ "initTimeStepContext"
int
initTimeStepContext(TstepCtx *tstepctx)
{
  int ierr; // Error return value

  PetscFunctionBegin;

  tstepctx->fnormInit     = 0.0;      // Initial function norm
  tstepctx->CFLInit       = 50.0;     // Initial CFL condition
  tstepctx->CFLMax        = 1.0e+05;  // Maximum CFL condition
  tstepctx->maxTimeSteps  = 50;       // Maximum SNES time steps
  tstepctx->fnormRatio    = 1.0e+10;  // Function norm ratio
  tstepctx->isLocal       = 1;        // Local/global time stepping
  /*
    User provided values in the petsc.opt file (runtime)
  */
  // Maximum SNES Time Steps
  ierr = PetscOptionsGetInt(NULL, NULL, "-maxTimeSteps", 
                              &tstepctx->maxTimeSteps, NULL ); 
  CHKERRQ(ierr);
  // Function norm ratio
  ierr = PetscOptionsGetReal(NULL, NULL, "-fnormRatio", 
                              &tstepctx->fnormRatio, NULL ); 
  CHKERRQ(ierr);
  // Initial CFL condition
  ierr = PetscOptionsGetReal(NULL, NULL, "-CFLInit", 
                              &tstepctx->CFLInit, NULL ); 
  CHKERRQ(ierr);
  // Maximum CFL condition
  ierr = PetscOptionsGetReal(NULL, NULL, "-CFLMax", 
                              &tstepctx->CFLMax, NULL ); 
  CHKERRQ(ierr);

/*
  // Maximum SNES Time Steps
  ierr = PetscOptionsGetInt(  NULL, "-maxTimeSteps", 
                              &tstepctx->maxTimeSteps, NULL ); 
  CHKERRQ(ierr);
  // Function norm ratio
  ierr = PetscOptionsGetReal( NULL, "-fnormRatio", 
                              &tstepctx->fnormRatio, NULL ); 
  CHKERRQ(ierr);
  // Initial CFL condition
  ierr = PetscOptionsGetReal( NULL, "-CFLInit", 
                              &tstepctx->CFLInit, NULL ); 
  CHKERRQ(ierr);
  // Maximum CFL condition
  ierr = PetscOptionsGetReal( NULL, "-CFLMax", 
                              &tstepctx->CFLMax, NULL ); 
  CHKERRQ(ierr);

*/
  PetscFunctionReturn(0);
}
