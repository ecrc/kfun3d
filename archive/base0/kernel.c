
#include "defs.h"
#include "header.h"

#define INC                   1.1 
#define PETSC_ERR_SNES_FAILED 1
#define f77FORCE              f77name(FORCE,force,force_)
/* 
  Solve non-linear system
*/
#undef  __FUNCT__
#define __FUNCT__ "update"
int 
update(SNES snes, void *ctx)
{
  AppCtx    *user       = (AppCtx *) ctx;
  GRID      *grid       = user->grid;
  GRIDPETSc *gridPETSc  = user->gridPETSc; 
  TstepCtx  *tsCtx      = user->tsCtx;
  /*
    Maximum time steps value for the nonlinear solve (SNES)
    It starts at 1 time step only before the preloading to get
    the executable pages into the main memory and then reexecute the 
    main kernels again. This preloading steps is important for
    PETSc profiler to get an accurate timings.
  */
  int     maxTimeSteps = 1;
  /*
    Local ratio of the 2-norm of function at current iterate
    Used by SNES convergence test of the solvers for systems of nonlinear
    equations
  */
  double  fnormRatio  = 1.0;
  /*
    Number of SNES failed time steps
  */
  int     nSNESFails  = 0; 
  int     nSNESFail   = 0;
  /*
    Solution vector, physical components, and PETSc error context
  */
  double  *qnode;
  double  clift, cdrag, cmom;
  int     ierr;
  /*
    MPI communicator
  */
  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  /*
    If the kernel is already executed once for the accurate profiling
    results, and this is the second execution. Preloading here is set
    to FALSE, means the maximum time steps has to be initialized back to 
    the origin.
  */ 
  if(!user->isPreLoading)
    maxTimeSteps = tsCtx->maxTimeSteps;
  /*
    Keep whatever stored in the solution vector backed up in the time
    step qnode context, which changes at every time step
  */
  ierr = VecCopy(gridPETSc->qnode, tsCtx->qnode); CHKERRQ(ierr);
  /*
    Start the kernel.
    Compute the nonlinear solve from 0 to the maximum time steps
    initialized at the beginning, unless the function norm returns
    that the nonlinear iterative kernels meet the convergence 
    condition, means that the problem has been solved before it
    reaches the maximum time step limit.
  */
  for((tsCtx->iTimeStep = 0);
      ((tsCtx->iTimeStep < maxTimeSteps) && 
       (fnormRatio <= tsCtx->fnormRatio));
      (tsCtx->iTimeStep++))
  {
    ierr = computePseudoTimeStep(snes, user); CHKERRQ(ierr);
    /*
      Solves a nonlinear system F(x) = b
      Null means b = 0 (constant part of the equation)
      x is the solution vector 
      global distributed solution vector
      Here FormJacobian is called 
    */
#if 0
  KSP ksp;
  ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);

  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);

  ierr = PCSetType(pc, PCILU); CHKERRQ(ierr);
#endif

    ierr = SNESSolve(snes, NULL, gridPETSc->qnode); CHKERRQ(ierr);
    /*
      Gets the number of unsuccessful steps attempted by 
      the nonlinear solver
    */
    ierr = SNESGetNonlinearStepFailures(snes, &nSNESFail); CHKERRQ(ierr);
    /*
      Counter for the total number of SNES failed steps
    */
    nSNESFails += nSNESFail; 
    nSNESFail   = 0;
    /*  
      Fails in the newton steps
    */
    if (nSNESFails >= 2)
    {
      /*
        Macro to be called when an error has been detected
      */
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SNES_FAILED, 
              "Unable to find a Newton Step");
    }
    
    ierr = PetscPrintf( comm, 
                        "At Time Step %d CFL = %g and fnorm = %g\n",
                        tsCtx->iTimeStep, tsCtx->CFL, tsCtx->fnorm );
    CHKERRQ(ierr);

    ierr = VecCopy(gridPETSc->qnode, tsCtx->qnode); CHKERRQ(ierr);
    
    ierr = VecGetArray(gridPETSc->qnode, &qnode); CHKERRQ(ierr);
    /*
      Compute the forces
      clift
      cdrag
      cmom
    */
    f77FORCE( &grid->nnodes, &grid->bound.solid.nfacet, qnode, 
              grid->coord.x, grid->coord.y, grid->coord.z, 
              grid->bound.solid.inode, grid->bound.solid.f2nt, 
              &clift, &cdrag, &cmom );

    ierr = PetscPrintf( comm, "%d\t%g\t%g\t%g\t%g\t%g\n", 
                        tsCtx->iTimeStep, tsCtx->CFL, tsCtx->fnorm, 
                        clift, cdrag, cmom ); CHKERRQ(ierr);
    
    ierr = VecRestoreArray(gridPETSc->qnode, &qnode); CHKERRQ(ierr);
    
    fnormRatio = tsCtx->fnormInit / tsCtx->fnorm;
  }

  ierr = PetscPrintf( comm, "CFL = %g fnorm = %g\n", 
                      tsCtx->CFL, tsCtx->fnorm ); CHKERRQ(ierr);
  ierr = PetscPrintf( comm, "clift = %g cdrag = %g cmom = %g\n", 
                      clift, cdrag, cmom); CHKERRQ(ierr);

  if (user->isPreLoading) 
  {
    tsCtx->fnormInit = 0.0;
    ierr = PetscPrintf(comm, "Preloading done ...\n"); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
/*
  Calculate pseudo-time step
*/
#undef  __FUNCT__
#define __FUNCT__ "computePseudoTimeStep"
int 
computePseudoTimeStep(SNES snes, void *ctx)
{
  AppCtx    *user   = (AppCtx *) ctx;
  TstepCtx  *tsCtx  = user->tsCtx;
  int       ierr;
  double    CFL;

  PetscFunctionBegin;
  /*
    The formFunction will calculates the time step for each cell through
    calling DELTAT2 routine
  */
  tsCtx->isPseudoTimeStep = 0;
  /* 
    Compute the gradients, pseudo time step, and the flux
  */
  ierr = formFunction(snes, tsCtx->qnode, tsCtx->res, user); 
  CHKERRQ(ierr);
  /*
    Set the flag to 1, means the pseudo time step has been calculated
  */
  tsCtx->isPseudoTimeStep = 1;
  /*
    Computes the vector norm
    Vector x; NORM 2 (the two norm, ||v|| = sqrt(sum_i (v_i)^2))
    the norm [output parameter]
  */
  ierr = VecNorm(tsCtx->res, NORM_2, &tsCtx->fnorm); 
  CHKERRQ(ierr);
  /* 
    First time through so compute initial function norm 
  */
  if (tsCtx->fnormInit == 0.0) 
  {
    tsCtx->fnormInit  = tsCtx->fnorm;
    tsCtx->CFL        = tsCtx->CFLInit;
  } 
  else 
  {
    /*
      Adjust the time step according to the CFL number.
      This stage happens between each of the first-order and 
      second-order phases of computation
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %                                                                %
      % (N_CFL)^l = (N_CFL)^0 * ((||f(u^0)||)/(||f(u^(l-1))||))^\sigma %
      %                                                                %
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    */
    CFL         = INC * tsCtx->CFLInit * tsCtx->fnormInit / tsCtx->fnorm;
    tsCtx->CFL  = PetscMin(CFL, tsCtx->CFLMax);
  }

  PetscFunctionReturn(0);
}
