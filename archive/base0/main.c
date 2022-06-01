
#include "defs.h" 
#include "main.h"
#include "header.h"
#include "ktime.h"
/*
  Finite volume flux split solver for general polygons 
*/
#undef  __FUNCT__
#define __FUNCT__ "main"
int 
main(int argc, char *argv[])
{

  struct ktime ktime;
  setktime(&ktime);

  // ====================================
  // Phase_2
  // ====================================
  
  double start_time;
  double exit_time;

  int       ierr;         // Error return value
  TstepCtx  tstepctx;     // Time step context
  GRID      grid;         // Grid information
  GRIDPETSc gridPETSc;    // PETSc data structure
  AppCtx    appctx;       // Application Context  
  SNES      snes;         // Non-linear solver context
  Mat       Jpc;          // Jacobian matrix
  double    sTime  = 0.0; // Start time
  double    eTime  = 0.0; // End time
  /*
    Initializes the PETSc database and MPI. 
    PetscInitialize() calls MPI_Init() if that has yet to be called, 
    so this routine should always be called near the beginning of your 
    program -- usually the very first line!
  */   
  ierr = PetscInitialize(&argc, &argv, "petsc.opt", help); 
  /*
    Checks error code, if non-zero it calls the error handler 
    and then returns
  */
  CHKERRQ(ierr);
  /*
    Routine that should be called soon AFTER the call to 
    PetscInitialize() if one is using a C main program that calls 
    Fortran routines that in turn call PETSc routines
  */
  ierr = PetscInitializeFortran(); CHKERRQ(ierr);
  /*
    The equivalent of the MPI_COMM_WORLD communicator which represents 
    all the processs that PETSc knows about
  */ 
  MPI_Comm comm = PETSC_COMM_WORLD;
  /*
    Determines the rank of the calling process in the communicator
  */
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  /*
    Determines the size of the group associated with a communicator
  */
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  /* 
    Tells PETSc to monitor the maximum memory usage so that 
    PetscMemoryGetMaximumUsage() will work
  */
  ierr = PetscMemorySetGetMaximumUsage(); CHKERRQ(ierr);

#ifdef HW_COUNTER
  ierr = initPAPI();
  if(ierr != 0)
    return 0;
#endif


  // ===================================
  // =================================== Phase_2

  start_time = 0.f; exit_time = 0.f;
  start_time = MPI_Wtime();
  /*
    1.  Initialize the time step context
    2.  Initialize the maximum number of threads, and the gradient time
        variables,
    3.  Initialize the global FORTRAN common variables
  */ 
  ierr = init(&tstepctx); CHKERRQ(ierr);

  exit_time = MPI_Wtime();


  ierr = PetscPrintf(comm, "Time step context, Fortran common blocks, "
                            "and global variables Initialized\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf( comm, "Time taken in this phased was %g\n",
                      exit_time - start_time); 
  CHKERRQ(ierr);
  /*
    Prints to standard out, only from the first processor in the 
    communicator. Calls from other processes are ignored
  */
  ierr = PetscPrintf(comm, "Running %d OpenMP Threads\n", nThreads);
  CHKERRQ(ierr);
  
  ierr = PetscTime(&sTime); CHKERRQ(ierr);  
  /* 
    get the grid information into local ordering 
  */







  ierr = initGrid(&grid); CHKERRQ(ierr); 

  ierr = PetscTime(&eTime); CHKERRQ(ierr);
  eTime -= sTime;

  ierr = PetscPrintf(comm, "Grid information initialized\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf( comm, "Time taken in this phase was %g\n",
                      eTime); 
  CHKERRQ(ierr);

  printktime(ktime, "Setup");

  /*
    Get the weights for calculating gradients using least squares
  */

  ierr = PetscTime(&sTime); CHKERRQ(ierr);

  setktime(&ktime);

  f77SUMGS( &grid.nnodes, &grid.nedge, &grid.thread.nedge, &nThreads, 
            grid.thread.eptr, grid.coord.x, grid.coord.y, grid.coord.z, 
            grid.thread.partitions, grid.thread.nedgeLoc, 
            grid.weight.r11, grid.weight.r12, grid.weight.r13, 
            grid.weight.r22, grid.weight.r23, grid.weight.r33, 
            grid.weight.r44);
/*
  FILE * f = fopen("me.txt", "w");
  int ii;  
  for(ii = 0; ii < grid.nnodes; ii++) 
  {
    fprintf(f, "w00: %g\n", grid.weight.r11[ii]);
    fprintf(f, "w01: %g\n", grid.weight.r12[ii]);
    fprintf(f, "w02: %g\n", grid.weight.r13[ii]);

    fprintf(f, "w11: %g\n", grid.weight.r22[ii]);
    fprintf(f, "w12: %g\n", grid.weight.r23[ii]);

    fprintf(f, "w22: %g\n", grid.weight.r33[ii]);

    fprintf(f, "w33: %g\n", grid.weight.r44[ii]);  
  }

  fclose(f);
*/
  ierr = PetscTime(&eTime); CHKERRQ(ierr);
  eTime = eTime - sTime;
  ierr = PetscPrintf(comm, "Weights for calculating the gradient using "
                            "least square calculated\n");
  ierr = PetscPrintf( comm, "Time taken in this phase was %g\n", 
                      eTime);
  /* 
    Points to the grid and the time step data struct
  */
  appctx.grid       = &grid;
  appctx.tsCtx      = &tstepctx;
  appctx.gridPETSc  = &gridPETSc;
  /* 
    Preload the executable to get accurate timings. 
    This runs the following chunk of code twice, first to get 
    the executable pages into memory and the second time for
    accurate timings.
  */
  /*
    Beginn a segment of code that may be preloaded (run twice) to 
    get accurate timings
    PETSC_TRUE: run the code twice
  */

  //PetscPreLoadBegin(PETSC_TRUE, "First Stage: Time Integration");
  appctx.isPreLoading = PETSC_FALSE;
  /*
    Create non-linear solver
  */  
  /*
    Initialize PETSc data structure
  */  
  ierr = initPETScDS(&appctx); CHKERRQ(ierr);

  /* 
    Creates a nonlinear solver context
  */  
  ierr = SNESCreate(comm, &snes); CHKERRQ(ierr);
  /*
    Sets the method for the nonlinear solver
  */  
  ierr = SNESSetType(snes, "newtonls"); CHKERRQ(ierr);  
  /*
    Set various routines and options 
    Sets the function evaluation routine and function 
    vector for use by the SNES routines in solving systems 
    of nonlinear equations; parameters:
    SNES context; vector to store function value (residual)
    Function evaluation routine ---> functional form used to 
    convey the nonlinear function to be solved by SNES 
    User-defined context for private data for the 
    function evaluation routine
    appctx:: grid and time step context 
    Newton => f'(x) x = - f(x) ::: f'(x) is Jacobian matrix
    and f(x) is the function
  */  
  ierr = SNESSetFunction( snes, appctx.gridPETSc->res, formFunction, 
                          &appctx ); CHKERRQ(ierr);
  /*
    Creates a matrix-free matrix context for use with 
    a SNES solver. This matrix can be used as the Jacobian 
    argument for the routine SNESSetJacobian()
    Use matrix-free to define Newton system; use explicit 
    (approx) Jacobian for preconditioner
  */   
  ierr = MatCreateSNESMF(snes, &Jpc); CHKERRQ(ierr);
  /*
    Sets the function to compute Jacobian as well as 
    the location to store the matrix.
    Parameters:
    SNES context; the matrix that defines the approximate 
    Jacobian; the matrix to be used in constructing the
    preconditioner, usually the same as previous one
    Jacobian evaluation routine; appctx defined context for
    private data
  */  
  ierr = SNESSetJacobian(snes, Jpc, appctx.gridPETSc->A, formJacobian, 
                        &appctx); CHKERRQ(ierr);
  /*
    The number of failed linear solve attempts allowed before 
    SNES returns with a diverged reason of SNES_DIVERGED_LINEAR_SOLVE
  */   
//  ierr = SNESSetMaxLinearSolveFailures(snes, MAXFAILS); CHKERRQ(ierr);
  /*
    Sets various SNES parameters from appctx options
  */
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
  /*
    Initialize the flow filed
  */
  ierr = formInitialGuess(&appctx); CHKERRQ(ierr);

  /*
    Solve nonlinear system
  */
  ierr = update(snes, &appctx); CHKERRQ(ierr);
  /*
    Memory usage Before deallocating PETSc data structures
  */ 
  ierr = displayMemUse("Memory usage before destroying\n"); 
  CHKERRQ(ierr);
  /*
    Deallocate PETSc DS
  */
  ierr = VecDestroy(&appctx.gridPETSc->qnode); CHKERRQ(ierr);  
  ierr = VecDestroy(&appctx.gridPETSc->qnodeLoc); CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.tsCtx->qnode);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.tsCtx->res); CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.gridPETSc->res); CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.gridPETSc->grad); CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.gridPETSc->A); CHKERRQ(ierr);
  ierr = MatDestroy(&Jpc); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  /*
    Memory usage after deallocating PETSc DS
  */
  ierr = displayMemUse("Memory usage after destroying\n");
  CHKERRQ(ierr);
   
  //PetscPreLoadEnd();
  /*
    Clear the memory
  */

  printktime(ktime, "Kernel Phase: ");

  free(appctx.grid->eptr);
  
  free(appctx.grid->normal.coord.x);
  free(appctx.grid->normal.coord.y);
  free(appctx.grid->normal.coord.z);
  free(appctx.grid->normal.len);

  free(appctx.grid->thread.partitions);
  free(appctx.grid->thread.nedgeLoc);
  free(appctx.grid->thread.eptr);
  free(appctx.grid->thread.normal.coord.x);
  free(appctx.grid->thread.normal.coord.y);
  free(appctx.grid->thread.normal.coord.z);
  free(appctx.grid->thread.normal.len);
  
  free(appctx.grid->coord.x);
  free(appctx.grid->coord.y);
  free(appctx.grid->coord.z);

  free(appctx.grid->area);
 
  free(appctx.grid->bound.solid.f2nt);
  free(appctx.grid->bound.solid.inode);
  free(appctx.grid->bound.solid.normal.x);
  free(appctx.grid->bound.solid.normal.y);
  free(appctx.grid->bound.solid.normal.z);

  free(appctx.grid->bound.viscs.f2nt);
  free(appctx.grid->bound.viscs.inode);
  free(appctx.grid->bound.viscs.normal.x);
  free(appctx.grid->bound.viscs.normal.y);
  free(appctx.grid->bound.viscs.normal.z);

  free(appctx.grid->bound.ffild.f2nt);
  free(appctx.grid->bound.ffild.inode);
  free(appctx.grid->bound.ffild.normal.x);
  free(appctx.grid->bound.ffild.normal.y);
  free(appctx.grid->bound.ffild.normal.z);

  free(appctx.grid->cdt);

  free(appctx.grid->weight.r11);
  free(appctx.grid->weight.r12);
  free(appctx.grid->weight.r13);
  free(appctx.grid->weight.r22);
  free(appctx.grid->weight.r23);
  free(appctx.grid->weight.r33);
  free(appctx.grid->weight.r44);

  cleanILUmem();

  ierr = PetscPrintf(comm, "Gradient time: %g sec.\n", gradTime); 
  CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Flux time: %g sec.\n", fluxTime); 
  CHKERRQ(ierr);

#ifdef HW_COUNTER
  ierr = PetscPrintf(comm, "PAPI hardware counter: %lld\n", handler); 
  CHKERRQ(ierr);
#endif

  ierr = PetscPrintf(comm, "================================\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Memory Usage ...................\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "================================\n"); 
  CHKERRQ(ierr);
  /*
    Checks for options to be called at the conclusion of the program. 
    MPI_Finalize() is called only if the appctx had not called MPI_Init() 
    before calling PetscInitialize()
  */
  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
} 
