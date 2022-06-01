
static char help[] = "FUN3D - 3-D, Unstructured Incompressible Euler Solver.\n\
originally written by W. K. Anderson of NASA Langley, \n\
ported into PETSc by D. K. Kaushik, ODU and ICASE, \n\
recently ported into PETSc-dev v3.4 by M. A, Al Farhan, KAUST, \n\
and then it is ported into Intel Xeon Phi Architecture (Many-Integrated Core MIC)\n\
by M. A. Al Farhan, KAUST (2014)\n\n";

#include "user.h" 

// Global Fortran variables
CINFO   *c_info;    // Pointer to COMMON INFO
CRUNGE  *c_runge;   // Pointer to COMMON RUNGE
CGMCOM  *c_gmcom;   // Pointer to COMMON GMCOM

double  memSize       = 0.0;    //
double  grad_time     = 0.0;    // Timing the gradient calculation phase
int     target_id     = 0;
int     num_devices   = 0;
int     kmp_affinity  = 0;

// Main Routine 
// Finite volume flux split solver for general polygons 
#undef  __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  int       ierr;
  PetscBool flg;
  TstepCtx  tsCtx;              // Time step context
  GRID      f_pntr;             // Grid information
  AppCtx    user;
  
  SNES      snes;               // Non-linear solver context
  Mat       Jpc;                // Jacobian matrix
  double    *qnode;
  double    time_ini, time_fin;
  int       maxfails = 10000;
   
  ierr = PetscInitialize(&argc, &argv, "petsc.opt", help); CHKERRQ(ierr);
  ierr = PetscInitializeFortran(); CHKERRQ(ierr);
  
  MPI_Comm comm = PETSC_COMM_WORLD;
  f77FORLINK();  // Link FORTRAN and C COMMONS
  
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);

  // Initialize stuff related to time stepping
  // ====================================================================
  tsCtx.fnorm_ini         = 0.0;  
  tsCtx.cfl_ini           = 50.0;    
  tsCtx.cfl_max           = 1.0e+05;
  tsCtx.max_steps         = 50;   
  tsCtx.max_time          = 1.0e+12; 
  tsCtx.iramp             = -50;
  tsCtx.dt                = -5.0; 
  tsCtx.fnorm_ratio       = 1.0e+07;
  tsCtx.LocalTimeStepping = 1;

  ierr = PetscOptionsGetInt(NULL, NULL, "-max_st", &tsCtx.max_steps,
                            NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-ts_rtol", &tsCtx.fnorm_ratio,
                             NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-cfl_ini", &tsCtx.cfl_ini,
                             NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-cfl_max", &tsCtx.cfl_max,
                             NULL); CHKERRQ(ierr);
  
  tsCtx.print_freq = tsCtx.max_steps; 
  ierr = PetscOptionsGetInt(NULL, NULL, "-print_freq", &tsCtx.print_freq,
                            &flg); CHKERRQ(ierr);

  c_info->alpha  = 3.0;
  c_info->beta   = 15.0;
  c_info->ivisc  = 0;

  c_gmcom->ilu0  = 1;
  c_gmcom->nsrch = 10;
                                       
  c_runge->nitfo = 0;

  f_pntr.jvisc   = c_info->ivisc;
  f_pntr.ileast  = 4;
  
  ierr = PetscOptionsGetReal(NULL, NULL, "-alpha", &c_info->alpha,
                             NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-beta", &c_info->beta,
                             NULL); CHKERRQ(ierr);
  // =====================================================================

  // Set the maximum number of threads for OpenMP 
# ifdef _OPENMP
  ierr = PetscOptionsGetInt(NULL, NULL, "-max_threads", &max_threads, &flg);
  CHKERRQ(ierr);
  omp_set_num_threads(max_threads);
  ierr = PetscPrintf(comm, "Using %d threads for each MPI process\n", 
                     max_threads); CHKERRQ(ierr);
# endif
# ifdef OFFLOAD_MODE
  ierr = PetscOptionsGetInt(NULL, NULL, "-target_id", &target_id, &flg);
  CHKERRQ(ierr);
  //ierr = PetscOptionsGetInt(NULL, "-kmp_affinity", &kmp_affinity, &flg);
  //CHKERRQ(ierr);
  //if(kmp_affinity == 0)
  //{
  //    kmp_set_defaults("KMP_AFFINITY=scatter");
  //    PetscPrintf(PETSC_COMM_WORLD, "KMP_AFFINITY is SCATTER\n");
 // }
  //else if(kmp_affinity == 1)
  //{
   //   kmp_set_defaults("KMP_AFFINITY=compact");
    //  PetscPrintf(PETSC_COMM_WORLD, "KMP_AFFINITY is COMPACT\n");
  //}
 // else if(kmp_affinity == 2)
  //{
    //  kmp_set_defaults("KMP_AFFINITY=balanced");
      //PetscPrintf(PETSC_COMM_WORLD, "KMP_AFFINITY is BALANCED\n");
  //}
  //else 
  //{
    //  kmp_set_defaults("KMP_AFFINITY=scatter");
   //   PetscPrintf(PETSC_COMM_WORLD, "KMP_AFFINITY is SCATTER\n");
  //}
  PetscPrintf(PETSC_COMM_WORLD, "Checking for Intel(R) Xeon Phi(TM) "
                                "(Target CPU) devices ... \n");
  num_devices = omp_get_num_devices();
  PetscPrintf(PETSC_COMM_WORLD, "Number of target devices installed: %d " 
                                "MICs\n", num_devices);
  PetscPrintf(PETSC_COMM_WORLD, "You are using Intel MIC #%d\n", target_id);
  omp_set_default_device(target_id);
  PetscPrintf(PETSC_COMM_WORLD, "Target #%d, has been set as the default device\n",
              omp_get_default_device());
  
# endif
# ifdef MEMORY_ALIGNMENT
  PetscPrintf(PETSC_COMM_WORLD, "64-Byte Memory Alignment\n");
# endif
  PetscPrintf(PETSC_COMM_WORLD, "========================================================\n");
  PetscPrintf(PETSC_COMM_WORLD, "========================================================\n");
  // get the grid information into local ordering
  ierr = GetLocalOrdering(&f_pntr); CHKERRQ(ierr);
  
  // Allocate Memory for some other grid arrays
  ierr = set_up_grid(&f_pntr); CHKERRQ(ierr);
  
  // If using least squares for the gradients, then calculate the r's
  if(f_pntr.ileast == 4)
  {
    // Get the weights for calculating gradients using least squares
    f77SUMGS(&f_pntr.nnodesLoc, &f_pntr.nedgeLoc, f_pntr.eptr,
             f_pntr.xyz, f_pntr.rxy, &rank, &f_pntr.nvertices);
  }
  // Points to the grid and the time step
  user.grid  = &f_pntr;
  user.tsCtx = &tsCtx;

# ifdef OFFLOAD_MODE
  // ********************************************************************
  // ********************************************************************
  //
  // Allocate the coprocessor memory
  // Transfer asynchronously the data to the coprocessor memory
  // Those the global data structure that are used in the 
  // flux evaluation routine written in Fortran
  // which is called every time the solver evaluate the 
  // residual of
  // the conservation law
  // evaluating the PDE
  // the other data structure will be sent in the Fortran code 
  // because they have been initialized and declared there
  // the following data structure are allocated once and they 
  // stay there
  // in the coprocessor memory for further usage 
  // (alloc_if, free_if)
  // and they will be deallocated at the end of the function
  // After loading the prgama we further check the 
  // offloading process
  // via a function that mainly make sure that the offloading has 
  // been preformed correctly
  int           nvertices   = user.grid->nvertices;
  double        *xyz        = user.grid->xyz;
# if defined(_OPENMP)
# if defined(HAVE_EDGE_COLORING)
  int           *eptr       = user.grid->eptr; 
  int           nedgeLoc    = user.grid->nedgeLoc;
  double        *xyzn       = user.grid->xyzn;
# pragma  offload_transfer target(mic:target_id) mandatory    \
          in(eptr       : length(2 * nedgeLoc)    ALLOC KEEP) \
          in(xyzn       : length(4 * nedgeLoc)    ALLOC KEEP) \
          in(xyz        : length(3 * nvertices)   ALLOC KEEP)
# elif defined(HAVE_REDUNDANT_WORK)
  int           *eptr       = user.grid->eptr; 
  int           nedgeLoc    = user.grid->nedgeLoc;
  double        *xyzn       = user.grid->xyzn;
# pragma  offload_transfer target(mic:target_id) mandatory    \
          in(eptr       : length(2 * nedgeLoc)    ALLOC KEEP) \
          in(xyzn       : length(4 * nedgeLoc)    ALLOC KEEP) \
          in(xyz        : length(3 * nvertices)   ALLOC KEEP)
# else
  int           nedgeAllThr = user.grid->nedgeAllThr; 
  int           *part_thr   = user.grid->part_thr;
  int           *nedge_thr  = user.grid->nedge_thr;
  int           *edge_thr   = user.grid->edge_thr;
  double        *xyzn_thr   = user.grid->xyzn_thr;
# pragma  offload_transfer target(mic:target_id) mandatory    \
          in(xyz        : length(3 * nvertices)   ALLOC KEEP) \
          in(part_thr   : length(nvertices)       ALLOC KEEP) \
          in(nedge_thr  : length(max_threads + 1) ALLOC KEEP) \
          in(edge_thr   : length(2 * nedgeAllThr) ALLOC KEEP) \
          in(xyzn_thr   : length(4 * nedgeAllThr) ALLOC KEEP)
# endif
# endif
# endif
  //
  // *********************************************************************
  // *********************************************************************

  // AMS stuff
  
  // Preload the executable to get accurate timings. 
  // This runs the following chunk of code twice, first to get 
  // the executable pages into memory and the second time for
  // accurate timings.  
  
  PetscPreLoadBegin(PETSC_TRUE, "Time integration");
  user.PreLoading = PetscPreLoading;
  // Create non-linear solver

  // Set PETSc data structure
  ierr = SetPetscDS(&f_pntr, &tsCtx); CHKERRQ(ierr);
  // Creates a nonlinear solver context
  ierr = SNESCreate(comm, &snes); CHKERRQ(ierr);
  // Sets the method for the nonlinear solver
  ierr = SNESSetType(snes, "newtonls"); CHKERRQ(ierr);
  
  //ierr = SNESSetType(snes, "anderson"); CHKERRQ(ierr);
  // Set various routines and options 

  // Sets the function evaluation routine and function 
  // vector for use by the SNES routines in solving systems 
  // of nonlinear equations; parameters:
  // SNES context; vector to store function value (residual)
  // Function evaluation routine ---> functional form used to 
  // convey the nonlinear function to be solved by SNES 
  // User-defined context for private data for the 
  // function evaluation routine
  // user:: grid and time step context 
  // Newton => f'(x) x = - f(x) ::: f'(x) is Jacobian matrix
  // and f(x) is the function
  ierr = SNESSetFunction(snes, user.grid->res, FormFunction, 
                         &user); CHKERRQ(ierr);
                          
  ierr = PetscOptionsHasName(NULL, NULL, "-matrix_free", &flg); 
  CHKERRQ(ierr);
  if(flg)
  {
    // Creates a matrix-free matrix context for use with 
    // a SNES solver. This matrix can be used as the Jacobian 
    // argument for the routine SNESSetJacobian()

    // Use matrix-free to define Newton system; use explicit 
    // (approx) Jacobian for preconditioner
    ierr = MatCreateSNESMF(snes, &Jpc); CHKERRQ(ierr);
    // Sets the function to compute Jacobian as well as 
    // the location to store the matrix.
    // Parameters:
    // SNES context; the matrix that defines the approximate 
    // Jacobian; the matrix to be used in constructing the
    // preconditioner, usually the same as previous one
    // Jacobian evaluation routine; user defined context for
    // private data
    ierr = SNESSetJacobian(snes, Jpc, user.grid->A, 
                           FormJacobian, &user); 
    CHKERRQ(ierr);
  } 
  else
  {
    // Use explicit (approx) Jacobian to define Newton system 
    // and preconditioner
    ierr = SNESSetJacobian(snes, user.grid->A, user.grid->A, 
                           FormJacobian, &user);
    CHKERRQ(ierr);
  }
  // the number of failed linear solve attempts allowed before 
  // SNES returns with a diverged reason of SNES_DIVERGED_LINEAR_SOLVE
  ierr = SNESSetMaxLinearSolveFailures(snes, maxfails);
  CHKERRQ(ierr);
  // Sets various SNES parameters from user options
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  // Initialize the flow filed
  ierr = FormInitialGuess(snes, user.grid); CHKERRQ(ierr);
  
  // Solve nonlinear system
  ierr = Update(snes, &user); CHKERRQ(ierr);

  ierr = VecRestoreArray(user.grid->qnode, &qnode); CHKERRQ(ierr);
   
  // Write restart file
  ierr = VecGetArray(user.grid->qnode, &qnode); CHKERRQ(ierr);
 
  ierr = VecRestoreArray(user.grid->qnode, &qnode); CHKERRQ(ierr);
  ierr = VecDestroy(&user.grid->qnode);     CHKERRQ(ierr);
  ierr = VecDestroy(&user.grid->qnodeLoc);  CHKERRQ(ierr);
  ierr = VecDestroy(&user.tsCtx->qold);     CHKERRQ(ierr);
  ierr = VecDestroy(&user.tsCtx->func);     CHKERRQ(ierr);
  ierr = VecDestroy(&user.grid->res);       CHKERRQ(ierr);
  ierr = VecDestroy(&user.grid->grad);      CHKERRQ(ierr);
  ierr = VecDestroy(&user.grid->gradLoc);   CHKERRQ(ierr);
  ierr = MatDestroy(&user.grid->A);         CHKERRQ(ierr);
  flg = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL, NULL, "-matrix_free", &flg); CHKERRQ(ierr);
  if (flg)
    ierr = MatDestroy(&Jpc); CHKERRQ(ierr);

  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user.grid->scatter); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user.grid->gradScatter); CHKERRQ(ierr);
  
   
  PetscPreLoadEnd();

# if defined(OFFLOAD_MODE)
  // *********************************************************************
  // *********************************************************************
  //
  // Deallocate the Phi memory
# if defined(_OPENMP)
# if defined(HAVE_EDGE_COLORING)
# pragma  offload_transfer target(mic:target_id) mandatory        \
          nocopy(eptr       : length(2 * nedgeLoc)    REUSE FREE) \
          nocopy(xyz        : length(3 * nvertices)   REUSE FREE) \
          nocopy(xyzn       : length(4 * nedgeLoc)    REUSE FREE)
# elif defined(HAVE_REDUNDANT_WORK)
# pragma  offload_transfer target(mic:target_id) mandatory        \
          nocopy(eptr       : length(2 * nedgeLoc)    REUSE FREE) \
          nocopy(xyzn       : length(4 * nedgeLoc)    REUSE FREE) \
          nocopy(xyz        : length(3 * nvertices)   REUSE FREE)
# else
# pragma  offload_transfer target(mic:target_id) mandatory        \
          nocopy(xyz        : length(3 * nvertices)   REUSE FREE) \
          nocopy(part_thr   : length(nvertices)       REUSE FREE) \
          nocopy(nedge_thr  : length(max_threads + 1) REUSE FREE) \
          nocopy(edge_thr   : length(2 * nedgeAllThr) REUSE FREE) \
          nocopy(xyzn_thr   : length(4 * nedgeAllThr) REUSE FREE)
# endif
# endif
# endif
  //
  // *********************************************************************
  // *********************************************************************

  /* allocated in set_up_grid() */
  free(user.grid->isface);
  free(user.grid->ivface);
  free(user.grid->ifface);
  free(user.grid->us);
  free(user.grid->vs);
  free(user.grid->as);

  /* Allocated in GetLocalOrdering() */
  free(user.grid->eptr);
  free(user.grid->ia);
  free(user.grid->ja);
  free(user.grid->loc2glo);
  free(user.grid->loc2pet);
  free(user.grid->xyzn);
# ifdef _OPENMP
# ifdef HAVE_REDUNDANT_WORK
  free(user.grid->resd);
# endif
# ifndef HAVE_EDGE_COLORING
# ifndef HAVE_REDUNDANT_WORK
  free(user.grid->part_thr);
  free(user.grid->nedge_thr);
  free(user.grid->edge_thr);
  free(user.grid->xyzn_thr);
# endif
# endif
# endif

  free(user.grid->xyz);
  free(user.grid->area);

  free(user.grid->nntet);
  free(user.grid->nnpts);
  free(user.grid->f2ntn);
  free(user.grid->isnode);
  free(user.grid->sxn);
  free(user.grid->syn);
  free(user.grid->szn);
  free(user.grid->sa);
  free(user.grid->sface_bit);

  free(user.grid->nvtet);
  free(user.grid->nvpts);
  free(user.grid->f2ntv);
  free(user.grid->ivnode);
  free(user.grid->vxn);
  free(user.grid->vyn);
  free(user.grid->vzn);
  free(user.grid->va);
  free(user.grid->vface_bit);

  free(user.grid->nftet);
  free(user.grid->nfpts);
  free(user.grid->f2ntf);
  free(user.grid->ifnode);
  free(user.grid->fxn);
  free(user.grid->fyn);
  free(user.grid->fzn);
  free(user.grid->fa);
  free(user.grid->cdt);
  free(user.grid->phi);
  free(user.grid->rxy);

  ierr = PetscPrintf(comm, "Time taken in gradient calculation %g sec.\n",
                     grad_time); CHKERRQ(ierr);


  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
} 

// Establishing the links between FORTRAN common blocks and C
EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "f77CLINK"
void PETSC_STDCALL f77CLINK(CINFO *p1, CRUNGE *p2, CGMCOM *p3)
{
  c_info  = p1;
  c_runge = p2;
  c_gmcom = p3;
}
EXTERN_C_END

// Create non-linear solver
#undef  __FUNCT__
#define __FUNCT__ "SetPetscDS"
int SetPetscDS(GRID *grid, TstepCtx *tsCtx)
{
  
  int       ierr, i, j, k;      // PETSc error & loop iterators
  int       jstart, jend;
  int       nbrs_diag; 
  int       nbrs_offd;       
  int       bs;                 // Number of elements in each block
  int       nnodes;             // Number of total nodes
  int       nnodesLoc;          // Number of local nodes
  int       nvertices;          // Number of local & ghost nodes 
  int       *loc2pet;           // Local to PETSc mapping
  int       *svertices;         // Global index of each local elem
  int       *val_diag;          // Diagonal elements 
  int       *val_offd;          // Off-diagonal elements
  IS        islocal, isglobal;
  // mappings from an arbitrary local ordering from 0 to n-1 to 
  // a global PETSc ordering used by a vector or matrix
  ISLocalToGlobalMapping  isl2g;   
  
  PetscBool flg;
  MPI_Comm  comm;
  
  comm  = PETSC_COMM_WORLD;  
  PetscFunctionBegin;
  // Initialization
  nnodes    = grid->nnodes;
  nnodesLoc = grid->nnodesLoc;
  nvertices = grid->nvertices;
  loc2pet   = grid->loc2pet;
  bs        = 4;              // 4 elements per block

  // Set up the PETSc datastructures

  /*
  
    // Creates a parallel vector and the parameters are:
    // Local vector length, global vector length, & vector
    // Global distributed solution vector
    ierr = VecCreateMPI(comm, bs * nnodesLoc, bs * nnodes,
                      &grid->qnode); CHKERRQ(ierr);
  */

  ierr = VecCreate(comm, &grid->qnode); CHKERRQ(ierr);
  ierr = VecSetSizes(grid->qnode, bs * nnodesLoc, bs * nnodes);
  CHKERRQ(ierr);
  ierr = VecSetBlockSize(grid->qnode, bs); CHKERRQ(ierr);
  ierr = VecSetType(grid->qnode, VECMPI);CHKERRQ(ierr);

  // Creates a new vector of the same type as an existing vector
  // Residual
  ierr = VecDuplicate(grid->qnode, &grid->res); CHKERRQ(ierr);
  // Global distributed solution vector
  ierr = VecDuplicate(grid->qnode, &tsCtx->qold); CHKERRQ(ierr);
  ierr = VecDuplicate(grid->qnode, &tsCtx->func); CHKERRQ(ierr);

  /*
    // Creates a standard, sequential array-style vector
    // Local sequential solution vector
    ierr = VecCreateSeq(MPI_COMM_SELF, bs * nvertices, 
                      &grid->qnodeLoc); CHKERRQ(ierr);
  */

  ierr = VecCreate(MPI_COMM_SELF, &grid->qnodeLoc); CHKERRQ(ierr);
  ierr = VecSetSizes(grid->qnodeLoc, bs * nvertices, bs * nvertices);
  CHKERRQ(ierr);
  ierr = VecSetBlockSize(grid->qnodeLoc, bs); CHKERRQ(ierr);
  ierr = VecSetType(grid->qnodeLoc, VECSEQ); CHKERRQ(ierr);

  /*
    // Gradient Vector
    ierr = VecCreateMPI(comm, 3 * bs * nnodesLoc, 3 * bs * nnodes,
                      &grid->grad); CHKERRQ(ierr);
  */

  ierr = VecCreate(comm, &grid->grad);
  ierr = VecSetSizes(grid->grad, 3 * bs * nnodesLoc, 3 * bs * nnodes);
  CHKERRQ(ierr);
  ierr = VecSetBlockSize(grid->grad, 3 * bs); CHKERRQ(ierr);
  ierr = VecSetType(grid->grad, VECMPI); CHKERRQ(ierr);

  /*
    // Local Gradient Vector
    ierr = VecCreateSeq(MPI_COMM_SELF, 3 * bs * nvertices, 
                      &grid->gradLoc); CHKERRQ(ierr);
  */

  ierr = VecCreate(MPI_COMM_SELF, &grid->gradLoc);
  ierr = VecSetSizes(grid->gradLoc,3 * bs * nvertices, 3 * bs * nvertices);
  CHKERRQ(ierr);
  ierr = VecSetBlockSize(grid->gradLoc, 3 * bs); CHKERRQ(ierr);
  ierr = VecSetType(grid->gradLoc, VECSEQ); CHKERRQ(ierr);

  // ==========================================================
  // Create scatter between the local and global vectors
  // 1. Create scatter for gnode
  
  // Creates a data structure for an index set containing a
  // list of evenly spaced integers [starting point and stride]
  // Parameters:
  // 1. Length of the locally owned portion of the IS
  // 2. First element of the locally owned portion of the IS
  // 3. step the chnage to the next index
  // 4. Output parameter --> new index set
  ierr = ISCreateStride(MPI_COMM_SELF, bs * nvertices, 0, 1,
                        &islocal);
# ifdef INTERLACING
# ifdef BLOCKING // interlacing and blocking
  ICALLOC(nvertices, &svertices);
  for(i = 0; i < nvertices; i++)
    svertices[i] = loc2pet[i];

  // Creates a data structure for an index set containing 
  // a list of integers. The indices are relative 
  // to entries, not blocks; the parameters are:
  // Number of elements in each block
  // Length of the index set (number of blocks)
  // List of integers, one for each block and count of block
  // not indices.
  // Determines how an array passed to certain functions 
  // is copied or retained
  // New index set
  ierr = ISCreateBlock(MPI_COMM_SELF, bs, nvertices, svertices,
                       PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr);
# else // interlacing and non-blocking
  ICALLOC(bs * nvertices, &svertices);
  for(i = 0; i < nvertices; i++)
    for(j = 0; j < bs; j++)
      svertices[j + bs * i] = j + bs * loc2pet[i];
  // Creates a data structure for an index set containing 
  // a list of integers; the parameters are:
  // the length of the index set
  // the list of integers
  // copy mode
  // new index set
  ierr = ISCreateGeneral(MPI_COMM_SELF, bs * nvertices, svertices,
                         PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr); 
# endif
# else // non-interlacing
  ICALLOC(bs * nvertices, &svertices);
  for(j = 0; j < bs; j++)
    for(i = 0; i < nvertices; i++)
      svertices[j * nvertices + i] = j * nvertices + loc2pet[i];

  ierr = ISCreateGeneral(MPI_COMM_SELF, bs * nvertices, svertices,
                         PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr); 
# endif
  free(svertices);
  // Creates a vector scatter context; parameters:
  // A vec that defines the shape of vectors from which we scatter
  // The indices of the scatter vector (from)
  // A vec that defines the shape of vectors to which we scatter
  // The indices of the scatter vector (to)
  ierr = VecScatterCreate(grid->qnode, isglobal, grid->qnodeLoc,
                          islocal, &grid->scatter); 
  CHKERRQ(ierr);
  ierr = ISDestroy(&isglobal);  CHKERRQ(ierr);
  ierr = ISDestroy(&islocal);   CHKERRQ(ierr);

  // 2. Create scatter for gradient vector of qnode

  ierr = ISCreateStride(MPI_COMM_SELF, 3 * bs * nvertices,
                        0, 1, &islocal); CHKERRQ(ierr);
# ifdef INTERLACING
# ifdef BLOCKING
  ICALLOC(nvertices,&svertices);
  for (i = 0; i < nvertices; i++)
    svertices[i] = loc2pet[i];
  
  ierr = ISCreateBlock(MPI_COMM_SELF, 3 * bs, nvertices, 
                       svertices, PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr);
# else
  ICALLOC(3 * bs * nvertices, &svertices);
  for (i = 0; i < nvertices; i++)
    for (j = 0; j < 3 * bs; j++)
      svertices[j + 3 * bs * i] = j + 3 * bs * loc2pet[i];
  
  ierr = ISCreateGeneral(MPI_COMM_SELF, 3 * bs * nvertices, 
                         svertices, PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr);
# endif
# else
  ICALLOC(3 * bs * nvertices, &svertices);
  for (j = 0; j < 3 * bs; j++)
    for (i = 0; i < nvertices; i++)
      svertices[j * nvertices + i] = j * nvertices + loc2pet[i];
  
  ierr = ISCreateGeneral(MPI_COMM_SELF, 3 * bs * nvertices, 
                         svertices, PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr);
# endif
  free(svertices);
  ierr = VecScatterCreate(grid->grad, isglobal, grid->gradLoc,
                          islocal, &grid->gradScatter);
  CHKERRQ(ierr);
  
  ierr = ISDestroy(&isglobal);  CHKERRQ(ierr);
  ierr = ISDestroy(&islocal);   CHKERRQ(ierr);
  
  // ===========================================================

  // Store the number of non-zeros per row
# ifdef INTERLACING
# ifdef BLOCKING
  ICALLOC(nnodesLoc, &val_diag);
  ICALLOC(nnodesLoc, &val_offd);
  for(i = 0; i < nnodesLoc; i++)
  {
    jstart = grid->ia[i] - 1;
    jend   = grid->ia[i + 1] - 1;
    nbrs_diag = 0;  // Number of diagonal elements
    nbrs_offd = 0;  // Number of off-diagonal elements
    for(j = jstart; j < jend; j++)
    {
      if((grid->ja[j] >= rstart) && 
         (grid->ja[j] < (rstart + nnodesLoc)))
        nbrs_diag++; // Increase number of diagonal
      else 
        nbrs_offd++; // Increase number of off-diagonal
    }
    val_diag[i] = nbrs_diag;
    val_offd[i] = nbrs_offd;
  }
  // Left hand side matrix (( A ))
  // Creates a sparse parallel matrix in block AIJ format 
  // (block compressed row); parameters:
  // Block size; number of local rows; number of local columns
  // Number of global rows; number of global columns
  // Number of non-zero blocks per block row in the diagonal
  // Array contains the number of non-zero blocks in the diagonal
  // Number of non-zero blocks per block row in the off diagonal
  // Array contains the number of non-zero blocks in the off-dia
  // Note::: Use PETSc default value in both cases
  ierr = MatCreateBAIJ(comm, bs, bs * nnodesLoc, bs * nnodesLoc, 
                       bs * nnodes, bs * nnodes, PETSC_DEFAULT, 
                       val_diag, PETSC_DEFAULT, val_offd, 
                       &grid->A); CHKERRQ(ierr);
# else
  ICALLOC(nnodesLoc * 4, &val_diag);
  ICALLOC(nnodesLoc * 4, &val_offd);
  for(i = 0; i < nnodesLoc; i++)
  {
    jstart = grid->ia[i] - 1;
    jend   = grid->ia[i + 1] - 1;
    nbrs_diag = 0;
    nbrs_offd = 0;
    for(j = jstart; j < jend; j++)
    {
      if((grid->ja[j] >= rstart) && 
         (grid->ja[j] < (rstart + nnodesLoc)))
        nbrs_diag++;
      else 
        nbrs_offd++;
    }
    for(j = 0; j < 4; j++)
    {
      int row = 4 * i + j;
      val_diag[row] = nbrs_diag * 4;
      val_offd[row] = nbrs_offd * 4;
    }
  }
  ierr = MatCreateMPIAIJ(comm, bs * nnodesLoc, bs * nnodesLoc, 
                       bs * nnodes, bs * nnodes, PETSC_DEFAULT, 
                       val_diag, PETSC_DEFAULT, val_offd, 
                       &grid->A); CHKERRQ(ierr);
# endif
  free(val_diag);
  free(val_offd);
# else
  if(size > 1)
  {
    SETERRQ(PETSC_COMM_SELF, 1, "Parallel case not supported " 
                                "in non-interlaced case\n");
  }
  ICALLOC(nnodes * 4, &val_diag);
  ICALLOC(nnodes * 4, &val_offd);
  for(j = 0; j < 4; j++)
  {
    for(i = 0; i < nnodes; i++)
    {
      int row = i + j * nnodes;
      jstart = grid->ia[i] - 1;
      jend   = grid->ia[i + 1] - 1;
      nbrs_diag = jend - jstart;
      val_diag[row] = nbrs_diag * 4;
      val_offd[row] = 0;
    }
  }
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF, bs * nnodes, 
                         bs * nnodes, PETSC_DEFAULT, 
                         val_diag, &grid->A); 
  CHKERRQ(ierr);
  
  free(val_diag);
  free(val_offd);

# endif
  
  // Set local to global mapping for setting the matrix
  // elements in local ordering: first set row by row mapping

# ifdef INTERLACING
  ICALLOC(nvertices, &svertices);
//  k = 0;
  for(i = 0; i < nvertices; i++)
      svertices[i] = loc2pet[i]; // global index
//    for(j = 0; j < bs; j++)
//
//
  // Creates a mapping between a local (0 to n) ordering 
  // and a global parallel ordering; parameters:
  // Number of local elements
  // Global index of each local element
  // Copy mode
  // new mapping data structure
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF, 
                                      bs, nvertices,
                                      svertices, 
                                      PETSC_COPY_VALUES,
                                      &isl2g); CHKERRQ(ierr);
  // Sets a local-to-global numbering for use by the 
  // routine MatSetValuesLocal() to allow users to insert 
  // matrix entries using a local (per-processor) numbering.
  // Parameters: 
  // Row mapping created with ISLocalToGlobalMappingCreate
  // Column mapping
  ierr = MatSetLocalToGlobalMapping(grid->A, isl2g, isl2g);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&isl2g); CHKERRQ(ierr);
  
  // Set the blockwise local to global mapping
# ifdef BLOCKING
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF,
                                      bs, nvertices,
                                      loc2pet,
                                      PETSC_COPY_VALUES,
                                      &isl2g); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(grid->A, isl2g, isl2g);
  CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&isl2g); CHKERRQ(ierr);
# endif
  free(svertices);
# endif
  PetscFunctionReturn(0);
}

// Evaluate Function F(x) 
// SNES context
// State at which to evaluate residual
// Vector to put residual (function value)
// used-defined function context
#undef  __FUNCT__
#define __FUNCT__ "FormFunction"
int FormFunction(SNES snes, Vec x, Vec f, void *dummy)
{
  AppCtx      *user   = (AppCtx *) dummy; // user-defined function context
  GRID        *grid   = user->grid;       // Working grid
  TstepCtx    *tsCtx  = user->tsCtx;      // Time step
  double      *qold;

  // Specifies that all variables declared subsequently
  // are available on the coprocessor
  // push: turns on the pragma.
  // It pushes target-name onto an internal compiler stack,
  // so that all subsequent variables are targeted for Intel®
  // MIC Architecture until the statement
  // #pragma offload_attribute (pop, target(target-name))
  // pop: turns off the pragma. Removes target-name from the
  // top of the internal compiler stack.
# ifdef OFFLOAD_MODE
# pragma offload_attribute(push, target(mic))
  double      *qnode;   // Nodes
  double      *res;     // Residual
  double      *grad;    // Gradient
# pragma offload_attribute(pop)
# else
  double      *qnode;   // Nodes
  double      *res;     // Residual
  double      *grad;    // Gradient
# endif
  
  double      temp;                 // Dummy array
  // Scatter context
  VecScatter  scatter     = grid->scatter;      // Node Vector
  VecScatter  gradScatter = grid->gradScatter;  // Gradient Vector 
  // PETSc vectors
  Vec         localX    = grid->qnodeLoc;   // Nodes
  Vec         localGrad = grid->gradLoc;    // Gradient
  
  int         i, j, ierr;         // Loop iterators and the PETSc error
  int         in;
  int         nbface;             // Number of faces
  int         ires;
  // Calculating the time
  double      time_ini, time_fin;  // Timer

  PetscFunctionBegin;
  // Get X into the local work vector
  // Begins a generalized scatter from one vector to another
  // Parameters: scatter context; vector from which we scatter
  // vector to which we scatter
  // INSERT_VALUES: any location not scattered to retains its old value
  // Scattering modey
  // For more information, visit: 
  // http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecScatterBegin.html
  ierr = VecScatterBegin(scatter, x, localX, INSERT_VALUES, 
                         SCATTER_FORWARD); CHKERRQ(ierr);
  // Ends a generalized scatter from one vector to another
  ierr = VecScatterEnd(scatter, x, localX, INSERT_VALUES,
                       SCATTER_FORWARD); CHKERRQ(ierr);
  // Access the local work f, grad, and input

  // Returns a pointer to a contiguous array that contains 
  // this processor's portion of the vector data
  ierr = VecGetArray(f, &res);            CHKERRQ(ierr);
  ierr = VecGetArray(grid->grad, &grad);  CHKERRQ(ierr);
  ierr = VecGetArray(localX, &qnode);     CHKERRQ(ierr);
  ires = tsCtx->ires;
  
  ierr = PetscTime(&time_ini); CHKERRQ(ierr);
  // user.h: #define f77LSTGS f77name(LSTGS,lstgs,lstgs_)
  // Calculates the Gradients at the nodes using weighted least squares
  // This subroutine solves using Gram-Schmidt
  f77LSTGS(&grid->nnodesLoc, &grid->nedgeLoc, grid->eptr,
           qnode, grad, grid->xyz, grid->rxy, &rank,
           &grid->nvertices);
  // Time this phase 
  ierr = PetscTime(&time_fin); CHKERRQ(ierr);
  grad_time += time_fin - time_ini; // Store the time in the global variable
  // Restores a vector after VecGetArray() has been called
  ierr = VecRestoreArray(grid->grad, &grad); CHKERRQ(ierr);

  // Local gradient
  ierr = VecScatterBegin(gradScatter, grid->grad, localGrad,
                         INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(gradScatter, grid->grad, localGrad,
                       INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecGetArray(localGrad, &grad); CHKERRQ(ierr);
  // Sum of faces: solid, viscous, and free faces
  nbface = grid->nsface + grid->nvface + grid->nfface;
  
# ifdef OFFLOAD_MODE
  // ********************************************************************
  // ********************************************************************
  //
  // Allocate the coprocessor memory
  // Transfer asynchronously the data to the coprocessor memory
  // Those the global data structure that are used in the 
  // flux evaluation routine written in Fortran
  // which is called every time the solver evaluate the 
  // residual of
  // the conservation law
  // evaluating the PDE
  // the other data structure will be sent in the Fortran code 
  // because they have been initialized and declared there
  // the following data structure are allocated once and they 
  // stay there
  // in the coprocessor memory for further usage 
  // (alloc_if, free_if)
  // and they will be deallocated at the end of the function
  // After loading the prgama we further check the 
  // offloading process
  // via a function that mainly make sure that the offloading has 
  // been preformed correctly
  int nvertices = grid->nvertices;
  # pragma  offload_transfer target(mic:target_id) mandatory  \
            in(grad : length(3 * 4 * nvertices) ALLOC KEEP)
  //
  // *********************************************************************
  // *********************************************************************
# endif
  // user.h: #define f77GETRES   f77name(GETRES,getres,getres_)
  // Calculates the residual
  f77GETRES(&grid->nnodesLoc, &grid->ncell, &grid->nedgeLoc, &grid->nsface,
            &grid->nvface, &grid->nfface, &nbface, &grid->nsnodeLoc, 
            &grid->nvnodeLoc, &grid->nfnodeLoc, grid->isface, grid->ivface,  
            grid->ifface, &grid->ileast, grid->isnode, grid->ivnode,  
            grid->ifnode, &grid->nnfacetLoc, grid->f2ntn, &grid->nnbound,
            &grid->nvfacetLoc, grid->f2ntv, &grid->nvbound, &grid->nffacetLoc,
            grid->f2ntf, &grid->nfbound, grid->eptr, grid->sxn, grid->syn, 
            grid->szn, grid->vxn, grid->vyn, grid->vzn, grid->fxn, grid->fyn, 
            grid->fzn, grid->xyzn, qnode, grid->cdt, grid->xyz, grid->area,
            grad, res, grid->turbre, grid->slen, grid->c2n, grid->c2e, grid->us,
            grid->vs, grid->as, grid->phi, grid->amut, &ires,
#           if defined(_OPENMP)
            &max_threads,
#           if defined(HAVE_EDGE_COLORING)
            &grid->ncolor, grid->ncount,
#           elif defined(HAVE_REDUNDANT_WORK)
            grid->resd,
#           else
            &grid->nedgeAllThr,
            grid->part_thr, grid->nedge_thr, grid->edge_thr, grid->xyzn_thr,
#           endif
#           if defined(OFFLOAD_MODE)
            &c_info->beta, &target_id,
#           endif
#           endif
            &tsCtx->LocalTimeStepping, &rank, &grid->nvertices);

# ifdef OFFLOAD_MODE
  // *********************************************************************
  // *********************************************************************
  //
  // Deallocate the Phi memory
  # pragma  offload_transfer target(mic:target_id) mandatory    \
            nocopy(grad : length(3 * 4 * nvertices) REUSE FREE)
  //
  // *********************************************************************
  // *********************************************************************
# endif
  // Add the contribution due to time stepping
  // f77GETRES routine calls first delta2 routine to get the time step for
  // each node and then it calls flux to calculate the residual 
  // This is my core contribution in the code because here we do the flux
  // calculation that will be ported in Xeon phi hardware
  if(ires == 1)
  {
    ierr = VecGetArray(tsCtx->qold, &qold); CHKERRQ(ierr);
#   ifdef INTERLACING
    for(i = 0; i < grid->nnodesLoc; i++)
    {
      temp = grid->area[i] / (tsCtx->cfl * grid->cdt[i]);
      for(j = 0; j < 4; j++)
      {
        in = 4 * i + j;
        res[in] += temp * (qnode[in] - qold[in]);
      }
    }
#   else
    for (j = 0; j < 4; j++) 
    {
      for (i = 0; i < grid->nnodesLoc; i++) 
      {
        temp = grid->area[i] / (tsCtx->cfl * grid->cdt[i]);
        in = grid->nnodesLoc * j + i;
        res[in] += temp * (qnode[in] - qold[in]);
      }
    }
#   endif
    ierr = VecRestoreArray(tsCtx->qold, &qold); CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(localX, &qnode);   CHKERRQ(ierr);
  ierr = VecRestoreArray(f, &res);          CHKERRQ(ierr);
  ierr = VecRestoreArray(localGrad, &grad); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Evaluate Jacobian F'(x)
// Input vector; matrix that defines the approximate Jacobian;
// matrix to be used to construct the preconditioner;
// flag indicating information about the preconditioner matrix structure
// user-defined context
#undef  __FUNCT__
#define __FUNCT__ "FormJacobian"
int FormJacobian(SNES snes, Vec x, Mat Jac, Mat B, void *dummy)
{
  AppCtx      *user   = (AppCtx *) dummy;
  GRID        *grid   = user->grid;
  TstepCtx    *tsCtx  = user->tsCtx;
  Mat         pc_mat  = B;             // preconditioner matrix
  Vec         localX  = grid->qnodeLoc;
  double      *qnode;
  int         ierr;                     // PETSc error

  PetscFunctionBegin;
 
  
  // Resets a factored matrix to be treated as unfactored
  ierr = MatSetUnfactored(pc_mat); CHKERRQ(ierr);
  ierr = VecGetArray(localX, &qnode); CHKERRQ(ierr);
  
  // Fill the nonzero term of the A matrix
  // user.F: #define f77FILLA  f77name(FILLA,filla,filla_)
  f77FILLA(&grid->nnodesLoc, &grid->nedgeLoc, grid->eptr, &grid->nsface,
           grid->isface, grid->fxn, grid->fyn, grid->fzn, grid->sxn,
           grid->syn, grid->szn, &grid->nsnodeLoc, &grid->nvnodeLoc,
           &grid->nfnodeLoc, grid->isnode, grid->ivnode, grid->ifnode,
           qnode, &pc_mat, grid->cdt, grid->area, grid->xyzn, &tsCtx->cfl,
           &rank, &grid->nvertices);
  
  ierr = VecRestoreArray(localX, &qnode); CHKERRQ(ierr);
  // Begins assembling the matrix
  ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

 // // Indicating information about the preconditioner matrix structure
// // *flag = SAME_NONZERO_PATTERN;
  // In order to execute this, we need print_freq less than the maximum 
  // time step, so that when the time step is dividable upon the print
  // frequency, this will execute the following routine
# ifdef MATRIX_VIEW
  if ((tsCtx->itstep != 0) && (tsCtx->itstep % tsCtx->print_freq) == 0)
  {
    PetscViewer viewer;
    char mat_file[PETSC_MAX_PATH_LEN];
    sprintf(mat_file, "mat_bin.%d", tsCtx->itstep);
    ierr = PetscViewerBinaryOpen(MPI_COMM_WORLD, mat_file,
                                 FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);
    ierr = MatView(pc_mat, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }
# endif
  PetscFunctionReturn(0);
}

// Form initial approximation
#undef  __FUNCT__
#define __FUNCT__ "FormInitialGuess"
int FormInitialGuess(SNES snes, GRID *grid)
{
  int         ierr;   // PETSc error
  double      *qnode;

  PetscFunctionBegin;
  ierr = VecGetArray(grid->qnode, &qnode); CHKERRQ(ierr);
  // user.h: extern void PETSC_STDCALL 
  // f77INIT(int*,PetscScalar*,PetscScalar*,PetscScalar*,int*,int*,int*);
  // Initializes the flow field
  f77INIT(&grid->nnodesLoc, qnode, grid->turbre, grid->amut, 
          &grid->nvnodeLoc, grid->ivnode, &rank);
  ierr = VecRestoreArray(grid->qnode, &qnode); CHKERRQ(ierr);

  //VecView(grid->qnode, PETSC_VIEWER_DEFAULT);

  PetscFunctionReturn(0);
}

// Solve non-linear system
#undef  __FUNCT__
#define __FUNCT__ "Update"
int Update(SNES snes, void *ctx)
{
  AppCtx      *user       = (AppCtx *) ctx;
  GRID        *grid       = user->grid;
  TstepCtx    *tsCtx      = user->tsCtx;
  VecScatter  scatter     = grid->scatter;
  Vec         localX      = grid->qnodeLoc;
  PetscBool   print_flag  = PETSC_FALSE;  
  FILE        *fptr       = 0;
  int         nfailsCum   = 0; 
  int         nfails      = 0;

  double      *qnode, *res;
  double      clift, cdrag, cmom;
  int         ierr, its;
  double      fratio;
  double      time1, time2, cpuloc, cpuglo;
  int         max_steps;

  /*  Scalar         cpu_ini,cpu_fin,cpu_time;*/
  /*  int            event0 = 14,event1 = 25,gen_start,gen_read;
      PetscScalar    time_start_counters,time_read_counters;
      long long      counter0,counter1;
  */

  PetscFunctionBegin;
  

  ierr = PetscOptionsHasName(NULL, NULL, "-print", &print_flag);
  CHKERRQ(ierr);
  if(print_flag)
  {
    ierr = PetscFOpen(PETSC_COMM_WORLD, "history.out", "w",
                      &fptr); CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_WORLD, fptr, 
                        "VARIABLES = iter, cfl, fnorm, clift,"
                        "cdrag, cmom, cpu\n"); CHKERRQ(ierr);
  }
  if(user->PreLoading)
    max_steps = 1;
  else
    max_steps = tsCtx->max_steps;
  fratio = 1.0;
  /* tsCtx->ptime = 0.0; */
  ierr = VecCopy(grid->qnode, tsCtx->qold); CHKERRQ(ierr);
  ierr = PetscTime(&time1); CHKERRQ(ierr);
  
  for(tsCtx->itstep = 0; 
      (tsCtx->itstep < max_steps) && (fratio <= tsCtx->fnorm_ratio);
      tsCtx->itstep++)
  {
    // Compute the time step (residuals and the flux) 
    // through calling FormFunction which evaluates F(u) = 0
    ierr = ComputeTimeStep(snes, tsCtx->itstep, user); CHKERRQ(ierr);
    // Solves a nonlinear system F(x) = b
    // Null means b = 0 (constant part of the equation)
    // x is the solution vector 
    // global distributed solution vector
    // Here FormJacobian is called 
    ierr = SNESSolve(snes, NULL, grid->qnode); CHKERRQ(ierr);
    // Gets the number of nonlinear iterations completed at this time
    ierr = SNESGetIterationNumber(snes, &its); CHKERRQ(ierr);
    // Gets the number of unsuccessful steps attempted by 
    // the nonlinear solver
    ierr = SNESGetNonlinearStepFailures(snes, &nfails); CHKERRQ(ierr);
    nfailsCum += nfails; 
    nfails = 0;
    // Fails in the newton steps
    if (nfailsCum >= 2) 
      SETERRQ(PETSC_COMM_SELF, 1, "Unable to find a Newton Step");
    // Output the resutls
    if (print_flag)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD, 
                         "At Time Step %d cfl = %g and fnorm = %g\n",
                         tsCtx->itstep, tsCtx->cfl, tsCtx->fnorm);
      CHKERRQ(ierr);
    }
    ierr = VecCopy(grid->qnode, tsCtx->qold); CHKERRQ(ierr);
    c_info->ntt = tsCtx->itstep + 1;
    ierr = PetscTime(&time2); CHKERRQ(ierr);
    cpuloc = time2 - time1;            
    cpuglo = 0.0;
    // Reduction operation to compute the total maximum time of the
    // non-linear solve time
    ierr = MPI_Allreduce(&cpuloc, &cpuglo, 1, MPIU_REAL, MPI_MAX,
                         PETSC_COMM_WORLD); CHKERRQ(ierr);
    c_info->tot = cpuglo;    // Total CPU time used upto this time step

    ierr = VecScatterBegin(scatter, grid->qnode, localX, INSERT_VALUES,
                           SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, grid->qnode, localX, INSERT_VALUES,
                         SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGetArray(grid->res, &res); CHKERRQ(ierr);
    ierr = VecGetArray(localX, &qnode); CHKERRQ(ierr);
    // user.h: #define f77FORCE f77name(FORCE,force,force_)
    // Calculate the forces
    f77FORCE(&grid->nnodesLoc, &grid->nedgeLoc, grid->isnode, grid->ivnode, 
             &grid->nnfacetLoc, grid->f2ntn, &grid->nnbound, &grid->nvfacetLoc,
             grid->f2ntv, &grid->nvbound, grid->eptr, qnode, grid->xyz, 
             grid->sface_bit, grid->vface_bit, &clift, &cdrag, &cmom, &rank,
             &grid->nvertices);   
    if(print_flag)
    {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "%d\t%g\t%g\t%g\t%g\t%g\n", 
                         tsCtx->itstep, tsCtx->cfl, tsCtx->fnorm, clift, cdrag,
                         cmom); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Wall clock time needed %g "
                         "seconds for %d time steps\n", cpuglo, tsCtx->itstep);
      CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_WORLD, fptr, "%d\t%g\t%g\t%g\t%g\t%g\t%g\n",
                          tsCtx->itstep, tsCtx->cfl, tsCtx->fnorm, clift, cdrag,
                          cmom, cpuglo); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(localX, &qnode); CHKERRQ(ierr);
    ierr = VecRestoreArray(grid->res, &res); CHKERRQ(ierr);
    fratio = tsCtx->fnorm_ini / tsCtx->fnorm;
    ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRQ(ierr);
  } // End of time step loop

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Total wall clock time needed %g seconds "
                     "for %d time steps\n", cpuglo, tsCtx->itstep); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "cfl = %g fnorm = %g\n", tsCtx->cfl, 
                     tsCtx->fnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "clift = %g cdrag = %g cmom = %g\n",
                     clift, cdrag, cmom); CHKERRQ(ierr);
  
  if (!rank && print_flag) 
    fclose(fptr); 
  
  if (user->PreLoading) 
  {
    tsCtx->fnorm_ini = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Preloading done ...\n");CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

// Calculate the time step
#undef  __FUNCT__
#define __FUNCT__ "ComputeTimeStep"
int ComputeTimeStep(SNES snes, int iter, void *ctx)
{
  AppCtx      *user   = (AppCtx *) ctx;
  TstepCtx    *tsCtx  = user->tsCtx;
  Vec         func    = tsCtx->func;
  double      inc     = 1.1;
  double      newcfl;
  int         ierr;

  PetscFunctionBegin;
  tsCtx->ires = 0;
  // Calculate the residual plus the fluxes 
  ierr = FormFunction(snes, tsCtx->qold, func, user); CHKERRQ(ierr);
  tsCtx->ires = 1;
  // Computes the vector norm
  // Vector x; NORM 2 (the two norm, ||v|| = sqrt(sum_i (v_i)^2))
  // the norm [output parameter]
  ierr = VecNorm(func, NORM_2, &tsCtx->fnorm); CHKERRQ(ierr);
  // First time through so compute initial function norm 
  if (tsCtx->fnorm_ini == 0.0) 
  {
    tsCtx->fnorm_ini = tsCtx->fnorm;
    tsCtx->cfl       = tsCtx->cfl_ini;
  } 
  else 
  {
    newcfl     = inc * tsCtx->cfl_ini * tsCtx->fnorm_ini / tsCtx->fnorm;
    tsCtx->cfl = PetscMin(newcfl, tsCtx->cfl_max);
  }


  PetscFunctionReturn(0);
}
