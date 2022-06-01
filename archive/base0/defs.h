
#include <petsc.h>
#include <petsc/private/fortranimpl.h>
#include <omp.h>
//#include <papi.h>
/*
  Global Variables
*/
double  gradTime; // Gradient calculation time  
double  fluxTime; // Flux evaluation timing
int     nThreads; // Number of threads at each MPI process
int     rank;     // MPI process ID
int     size;     // MPI number of running processes

#ifdef HW_COUNTER
int       eventset; // PAPI counter context
long long handler;  // PAPI counter return value 
#endif

/*
  FORTRAN routine definitions
*/
#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define f77name(ucase,lcase,lcbar) lcbar
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define f77name(ucase,lcase,lcbar) ucase
#else
#define f77name(ucase,lcase,lcbar) lcase
#endif
/*
  Node coordinates in a 3D mesh
*/
struct COORD
{
  double *x;  // X component
  double *y;  // Y component
  double *z;  // Z component
};
/*
  Normal to faces and length
*/
struct NORMAL
{
  double  *len;         // Length
  struct  COORD coord;  // X, Y, and Z components
};
/*
  Weights for calculating the gradients using least square
*/
struct WEIGHT
{
  double *r11;  // w11
  double *r12;  // r12r11
  double *r13;  // r13r11
  double *r22;  // w22
  double *r23;  // r23r22
  double *r33;  // w33
  double *r44;  // rmess
};
/*
  MeTiS partitioning using for OpenMP
*/
struct THREAD
{
  int     nedge;          // Number edge for all threads with replication
  int     *partitions;    // MeTiS nodes partitioning
  int     *nedgeLoc;      // Number of edges local for each thread
  int     *eptr;          // Edge pointers
  struct  NORMAL  normal; // Normal to faces and length
};
/*
  A boundary's elements
*/
struct BOUNDELM
{
  int     nnode;        // Number of nodes
  int     nfacet;       // Number of facets
  int     *f2nt;        // Facet to tetrahedron indices
  int     *inode;       // nodes pointers
  struct  COORD normal; // Normal to faces
};
/*
  Boundaries
*/
struct BOUND
{
  int     nsbound;            // Number of solid boundaries
  int     nvbound;            // Number of viscous boundaries
  int     nfbound;            // Number of free boundaries
  struct  BOUNDELM solid;     // Solid boundary elements
  struct  BOUNDELM viscs;     // Viscous boundary elements
  struct  BOUNDELM ffild;     // Free boundary elements
};
/*
  Grid information
*/
typedef struct grid_t
{
  int     nnodes;           // Number of mesh nodes
  int     nedge;            // Number of mesh edges
  int     *eptr;            // Edge pointers array   
  struct  NORMAL  normal;   // Normal to faces and length
  struct  COORD   coord;    // Node Coordinates
  struct  THREAD  thread;   // MeTiS subdomain partitioning elements
  struct  BOUND   bound;    // Boundaries elements
  struct  WEIGHT  weight;   // Least square weights
  double  *area;            // Area of control volumes
  double  *cdt;             // Pseudo-time step for each cell
} GRID;                                            
/* 
  PETSc data structures
*/
typedef struct gridPETSc_t
{
  Vec   qnode;        // Global distributed solution vector
  Vec   qnodeLoc;     // Local sequential solution vector
  Vec   res;          // Residual vector
  Vec   grad;         // Gradient vector
  Mat   A;            // Left hand side matrix
} GRIDPETSc;
/*
  Time step context
*/
typedef struct tstepctx_t 
{
  /*
    A vector state at which to evaluate residual (x) at every
    time step (hold the old x vector before proceeding to the
    next SNES time step)
  */
  Vec     qnode;
  /*
    A vector to put residual (function value) (f) used at every 
    time step calculation
  */
  Vec     res;
  /*
    Courant–Friedrichs–Lewy stability restriction condition
  */
  double  CFL;
  double  CFLInit;
  double  CFLMax; 
  /*
    2-norm of function at current iterate
  */
  double  fnorm;
  double  fnormInit; 
  double  fnormRatio;
  /*
    A flag to indicate that the formFunction is called to compute the 
    pseudo time step
  */
  int     isPseudoTimeStep;
  /*
    Time step counting variable: time step loop iterator (i)
  */
  int     iTimeStep;
  /*
    Maximum time steps for the nonlinear solve
  */
  int     maxTimeSteps;
  /*
    A flag for local or global time step
  */  
  int     isLocal;
} TstepCtx;
/*
  Encapsulate the grid data structure information and time step
  data structure information
*/
typedef struct appctx_t 
{
  GRID        *grid;        // Pointer to grid info
  GRIDPETSc   *gridPETSc;   // PETSc data structure context
  TstepCtx    *tsCtx;       // Pointer to time stepping context
  PetscBool   isPreLoading; // Preloading flag (single time step)
} AppCtx;
