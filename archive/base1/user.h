#include <petsc.h>
#include <petsc/private/fortranimpl.h>

#define MEMORY_ALIGNMENT
//#define OFFLOAD_MODE
//#define HAVE_REDUNDANT_WORK
//#define HAVE_EDGE_COLORING

#ifdef _OPENMP
#include <omp.h>
#ifdef OFFLOAD_MODE
#include <offload.h>
#endif
#ifndef HAVE_EDGE_COLORING
#include <metis.h> 
#endif
#endif
                  
#define max_colors    200
#define MAX_FILE_SIZE 256
#define mem_alignment 64

#define REAL          double

#if defined(MEMORY_ALIGNMENT)
#define ICALLOC(size, y) ierr = \
        posix_memalign((void **) y, mem_alignment, PetscMax(size, 1) * sizeof(int)); \
        CHKERRQ(ierr);
#define FCALLOC(size, y) ierr = \
        posix_memalign((void **) y, mem_alignment, PetscMax(size, 1) * sizeof(double)); \
        CHKERRQ(ierr);
#else
#define ICALLOC(size, y) ierr = \
        PetscMalloc((PetscMax(size, 1)) * sizeof(int), y);\
        CHKERRQ(ierr);
#define FCALLOC(size, y) ierr = \
        PetscMalloc((PetscMax(size, 1)) * sizeof(double), y);\
        CHKERRQ(ierr);
#endif

#if defined(OFFLOAD_MODE)
#pragma offload_attribute(push, target(mic))
int max_threads;
#pragma offload_attribute(pop)
#else
int max_threads;
#endif

#ifdef OFFLOAD_MODE
#define  ALLOC  alloc_if(1)
#define  REUSE  alloc_if(0)
#define  FREE   free_if(1)
#define  KEEP   free_if(0)
#endif 

int     rank;                 // MPI process rank
int     size;                 // Number of MPI processes
int     rstart;               // Used for identifying the ghost points in the mesh 

typedef struct gxy{                                   /* GRID STRUCTURE             */
  int         nnodes;                                 /* Number of nodes            */
  int         ncell;                                  /* Number of cells            */
  int         nedge;                                  /* Number of edges            */
  int         nnbound;
  int         nvbound;
  int         nfbound;
  int         nnfacet;
  int         nvfacet;
  int         nffacet;
  int         ntte;
  int         nsface;                                 /* Total # of solid faces     */
  int         nvface;                                 /* Total # of viscous faces   */
  int         nfface;                                 /* Total # of far field faces */
  int         nsnode;                                 /* Total # of solid nodes     */
  int         nvnode;                                 /* Total # of viscous nodes   */
  int         nfnode;                                 /* Total # of far field nodes */
  double      *rxy;
# if defined(OFFLOAD_MODE)                            /* Offloading mode            */
# pragma offload_attribute(push, target(mic))         /* Define in the coprocessor  */
  int         nnodesLoc;                              /* Local nodes                */
  int         nedgeLoc;                               /* Local edges                */
  int         nvertices;                              /* Local nodes + ghost nodes  */
  double      *xyz;                                   /* Node Coordinates           */
  int         *eptr;                                  /* Edge pointers              */
# ifndef HAVE_EDGE_COLORING                           /* Using MeTiS                */
  int         nedgeAllThr;
  int         *part_thr;
  int         *nedge_thr;
  int         *edge_thr;
  double      *xyzn_thr;
# else                                                /* Using edge coloring        */
  int         ncolor;                                 /* Number of colors           */
  int         ncount[max_colors];                     /* No. of faces in color      */
# endif
# pragma offload_attribute(pop)
# else                                                /* Not Offloading mode        */
  int         nnodesLoc;                              /* Local nodes                */
  int         nedgeLoc;                               /* Local edges                */
  int         nvertices;                              /* Local nodes + ghost nodes  */
  double      *xyz;                                   /* Node Coordinates           */
  int         *eptr;                                  /* Edge pointers              */
# ifndef HAVE_EDGE_COLORING
  int         nedgeAllThr;  
  int         *part_thr;
  int         *nedge_thr;
  int         *edge_thr;
  double      *xyzn_thr;
# else
  int         ncolor;
  int         ncount[max_colors];
# endif
# endif  

# ifdef HAVE_REDUNDANT_WORK   
  double      *resd;  
# endif     
  int         jvisc;                                  /* 0 = Euler                  */
                                                      /* 1 = laminar no visc LHS    */
                                                      /* 2 = laminar w/ visc LHS    */
                                                      /* 3 = turb B-B no visc LHS   */
                                                      /* 4 = turb B-B w/ visc LHS   */
                                                      /* 5 = turb Splrt no visc LHS */
                                                      /* 6 = turb Splrt w/ visc LHS */
  int         ileast;                                 /* 1 = Lst square gradient    */
  int         *isface;                                /* Face # of solid faces      */
  int         *ifface;                                /* Face # of far field faces  */
  int         *ivface;                                /* Face # of viscous faces    */
  int         *isford;                                /* Copies of isface, ifface,  */
  int         *ifford;                                /*  and ivface used for       */
  int         *ivford;                                /*  ordering                  */
  int         *isnode;                                /* Node # of solid nodes      */
  int         *ivnode;                                /* Node # of viscous nodes    */
  int         *ifnode;                                /* Node # of far field nodes  */
  int         *c2n;                                   /* Cell-to-node pointers      */
  int         *c2e;                                   /* Cell-to-edge pointers      */
  int         *ia, *ja;                               /* Stuff for ILU(0)           */ 
  int         *loc2pet;                               /* local to PETSc mapping     */
  int         *loc2glo;                               /* local to global mapping    */
  double      *area;                                  /* Area of control volumes    */
  int         *nntet,*nnpts;
  int         *f2ntn;
  double      *sxn, *syn, *szn, *sa;                  /* Normals at solid nodes     */
  int         *v2p;				                            /* Vertex to processor mapping*/
  int         *sface_bit, *vface_bit;
  int         *nvtet,*nvpts,*nftet,*nfpts;
  int         *f2ntv,*f2ntf;
  double      *vxn, *vyn, *vzn, *va;                  /* Normals at viscous nodes   */
  double      *fxn, *fyn, *fzn, *fa;                  /* Normals at far field nodes */
  double      *xyzn;                                  /* Normal to faces and length */
  double      *cdt;                                   /* Local time step            */
  double      *phi;                                   /* Flux limiter               */
  double      *dft1, *dft2;                           /* Turb mod linearization     */
  double      *slen;                                  /* Generalized distance       */
  double      *turbre;                                /* nu x turb Reynolds #       */
  double      *amut;                                  /* Turbulent mu (viscosity)   */
  double      *turbres;                               /* Turbulent residual         */
  double      *us, *vs, *as;                          /* For linearizing viscous    */
  int         nsnodeLoc, nvnodeLoc, nfnodeLoc;
  int         nnfacetLoc, nvfacetLoc, nffacetLoc; 

  /* PETSc datastructures and other related info */
  Vec         qnode;                                  /* Global distributed solution vector        */
  Vec         qnodeLoc;                               /* Local sequential solution vector          */
  Vec         qold;                                   /* Global distributed solution vector        */
  Vec         res;                                    /* Residual                                  */
  Vec         grad;                                   /* Gradient Vector	                         */
  Vec         gradLoc;                  	            /* Local Gradient Vector	                   */
  Vec         B;                                      /* Right hand side                           */
  Mat         A;                                      /* Left hand side                            */
  VecScatter  scatter, gradScatter;                   /* Scatter between local and global vectors  */

} GRID;                                            

typedef struct{                                 /* GENERAL INFORMATION        */
  double      title[20];                        /* Title line                 */
# ifdef OFFLOAD_MODE
# pragma offload_attribute(push, target(mic))
  double      beta;                             /* Artificial Compress. Param */
# pragma offload_attribute(pop)
# else
  double      beta;                             /* Artificial Compress. Param */
# endif
  double      alpha;                            /* Angle of attack            */
  double      Re;                               /* Reynolds number            */
  double      dt;                               /* Input cfl                  */
  double      tot;                              /* total computer time        */
  double      res0;                             /* Begining residual          */
  double      resc;                             /* Current residual           */
  int         ntt;                              /* A counter                  */
  int         mseq;                             /* Mesh sequencing            */
  int         ivisc;                            /* 0 = Euler                  */
                                                /* 1 = laminar no visc LHS    */
                                                /* 2 = laminar w/ visc LHS    */
                                                /* 3 = turb BB no visc LHS    */
                                                /* 4 = turb BB w/ visc LHS    */
                                                /* 5 = turb SA w/ visc LHS    */
                                                /* 6 = turb SA w/ visc LHS    */
  int         irest;                            /* for restarts irest = 1     */
  int         icyc;                             /* iterations completed       */
  int         ihane;                            /* ihane = 0 for van leer fds */
                                                /*       = 1 for hanel flux   */
                                                /*       = 2 for Roe's fds    */
  int         ntturb;                           /* Counter for turbulence     */
} CINFO;                                        /* COMMON INFO                */

typedef struct {                                /* FLOW SOLVER VARIABLES      */
  double      cfl1;                             /* Starting CFL number        */
  double      cfl2;                             /* Ending   CFL number        */
  int         nsmoth;                           /* How many its for Res smooth*/
  int         iflim;                            /* 1=use limiter 0=no limiter */
  int         itran;                            /* 1=transition (spalart only)*/
  int         nbtran;                           /* No. of transition points   */
  int         jupdate;                          /* For freezing Jacobians */
  int         nstage;                           /* Number of subiterations    */
  int         ncyct;                            /* Subiterations for turb mod */
  int         iramp;                            /* Ramp CFL over iramp iters  */
  int         nitfo;                            /* Iterations first order     */
} CRUNGE;                                       /* COMMON RUNGE               */

typedef struct{                                 /*============================*/
  double      gtol;                             /* linear system tolerence    */
  int         icycle;                           /* Number of GMRES iterations */
  int         nsrch;                            /* Dimension of Krylov        */
  int         ilu0;                             /* 1 for ILU(0)               */
  int         ifcn;                             /* 0=fcn2 1=fcneval(nwt Krlv) */
} CGMCOM;                                       /* COMMON GMCOM               */

// Time stepping
typedef struct {
  Vec         qnew, qold, func;
  double      fnorm_ini, dt_ini, cfl_ini;
  double      ptime;
  double      cfl_max, max_time;
  double      fnorm, dt, cfl;
  double      fnorm_ratio;
  int         ires, iramp, itstep;
  int         max_steps, print_freq;
  int         LocalTimeStepping;                         
} TstepCtx;

// Encapsulate the grid data structure information and time step
// data structure information
typedef struct {
  GRID        *grid;      // Pointer to grid info
  TstepCtx    *tsCtx;     // Pointer to time stepping context
  PetscBool   PreLoading;
} AppCtx;

// C routines
// =============================
int GetLocalOrdering(GRID *);
#ifdef _OPENMP 
#ifdef HAVE_EDGE_COLORING
int EdgeColoring(int nnodes, int nedge, int *e2n, int *eperm, int *ncolor, int *ncount);
#endif
#endif
int SetPetscDS(GRID *, TstepCtx *);
int FormJacobian(SNES, Vec, Mat, Mat, void *);
int FormFunction(SNES, Vec, Vec, void *);
int FormInitialGuess(SNES, GRID *);
int Update(SNES, void *);
int ComputeTimeStep(SNES, int, void *);

/* =================  Fortran routines called from C ======================= */

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define f77name(ucase,lcase,lcbar) lcbar
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#define f77name(ucase,lcase,lcbar) ucase
#else
#define f77name(ucase,lcase,lcbar) lcase
#endif
#define f77FORLINK  f77name(FORLINK,forlink,forlink_)
#define f77CLINK    f77name(CLINK,clink,clink_)
#define f77GETIA    f77name(GETIA,getia,getia_)
#define f77GETJA    f77name(GETJA,getja,getja_)
#define f77SORTER   f77name(SORTER,sorter,sorter_)
#define f77SUMGS    f77name(SUMGS,sumgs,sumgs_)
#define f77INIT     f77name(INIT,init,init_)
#define f77LSTGS    f77name(LSTGS,lstgs,lstgs_)
#define f77GETRES   f77name(GETRES,getres,getres_)
#define f77FILLA    f77name(FILLA,filla,filla_)
#define f77FORCE    f77name(FORCE,force,force_)

EXTERN_C_BEGIN

extern void PETSC_STDCALL f77FORLINK(void);
extern void PETSC_STDCALL f77GETIA(int*,int*,int*,int*,int*,int*);
extern void PETSC_STDCALL f77GETJA(int*,int*,int*,int*,int*,int*,int*);
extern void PETSC_STDCALL f77SUMGS(int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*);
extern void PETSC_STDCALL f77INIT(int*,PetscScalar*,PetscScalar*,PetscScalar*,int*,int*,
                                  int*);
extern void PETSC_STDCALL f77LSTGS(int*,int*,int*,PetscScalar*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,int*,int*);
extern void PETSC_STDCALL f77GETRES(int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,
                                   int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,
                                   int*,int*,int*,int*,int*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,int*,
                                   int*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,int*,
#                                  if defined(_OPENMP) 
                                   int*,
#                                  if defined(HAVE_EDGE_COLORING)
                                   int*, int*,
#                                  elif defined(HAVE_REDUNDANT_WORK)     
                                   PetscScalar*,       
#                                  else
                                   int*,
                                   int*,int*,int*,PetscScalar*,
#                                  endif
#                                  ifdef OFFLOAD_MODE
                                   PetscScalar*, int*, 
#                                  endif
#                                  endif 
                                   int*,int*,int*
                                   );
extern void PETSC_STDCALL f77FILLA(int*,int*,int*,int*,int*,PetscScalar*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,PetscScalar*,PetscScalar*,int*,int*,int*,int*,int*,
                                   int*,PetscScalar*,Mat*,PetscScalar*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,int*,int*);
extern void PETSC_STDCALL f77FORCE(int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,
                                   PetscScalar*,PetscScalar*,int*,int*,PetscScalar*,PetscScalar*,
                                   PetscScalar*,int*,int*);
EXTERN_C_END
