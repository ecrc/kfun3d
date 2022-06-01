
/*
  find_sorted_permutation.c
*/
int 
imain(size_t, unsigned int*, unsigned int*);
/*
  init.c
*/
int
init(TstepCtx *);
int
initTimeStepContext(TstepCtx *);
/*
  gridinfo.c
*/
int   
initGrid(GRID *);
int   
initPETScDS(void *);
int 
processEdges(int, int, int*, struct NORMAL);
int 
constructILU(int, int, int*);
int
remapCoordinates(int, int, struct COORD, double*);
int
partitionSubdomainUsingMeTiS(int, int, int*, int*, int*);
int
processEdgesLoc(int, int, int*, struct NORMAL, int*, int*, int*, 
                struct NORMAL);
void
cleanILUmem();
/*
  formfunc.c
*/
int   
formJacobian(SNES, Vec, Mat , Mat , void *);
int   
formFunction(SNES, Vec, Vec, void *);
int   
formInitialGuess(void *);
/*
  kernel.c
*/
int   
update(SNES, void *);
int 
computePseudoTimeStep(SNES, void *);

//
//int   initPAPI();
/*
  Fortran routines
*/
EXTERN_C_BEGIN
/*
  ilu.F
*/
extern void PETSC_STDCALL 
f77GETIA(int*, int*, int*, int*, int*);
extern void PETSC_STDCALL 
f77GETJA(int*, int*, int*, int*, int*, int*);
/*
  grad.F
*/
extern void PETSC_STDCALL 
f77SUMGS( int*, int*, int*, int*, int*, double*, double*, double*, int*, 
          int*, double*, double*, double*, double*, double*, double*,
          double* );
extern void PETSC_STDCALL 
f77LSTGS( int*, int*, int*, int*, int*, double*, double*, double*, 
          double*, double*, double*, double*, double*, double*, 
          double*, double*, int*, int*, double*);
/*
  initfort.F   
*/
extern void PETSC_STDCALL 
f77INITCOMM();
extern void PETSC_STDCALL 
f77INIT(int*, double*);
/*
  timestep.F
*/
extern void PETSC_STDCALL 
f77DELTAT2( int*, int*, int*, int*, int*, int*, int*, int*, double*,
            double*, double*, double*, double*, double*, double*, 
            double*, double*, double*, double*, double*, int*, int*,
            int*, int*, double* );
/*
  flux.F
*/            
extern void PETSC_STDCALL 
f77FLUX(int*, int*, int*, int*, int*, int*, int*, int*, int*, double*, 
        double*, double*, double*, double*, double*, double*, double*, 
        double*, double*, double*, double*, int*, int*, int*, int*, 
        int*, int*, double*, double*
#ifdef HW_COUNTER        
        ,int*, long long*
#endif
        );
/*
  fill.F
*/
extern void PETSC_STDCALL 
f77FILLA( int*, int*, int*, int*, int*, double*, int*, int*, double*,
          double*, double*, double*, double*, double*, double*, 
          double*, double*, double*, double*, double*, double*, 
          int*, int*, int*, int*, Mat* );
/*
  force.F
*/
extern void PETSC_STDCALL 
f77FORCE( int*, int*, double*, double*, double*, double*, 
          int*, int*, double*, double*, double* );

EXTERN_C_END
