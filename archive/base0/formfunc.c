
#include "defs.h"
#include "formfunc.h"
#include "header.h"
/*
  Evaluate Function F(x): Functional form used to convey the 
  nonlinear function to be solved by PETSc SNES
*/  
#undef  __FUNCT__
#define __FUNCT__ "formFunction"
int 
formFunction(SNES snes, Vec x, Vec f, void *ctx)
{
  AppCtx      *user       = (AppCtx *) ctx;
  GRID        *grid       = user->grid;
  GRIDPETSc   *gridPETSc  = user->gridPETSc;
  TstepCtx    *tsCtx      = user->tsCtx;
  double      *qold;
  double      *qnode;
  double      *res;
  double      *grad;
  double      temp;
  int         i, j, ierr;
  double      sTime, eTime;
  double      fluxTimeLoc = 0.0;

#ifdef HW_COUNTER  
  long long   val = 0;
#endif

  Vec         qnodeLoc      = gridPETSc->qnodeLoc;


  PetscFunctionBegin;

  ierr = VecCopy(x, qnodeLoc);

  ierr = VecGetArray(f, &res);                  CHKERRQ(ierr);
  ierr = VecGetArray(gridPETSc->grad, &grad);   CHKERRQ(ierr);
  ierr = VecGetArray(qnodeLoc, &qnode);         CHKERRQ(ierr);
  
  ierr = PetscTime(&sTime); CHKERRQ(ierr);
  /*  
    Calculates the Gradients at the nodes using weighted least squares
    This subroutine solves using Gram-Schmidt
  */
  f77LSTGS( &grid->nnodes, &grid->nedge, &grid->thread.nedge, &nThreads,
            grid->thread.eptr, qnode, grid->coord.x, grid->coord.y, 
            grid->coord.z, grid->weight.r11, grid->weight.r12,
            grid->weight.r13, grid->weight.r22, grid->weight.r23,
            grid->weight.r33, grid->weight.r44, grid->thread.partitions, 
            grid->thread.nedgeLoc, grad );

  ierr = PetscTime(&eTime); CHKERRQ(ierr);
  gradTime += eTime - sTime;
  
  if(tsCtx->isPseudoTimeStep == 0)
  {
    f77DELTAT2( &grid->nnodes, &grid->nedge, &grid->thread.nedge,
                &grid->bound.solid.nnode, &grid->bound.ffild.nnode,
                &tsCtx->isLocal, &nThreads, grid->thread.eptr, qnode,
                grid->thread.normal.coord.x, grid->thread.normal.coord.y, 
                grid->thread.normal.coord.z, grid->thread.normal.len, 
                grid->area, grid->bound.solid.normal.x, 
                grid->bound.solid.normal.y, grid->bound.solid.normal.z, 
                grid->bound.ffild.normal.x, grid->bound.ffild.normal.y, 
                grid->bound.ffild.normal.z, grid->bound.solid.inode, 
                grid->bound.ffild.inode, grid->thread.partitions, 
                grid->thread.nedgeLoc, grid->cdt );
  }
  /* 
    Calculates the flux
  */
  f77FLUX(  &grid->nnodes, &grid->nedge, &grid->thread.nedge, 
            &grid->bound.solid.nnode, &grid->bound.ffild.nnode, 
            &grid->bound.solid.nfacet, &grid->bound.ffild.nfacet, 
            &nThreads, grid->thread.eptr, qnode, grid->coord.x, 
            grid->coord.y, grid->coord.z, grid->thread.normal.coord.x, 
            grid->thread.normal.coord.y, grid->thread.normal.coord.z, 
            grid->thread.normal.len, grad, grid->bound.ffild.normal.x, 
            grid->bound.ffild.normal.y, grid->bound.ffild.normal.z, 
            grid->bound.solid.inode, grid->bound.ffild.inode, 
            grid->bound.solid.f2nt, grid->bound.ffild.f2nt, 
            grid->thread.partitions, grid->thread.nedgeLoc, res, &fluxTimeLoc
#ifdef HW_COUNTER
            ,&eventset, &val 
#endif
            );

  fluxTime += fluxTimeLoc;

#ifdef HW_COUNTER
  handler += val;
#endif

  if(tsCtx->isPseudoTimeStep == 1)
  {
    ierr = VecGetArray(tsCtx->qnode, &qold); CHKERRQ(ierr);
    for(i = 0; i < grid->nnodes; i++)
    {
      temp = grid->area[i] / (tsCtx->CFL * grid->cdt[i]);
      for(j = 0; j < 4; j++)
        res[4 * i + j] += temp * (qnode[4 * i + j] - qold[4 * i + j]);
    }
    ierr = VecRestoreArray(tsCtx->qnode, &qold); CHKERRQ(ierr);
  }

  ierr = VecRestoreArray(qnodeLoc, &qnode);   CHKERRQ(ierr);
  ierr = VecRestoreArray(f, &res);          CHKERRQ(ierr);
  ierr = VecRestoreArray(gridPETSc->grad, &grad); CHKERRQ(ierr);


  PetscFunctionReturn(0);
}
/*
  Function used to convey the nonlinear Jacobian of the 
  function to be solved by SNES 
  
  Evaluate Jacobian F'(x)
  Input vector; matrix that defines the approximate Jacobian;
  matrix to be used to construct the preconditioner;
  flag indicating information about the preconditioner matrix structure
  user-defined context
*/
#undef  __FUNCT__
#define __FUNCT__ "formJacobian"
int 
formJacobian(SNES snes, Vec x, Mat Amat, Mat Pmat, void *ctx)
{
  AppCtx      *user       = (AppCtx *) ctx;
  GRID        *grid       = user->grid;
  GRIDPETSc   *gridPETSc  = user->gridPETSc;
  TstepCtx    *tsCtx      = user->tsCtx;
  Vec         qnodeLoc    = gridPETSc->qnodeLoc;
  double      *qnode;
  int         ierr;

  PetscFunctionBegin;

//
  /* 
    Resets a factored matrix to be treated as unfactored
  */
  ierr = MatSetUnfactored(Pmat); CHKERRQ(ierr);

  ierr = VecGetArray(qnodeLoc, &qnode); CHKERRQ(ierr);
  /* 
    Fill the nonzero term of the A matrix
  */
  f77FILLA( &grid->nnodes, &grid->nedge, &grid->thread.nedge, 
            &grid->bound.solid.nnode, &grid->bound.ffild.nnode, 
            &tsCtx->CFL, &nThreads, grid->eptr, qnode, 
            grid->normal.coord.x, grid->normal.coord.y, 
            grid->normal.coord.z, grid->normal.len, grid->area, 
            grid->cdt, grid->bound.solid.normal.x, 
            grid->bound.solid.normal.y, grid->bound.solid.normal.z, 
            grid->bound.ffild.normal.x, grid->bound.ffild.normal.y, 
            grid->bound.ffild.normal.z, grid->bound.solid.inode, 
            grid->bound.ffild.inode, grid->thread.partitions, 
            grid->thread.nedgeLoc, &Pmat );
  
  ierr = VecRestoreArray(qnodeLoc, &qnode); CHKERRQ(ierr);
  /* 
    Begins assembling the matrix
  */
  ierr = MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY);  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY);    CHKERRQ(ierr);

  printf("**** Jacobian is invoked\n\n");

/*  PetscViewer v;

  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "Amat.m", &v);
  PetscViewerPushFormat(v, PETSC_VIEWER_ASCII_INFO_DETAIL);  //PETSC_VIEWER_ASCII_DENSE);//PETSC_VIEWER_DEFAULT);//PETSC_VIEWER_ASCII_MATLAB);
  MatView(Pmat,v);
  
  PetscViewerDestroy(&v);

  exit(0);*/

  PetscFunctionReturn(0);
}
/* 
  Form initial approximation
*/
#undef  __FUNCT__
#define __FUNCT__ "formInitialGuess"
int 
formInitialGuess(void *ctx)
{
  AppCtx      *user       = (AppCtx *) ctx;
  GRID        *grid       = user->grid;
  GRIDPETSc   *gridPETSc  = user->gridPETSc;
  int         ierr;
  double      *qnode;

  PetscFunctionBegin;

  ierr = VecGetArray(gridPETSc->qnode, &qnode); CHKERRQ(ierr);
  /*
    Initializes the flow field
  */  
  f77INIT(&grid->nnodes, qnode);

  ierr = VecRestoreArray(gridPETSc->qnode, &qnode); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
