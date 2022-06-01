
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscsnes.h>
#include <omp.h>
//#include <mkl.h>
#include <geometry.h>
#include <ktime.h>
#include <allocator.h>
#ifdef __USE_HW_COUNTER
#include <perf.h>
#include <kperf.h>
#endif
#include <kernel.h>
#include <phy.h>
#include <main.h>

#include "petsc_stuffs.h"

/* Compute the Pseudo Time Step */
static inline int 
computePsiTS(struct ctx *restrict c)
{
  int ierr;

//  const double *restrict q;
//  ierr = VecGetArrayRead(c->ts->q, (const PetscScalar **) &q);
//  CHKERRQ(ierr);

//  double *restrict r;
//  ierr = VecGetArray(c->ts->r, (PetscScalar **) &r);
//  CHKERRQ(ierr);

  struct delta delta;
  {
    delta.nnodes = c->g->n->sz;
    delta.nsnodes = c->g->b->s->n->sz;
    delta.nfnodes = c->g->b->f->n->sz;
    delta.nsptr = c->g->b->s->n->nptr;
    delta.nfptr = c->g->b->f->n->nptr;
    delta.s_xyz0 = c->g->b->s->n->xyz->x0;
    delta.s_xyz1 = c->g->b->s->n->xyz->x1;
    delta.s_xyz2 = c->g->b->s->n->xyz->x2;
    delta.f_xyz0 = c->g->b->f->n->xyz->x0;
    delta.f_xyz1 = c->g->b->f->n->xyz->x1;
    delta.f_xyz2 = c->g->b->f->n->xyz->x2;
    delta.area = c->g->n->area;
    delta.q = c->ts->q;
    delta.ie = c->g->s->ie;
    delta.part = c->g->s->part;
    delta.n0 = c->g->e->eptr->n0;
    delta.n1 = c->g->e->eptr->n1;
    delta.x0 = c->g->e->xyzn->x0;
    delta.x1 = c->g->e->xyzn->x1;
    delta.x2 = c->g->e->xyzn->x2;
    delta.x3 = c->g->e->xyzn->x3;
    delta.bsz = c->g->c->bsz;
    delta.cdt = c->ts->cdt;
    delta.t = &c->t->deltat2;
#ifdef __USE_HW_COUNTER
    delta.perf_counters = c->perf_counters;
#endif
  }
  compute_deltat2(&delta);

  struct residual res;
  {
    res.w0termsx = c->g->e->w->w0->x0;
    res.w0termsy = c->g->e->w->w0->x1;
    res.w0termsz = c->g->e->w->w0->x2;
    res.w1termsx = c->g->e->w->w1->x0;
    res.w1termsy = c->g->e->w->w1->x1;
    res.w1termsz = c->g->e->w->w1->x2;
    res.gradx0 = c->grad->x0;
    res.gradx1 = c->grad->x1;
    res.gradx2 = c->grad->x2;

#if 0
    grad.t = &c->t->grad;
#ifdef __USE_HW_COUNTER
    grad.perf_counters = c->perf_counters;
#endif
  }
  compute_grad(&grad);

  struct flux flux;
  {
#endif

    res.bsz = c->g->c->bsz;
    res.nfnodes = c->g->b->f->n->sz;
    res.dofs = c->g->c->sz;
    res.snfc = c->g->s->snfc;
    res.pressure = c->iv->p;
    res.velocity_u = c->iv->u;
    res.velocity_v = c->iv->v;
    res.velocity_w = c->iv->w;
    res.f_xyz0 = c->g->b->f->n->xyz->x0;
    res.f_xyz1 = c->g->b->f->n->xyz->x1;
    res.f_xyz2 = c->g->b->f->n->xyz->x2;
    res.xyz0 = c->g->n->xyz->x0;
    res.xyz1 = c->g->n->xyz->x1;
    res.xyz2 = c->g->n->xyz->x2;
    res.ie = c->g->s->ie;
    res.part = c->g->s->part;
    res.snfic = c->g->s->snfic;
    res.n0 = c->g->e->eptr->n0;
    res.n1 = c->g->e->eptr->n1;
    res.nfptr = c->g->b->f->n->nptr;
    res.sn0 = c->g->b->snfptr->n0;
    res.sn1 = c->g->b->snfptr->n1;
    res.sn2 = c->g->b->snfptr->n2;
    res.x0 = c->g->e->xyzn->x0;
    res.x1 = c->g->e->xyzn->x1;
    res.x2 = c->g->e->xyzn->x2;
    res.x3 = c->g->e->xyzn->x3;
    res.q = c->ts->q;
    res.gradx0 = c->grad->x0;
    res.gradx1 = c->grad->x1;
    res.gradx2 = c->grad->x2;
    res.r = c->ts->r;
    res.t = &c->t->flux;
#ifdef __USE_HW_COUNTER
    res.perf_counters = c->perf_counters;
#endif
  }
  compute_residual(&res);

#ifdef __USE_MAN_FLOPS_COUNTER
  uint64_t grad_flops = 56 * c->g->e->sz;
  uint64_t flux_flops = 0;
  flux_flops += 335 * c->g->e->sz;
  flux_flops += 53 * c->g->b->s->f->sz;
  flux_flops += 210 * c->g->b->f->n->sz;

  c->t->flux.flops += flux_flops + grad_flops;
  //c->t->grad.flops += grad_flops;
#endif

//  ierr = VecRestoreArrayRead(c->ts->q, (const PetscScalar **) &q);
//  CHKERRQ(ierr);

  //c->ts->fnorm = cblas_dnrm2((int) c->g->c->sz, r, 1);

//  ierr = VecNorm(c->ts->r, NORM_2, &c->ts->fnorm);
//  CHKERRQ(ierr);

  double norm = 0;

  int i = 0;
  for(;i < c->g->c->sz; i++) norm += c->ts->r[i] * c->ts->r[i];

  c->ts->fnorm = sqrt(norm);

//  ierr = VecRestoreArray(c->ts->r, (PetscScalar **) &r);
//  CHKERRQ(ierr);


  if(c->ts->fnorm_init == 0.f)
  {
    c->ts->fnorm_init = c->ts->fnorm;
    c->ts->cfl = c->ts->cfl_init;
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
    double cfl = 1.1 * c->ts->cfl_init * c->ts->fnorm_init;
    cfl /= c->ts->fnorm;
    c->ts->cfl = (cfl < MAX_CFL) ? cfl : MAX_CFL;
  }

  return 0;
}

//extern int
//MatCreateSeqBAIJ1(PetscInt, PetscInt, PetscInt, PetscInt*, PetscInt*, double *, PetscInt,
 //               PetscInt, Mat *);

extern int ComputeNewton(GMRES *, void *, /*Vec,*/ double *, double *, Jacobian *);//, Mat);

//extern int SNESCreate1(SNES*);

//extern int SNESSetFunction1(SNES, Vec, PetscErrorCode (*f)(SNES,Vec,Vec,void*),void*);

//extern int MatCreateSNESMF1(Vec, Vec, Vec, Mat *);

extern Jacobian * CreateJacobianTable(const size_t);

//extern int SNESSetJacobian1(SNES, Mat, Mat, PetscErrorCode (*J)(SNES,Vec,Mat,Mat,void*),void *);



//extern int SNESSetFromOptions1(KSP *);

extern int KSPSetUp_GMRES1(GMRES *, size_t);
extern void ComputeSymbolicILU(const int n,
  const int m,
  const int bs2,
  const int *ai,
  int *aj,
  int **bii,
  int **bjj,
  double **baa,
  int **bbdiag);

int
#ifdef __USE_HW_COUNTER
ikernel(int argc, char * argv[], const struct geometry *restrict g,
        struct kernel_time *restrict kt,
        struct perf_counters *restrict perf_counters)
#else
ikernel(int argc, char * argv[], const struct geometry *restrict g,
        struct kernel_time *restrict kt)
#endif
{

  /* Compute the weights for weighted least square */
  struct xyz *restrict grad;
  kmalloc(1, sizeof(struct xyz), (void *) &grad);

  kmalloc(g->c->sz, sizeof(double), (void *) &grad->x0);
  kmalloc(g->c->sz, sizeof(double), (void *) &grad->x1);
  kmalloc(g->c->sz, sizeof(double), (void *) &grad->x2);

  /* Time step */  
  struct ts *restrict ts;
  kmalloc(1, sizeof(struct ts), (void *) &ts);

  kmalloc(g->n->sz, sizeof(double), (void *) &ts->cdt);

  ts->fnorm_init = 0.f;
  ts->cfl_init = 50.f;

  ts->q = (double *) malloc(g->c->sz * sizeof(double));
  double *q = (double *) malloc(g->c->sz * sizeof(double));
  double *x = (double *) malloc(g->c->sz * sizeof(double));
  double *r = (double *) malloc(g->c->sz * sizeof(double));
  ts->r = (double *) malloc(g->c->sz * sizeof(double));





  int ierr;

  ierr = PetscInitialize(&argc, &argv, "petsc.opt", NULL);
  CHKERRQ(ierr);
  

  /* Q node state vector */
//  Vec q;
//  ierr = VecCreateSeq(PETSC_COMM_SELF, g->c->sz, &q);
//  CHKERRQ(ierr);

//  ierr = VecCreateSeq(PETSC_COMM_SELF, g->c->sz, &ts->q);
//  CHKERRQ(ierr);


//  ierr = VecCreateSeq(PETSC_COMM_SELF, g->c->sz, &ts->r);
//  CHKERRQ(ierr);

  /* Right-hand side matrix */
//  Mat A = NULL;
//  ierr = MatCreateSeqBAIJ1(g->c->bsz, g->c->sz, g->c->sz, (PetscInt*) g->c->ia,
//                          (PetscInt*) g->c->ja, g->c->aa,
//                          PETSC_DEFAULT, (int) g->c->nnz, &A);
//  CHKERRQ(ierr);

//  kfree(g->c->nnz);

//  double *restrict q0;
//  ierr = VecGetArray(q, (PetscScalar **) &q0);
//  CHKERRQ(ierr);
   
//  double *restrict q1;
//  ierr = VecGetArray(ts->q, (PetscScalar **) &q1);
//  CHKERRQ(ierr);

  /* Set the physical parameters */

  struct ivals *restrict iv;
  kmalloc(1, sizeof(struct ivals), (void *) &iv);

  struct igtbl ig;
  {
    ig.sz = g->n->sz;
    ig.bsz = g->c->bsz;
    ig.iv = iv;
    ig.q0 = q;
    ig.q1 = ts->q;
    ig.t = kt;
#ifdef __USE_HW_COUNTER
    ig.perf_counters = perf_counters;
#endif
  }

  iguess(&ig);

//  ierr = VecRestoreArray(q, (PetscScalar **) &q0);
//  CHKERRQ(ierr);

//  ierr = VecRestoreArray(ts->q, (PetscScalar **) &q1);
//  CHKERRQ(ierr);

//  SNES snes;
//  ierr = SNESCreate1(&snes);
//  CHKERRQ(ierr);

//  Vec r;
//  ierr = VecCreateSeq(PETSC_COMM_SELF, g->c->sz, &r);
//  CHKERRQ(ierr);

//  Vec r;
//  ierr = VecDuplicate(q, &r);
//  CHKERRQ(ierr);

//  Vec x;
//  ierr = VecDuplicate(r, &x);
//  CHKERRQ(ierr);

//  Vec u;
//  ierr = VecDuplicate(q, &u);
//  CHKERRQ(ierr);

//  Vec w;
//  ierr = VecDuplicate(q, &w);
//  CHKERRQ(ierr);

  ILUTable *ilu = (ILUTable *) malloc(sizeof(ILUTable));

  ilu->isCalled = 0;

  struct ctx ctx;
  {
    ctx.ilu = ilu;
    ctx.grad = grad; // Gradient vector
    ctx.g = g; // Geometry
    ctx.iv = iv; // Physics values
    ctx.ts = ts; // Time step table
    ctx.q = q; // Q node vector: State variables
    ctx.t = kt; // Kernel timing structure
#ifdef __USE_HW_COUNTER
    ctx.perf_counters = perf_counters;
#endif
  }

//  ierr = SNESSetFunction1(snes, r, ComputeResidual, &ctx);
//  CHKERRQ(ierr);
  
//  Mat amat;
//  ierr = MatCreateSNESMF1(u, w, r, &amat);
//  CHKERRQ(ierr);

  Jacobian *amat = CreateJacobianTable(g->c->sz);

//  ierr = SNESSetJacobian1(snes, amat, A, jfunc, &ctx);
//  CHKERRQ(ierr);

//  KSP ksp;

//  ierr = SNESSetFromOptions1(&ksp);//, A);
//  CHKERRQ(ierr);

  double fnorm_ratio = 1.f;

  double clift; // LIFT force 
  double cdrag; // DRAG force
  double cmomn; // MOMENTUM

  GMRES *gmres = (GMRES *) malloc(sizeof(GMRES));

  KSPSetUp_GMRES1(gmres, g->c->sz);

  FillPreconditionerMatrix(&ctx);

//   if(!c->ilu->isCalled)
  //     {
         ComputeSymbolicILU(
         //  at->nz,
         g->n->sz,//at->mbs,
         g->n->sz,//at->nbs,
         g->c->bsz2,//at->bs2,
         (int *) g->c->ia,//at->i,
         (int *) g->c->ja,//at->j,
           &ilu->ia,//&bt->i,
             &ilu->ja,//&bt->j,
               &ilu->aa,//&bt->a,
                 &ilu->diag);//&bt->diag);

  //                   c->ilu->isCalled = 1;
//                       }

  
  uint32_t i;
  for(i = 0; (i < MAX_TS && fnorm_ratio <= FNORM_RATIO); i++)
  {
    ierr = computePsiTS(&ctx);
    CHKERRQ(ierr);
   /*
      Solves a nonlinear system F(x) = b
      Null means b = 0 (constant part of the equation)
      x is the solution vector 
      global distributed solution vector
      Here FormJacobian is called 
    */
    ierr = ComputeNewton(gmres, &ctx/*, q*/, r, x, amat);//, A);
    CHKERRQ(ierr);
    
    printf("At Time Step %d CFL = %g and function norm = %g\n",
          i, ts->cfl, ts->fnorm);

//    ierr = VecCopy(q, ts->q);
//    CHKERRQ(ierr);

    
  //  ierr = VecGetArrayRead(q, (const PetscScalar **) &q0);
  //  CHKERRQ(ierr);

   memcpy(ts->q, q, g->c->sz * sizeof(double));
    struct force f;
    {
      f.g = g;
      f.iv = iv;
      f.q = q;
      f.clift = &clift;
      f.cdrag = &cdrag;
      f.cmomn = &cmomn;
      f.t = kt;
#ifdef __USE_HW_COUNTER
      f.perf_counters = perf_counters;
#endif
    }
    compute_force(&f);

    printf("%d\t%g\t%g\t%g\t%g\t%g\n", 
          i, ts->cfl, ts->fnorm, clift, cdrag, cmomn);
    
//    ierr = VecRestoreArrayRead(q, (const PetscScalar **) &q0);
//    CHKERRQ(ierr);
    
    fnorm_ratio = ts->fnorm_init / ts->fnorm;
  }

  printf("CFL = %g fnorm = %g\n", ts->cfl, ts->fnorm);
  printf("clift = %g cdrag = %g cmom = %g\n", clift, cdrag, cmomn);

//  ierr = MatDestroy(&A);
//  CHKERRQ(ierr);
//  ierr = MatDestroy(&amat);
//  CHKERRQ(ierr);

//  ierr = SNESDestroy(&snes);
//  CHKERRQ(ierr);


  //ierr = VecDestroy(&r);
//  CHKERRQ(ierr);
  //ierr = VecDestroy(&q);
//  CHKERRQ(ierr);

//  ierr = VecDestroy(&ts->q);
//  CHKERRQ(ierr);
//  ierr = VecDestroy(&ts->r);
//  CHKERRQ(ierr);


  kfree(iv);
  kfree(ts->cdt);
  kfree(ts);

  kfree(grad->x0);
  kfree(grad->x1);
  kfree(grad->x2);
  kfree(grad);


//  ierr = PetscFinalize();
//  CHKERRQ(ierr);

  return 0;
}
