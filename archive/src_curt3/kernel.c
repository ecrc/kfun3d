
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscsnes.h>
#include <omp.h>
#include <mkl.h>
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


/* Compute the Pseudo Time Step */
static inline int 
computePsiTS(struct ctx *restrict c)
{
  int ierr;

  const double *restrict q;
  ierr = VecGetArrayRead(c->ts->q, (const PetscScalar **) &q);
  CHKERRQ(ierr);

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
    delta.q = q;
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

  struct grad grad;
  {
    grad.bsz = c->g->c->bsz;
    grad.dofs = c->g->c->sz;
    grad.ie = c->g->s->ie;
    grad.part = c->g->s->part;
    grad.n0 = c->g->e->eptr->n0;
    grad.n1 = c->g->e->eptr->n1;
    grad.w0termsx = c->g->e->w->w0->x0;
    grad.w0termsy = c->g->e->w->w0->x1;
    grad.w0termsz = c->g->e->w->w0->x2;
    grad.w1termsx = c->g->e->w->w1->x0;
    grad.w1termsy = c->g->e->w->w1->x1;
    grad.w1termsz = c->g->e->w->w1->x2;
    grad.q = q;
    grad.gradx0 = c->grad->x0;
    grad.gradx1 = c->grad->x1;
    grad.gradx2 = c->grad->x2;
    grad.t = &c->t->grad;
#ifdef __USE_HW_COUNTER
    grad.perf_counters = c->perf_counters;
#endif
  }
  compute_grad(&grad);

  double *restrict r;
  ierr = VecGetArray(c->ts->r, (PetscScalar **) &r);
  CHKERRQ(ierr);

  struct flux flux;
  {
    flux.bsz = c->g->c->bsz;
    flux.nfnodes = c->g->b->f->n->sz;
    flux.dofs = c->g->c->sz;
    flux.snfc = c->g->s->snfc;
    flux.pressure = c->iv->p;
    flux.velocity_u = c->iv->u;
    flux.velocity_v = c->iv->v;
    flux.velocity_w = c->iv->w;
    flux.f_xyz0 = c->g->b->f->n->xyz->x0;
    flux.f_xyz1 = c->g->b->f->n->xyz->x1;
    flux.f_xyz2 = c->g->b->f->n->xyz->x2;
    flux.xyz0 = c->g->n->xyz->x0;
    flux.xyz1 = c->g->n->xyz->x1;
    flux.xyz2 = c->g->n->xyz->x2;
    flux.ie = c->g->s->ie;
    flux.part = c->g->s->part;
    flux.snfic = c->g->s->snfic;
    flux.n0 = c->g->e->eptr->n0;
    flux.n1 = c->g->e->eptr->n1;
    flux.nfptr = c->g->b->f->n->nptr;
    flux.sn0 = c->g->b->snfptr->n0;
    flux.sn1 = c->g->b->snfptr->n1;
    flux.sn2 = c->g->b->snfptr->n2;
    flux.x0 = c->g->e->xyzn->x0;
    flux.x1 = c->g->e->xyzn->x1;
    flux.x2 = c->g->e->xyzn->x2;
    flux.x3 = c->g->e->xyzn->x3;
    flux.q = q;
    flux.gradx0 = c->grad->x0;
    flux.gradx1 = c->grad->x1;
    flux.gradx2 = c->grad->x2;
    flux.r = r;
    flux.t = &c->t->flux;
#ifdef __USE_HW_COUNTER
    flux.perf_counters = c->perf_counters;
#endif
  }
  compute_flux(&flux);

#ifdef __USE_MAN_FLOPS_COUNTER
  uint64_t grad_flops = 56 * c->g->e->sz;
  uint64_t flux_flops = 0;
  flux_flops += 335 * c->g->e->sz;
  flux_flops += 53 * c->g->b->s->f->sz;
  flux_flops += 210 * c->g->b->f->n->sz;

  c->t->flux.flops += flux_flops;
  c->t->grad.flops += grad_flops;
#endif

  ierr = VecRestoreArrayRead(c->ts->q, (const PetscScalar **) &q);
  CHKERRQ(ierr);

  c->ts->fnorm = cblas_dnrm2((int) c->g->c->sz, r, 1);

  ierr = VecRestoreArray(c->ts->r, (PetscScalar **) &r);
  CHKERRQ(ierr);

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
  int ierr;

  ierr = PetscInitialize(&argc, &argv, "petsc.opt", NULL);
  CHKERRQ(ierr);
  
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

  /* Q node state vector */
  Vec q;
  ierr = VecCreateSeq(PETSC_COMM_SELF, g->c->sz, &q);
  CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, g->c->sz, &ts->q);
  CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF, g->c->sz, &ts->r);
  CHKERRQ(ierr);

  /* Right-hand side matrix */
  Mat A = NULL;
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF, g->c->bsz, g->c->sz, g->c->sz,
                          PETSC_DEFAULT, (const int *) g->c->nnz, &A);
  CHKERRQ(ierr);

  kfree(g->c->nnz);

  double *restrict q0;
  ierr = VecGetArray(q, (PetscScalar **) &q0);
  CHKERRQ(ierr);
   
  double *restrict q1;
  ierr = VecGetArray(ts->q, (PetscScalar **) &q1);
  CHKERRQ(ierr);

  /* Set the physical parameters */

  struct ivals *restrict iv;
  kmalloc(1, sizeof(struct ivals), (void *) &iv);

  struct igtbl ig;
  {
    ig.sz = g->n->sz;
    ig.bsz = g->c->bsz;
    ig.iv = iv;
    ig.q0 = q0;
    ig.q1 = q1;
    ig.t = kt;
#ifdef __USE_HW_COUNTER
    ig.perf_counters = perf_counters;
#endif
  }

  iguess(&ig);

  ierr = VecRestoreArray(q, (PetscScalar **) &q0);
  CHKERRQ(ierr);

  ierr = VecRestoreArray(ts->q, (PetscScalar **) &q1);
  CHKERRQ(ierr);

  SNES snes;
  ierr = SNESCreate(PETSC_COMM_SELF, &snes);
  CHKERRQ(ierr);

  Vec r;
  ierr = VecDuplicate(q, &r);
  CHKERRQ(ierr);

  struct ctx ctx;
  {
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

  ierr = SNESSetFunction(snes, r, ffunc, &ctx);
  CHKERRQ(ierr);
  
  Mat amat;
  ierr = MatCreateSNESMF(snes, &amat);
  CHKERRQ(ierr);

  ierr = SNESSetJacobian(snes, amat, A, jfunc, &ctx);
  CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);
  CHKERRQ(ierr);

  double fnorm_ratio = 1.f;

  double clift; // LIFT force 
  double cdrag; // DRAG force
  double cmomn; // MOMENTUM
  
  uint32_t nfails = 0;

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
    ierr = SNESSolve(snes, NULL, q);
    CHKERRQ(ierr);
    /*
      Gets the number of unsuccessful steps attempted by 
      the nonlinear solver
    */
    int fail = 0;
    ierr = SNESGetNonlinearStepFailures(snes, &fail);
    CHKERRQ(ierr);

    nfails += fail;

    if(nfails >= 2) SETERRQ(PETSC_COMM_SELF, 1, "Can't find a Newton Step");
    
    printf("At Time Step %d CFL = %g and function norm = %g\n",
          i, ts->cfl, ts->fnorm);


    ierr = VecCopy(q, ts->q);
    CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(q, (const PetscScalar **) &q0);
    CHKERRQ(ierr);

    struct force f;
    {
      f.g = g;
      f.iv = iv;
      f.q = q0;
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
    
    ierr = VecRestoreArrayRead(q, (const PetscScalar **) &q0);
    CHKERRQ(ierr);
    
    fnorm_ratio = ts->fnorm_init / ts->fnorm;
  }

  printf("CFL = %g fnorm = %g\n", ts->cfl, ts->fnorm);
  printf("clift = %g cdrag = %g cmom = %g\n", clift, cdrag, cmomn);

  ierr = MatDestroy(&A);
  CHKERRQ(ierr);
  ierr = MatDestroy(&amat);
  CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);
  CHKERRQ(ierr);
  ierr = VecDestroy(&r);
  CHKERRQ(ierr);
  ierr = VecDestroy(&q);
  CHKERRQ(ierr);
  ierr = VecDestroy(&ts->q);
  CHKERRQ(ierr);
  ierr = VecDestroy(&ts->r);
  CHKERRQ(ierr);

  kfree(iv);
  kfree(ts->cdt);
  kfree(ts);

  kfree(grad->x0);
  kfree(grad->x1);
  kfree(grad->x2);
  kfree(grad);

  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return 0;
}
