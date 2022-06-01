
//#include <petscmat.h>
//#include <petscsnes.h>
//#include <petscvec.h>

#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include <geometry.h>
#include <ktime.h>
#ifdef __USE_HW_COUNTER
#include <perf.h>
#include <kperf.h>
#endif
#include <kernel.h>
#include <phy.h>

//extern int
//InitJacobian(Vec, Mat);

/*
  Evaluate Function F(x): Functional form used to convey the 
  nonlinear function to be solved by PETSc SNES
*/  
void 
//ffunc(SNES snes, Vec x, Vec f, void *restrict ctx)
//ComputeResidual(SNES snes, Vec x, Vec f, void *restrict ctx)
//ComputeResidual(Vec x, Vec f, void * ctx)
ComputeResidual(const double *q, double *r, void * ctx)
{
  struct ctx *restrict c = (struct ctx *) ctx;

  const size_t nnodes = c->g->n->sz;
  const size_t bsz = c->g->c->bsz;

//  int ierr;

//  const double *restrict q;
//  ierr = VecGetArrayRead(x, (const PetscScalar **) &q);
//  CHKERRQ(ierr);

//  double *restrict r;
//  ierr = VecGetArray(f, (PetscScalar **) &r);
//  CHKERRQ(ierr);

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
    res.q = q;
    res.gradx0 = c->grad->x0;
    res.gradx1 = c->grad->x1;
    res.gradx2 = c->grad->x2;
    res.r = r;
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

#ifdef __USE_HW_COUNTER
  const struct fd fd = c->perf_counters->fd;

  struct counters start;
  perf_read(fd, &start);

  const uint64_t icycle = __rdtsc();
#endif

  struct ktime ktime;
  setktime(&ktime); 

//  const double *restrict q_;
//  ierr = VecGetArrayRead(c->ts->q, (const PetscScalar **) &q_);
//  CHKERRQ(ierr);

  const double *restrict area = c->g->n->area;
  double *restrict cdt = c->ts->cdt;
  const double cfl = c->ts->cfl;

  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < nnodes; i++)
  {
    const double t = area[i] / (cfl * cdt[i]);
    const uint32_t idx = bsz * i;

    r[idx + 0] += t * (q[idx + 0] - c->ts->q[idx + 0]);
    r[idx + 1] += t * (q[idx + 1] - c->ts->q[idx + 1]);
    r[idx + 2] += t * (q[idx + 2] - c->ts->q[idx + 2]);
    r[idx + 3] += t * (q[idx + 3] - c->ts->q[idx + 3]);
  }

//  ierr = VecRestoreArrayRead(c->ts->q, (const PetscScalar **) &q_);
//  CHKERRQ(ierr);
 
  compute_time(&ktime, &c->t->tstep_contr);

//  ierr = VecRestoreArray(f, (PetscScalar **) &r);
//  CHKERRQ(ierr);

//  ierr = VecRestoreArrayRead(x, (const PetscScalar **) &q);
//  CHKERRQ(ierr);

#ifdef __USE_HW_COUNTER
  const uint64_t cycle = __rdtsc() - icycle;

  struct counters end;
  perf_read(fd, &end);

  struct tot tot;
  perf_calc(start, end, &tot);

  c->perf_counters->ctrs->timestep.cycles += cycle;
  c->perf_counters->ctrs->timestep.tot.imcR += tot.imcR;
  c->perf_counters->ctrs->timestep.tot.imcW += tot.imcW;
  c->perf_counters->ctrs->timestep.tot.edcR += tot.edcR;
  c->perf_counters->ctrs->timestep.tot.edcW += tot.edcW;
#endif

//  return 0;
}

#if 0
/*
  Function used to convey the nonlinear Jacobian of the 
  function to be solved by SNES 
  
  Evaluate Jacobian F'(x)
  Input vector; matrix that defines the approximate Jacobian;
  matrix to be used to construct the preconditioner;
  flag indicating information about the preconditioner matrix structure
  user-defined context
*/
int
//jfunc(SNES snes, Vec x, Mat Amat, Mat Pmat, void *restrict ctx)
//ComputeJacobian(Vec x, Mat Amat, Mat Pmat, void * ctx)
//ComputeJacobian(const double *q, Mat Pmat, void *ctx)

FillPreconditionerMatrix(const double *q, Mat Pmat, void *ctx)
{
  struct ctx *restrict c = (struct ctx *) ctx;

  /* 
    Resets a factored matrix to be treated as unfactored
  */

  int ierr;

  //ierr = MatSetUnfactored(Pmat);
  //CHKERRQ(ierr);

//  const double *restrict q;
//  ierr = VecGetArrayRead(x, (const PetscScalar **) &q);
//  CHKERRQ(ierr);

  /* 
    Fill the nonzero term of the A matrix
  */

  struct fill fill;
  {
    fill.q = q;
    fill.g = c->g;
    fill.ts = c->ts;
    fill.iv = c->iv;
    fill.A = Pmat;
    fill.t = c->t;
#ifdef __USE_HW_COUNTER
    fill.perf_counters = c->perf_counters;
#endif
  }

  ierr = fill_mat(&fill);
  CHKERRQ(ierr);

//  ierr = VecRestoreArrayRead(x, (const PetscScalar **) &q);
//  CHKERRQ(ierr);

//  ierr = MatAssemblyBegin1(Amat, MAT_FINAL_ASSEMBLY);
//  CHKERRQ(ierr);
//  ierr = MatAssemblyEnd1(Amat, MAT_FINAL_ASSEMBLY);
//  CHKERRQ(ierr);

//  ierr = InitJacobian(x, Amat);
//  CHKERRQ(ierr);

  return 0;
}
#endif
