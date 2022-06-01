
/*
  Author: Mohammed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>
#include <omp.h>
#include "inc/geometry.h"
#include "inc/ktime.h"
#include "inc/ker/kernel.h"
#include "inc/ker/phy.h"

/*
  Evaluate Function F(x): Functional form used to convey the 
  nonlinear function to be solved by PETSc SNES
*/  
int 
ffunc(SNES snes, Vec x, Vec f, void *restrict ctx)
{
  struct ctx *restrict c = (struct ctx *) ctx;

  const size_t nnodes = c->g->n->sz;
  const size_t bsz = c->g->c->bsz;

  int ierr;

  const double *restrict q;
  ierr = VecGetArrayRead(x, (const PetscScalar **) &q);
  CHKERRQ(ierr);

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
  }
  compute_grad(&grad);

  double *restrict r;
  ierr = VecGetArray(f, (PetscScalar **) &r);
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
  }
  compute_flux(&flux);

  struct ktime ktime;
  setktime(&ktime); 

  const double *restrict q_;
  ierr = VecGetArrayRead(c->ts->q, (const PetscScalar **) &q_);
  CHKERRQ(ierr);

  const double *restrict area = c->g->n->area;
  double *restrict cdt = c->ts->cdt;
  const double cfl = c->ts->cfl;

  uint32_t i;
#pragma omp parallel for
  for(i = 0; i < nnodes; i++)
  {
    const double t = area[i] / (cfl * cdt[i]);
    const uint32_t idx = bsz * i;

    r[idx + 0] += t * (q[idx + 0] - q_[idx + 0]);
    r[idx + 1] += t * (q[idx + 1] - q_[idx + 1]);
    r[idx + 2] += t * (q[idx + 2] - q_[idx + 2]);
    r[idx + 3] += t * (q[idx + 3] - q_[idx + 3]);
  }

  ierr = VecRestoreArrayRead(c->ts->q, (const PetscScalar **) &q_);
  CHKERRQ(ierr);
 
  compute_time(&ktime, &c->t->tstep_contr);

  ierr = VecRestoreArray(f, (PetscScalar **) &r);
  CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(x, (const PetscScalar **) &q);
  CHKERRQ(ierr);

  return 0;
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
int
jfunc(SNES snes, Vec x, Mat Amat, Mat Pmat, void *restrict ctx)
{
  struct ctx *restrict c = (struct ctx *) ctx;

  /* 
    Resets a factored matrix to be treated as unfactored
  */

  int ierr;

  ierr = MatSetUnfactored(Pmat);
  CHKERRQ(ierr);

  const double *restrict q;
  ierr = VecGetArrayRead(x, (const PetscScalar **) &q);
  CHKERRQ(ierr);

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
  }

  ierr = fill_mat(&fill);
  CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(x, (const PetscScalar **) &q);
  CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  
  return 0;
}
