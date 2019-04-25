#ifdef __cplusplus
extern "C" {
#endif
extern void CXX_Matrix_Delete(void *ptr);
#ifdef __cplusplus
}
#endif

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include "bench.h"
#include "allocator.h"
#include "geometry.h"

extern void Kernel(GEOMETRY *);
extern void Setup(const char *, GEOMETRY *);

int
main(int argc, char * argv[])
{
  BENCH start_global_bench = rdbench();

  fun3d_init_log();
  
  BENCH start_bench;

  start_bench = rdbench();

  assert(argc == 5);

  char mesh[2]; // To avoid buffer overflow
  strcpy((char *) &mesh, argv[2]);

  // Set the file full pathname
  char fpath[22]; // To avoid buffer overflow
  sprintf(fpath, "./mesh/uns3d%c.msh", mesh[0]);
  printf("%s\n", fpath);

  uint32_t threads = (uint32_t) atoi(argv[4]);

  // Set OpenMP threads
  omp_set_num_threads((int) threads);

  printf("Number of Threads: %d\n", threads);

  GEOMETRY *g = (GEOMETRY *) fun3d_malloc(1, sizeof(GEOMETRY));

  // Read the mesh file and store in the local data structure
  Setup(fpath, g);

  fun3d_log(start_bench, KERNEL_SETUP);

  start_bench = rdbench();

  Kernel(g);

  fun3d_log(start_bench, KERNEL_CORE);

  /* Clean up the memory */

  fun3d_free(g->e->xyzn->x0);
  fun3d_free(g->e->xyzn->x1);
  fun3d_free(g->e->xyzn->x2);
  fun3d_free(g->e->xyzn->x3);
  fun3d_free(g->e->xyzn);
  fun3d_free(g->e->w->w0->x0);
  fun3d_free(g->e->w->w0->x1);
  fun3d_free(g->e->w->w0->x2);
  fun3d_free(g->e->w->w0);
  fun3d_free(g->e->w->w1->x0);
  fun3d_free(g->e->w->w1->x1);
  fun3d_free(g->e->w->w1->x2);
  fun3d_free(g->e->w->w1);
  fun3d_free(g->e->w);
  fun3d_free(g->e->eptr->n0);
  fun3d_free(g->e->eptr->n1);
  fun3d_free(g->e->eptr);
  fun3d_free(g->e);
  fun3d_free(g->s->i);
  fun3d_free(g->s);
  fun3d_free(g->t->i);
  fun3d_free(g->t);
  //fun3d_free(g->c->mat->i);
  //fun3d_free(g->c->mat->a);
  //fun3d_free(g->c->mat->j);
  //fun3d_free(g->c->mat->d);
  //fun3d_free(g->c->mat->w);
  //fun3d_free(g->c->mat);
  fun3d_free(g->c);
  fun3d_free(g->b->s->xyz->x0);
  fun3d_free(g->b->s->xyz->x1);
  fun3d_free(g->b->s->xyz->x2);
  fun3d_free(g->b->s->xyz);
  fun3d_free(g->b->s->nptr);
  fun3d_free(g->b->s);
  fun3d_free(g->b->f->xyz->x0);
  fun3d_free(g->b->f->xyz->x1);
  fun3d_free(g->b->f->xyz->x2);
  fun3d_free(g->b->f->xyz);
  fun3d_free(g->b->f->nptr);
  fun3d_free(g->b->f);
  fun3d_free(g->b->fc->fptr->n0);
  fun3d_free(g->b->fc->fptr->n1);
  fun3d_free(g->b->fc->fptr->n2);
  fun3d_free(g->b->fc->fptr);
  fun3d_free(g->b->fc);
  fun3d_free(g->b);
  fun3d_free(g->n->xyz->x0);
  fun3d_free(g->n->xyz->x1);
  fun3d_free(g->n->xyz->x2);
  fun3d_free(g->n->xyz);
  fun3d_free(g->n->cdt);
  fun3d_free(g->n->part);
  fun3d_free(g->n);
  fun3d_free(g->q->q);
  fun3d_free(g->q);
  CXX_Matrix_Delete(g->matrix);
  fun3d_free(g);

  fun3d_log(start_global_bench, FUN3D);
  
  fun3d_close_log();

  return 0;
}