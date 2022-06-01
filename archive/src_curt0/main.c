
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>
#include "inc/allocator.h"
#include "inc/geometry.h"
#include "inc/ktime.h"
#include "inc/main.h"

int
main(int argc, char * argv[])
{
  struct ktime ktime1;
  setktime(&ktime1);

  struct kernel_time kt;

  zero_kernel_time(&kt);

  struct ktime ktime;
  setktime(&ktime);

  uint8_t mesh; // Mesh filename

  // Read the mesh filename
  if(argc < 3 || strcmp(argv[1], "-m") != 0) panic("mesh type");
  else strcpy((char *) &mesh, (void *) argv[2]);

  // Set the file full pathname
  uint8_t fpath[21];
  sprintf((char *) fpath, "../../mesh/uns3d%c.msh", mesh);

  // Data type is int so that we can validate negative input
  int32_t threads = 2;
  if(argc < 5 || strcmp(argv[3], "-t") != 0) panic("number of threads");
  else
  {
    threads = (int32_t) atoi(argv[4]);
    // Threads must be greater than 0
    if(threads <= 0) panic("number of threads <= 0");
  }

  // Set OpenMP threads
  omp_set_num_threads(threads);

  printf("Number of Threads: %d\n", threads);

  struct geometry *restrict g;
  kmalloc(1, sizeof(struct geometry), (void *) &g);

  // Read the mesh file and store in the local data structure
  imesh(fpath, g);

  /* CSR graph */
  kfree(g->c->ia);
  kfree(g->c->ja);

  kfree(g->b->s->f->fptr);

  kfree(g->b->f->f->fptr);
  kfree(g->b->f->f);
  
  compute_time(&ktime, &kt.setup);

  setktime(&ktime);

  ikernel(argc, argv, g, &kt);

  kfree(g->e->xyzn->x0);
  kfree(g->e->xyzn->x1);
  kfree(g->e->xyzn->x2);
  kfree(g->e->xyzn->x3);
  kfree(g->e->xyzn);

  kfree(g->e->w->w0->x0);
  kfree(g->e->w->w0->x1);
  kfree(g->e->w->w0->x2);
  kfree(g->e->w->w0);
  kfree(g->e->w->w1->x0);
  kfree(g->e->w->w1->x1);
  kfree(g->e->w->w1->x2);
  kfree(g->e->w->w1);
  kfree(g->e->w);

  kfree(g->e->eptr->n0);
  kfree(g->e->eptr->n1);
  kfree(g->e->eptr);
  kfree(g->e);

  kfree(g->s->part);
  kfree(g->s->ie);
  kfree(g->s->snfic);
  kfree(g->s);

  kfree(g->c);

  /* Boundaries */
  /* == Solid boundaries */
  kfree(g->b->s->n->xyz->x0);
  kfree(g->b->s->n->xyz->x1);
  kfree(g->b->s->n->xyz->x2);
  kfree(g->b->s->n->xyz);
  kfree(g->b->s->n->nptr);
  kfree(g->b->s->n);
  kfree(g->b->s->f);
  kfree(g->b->s);
  /* == Free boundaries */
  kfree(g->b->f->n->xyz->x0);
  kfree(g->b->f->n->xyz->x1);
  kfree(g->b->f->n->xyz->x2);
  kfree(g->b->f->n->xyz);
  kfree(g->b->f->n->nptr);
  kfree(g->b->f->n);
  kfree(g->b->f);
  /* Boundaries */
  kfree(g->b->snfptr->n0);
  kfree(g->b->snfptr->n1);
  kfree(g->b->snfptr->n2);
  kfree(g->b->snfptr);
  kfree(g->b);

  /* Nodes */
  kfree(g->n->xyz->x0);
  kfree(g->n->xyz->x1);
  kfree(g->n->xyz->x2);
  kfree(g->n->xyz);
  kfree(g->n->area);
  kfree(g->n);

  /* Geometry */
  kfree(g);

  compute_time(&ktime, &kt.kernel);

  compute_time(&ktime1, &kt.overall);

  print_kernel_time(kt);

  return 0;
}
