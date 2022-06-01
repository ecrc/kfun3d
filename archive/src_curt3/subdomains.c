
/*
  Author: Mohammed Ahmed Al Farhan
  Email:  mohammed.farhan@kaust.edu.sa
*/

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include "inc/allocator.h"
#include "inc/geometry.h"
#ifdef __USE_METIS
#include <metis.h>
#endif

static void
emalloc_sub(struct geometry *restrict g)
{
  const uint32_t threads = omp_get_max_threads();

  struct etbl *restrict e;
  kmalloc(1, sizeof(struct etbl), (void *) &e);

  struct edge *restrict eptr;
  kmalloc(1, sizeof(struct edge), (void *) &eptr);

  struct xyzn *restrict xyzn;
  kmalloc(1, sizeof(struct xyzn), (void *) &xyzn);

  /* Number of the edges with replications means that 
   * some edges have been replicated (redundantly copied) to 
   * each thread to avoid communication */
  e->sz = g->s->ie[threads];
  
  kmalloc(e->sz, sizeof(uint32_t), (void *) &eptr->n0);
  kmalloc(e->sz, sizeof(uint32_t), (void *) &eptr->n1);

  e->eptr = eptr;

  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x0);
  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x1);
  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x2);
  kmalloc(e->sz, sizeof(double), (void *) &xyzn->x3);

  e->xyzn = xyzn;

  uint32_t *restrict buf;
  kmalloc(threads, sizeof(uint32_t), (void *) &buf);

  /* Copy the data from the original start edges index to the temporary
   * buffer */
  __assume_aligned(buf, MEMALIGN);
  memcpy(buf, g->s->ie, threads * sizeof(uint32_t));
  
  /* Scan the edges and color them based on their subdomains
   * that they belong to */
  uint32_t i;
  for(i = 0; i < g->e->sz; i++)
  {
    /* Get the thread IDs that this edge belongs to. The edge could
     * could be belong to two different threads. */
    uint32_t t0 = g->s->part[g->e->eptr->n0[i]];
    uint32_t t1 = g->s->part[g->e->eptr->n1[i]];

    /* Do the first thread */
    
    /* Store the edges' endpoints pointers in a new array that 
     * is indexed based upon their partition indices. */
    e->eptr->n0[buf[t0]] =  g->e->eptr->n0[i];
    e->eptr->n1[buf[t0]] =  g->e->eptr->n1[i];

    /* Store the edges' normals in a new array that is indexed
     * based upon their partition indices. */
    e->xyzn->x0[buf[t0]] = g->e->xyzn->x0[i];
    e->xyzn->x1[buf[t0]] = g->e->xyzn->x1[i];
    e->xyzn->x2[buf[t0]] = g->e->xyzn->x2[i];
    e->xyzn->x3[buf[t0]] = g->e->xyzn->x3[i];

    /* Move the index pointer by one so that we do not overwrite the
     * initial results of the current thread data.  */
    buf[t0]++;

    /* Do the second thread */

    /* Do the same processing if the second endpoint belongs to a 
     * different thread */
    if(t0 != t1)
    {
      /* Edge nodes pointers */
      e->eptr->n0[buf[t1]] = g->e->eptr->n0[i];
      e->eptr->n1[buf[t1]] = g->e->eptr->n1[i];

      /* Edge normals */
      e->xyzn->x0[buf[t1]] = g->e->xyzn->x0[i];
      e->xyzn->x1[buf[t1]] = g->e->xyzn->x1[i];
      e->xyzn->x2[buf[t1]] = g->e->xyzn->x2[i];
      e->xyzn->x3[buf[t1]] = g->e->xyzn->x3[i];

      buf[t1]++; // Move the index of the second thread
    }

  }

  kfree(buf); // Done from the temporary buffer

  kfree(g->e->eptr->n0);
  kfree(g->e->eptr->n1);
  kfree(g->e->eptr);

  kfree(g->e->xyzn->x0);
  kfree(g->e->xyzn->x1);
  kfree(g->e->xyzn->x2);
  kfree(g->e->xyzn->x3);
  kfree(g->e->xyzn);

  kfree(g->e);

  g->e = e;
}

void
isubdomain(struct geometry *restrict g)
{
  uint32_t *restrict part;
#if defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW)
  part = calloc(g->n->sz, sizeof(uint32_t));
#else
  kcalloc(g->n->sz, sizeof(uint32_t), (void *) &part);
#endif

  uint32_t threads = omp_get_max_threads();

  if(threads > 1)
  {
#ifdef __USE_METIS
    /* Number of balancing constraints that METIS needs to partitioning
     * the graph. The minimum is 1, which means you want to equidistribute 
     * the workload between the nodes as much as possible. 
     * So, one is the default.
     *
     * For more information, refer to METIS user manual
     * */
    int32_t ncon = 1;
    int32_t ncuts = 0;

    g->c->ia[g->n->sz]++; // Just for METIS

    METIS_PartGraphKway((int32_t*) &g->n->sz, &ncon,
                        (int32_t *) g->c->ia, (int32_t *) g->c->ja,
                        NULL, NULL, NULL, (int32_t *) &threads, NULL,
                        NULL, NULL, &ncuts, (int32_t *) part);

    g->c->ia[g->n->sz]--; // Just for METIS

    printf("Number edge cuts, generated by METIS is: %d\n", ncuts);
#endif
  }

  uint32_t *restrict ie;
  kcalloc(threads+1, sizeof(uint32_t), (void *) &ie);

  uint32_t i;
  for(i = 0; i < g->e->sz; i++)
  {
    /* Add 1 is to change the indexing from 0 to 1  */
    uint32_t t0 = part[g->e->eptr->n0[i]]; // Thread id owns the node
    uint32_t t1 = part[g->e->eptr->n1[i]]; // Thread id owns the node

    /* Add work to the first thread */
    ie[t0]++;
    /* If two threads are owning each part of the edge, means 
     * that each thread has one end point then add the second 
     * one as well */
    if(t0 != t1) ie[t1]++;
  }
 
  uint32_t sum = 0;
  for(i = 0; i < threads; i++)
  {
    uint32_t temp = ie[i];
    ie[i] = sum;
    sum += temp;
  }
  ie[threads] = sum; // Total number of nonzero blocks

  g->s->ie = ie;

#if defined(__USE_MEMKIND) && defined(__USE_POSIX_HBW)
  kcalloc(g->n->sz, sizeof(uint32_t), (void *) &g->s->part);

  memcpy(g->s->part, part, g->n->sz * sizeof(uint32_t));

  free(part);
#else
  g->s->part = part;
#endif

  emalloc_sub(g);
}
