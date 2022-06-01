
#include <stdlib.h>

typedef struct jacobian {
  /* previous sqrt(1.0 + || U ||) */
  double q_norm;
  double *w;
  double *q;
} Jacobian;

typedef struct csr_t {
  double *a;
  int *j;
  int *i;
} CSRTable; 

typedef struct bcsr_t {
  int *d;
  CSRTable *m;
} BCSRTable;

typedef struct precon_t {
  int bsz;
  int bsz2;
  size_t sz;
  BCSRTable *A;
  BCSRTable *LU; 
} PRECONTable;

//#include <petscvec.h>

typedef struct gmres_t {
  double *orthogwork;
  double *hh_origin;
  double *hs_origin;
  double *rs_origin;
  double *cc_origin;
  double *ss_origin;
  double **vecs;
} GMRES;

#define HH(a,b)  (gmres->hh_origin + (b)*(30+2)+(a))
#define HES(a,b) (gmres->hs_origin + (b)*(30+1)+(a))
#define CC(a)    (gmres->cc_origin + (a))
#define SS(a)    (gmres->ss_origin + (a))
#define GRS(a)   (gmres->rs_origin + (a))

#define VEC_OFFSET     2
#define VEC_TEMP       gmres->vecs[0]
#define VEC_TEMP_MATOP gmres->vecs[1]
#define VEC_VV(i)      gmres->vecs[VEC_OFFSET+i]
