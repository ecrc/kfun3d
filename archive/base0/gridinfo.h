
#include <metis.h>

#define MAX_FILE_SIZE 256
#define GRIDPARAM     13
#define BS            4
/*
  Memory allocation with 64-byte alignment for a double data structure
*/
#define FCALLOC(size, y) ierr =                             \
        posix_memalign((void **) y, 64,                     \
                      PetscMax(size, 1) * sizeof(double));  \
        CHKERRQ(ierr);
/*
  Memory allocation with 64-byte alignment for an integer data structure
*/
#define ICALLOC(size, y) ierr =                             \
        posix_memalign((void **) y, 64,                     \
                      PetscMax(size, 1) * sizeof(int));     \
        CHKERRQ(ierr);

#define f77GETIA  f77name(GETIA,getia,getia_)
#define f77GETJA  f77name(GETJA,getja,getja_)

/*
  Stuffs for Incomplete LU (ILU)
  Compressed Sparse Row (CSR) representation
*/
typedef struct ilu_t
{
  int *ia;          // Starting position of each row in the matrix
  int *ja;          // The column indices
  int *ndiagElems;  // Number of the non zeros per row in the diagonal
  int *noffdElems;  // Number of the non zeros per row in the off diagonal
} ILU;
/*
  Global pointer to the ILU struct
*/
ILU ilu;
