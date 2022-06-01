
/*
  C routine that compute the memory usage by the application
  It calculates up to the point, where function is called
*/
#include "defs.h"

int
displayMemUse(char *msg)
{
  int       ierr;
  MPI_Comm  comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  
  ierr = PetscPrintf(comm, "================================\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Memory Usage ...................\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "================================\n"); 
  CHKERRQ(ierr);

/*  ierr = PetscMemoryShowUsage(PETSC_VIEWER_STDOUT_WORLD, msg);
  CHKERRQ(ierr);*/

  ierr = PetscMemoryView(PETSC_VIEWER_STDOUT_WORLD, msg);
  CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "================================\n"); 
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
