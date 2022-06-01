
#include "defs.h"
#include "header.h"
#include "gridinfo.h"

/* 
  Get the grid information into local ordering
*/
#undef  __FUNCT__
#define __FUNCT__ "initGrid"
int
initGrid(GRID *grid)
{
  int       ierr;
  int       *temp;
  char      meshFile[PETSC_MAX_PATH_LEN] = "";
  PetscBool flag;
  int       fd = 0;                   // File descriptor
  
  MPI_Comm  comm = PETSC_COMM_WORLD;

  int       nnodes;                   // Number of nodes
  int       nedge;                    // Number of edges
  int       nsfacet;                  // Number of node facets
  int       nvfacet;                  // Number of viscous facets
  int       nffacet;                  // Number of far field facets
  int       nsbound;                  // Number of boundary nodes
  int       nvbound;                  // Number of viscous nodes
  int       nfbound;                  // Number of far field nodes
  int       nsnode;                   // Number of solid nodes
  int       nvnode;                   // Number of viscous nodes
  int       nfnode;                   // Number of far field nodes

  double    sTime, eTime;
  int       *nntet, *nnpts;
  int       *nvtet, *nvpts;
  int       *nftet, *nfpts;
  /*
    First executable line of each PETSc function used 
    for error handling (for traceback). 
  */
  PetscFunctionBegin;
  /*
    Allocate a temporary storage for the grid parameters
  */
  ICALLOC(GRIDPARAM, &temp);
  /*
    Gets the string value for a particular option in the database
  */   
  ierr = PetscOptionsGetString( NULL, NULL, "-mesh", meshFile, MAX_FILE_SIZE,
                                &flag ); CHKERRQ(ierr);

/*  ierr = PetscOptionsGetString( NULL, "-mesh", meshFile, MAX_FILE_SIZE,
                                &flag ); CHKERRQ(ierr);*/

                                
  /* 
    Check to see if the path is a valid regular FILE
  */
  ierr = PetscTestFile(meshFile, 'r', &flag); CHKERRQ(ierr);
  /*
    The path is not a regular file path
  */ 
  if(!flag)
  { 
    /*
      "uns3d.msh" is the default filename with cwd path
      Copy a string
    */   
    ierr = PetscStrcpy(meshFile, "uns3d.msh"); CHKERRQ(ierr);
  }
  /*
    Opens a PETSc binary file (read only)
  */
  ierr = PetscBinaryOpen(meshFile, FILE_MODE_READ, &fd); CHKERRQ(ierr);
  /*
    Read grid parameters from the mesh file (13 parameters)
    Reads from a binary file (Collective on MPI_Comm)
  */
  ierr = PetscBinarySynchronizedRead( comm, fd, temp, GRIDPARAM, 
                                      PETSC_INT ); CHKERRQ(ierr);
  /*
    Store the grid parameters for global usage
  */
  nnodes  = grid->nnodes              = temp[1];  // Number of nodes
  nedge   = grid->nedge               = temp[2];  // Number of edges
  nsbound = grid->bound.nsbound       = temp[3];  // Solid boundaries
  nvbound = grid->bound.nvbound       = temp[4];  // Viscous boundaries
  nfbound = grid->bound.nfbound       = temp[5];  // Far field boundaries
  nsfacet = grid->bound.solid.nfacet  = temp[6];  // Solid facets
  nvfacet = grid->bound.viscs.nfacet  = temp[7];  // Viscous facets
  nffacet = grid->bound.ffild.nfacet  = temp[8];  // Far field facets
  nsnode  = grid->bound.solid.nnode   = temp[9];  // Solid nodes
  nvnode  = grid->bound.viscs.nnode   = temp[10]; // Viscous nodes
  nfnode  = grid->bound.ffild.nnode   = temp[11]; // Far field nodes
  
  /*
    Free the temporary storage for further usage
  */
  free(temp);  
  /*
    Returns the current time of day in seconds
  */
  ierr = PetscTime(&sTime); CHKERRQ(ierr);
  /*
    Edge pointers array
  */
  ICALLOC(2 * nedge, &grid->eptr);
  /*
    normal to faces and length
  */
  FCALLOC(nedge, &grid->normal.coord.x);
  FCALLOC(nedge, &grid->normal.coord.y);
  FCALLOC(nedge, &grid->normal.coord.z);
  FCALLOC(nedge, &grid->normal.len);
  /*
    Read edge pointers and store them
    Read the normal to faces and length and store them
  */
  ierr = processEdges(nedge, fd, grid->eptr, grid->normal);
  CHKERRQ(ierr);

  ierr = PetscTime(&eTime); CHKERRQ(ierr);
  eTime -= sTime;

  ierr = PetscPrintf(comm,  "Edges stored and reordered for better "
                            "cache locality\n"); 
  CHKERRQ(ierr);                           
  ierr = PetscPrintf(comm,  "Edge normals partitioned, reordered, "
                            "and stored for better cache locality\n");
  CHKERRQ(ierr);
  ierr = PetscPrintf( comm,  "Time taken in this phase was %g\n", 
                      eTime ); 
  CHKERRQ(ierr);

  ierr = PetscTime(&sTime); CHKERRQ(ierr);
  /*
    Stuffs for ILU(0)
    ia, ja
    Now make the local 'ia' and 'ja' arrays
  */
  ierr = constructILU(nnodes, nedge, grid->eptr); CHKERRQ(ierr);
 
  ierr = PetscTime(&eTime); CHKERRQ(ierr);
  eTime -= sTime;

  ierr = PetscPrintf( comm, "The Jacobian has %d non-zero blocks "
                            "with block size = %d\n", 
                      ilu.ia[nnodes] - 1, BS); 
  CHKERRQ(ierr);  
  ierr = PetscPrintf( comm,  "Time taken in this phase was %g\n", 
                      eTime ); 
  CHKERRQ(ierr);

  ierr = PetscTime(&sTime); CHKERRQ(ierr);
  /*
    Remap coordinates
  */
  FCALLOC(nnodes, &grid->coord.x);
  FCALLOC(nnodes, &grid->coord.y);
  FCALLOC(nnodes, &grid->coord.z);

  FCALLOC(nnodes, &grid->area);

  ierr = remapCoordinates(nnodes, fd, grid->coord, grid->area); 
  CHKERRQ(ierr);
 
  ierr = PetscTime(&eTime); CHKERRQ(ierr);
  eTime -= sTime;
  
  ierr = PetscPrintf(comm, "Coordinates remapped\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Time take in this phase was %g\n",
                     eTime); 
  CHKERRQ(ierr);

  ierr = PetscTime(&sTime); CHKERRQ(ierr);
  /*
    A vector that stores the partition vector of the graph generated
    by MeTiS
  */
  ICALLOC(nnodes, &grid->thread.partitions);
  /*
    Number of local edges for each threads
  */
  ICALLOC((nThreads + 1), &grid->thread.nedgeLoc);

  ierr = partitionSubdomainUsingMeTiS(nnodes, nedge, grid->eptr, 
                                      grid->thread.partitions, 
                                      grid->thread.nedgeLoc);
  CHKERRQ(ierr);
  /*
    Total Number of edges assigned to all threads
    nedge + edge replications
  */
  grid->thread.nedge = grid->thread.nedgeLoc[nThreads] - 1;
  /*
    Local edge pointers
  */
  ICALLOC(2 * grid->thread.nedge, &grid->thread.eptr);
  /*
    Local face normals and length
  */
  FCALLOC(grid->thread.nedge, &grid->thread.normal.coord.x);
  FCALLOC(grid->thread.nedge, &grid->thread.normal.coord.y);
  FCALLOC(grid->thread.nedge, &grid->thread.normal.coord.z);
  FCALLOC(grid->thread.nedge, &grid->thread.normal.len);
  
  ierr = processEdgesLoc( grid->nedge, grid->thread.nedge, grid->eptr, 
                          grid->normal, grid->thread.partitions, 
                          grid->thread.nedgeLoc, grid->thread.eptr, 
                          grid->thread.normal ); CHKERRQ(ierr);

  ierr = PetscTime(&eTime); CHKERRQ(ierr);
  eTime -= sTime;

  ierr = PetscPrintf(comm, "Subdomain partitioned\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Time take in this phase was %g\n",
                     eTime); 
  CHKERRQ(ierr);
 
  ierr = PetscTime(&sTime); CHKERRQ(ierr);
  /*
    Do the boundaries
  */
  /*
    ============================================ Solid Boundaries
  */
  ICALLOC(nsbound, &nntet);
  ICALLOC(nsbound, &nnpts);
  /*
    Facet to a solid tetrahedron  
  */
  ICALLOC(4 * nsfacet, &grid->bound.solid.f2nt);
  /* 
    Node number of solid nodes
  */
  ICALLOC(nsnode, &grid->bound.solid.inode);
  /*
    Normals
  */
  FCALLOC(nsnode, &grid->bound.solid.normal.x);
  FCALLOC(nsnode, &grid->bound.solid.normal.y);
  FCALLOC(nsnode, &grid->bound.solid.normal.z);
  /*
    Read from the mesh file
  */
  ierr = PetscBinarySynchronizedRead( comm, fd, nntet, nsbound, 
                                      PETSC_INT); 
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, nnpts, nsbound,
                                      PETSC_INT); 
  CHKERRQ(ierr);
   
  ierr = PetscBinarySynchronizedRead( comm, fd, grid->bound.solid.f2nt,
                                      4 * nsfacet, PETSC_INT);
  CHKERRQ(ierr);
  

  ierr = PetscBinarySynchronizedRead( comm, fd, grid->bound.solid.inode,
                                      nsnode, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, 
                                      grid->bound.solid.normal.x,
                                      nsnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, 
                                      grid->bound.solid.normal.y,
                                      nsnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, 
                                      grid->bound.solid.normal.z,
                                      nsnode, PETSC_SCALAR);
  CHKERRQ(ierr);

  free(nntet);
  free(nnpts);

  ierr = PetscPrintf(comm, "Solid boundaries partitioned\n");
  CHKERRQ(ierr);
  /*
    ============================================ Viscous Boundaries
  */
  ICALLOC(nvbound, &nvtet);
  ICALLOC(nvbound, &nvpts);
  ICALLOC(4 * nvfacet, &grid->bound.viscs.f2nt);
  ICALLOC(nvnode, &grid->bound.viscs.inode);

  FCALLOC(nvnode, &grid->bound.viscs.normal.x);
  FCALLOC(nvnode, &grid->bound.viscs.normal.y);
  FCALLOC(nvnode, &grid->bound.viscs.normal.z);

  ierr = PetscBinarySynchronizedRead( comm, fd, nvtet,
                                      nvbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, nvpts,
                                      nvbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, grid->bound.viscs.f2nt,
                                      4 * nvfacet, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, grid->bound.viscs.inode,
                                      nvnode, PETSC_INT);
  CHKERRQ(ierr);

  ierr = PetscBinarySynchronizedRead( comm, fd,
                                      grid->bound.viscs.normal.x,
                                      nvnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd,
                                      grid->bound.viscs.normal.y,
                                      nvnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd,
                                      grid->bound.viscs.normal.z,
                                      nvnode, PETSC_SCALAR);
  CHKERRQ(ierr);

  free(nvtet);
  free(nvpts);
  
  ierr = PetscPrintf(comm, "Viscous boundaries partitioned\n");
  CHKERRQ(ierr);
  /*
    ============================================ Viscous Boundaries
  */
  ICALLOC(nfbound, &nftet);
  ICALLOC(nfbound, &nfpts);
  ICALLOC(4 * nffacet, &grid->bound.ffild.f2nt);
  ICALLOC(nfnode, &grid->bound.ffild.inode);

  FCALLOC(nfnode, &grid->bound.ffild.normal.x);
  FCALLOC(nfnode, &grid->bound.ffild.normal.y);
  FCALLOC(nfnode, &grid->bound.ffild.normal.z);

  ierr = PetscBinarySynchronizedRead( comm, fd, nftet,
                                      nfbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, nfpts,
                                      nfbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, grid->bound.ffild.f2nt,
                                      4 * nffacet, PETSC_INT);
  CHKERRQ(ierr);
  
  ierr = PetscBinarySynchronizedRead( comm, fd, grid->bound.ffild.inode,
                                      nfnode, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, 
                                      grid->bound.ffild.normal.x,
                                      nfnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, 
                                      grid->bound.ffild.normal.y,
                                      nfnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead( comm, fd, 
                                      grid->bound.ffild.normal.z,
                                      nfnode, PETSC_SCALAR);
  CHKERRQ(ierr);

  free(nftet);
  free(nfpts);

  ierr = PetscPrintf(comm, "Free boundaries partitioned\n");
  CHKERRQ(ierr);

  ierr = PetscTime(&eTime);
  eTime -= sTime;

  ierr = PetscPrintf( comm, "Time take in this phase was %g\n", 
                      eTime);
  CHKERRQ(ierr);

  ierr = displayMemUse("Memory usage after partitioning\n"); 
  CHKERRQ(ierr);
  /*
    The time step vector for each cell
  */ 
  FCALLOC(nnodes, &grid->cdt);
  /*
    The weights vector of the gradient
  */
  FCALLOC(nnodes, &grid->weight.r11);
  FCALLOC(nnodes, &grid->weight.r12);
  FCALLOC(nnodes, &grid->weight.r13);
  FCALLOC(nnodes, &grid->weight.r22);
  FCALLOC(nnodes, &grid->weight.r23);
  FCALLOC(nnodes, &grid->weight.r33);
  FCALLOC(nnodes, &grid->weight.r44);
  /*
    Put different mappings and other info into grid
  */
  Print:
  {   
    ierr = PetscPrintf(comm,"==============================\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Partitioning quality info ....\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"==============================\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"-----------------------------------\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Item                       Total\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"-----------------------------------\n");
    CHKERRQ(ierr);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of nodes       %9d\n", nnodes);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of edges       %9d\n", nedge);
    CHKERRQ(ierr);
    ierr = PetscPrintf( comm, "Edges+replication     %9d\n", 
                        grid->thread.nedge); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of solid faces %9d\n", nsfacet);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of viscous faces %7d\n", nvfacet);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of free faces  %9d\n", nffacet);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of solid nodes %9d\n", nsnode);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of viscous nodes %7d\n", nvnode);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of free nodes  %9d\n", nfnode);
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"-----------------------------------\n");
    CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0); 
}
/*
  1.1 Read the edge pointers from the input mesh file
  1.2 Order them based on their permutation array
  1.3 Field interlacing
  1.4 Store them in the grid struct
  2.1 Read the normals and the length from the input mesh file
  2.2 Field interlacing
  2.3 Store them in the grid struct
*/
#undef  __FUNCT__
#define __FUNCT__ "processEdges"
int
processEdges(int nedge, int fd, int *eptr, struct NORMAL normal)
{
  int     *temp1;       // Temporary array for the edge pointers
  int     *eperm;       // Edge permutation array
  double  *temp2;       // Temporary array for the normals and length
  int     ierr;         // Error return value
  int     i = 0, k = 0; // Loop iterators
  int     node1, node2; // An edge endpoints
  /*
    MPI communicator
  */
  MPI_Comm  comm = PETSC_COMM_WORLD;

  PetscFunctionBegin;
  /*
    Edge pointers temporary array
  */
  ICALLOC(2 * nedge, &temp1);
  /*
    Edge permutation array
  */
  ICALLOC(nedge, &eperm);
  /*
    Normal to faces and length temporary array
  */
  FCALLOC(4 * nedge, &temp2);  
  /*
    Read edge pointers from the mesh file
  */
  ierr = PetscBinarySynchronizedRead( comm, fd, temp1, 2 * nedge,
                                      PETSC_INT ); CHKERRQ(ierr);  
  /*
    Initializes the permutation array from 0:n-1
  */
  for(i = 0; i < nedge; i++) eperm[i] = i;
  /*
    Now reorder the edges for better cache locality
  */
  /*
    Computes the permutation of values that gives a sorted sequence.
  */
  double start_time = 0.f, end_time = 0.f;

  ierr = PetscTime(&start_time); CHKERRQ(ierr);

  PetscPrintf(comm, "\t **** Find the permutation array using PETSc\n");
  CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(nedge, temp1, eperm); 
  CHKERRQ(ierr);

  //imain(nedge, temp1, eperm);

  ierr = PetscTime(&end_time); CHKERRQ(ierr);

  end_time -= start_time;

  PetscPrintf(comm, "\t **** Time spent in finding the permutation ");
  CHKERRQ(ierr);
  PetscPrintf(comm, "array is: %g sec.\n", end_time);
  CHKERRQ(ierr);

  /*
    Test the memory for corruption. This can be used to check 
    for memory overwrites
  */
  ierr = PetscMallocValidate(__LINE__, __FUNCT__, __FILE__); 
  CHKERRQ(ierr);
  /*
    Loop over edges and store them with field Interlacing for better 
    cache locality using edge pointers array
  */

  k = 0;
  for(i = 0; i < nedge; i++)
  {
    /*
      Read an edge
    */
    node1 = temp1[eperm[i]];
    node2 = temp1[nedge + eperm[i]];
    /*
      Reorder the edge; store the edge nodes consecutively
    */
    eptr[k++] = node1;
    eptr[k++] = node2;

  } 
  /*
    Renumber unit normals of dual face
    and the area of the dual mesh face
  */
  /*
    x-component
  */
  ierr = PetscBinarySynchronizedRead( comm, fd, temp2, 
                                      4 * nedge, PETSC_SCALAR );
  CHKERRQ(ierr);

  for(i = 0; i < nedge; i++){
    normal.coord.x[i] = temp2[eperm[i]]; 
  /*
    y-component
  */
 // ierr = PetscBinarySynchronizedRead( comm, fd, temp2, 
 //                                     nedge, PETSC_SCALAR );
//  CHKERRQ(ierr);
  
//  for(i = 0; i < nedge; i++)
    normal.coord.y[i] = temp2[eperm[i]+nedge];
  /*
    z-component
  */
//  ierr = PetscBinarySynchronizedRead( comm, fd, temp2, 
//                                      nedge, PETSC_SCALAR );
//  CHKERRQ(ierr);
  
//  for(i = 0; i < nedge; i++)
    normal.coord.z[i] = temp2[eperm[i]+nedge+nedge]; 
  /*
    Length
  */
//  ierr = PetscBinarySynchronizedRead( comm, fd, temp2, 
  //                                    nedge, PETSC_SCALAR );
//  CHKERRQ(ierr);

//  for(i = 0; i < nedge; i++)
    normal.len[i] = temp2[eperm[i]+nedge+nedge+nedge]; 
}
  free(eperm);
  free(temp2);
  free(temp1);
#if 0
  FILE * vf = fopen("k.in", "w");

  int myk = 0;

  for(i = 0; i < nedge; i++)
  {
    fprintf(vf,"%d : %d\n", 
                eptr[myk++]-1, eptr[myk++]-1);
    fprintf(vf,"%lf : %lf : %lf : %lf\n",
                normal.coord.x[i], normal.coord.y[i], normal.coord.z[i],
                normal.len[i]);

  }
  fclose(vf);

/*
  for(i = 0; i < 10; i++) 
  {
    printf("%f\n", normal.coord.x[i]);
    printf("%f\n", normal.coord.y[i]);
    printf("%f\n", normal.coord.z[i]);
    printf("%f\n", normal.len[i]);
  }
*/
#endif

  PetscFunctionReturn(0);
}
/*
  Construct the Incomplete LU stffs
  ia: starting position of each row
  ja: Column indices
*/
#undef  __FUNCT__
#define __FUNCT__ "constructILU"
int
constructILU(int nnodes, int nedge, int *eptr)
{
  int *temp;
  int nnz;        // Number of the non-zero elements in the matrix A
  int i;
  int jstart;     // Start index in the ia
  int jend;       // End index in the ja
  int ndiagElems; // Number of the non zeros elements in the diagonal
  int noffdElems; // Number of the non zeros elements in the off diagonal
  int j;
  int ierr;

  PetscFunctionBegin;
  /*    
    IA is the row pointer array
    represents the starting positions of each row of the matrix A
    The last element is the number of nonzero in the matrix
  */
  ICALLOC(nnodes + 1, &ilu.ia);
  ICALLOC(nnodes, &temp);
  /*
    Constructs ia
  */
  f77GETIA(&nnodes, &nedge, eptr, temp, ilu.ia); 

  /*
    Number of non zeros
    used to allocated JA array
    column indices of non zero elements in the
    sparse matrix
  */
  nnz = ilu.ia[nnodes] - 1;
  /*
    The column indices
  */
  ICALLOC(nnz, &ilu.ja); 
  /*
    temp1: is iwork array used during the calculation of the column index
    ja: is the column indices 
  */
  f77GETJA(&nnodes, &nedge, eptr, ilu.ia, temp, ilu.ja);

  free(temp);

/*
  f = fopen("ia.me", "w");

  for(i = 0; i < nnodes+1; i++) fprintf(f, "%d\n", ilu.ia[i]-1);

  fclose(f);
*/

  /*
    Adjust the ja array
  */
  for (i = 0; i < nnz; i++)
    ilu.ja[i] = ilu.ja[i] - 1;

 for (i = 0; i < nnodes; i++)
    ilu.ia[i] = ilu.ia[i] - 1;

  /*
    Store the number of non-zeros per row
  */
  ICALLOC(nnodes, &ilu.ndiagElems);
  ICALLOC(nnodes, &ilu.noffdElems);
  /*
    Loop over ia and ja to count the number of the none zeros
  */
  for(i = 0; i < nnodes; i++)
  {
    jstart = ilu.ia[i];// - 1;
    jend   = ilu.ia[i + 1];// - 1;
    
    if(jend == ilu.ia[nnodes]) jend--;

    ndiagElems = 0;  // Number of diagonal elements
    noffdElems = 0;  // Number of off-diagonal elements
    for(j = jstart; j < jend; j++)
    {
      if(ilu.ja[j] < nnodes) 
        ndiagElems++;
      else 
        noffdElems++;
    }
    ilu.ndiagElems[i] = ndiagElems;
    ilu.noffdElems[i] = noffdElems;
  }

#if 0
  FILE * f = fopen("ja.me", "w");

  for(i = 0; i < nnodes; i++) fprintf(f, "%d\n", ilu.ndiagElems[i]);

  fclose(f);
#endif
 
  PetscFunctionReturn(0);
}
/*
  Remap XYZ coordinates and the area
*/
#undef  __FUNCT__
#define __FUNCT__ "remapCoordinates"
int
remapCoordinates(int nnodes, int fd, struct COORD coord, double *area)
{
  double  *temp;
  int     ierr;
  int     i;

  MPI_Comm comm = PETSC_COMM_WORLD;
  
  PetscFunctionBegin;

  FCALLOC(4 * nnodes, &temp);
  /* 
    x-component
  */   
  ierr = PetscBinarySynchronizedRead( comm, fd, temp, 4 * nnodes, 
                                      PETSC_SCALAR ); CHKERRQ(ierr);

  for(i = 0; i < nnodes; i++)
  {
    coord.x[i] = temp[i];
  /*
    y-component
  */
//  ierr = PetscBinarySynchronizedRead( comm, fd, temp, nnodes, 
  //                                    PETSC_SCALAR ); CHKERRQ(ierr);
 // for(i = 0; i < nnodes; i++)
    coord.y[i] = temp[i + nnodes];  
  /* 
    z-component
  */
//  ierr = PetscBinarySynchronizedRead( comm, fd, temp, nnodes, 
  //                                    PETSC_SCALAR ); CHKERRQ(ierr);
//  for(i = 0; i < nnodes; i++)
    coord.z[i] = temp[i + nnodes + nnodes];
  /*
    Renumber dual volume
  */
//  ierr = PetscBinarySynchronizedRead( comm, fd, temp, nnodes, 
  //                                    PETSC_SCALAR ); CHKERRQ(ierr);
//  for(i = 0; i < nnodes; i++)
    area[i] = temp[i + nnodes + nnodes + nnodes];
  }
  free(temp);

/*
  FILE * f = fopen("co.txt", "w");
  for(i = 0; i < nnodes; i++) 
    fprintf(f, "%f\t%f\t%f\t%f\n",
            coord.x[i], coord.y[i], coord.z[i], area[i]);
  
  fclose(f);

*/
  PetscFunctionReturn(0);
}
/*
  Get the local adjacency structure of the graph
  for partitioning the local graph into
  nThreads pieces
*/
#undef  __FUNCT__
#define __FUNCT__ "partitionSubdomainUsingMeTiS"
int
partitionSubdomainUsingMeTiS( int nnodes, int nedge, int *eptr, 
                              int *partitions, int *nedgeLoc )
{
  int ierr;

  //int *ia, *ja;
  int options[5]; 
  int *vwtg   = 0;
  int *adjwgt = 0;
  int numflag = 0;
  int wgtflag = 0;  
  int edgecut; 
 
  int j, i, k, node1, node2; 
  int inode, jstart, jend;  
  int thr1, thr2, ned1, ned2;

  MPI_Comm comm = PETSC_COMM_WORLD;
  
  PetscFunctionBegin;
 
  options[0] = 0;    
   
//  ICALLOC((nnodes + 1), &ia);
//  ICALLOC((2 * nedge), &ja);
//  ia[0] = 0;

//  for(i = 0; i < nnodes; i++)
  //  ilu.ia[i]--;// - i - 1;


 /*
  for(i = 1; i <= nnodes; i++)
    ia[i] = ilu.ia[i] - i - 1;
  
  for(i = 0; i < nnodes; i++)
  {
    jstart  = ilu.ia[i] - 1;
    jend    = ilu.ia[i + 1] - 1;

    k = ia[i];
    
    for(j = jstart; j < jend; j++)
    {
      inode = ilu.ja[j];
      if(inode != i)
        ja[k++] = inode;
    }
  }
  */
  for(i = 0; i < nnodes; i++)
    partitions[i] = 0;
  
  if(nThreads > 1)
  {
    /*
       
      METIS_PartGraphRecursive(&nnodes, ia, ja, vwtg,
                               adjwgt, &wgtflag, &numflag,
                               &nThreads, options, 
                               &edgecut, gridLoc->partitions); 
    */
/*    METIS_PartGraphKway(&nnodes, ia, ja, vwtg, adjwgt, &wgtflag, 
                        &numflag, &nThreads, options, 
                        &edgecut, partitions);*/

    int ncon = 1;

    METIS_PartGraphKway(&nnodes, 
                        &ncon, 
                        ilu.ia, 
                        ilu.ja, 
                        NULL, 
                        NULL, 
                        NULL, 
                        &nThreads, 
                        NULL, 
                        NULL,
                        NULL,
                        &edgecut, 
                        partitions);

                        
  }

  ierr = PetscPrintf( comm, "The number of cut edges is %d\n", 
                      edgecut); CHKERRQ(ierr);
  
  for(i = 0; i < (nThreads + 1); i++)
    nedgeLoc[i] = 0;
    
  k = 0;
  for(i = 0; i < nedge; i++)
  {
    node1 = eptr[k++] - 1;
    node2 = eptr[k++] - 1;
    
    thr1  = partitions[node1] + 1;
    thr2  = partitions[node2] + 1;
    
    nedgeLoc[thr1] += 1;
    
    if(thr1 != thr2)
      nedgeLoc[thr2] += 1;
  }


//  ned1 = nedgeLoc[0];
  nedgeLoc[0] = 1;
  
  for(i = 1; i <= nThreads; i++)
  {
//    ned2 = nedgeLoc[i];

    nedgeLoc[i] += nedgeLoc[i - 1];// + ned1;

//    ned1 = ned2;
  }

  PetscFunctionReturn(0);
}
/*
  Partition the edges, normals to faces, and length between 
  the OpenMP threads
*/
#undef  __FUNCT__
#define __FUNCT__ "processEdgesLoc"
int
processEdgesLoc(int nedge, int nedgeThreads, int *eptr, 
                struct NORMAL normal, int *partitions, 
                int *nedgeLoc, int *eptrLoc, struct NORMAL normalLoc)
{
  int ierr;
  int *temp;
  int i, k;
  int ie1, ie2, ie3;
  int node1, node2, thr1, thr2;

  PetscFunctionBegin;
  
  ICALLOC(nThreads, &temp);
  
  for(i = 0; i < nThreads; i++)
    temp[i] = nedgeLoc[i] - 1;
    
  k = 0;
  for(i = 0; i < nedge; i++)
  {
    node1 = eptr[k++];
    node2 = eptr[k++];
        
    thr1 = partitions[node1 - 1];
    thr2 = partitions[node2 - 1];

    ie1 = 2 * temp[thr1];
    ie2 = temp[thr1];
    ie3 = i;
      
    eptrLoc[ie1]     = node1;
    eptrLoc[ie1 + 1] = node2;

    normalLoc.coord.x[ie2]    = normal.coord.x[ie3];
    normalLoc.coord.y[ie2]    = normal.coord.y[ie3];
    normalLoc.coord.z[ie2]    = normal.coord.z[ie3];
    normalLoc.len[ie2]        = normal.len[ie3];
      
    temp[thr1] += 1;
      
    if(thr1 != thr2)
    {
      ie1 = 2 * temp[thr2];
      ie2 = temp[thr2];

      eptrLoc[ie1]      = node1;
      eptrLoc[ie1 + 1]  = node2;

      normalLoc.coord.x[ie2]    = normal.coord.x[ie3];
      normalLoc.coord.y[ie2]    = normal.coord.y[ie3];
      normalLoc.coord.z[ie2]    = normal.coord.z[ie3];
      normalLoc.len[ie2]        = normal.len[ie3];
        
      temp[thr2] += 1;
    } 
  }

  PetscFunctionReturn(0);
}
/*  
  Create non-linear solver
*/
#undef  __FUNCT__
#define __FUNCT__ "initPETScDS"
int initPETScDS(void *ctx)
{
  AppCtx    *user       = (AppCtx *) ctx;
  GRID      *grid       = user->grid;
  GRIDPETSc *gridPETSc  = user->gridPETSc;
  TstepCtx  *tsCtx      = user->tsCtx;
  
  int   nnodes = grid->nnodes;
  int   *indices;
  int   ierr, i; 
  char  *msg;
  
  ISLocalToGlobalMapping  isl2g;
  MPI_Comm                comm = PETSC_COMM_WORLD;
  
  PetscFunctionBegin;
  /* 
    Creates a standard, sequential array-style vector
  */ 
  ierr = VecCreateSeq(comm, BS * nnodes, &gridPETSc->qnode); 
  CHKERRQ(ierr);
  ierr = VecCreateSeq(comm, 3 * BS * nnodes, &gridPETSc->grad);  
  CHKERRQ(ierr);
  /*
    Creates a new vector of the same type as an existing vector
  */
  ierr = VecDuplicate(gridPETSc->qnode, &gridPETSc->qnodeLoc);    
  CHKERRQ(ierr);
  ierr = VecDuplicate(gridPETSc->qnode, &gridPETSc->res);   
  CHKERRQ(ierr);
  ierr = VecDuplicate(gridPETSc->qnode, &tsCtx->res); 
  CHKERRQ(ierr);
  ierr = VecDuplicate(gridPETSc->qnode, &tsCtx->qnode); 
  CHKERRQ(ierr);
  /* 
    Left hand side matrix (( A ))
    Creates a sparse parallel matrix in block AIJ format 
    (block compressed row); parameters:
    Block size; number of local rows; number of local columns
    Number of global rows; number of global columns
    Number of non-zero blocks per block row in the diagonal
    Array contains the number of non-zero blocks in the diagonal
    Number of non-zero blocks per block row in the off diagonal
    Array contains the number of non-zero blocks in the off-dia
    Note::: Use PETSc default value in both cases
  */
  ierr = MatCreateBAIJ( comm, BS, BS * nnodes, BS * nnodes, BS * nnodes, 
                        BS * nnodes, PETSC_DEFAULT, ilu.ndiagElems, 
                        PETSC_DEFAULT, ilu.noffdElems, &gridPETSc->A ); 
  CHKERRQ(ierr);

  msg = "Memory usage after allocating PETSc data structures\n";

  ierr = displayMemUse(msg); CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "================================\n"); 
  CHKERRQ(ierr);
  
  /* 
    Set local to global mapping for setting the matrix
    elements in local ordering: first set row by row mapping
  */
  ICALLOC(BS * nnodes, &indices);
  for(i = 0; i < BS * nnodes; i++)
    indices[i] = i;  
  /*
    Creates a mapping between a local (0 to n) ordering 
    and a global parallel ordering; parameters:
    Number of local elements
    Global index of each local element
    Copy mode
    new mapping data structure
  */
  ierr = ISLocalToGlobalMappingCreate(comm, BS, nnodes, indices, 
                                      PETSC_COPY_VALUES, &isl2g); 
  CHKERRQ(ierr);
  /* 
    Sets a local-to-global numbering for use by the 
    routine MatSetValuesLocal() to allow users to insert 
    matrix entries using a local (per-processor) numbering.
    Parameters: 
    Row mapping created with ISLocalToGlobalMappingCreate
    Column mapping
  */
  ierr = MatSetLocalToGlobalMapping(gridPETSc->A, isl2g, isl2g); 
  CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(&isl2g); CHKERRQ(ierr);
  
  free(indices);

  PetscFunctionReturn(0);
}
/*
  Deallocate the ia and ja at the end of the kernel
*/
#undef  __FUNCT__
#define __FUNCT__ "cleanILUmem"
void
cleanILUmem()
{
  free(ilu.ia);
  free(ilu.ja);
  free(ilu.ndiagElems);
  free(ilu.noffdElems);
}
