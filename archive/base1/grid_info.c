#include "user.h"

#define P_LEN   PETSC_MAX_PATH_LEN

// Get the grid information into local ordering
#undef  __FUNCT__
#define __FUNCT__ "GetLocalOrdering"
PetscErrorCode GetLocalOrdering(GRID *grid)
{
  int             ierr;                     // Return error code from PETSc functions
  int             *tmp;                     // Dummy integer array
  int             grid_param = 13;          // Number of grid parameters (for reading the mesh file)
  char            mesh_file[P_LEN] = "";    // Mesh file
  PetscBool       flg;                      // Generic existence flag
  PetscBool       exists;                   // File existence flag 
  int             fdes = 0;                 // File descriptor
  MPI_Comm        comm = PETSC_COMM_WORLD;  // MPI communicator object
  int             nnodes;                   // Number of nodes
  int             nedge;                    // Number of edges
  int             nnfacet;
  int             nvfacet;
  int             nffacet;
  int             nnbound;
  int             nvbound; 
  int             nfbound;
  int             nsnode;
  int             nvnode;
  int             nfnode;
  int             *l2a;                     // Local to global mapping of nodes
  int             *v2p;                     // MeTiS partitioning (process's rank) [GLOBAL MAPPING]
  int             *a2l;                     // Index of local and global (-1) node [LOCAL MAPPING]
  int             nnodesLoc = 0;            // Number of local nodes for each MPI process
  FILE            *fptr;
  int             i, j, k, inode;           // Loop iterators
  double          time_ini, time_fin;       // Start/Exit time
  int             node1, node2, node3;      // Mesh node (from the edge pointer array)
  int             *pordering;               // PETSc ordering
  AO              ao;                       // Application ordering
  int             nedgeLoc;                 // Number of local edges for each MPI process
  int             nvertices;                // Number of local nodes + ghost nodes (each MPI proc)
  int             nedgeLocEst;              // Estimated number of local edges
  int             remEdges;                 // Remaining number of edges 
  off_t           currentPos = 0;           // File offset position (current)
  int             readEdges;                // Number of edges have been read from the mesh file
  off_t           newPos = 0;               // File offset position (next)
  int             *eperm;                   // Edge permutation array, initialized from (0-n-1)
  int             *edge_bit;                // Global edge index
  int             cross_edges = 0;          // Number of edges with endpoints in diff components 
  int             nnz;                      // Number of none zeros in the matrix
  int             bs = 4;                   // Matrix block size
  int             *l2p;                     // Local to PETSc mapping
  int             *p2l;                     // PETSc to Local mapping
  double          *ftmp;
  double          *ftmp1;
  int             nnodesLocEst;             // Estimated number of local nodes
  int             remNodes;                 // Remaining number of nodes
  int             readNodes;                // # of nodes that have been read from the mesh file
  int             isurf;                    // Local surface
  int             nsnodeLoc;                // Local solid nodes
  int             nnfacetLoc;               // local solid faces
  int             nte;
  int             nb;
  int             *tmp1;
  int             *tmp2; 
  int             nvnodeLoc;                // Local viscous nodes
  int             nvfacetLoc;               // Local viscous faces
  int             nfnodeLoc;                // Local far field nodes
  int             nffacetLoc;               // Local far field faces
  FILE            *fptr1;
  int             jstart, jend;

  // First executable line of each PETSc function used for error handling (for traceback). 
  PetscFunctionBegin;
  // Read the integer grid parameters
  // 13 grid parameters with integer datatype
  ICALLOC(grid_param, &tmp);
  // First MPI process. Done only by one process, others will skip this
  if(!rank) // rank == 0
  {
    // Gets the string value for a particular option in the database
    ierr = PetscOptionsGetString(NULL, NULL, "-mesh", mesh_file, MAX_FILE_SIZE,
                                 &flg); CHKERRQ(ierr);
    // Check to see if the path is a valid regular FILE
    // File mode => r: readable; w: writable; e: executable
    ierr = PetscTestFile(mesh_file, 'r', &exists); CHKERRQ(ierr);
    if(!exists)
    { // try uns3d.msh as the default file name
      // Copy a string
      ierr = PetscStrcpy(mesh_file, "uns3d.msh"); CHKERRQ(ierr);
    }
    // Opens a PETSc binary file (read only)
    ierr = PetscBinaryOpen(mesh_file, FILE_MODE_READ, &fdes); CHKERRQ(ierr); 
  }
  // Read grid parameters from the mesh file (13 parameters)
  // Reads from a binary file (Collective on MPI_Comm)
  ierr = PetscBinarySynchronizedRead(comm, fdes, tmp, grid_param, PETSC_INT); CHKERRQ(ierr);
  
  // Store the grid parameters in the grid variables of the grid struct 
  grid->ncell   = tmp[0];   // Number of cells
  grid->nnodes  = tmp[1];   // Number of nodes
  grid->nedge   = tmp[2];   // Number of edges
  grid->nnbound = tmp[3];   // Number of nodes in the boundary
  grid->nvbound = tmp[4];   // Number of 
  grid->nfbound = tmp[5];   // Number of faces in the boundary
  grid->nnfacet = tmp[6];
  grid->nvfacet = tmp[7];   
  grid->nffacet = tmp[8];   
  grid->nsnode  = tmp[9];   // Number of solid nodes
  grid->nvnode  = tmp[10];  // Number of viscous nodes
  grid->nfnode  = tmp[11];  // Number of far field nodes
  grid->ntte    = tmp[12];
  grid->nsface  = 0;        // Number of solid faces
  grid->nvface  = 0;        // Number of viscous faces
  grid->nfface  = 0;        // Number of far field faces
  
  // Free memory
  free(tmp);
  // Prints to standard out, only from the first processor in the communicator 
  // Calls from other processes are ignored
  ierr = PetscPrintf(comm,"nnodes = %d, nedge = %d, nnfacet = %d, nsnode = %d, "
                          "nfnode = %d\n", 
                     grid->nnodes, grid->nedge, grid->nnfacet, 
                     grid->nsnode, grid->nfnode); CHKERRQ(ierr);
  // Point to the grid variables to deal with them locally within the function
  nnodes  = grid->nnodes;
  nedge   = grid->nedge;
  nnfacet = grid->nnfacet;
  nvfacet = grid->nvfacet;
  nffacet = grid->nffacet;
  nnbound = grid->nnbound;
  nvbound = grid->nvbound;
  nfbound = grid->nfbound;
  nsnode  = grid->nsnode;
  nvnode  = grid->nvnode;
  nfnode  = grid->nfnode;

  // ==============================================================
  // v2p: Vertex to Processor Mapping 
  // Reading the generated vector mapping of MPI process ranks that
  // is generated by MeTiS partitioning
  // ==============================================================
  // Examples
  //********************************
  //v2p -> [0 0 1 0 1 2 3 0 2 2]
  //********************************
  //P0: a2l -> [0 1 -1 2 -1 -1  -1  3  -1]
  //P0: l2a -> [1 2 7]
  //********************************
  //P1: a2l -> [-1 -1 0 -1 1 -1 -1  -1 -1]
  //P1: l2a -> [2 4]
  //********************************
  //P2: a2l -> [-1 -1 -1 -1 -1 0 -1  1  2] 
  //P2: l2a -> [5 8 9]
  //********************************
  //P3: a2l -> [-1 -1 -1 -1 -1 -1 0 -1 -1]
  //P3: l2a -> [6]
  //********************************
  // Read the partitioning vector generated by MeTiS
  ICALLOC(nnodes, &l2a); // size of local nodes
  // It contains the indices of each node in the v2p vector
  // Includes nnodes lines each line has a process number 
  // represents which process that node belongs to 
  ICALLOC(nnodes, &v2p); // partitioning vector generated by MeTiS
  ICALLOC(nnodes, &a2l); // [number of nodes]
  // It contains the indices of the l2a array
  nnodesLoc = 0; // Local number of nodes assigned to each process
  // Initialize a2l vector to -1's
  for(i = 0; i < nnodes; i++)
    a2l[i] = -1;
  // Timing:  reading partitioning vector task
  // Returns the current time of day in seconds
  ierr = PetscTime(&time_ini); CHKERRQ(ierr);
  // Rank 0 ( first MPI process)
  if(!rank)
  {
    if(size == 1) // Only 1 MPI process (Non distributed memory code)
    {
      // Avoid the overhead of reading the partitioning vector
      // All nodes are assigned to the rank 0 process
      // (No partitioning)
      // Zeros the specified memory
      ierr = PetscMemzero(v2p, nnodes * sizeof(int)); CHKERRQ(ierr);
    }
    else // More than 1 MPI process (Distributed memory)
    {
      char spart_file[P_LEN], part_file[P_LEN];
      // Read the partitioning vector file into spart_file
      // "spart_file" accepts any user input name
      // This is a protection to make sure that PETSc will read an existing file 
      // that is generated by MeTiS 
      ierr = PetscOptionsGetString(NULL, NULL, "-partition", spart_file,
                                   P_LEN, &flg); CHKERRQ(ierr);
      ierr = PetscTestFile(spart_file, 'r', &exists); CHKERRQ(ierr);
      // The structure of the file that is generated by MeTiS is:
      // Each line has a number from 0 ... size (number of MPI processes)
      // That number represents the process that this node belongs to
      // The total number of lines in the file equals to the number of nodes.
      if(!exists) // Mistake in the file name
      { // Append the number of processes
        // The standard format of the generated file by MeTiS after partitioning
        // "part_vec.part.<number of MPI processes>
        sprintf(part_file, "part_vec.part.%d", size);
        ierr = PetscStrcpy(spart_file, part_file); CHKERRQ(ierr);
      }
      // Open formated file
      fptr = fopen(spart_file, "r");
      if(!fptr)
        SETERRQ1(PETSC_COMM_SELF, 1, "Cannot open file %s\n", part_file);
      // Read the input file 
      for(inode = 0; inode < nnodes; inode++)
      {
        // Rank of each node (from node 0 to node N)
        fscanf(fptr, "%d\n", &node1);
        v2p[inode] = node1; // add it to the vector
      }
      fclose(fptr);
    }
  }  
  // Broadcast the partitioning vector data from the first MPI process 
  // (rank 0) to all process in the communicator
  ierr = MPI_Bcast(v2p, nnodes, MPI_INT, 0, comm); CHKERRQ(ierr);
  // Read the broadcasted vector by each MPI process
  for(inode = 0; inode < nnodes; inode++)
  {
    // If this node belongs to this process (characterized by the rank)
    if(v2p[inode] == rank)
    {
      // Store the index of this node in the process's local array
      // This array contains only the indices of the local nodes
      // The max size of this array is the number of nodes
      // However, it is dynamic array stores only the local node indices 
      l2a[nnodesLoc] = inode;
      // This array contains all the mesh nodes (-1s)
      // If this node belongs to this process, then
      // Store the node index, otherwise, keep it -1, 
      // which represents this node is not belong to this process
      // Fixed size associated to the total number of nodes in the mesh
      a2l[inode] = nnodesLoc;
      // Keeps track of the total number of local nodes for each MPI process
      nnodesLoc++; // Number of local nodes
    }
  }
  // End timing
  ierr = PetscTime(&time_fin); CHKERRQ(ierr);
  time_fin -= time_ini;
  ierr = PetscPrintf(comm, "Partition Vector read successfully\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Time taken in this phase was %g\n", time_fin);
  CHKERRQ(ierr);
  // Count total number of neighbor nodes
  // ***********************************
  // Compute partial reduction 
  // Sum total number of nodes in the mesh from each process 
  // rstart should equals to the total number of nodes in the mesh
  ierr = MPI_Scan(&nnodesLoc, &rstart, 1, MPI_INT, MPI_SUM, comm); 
  CHKERRQ(ierr);
  rstart -= nnodesLoc; // Remaining nodes (not belongs to this process)
  ICALLOC(nnodesLoc, &pordering); // PETSc ordering for the nodes
  // create new ordering array for the nodes based upon total number 
  // of nodes that are not belonging to this MPI process
  for(i = 0; i < nnodesLoc; i++)
    pordering[i] = rstart + i;

  // AO map between various global numbering schemes 
  // It is PETSc parallel data structure for handling unstructured meshes context
  // Create PETSc ordering based upon two arrays:
  // l2a which contains indices of the local nodes
  // pordering which contains new ordering starts from number of nodes
  // that are not belong to this process up to the total number of local nodes
  // Creates a basic application ordering using two integer arrays
  ierr = AOCreateBasic(comm, nnodesLoc, l2a, pordering, &ao); CHKERRQ(ierr);
  free(pordering);
  /* Now count the local number of edges - including edges with ghost nodes
   * but edges between ghost nodes are NOT counted*/
  nedgeLoc = 0; // Local number of edges
  nvertices = nnodesLoc; // Local number of nodes
  /* Choose an estimated number of local edges.
   * The choice nedgeLocEst = 1000000 looks reasonable as it
   * will read the edge and edge normal arrays in 8 MB chunks
   * nedgeLocEst = nedge/size */
  nedgeLocEst = PetscMin(nedge, 1000000);
  remEdges = nedge; // Initialization
  ICALLOC(2 * nedgeLocEst, &tmp); // number of edges and edge normal arrays
  // Move the file offset pointer to start reading the edges from the mesh file
  // PETSC_BINARY_SEEK_CUR sets the file offset pointer to the current pointer
  // The current pointer equals to grid_param = 13 * sizeof(int) (= 52)
  ierr = PetscBinarySynchronizedSeek(comm, fdes, 0, PETSC_BINARY_SEEK_CUR,
                                     &currentPos); CHKERRQ(ierr);
  // Start timing this phase
  ierr = PetscTime(&time_ini); CHKERRQ(ierr);
  // Traverse over edges
  // remEdges == nedge
  // ********* The purpose of this while loop is to count the number of local
  // edges belongs to each MPI process as well as count the number of the ghost
  // nodes that are belong to each MPI process 
  while(remEdges > 0)
  {
    // Edges that have been read
    readEdges = PetscMin(remEdges, nedgeLocEst);
    // Read edges value provided in the mesh file starting right after the grid 
    // parameters and store them in a tmp vector
    ierr = PetscBinarySynchronizedRead(comm, fdes, tmp, readEdges, PETSC_INT);
    CHKERRQ(ierr);
    // Move the file offset pointer to continue reading the edges from the mesh file 
    ierr = PetscBinarySynchronizedSeek(comm, fdes, (nedge - readEdges) * 
                                       PETSC_BINARY_INT_SIZE, PETSC_BINARY_SEEK_CUR, 
                                       &newPos); CHKERRQ(ierr);
    // Read same number of edges twice from the mesh file
    // Read the remaining edges in the file 
    ierr = PetscBinarySynchronizedRead(comm, fdes, tmp + readEdges, readEdges, 
                                       PETSC_INT); CHKERRQ(ierr);
    // Set the file offset pointer
    ierr = PetscBinarySynchronizedSeek(comm, fdes, -nedge * PETSC_BINARY_INT_SIZE,
                                       PETSC_BINARY_SEEK_CUR, &newPos); 
    CHKERRQ(ierr);
    // Number of edges
    for(j = 0; j <  readEdges; j++)
    {
      // Traverse over edges, by picking two nodes that represent each edge 
      // Position of the node pointer in the mesh file depends according to
      // the structure of the mesh file
      // here, we assume that the first node is at the beginning of the file
      // and the second node is at middle of the file (non-interlaced)
      // [0, ....., readEdges -1, readEdges, ....., 2 * readEdges - 1]
      node1 = tmp[j] - 1; // In the first read stream 
      node2 = tmp[j + readEdges] - 1; // In the second read stream
      // Either node has to be belonged to this MPI process
      // If two of them belongs to it, then the edge belongs to this process
      // If one of them belongs to it, then the second one is a ghost node
      // So, we consider that edge in each process has one end of that node
      if((v2p[node1] == rank) || (v2p[node2] == rank))
      {
        nedgeLoc++; // Increment number of local edges
        // count local number of edges
        // These two conditions ensure that the edges between the
        // ghost nodes will be considered 
        // However, the edges between the ghost nodes within each process 
        // will be considered
        // node1 is a ghost node
        // node1 belongs to other MPI process
        if(a2l[node1] == -1)
        {
          // considered as a ghost node
          // Add this node to the current process structure as a ghost node
          // Add the index of this process
          // Add to the number of the total local nodes
          l2a[nvertices] = node1; // Add this node to the current process
          a2l[node1] = nvertices; // Store the index of the new added node
          nvertices++; // Increment local nodes
        }
        // node2 is a ghost node
        // node2 belongs to other MPI process
        if(a2l[node2] == -1)
        {
          // Same procedure with the first node
          l2a[nvertices] = node2; // add this node to the current process 
          a2l[node2] = nvertices; // Store the index of the new added node
          nvertices++; // Increment local nodes
        }
      }
    }
    // Guess: this main while loop will be executed once even if we have
    // more than one MPI process, so far I have not seen a case where 
    // this loop has to be executed more than once. 
    // JUST A PERSONAL OBSERVATION +++++ NEEDS TO BE PROVEN
    remEdges -= readEdges;
    ierr = MPI_Barrier(comm);
  }
  // End the task
  ierr = PetscTime(&time_fin); CHKERRQ(ierr);
  time_fin -= time_ini;
  ierr = PetscPrintf(comm, "Local edges counted with MPI_Bcast %d\n",
                     nedgeLoc); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Local vertices counted %d\n", nvertices);
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Time taken in this phase was %g\n", time_fin);
  CHKERRQ(ierr); 
  // Store the local edges
  ICALLOC(2 * nedgeLoc, &grid->eptr); // Edge pointers that reference to l2a
  ICALLOC(nedgeLoc, &edge_bit); // global index
  ICALLOC(nedgeLoc, &eperm); // permutation array
  i = 0; j = 0; k = 0;
  remEdges = nedge;
  // Move the offset pointer to the position of the file after the grid 
  // parameters, so that we can start reading edges from the file to 
  // store them 
  ierr = PetscBinarySynchronizedSeek(comm, fdes,  currentPos, 
                                     PETSC_BINARY_SEEK_SET, &newPos);
  CHKERRQ(ierr);
  currentPos = newPos; // Move the pointer
  ierr = PetscTime(&time_ini); CHKERRQ(ierr); // Start timing
  while(remEdges > 0)
  {
    readEdges = PetscMin(remEdges, nedgeLocEst);
    // Start reading the edges again from the file right after the
    // grid parameters
    ierr = PetscBinarySynchronizedRead(comm, fdes, tmp, readEdges,
                                       PETSC_INT); CHKERRQ(ierr);
    // Move the pointer forward to read the rest of the edges
    ierr = PetscBinarySynchronizedSeek(comm, fdes, (nedge - readEdges) * 
                                       PETSC_BINARY_INT_SIZE, 
                                       PETSC_BINARY_SEEK_CUR, &newPos);
    CHKERRQ(ierr);
    // Read the remaining edges
    ierr = PetscBinarySynchronizedRead(comm, fdes, tmp + readEdges, 
                                       readEdges, PETSC_INT);
    CHKERRQ(ierr);
    // Move the pointer offset
    ierr = PetscBinarySynchronizedSeek(comm, fdes, -nedge * 
                                       PETSC_BINARY_INT_SIZE,
                                       PETSC_BINARY_SEEK_CUR,
                                       &newPos); CHKERRQ(ierr);
    for(j = 0; j < readEdges; j++)
    {
      node1 = tmp[j] - 1; // first node
      node2 = tmp[j + readEdges] - 1; // second node
      // Any Node: in case of ghost nodes
      if((v2p[node1] == rank) || (v2p[node2] == rank))
      {
        // Edge pointers
        grid->eptr[k] = a2l[node1]; // Global node1 index
        // Store the index of the 1st node in l2a array
        grid->eptr[k + nedgeLoc] = a2l[node2]; // Global node2 index
        // Store the index of the 2nd node in l2a array
        // Record global file index of the edge
        edge_bit[k] = i; // global edge index
        eperm[k] = k;  // Permutation array for ordering, includes index of the edge array
        // the edges to a new sorted list
        // So, it's an array with size of the total number of 
        // the local edges that contains values from 0 to 
        // the number of local edges
        k++;
      }
      i++;
    }
    // Same observation
    remEdges -= readEdges;
    ierr = MPI_Barrier(comm);
  }
  // Move the file offset pointer to the end of edges values
  ierr = PetscBinarySynchronizedSeek(comm, fdes, currentPos + 2 * 
                                     nedge * PETSC_BINARY_INT_SIZE, 
                                     PETSC_BINARY_SEEK_SET, &newPos);
  CHKERRQ(ierr);
  ierr = PetscTime(&time_fin); CHKERRQ(ierr);
  time_fin -= time_ini;
  ierr = PetscPrintf(comm, "Local edges stored\n"); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Time taken in this phase was %g\n", 
                     time_fin); CHKERRQ(ierr);
  free(tmp);
  ICALLOC(2 * nedgeLoc, &tmp); // local edges
  // Copy the edges from the grid->eptr to the tmp variable
  // To manipulate the grid->eptr data structure so that we can
  // reorder the edges for better cache locality
  ierr = PetscMemcpy(tmp, grid->eptr, 2 * nedgeLoc * sizeof(int)); 
  CHKERRQ(ierr);

# if defined(_OPENMP) && defined(HAVE_EDGE_COLORING)
  ierr = EdgeColoring(nvertices, nedgeLoc, grid->eptr, eperm, 
                      &grid->ncolor, grid->ncount);
  CHKERRQ(ierr);
  PetscPrintf(comm, "%d\n", grid->ncolor);
//  for(i = 0; i < 200; i++)
  //  PetscPrintf(comm, "%d\n", grid->ncount[i]);
# else
  // Now reorder the edges for better cache locality

  /*
   *  tmp[0]=7;tmp[1]=6;tmp[2]=3;tmp[3]=9;tmp[4]=2;tmp[5]=0;
   *  ierr = PetscSortIntWithPermutation(6,tmp,eperm);
   *  for (i=0; i<6; i++)
   *    printf("%d %d %d\n",i,tmp[i],eperm[i]);
   */
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(0, "-no_edge_reordering", &flg, NULL); 
  CHKERRQ(ierr);
  if(!flg)
  {
    // Reorder the edges using PETSc routine
    // Computers the permutation of values that gives a sorted 
    // sequence 
    ierr = PetscSortIntWithPermutation(nedgeLoc, tmp, eperm);
    CHKERRQ(ierr);
  }
# endif
# ifdef PETSC_DEVELOPMENT_VERSION
  ierr = PetscMallocValidate(__LINE__, __FUNCT__, __FILE__, 0); CHKERRQ(ierr);
# else
  ierr = PetscMallocValidate(__LINE__, __FUNCT__, __FILE__); CHKERRQ(ierr);
# endif
  k = 0;
  // loop over edges
  for(i = 0; i < nedgeLoc; i++)
  {
    int cross_node = nnodesLoc / 2; 
    node1 = tmp[eperm[i]] + 1; // read the first node 
    node2 = tmp[nedgeLoc + eperm[i]] + 1; // 2nd node
    // Interlaced: v1, u1, w1, p1, v2, u2, w2, p2, ... etc
#   ifdef INTERLACING
    grid->eptr[k++] = node1;
    grid->eptr[k++] = node2;
    // Not Interlaced: v1, ..., u1, ..., w1, ..., p1, ... etc
#   else
    grid->eptr[i] = node1;
    grid->eptr[nedgeLoc + i] = node2;
#   endif
    // cross_edge means that the first node must lay on 
    // the first half of the nodes
    // and the second node must lay on the second 
    // half of the nodes
    /* if (node1 > node2)
     *  printf("On processor %d, for edge %d node1 = %d, node2 = %d\n",
     *          rank,i,node1,node2); CHKERRQ(ierr);
     */
    if ((node1 <= cross_node) && (node2 > cross_node)) 
      cross_edges++; 
      // Edges with endpoints in different components
  }
  ierr = PetscPrintf(comm, "Number of cross edges %d\n", 
                     cross_edges); CHKERRQ(ierr);
  free(tmp);
# if defined(_OPENMP)               && \
     !defined(HAVE_REDUNDANT_WORK)  && \
     !defined(HAVE_EDGE_COLORING)
  // With ghost nodes
  // Stuffs for ILU(0)
  // ia, ja, iau, fhelp
  // Now make the local 'ia' and 'ja' arrays   
  // IA is the row pointer array
  // represents the starting positions of each row of matrix
  // A
  ICALLOC(nvertices + 1, &grid->ia);
  // Use tmp for a work array
  ICALLOC(nvertices, &tmp);
  // user.h: #define f77GETIA f77name(GETIA,getia,getia_)
  // Get the IA, JA, and IAU arrays
  f77GETIA(&nvertices, &nedgeLoc, grid->eptr,
           grid->ia, tmp, &rank);
  // Number of non zeros
  // used to allocated JA array
  // column indices of non zero elements in the
  // sparse matrix
  nnz = grid->ia[nvertices] - 1;
  
# ifdef BLOCKING
  ierr = PetscPrintf(comm, "The Jacobian has %d non-zero blocks "
                           "with block size = %d\n", nnz, bs);
  CHKERRQ(ierr);
# else
  ierr = PetscPrintf(comm, "The Jacobian has %d non-zeros\n", nnz);
  CHKERRQ(ierr);
# endif
   
  ICALLOC(nnz, &grid->ja);
  // user.h: #define f77GETJA f77name(GETJA,getja,getja_)
  f77GETJA(&nvertices, &nedgeLoc, grid->eptr, grid->ia, 
           grid->ja, tmp, &rank);
  free(tmp);
  PetscPrintf(comm, "ILU with ghost nodes\n");
# else
  // Without ghost nodes
  // Now make the local 'ia' and 'ja' arrays
  ICALLOC(nnodesLoc + 1, &grid->ia);
  // Use tmp for a work array
  ICALLOC(nnodesLoc, &tmp);
  f77GETIA(&nnodesLoc, &nedgeLoc, grid->eptr, 
           grid->ia, tmp, &rank);
  nnz = grid->ia[nnodesLoc] - 1;

# ifdef BLOCKING
  ierr = PetscPrintf(comm, "The Jacobian has %d non-zero blocks "
                           "with block size = %d\n", nnz, bs);
  CHKERRQ(ierr);
# else
  ierr = PetscPrintf(comm, "The Jacobian has %d non-zeros\n", nnz);
  CHKERRQ(ierr);
# endif

  ICALLOC(nnz, &grid->ja);
  f77GETJA(&nnodesLoc, &nedgeLoc, grid->eptr,
           grid->ia, grid->ja, tmp, &rank);
  free(tmp);
  PetscPrintf(comm, "ILU without ghost nodes\n");
# endif
  // Allocate local to global mapping
  ICALLOC(nvertices, &grid->loc2glo);
  // local to global mapping is the same as l2a
  // which takes the global number each mesh node that represents 
  // the MPI process rank that belongs to
  // and assign them locally to that process
  ierr = PetscMemcpy(grid->loc2glo, l2a, nvertices*sizeof(int)); CHKERRQ(ierr);
  free(l2a);
  l2a = grid->loc2glo; // Points to the same object
  // local to PETSc mapping
  ICALLOC(nvertices, &grid->loc2pet);
  l2p = grid->loc2pet;
  // l2p has the same content as l2a
  ierr = PetscMemcpy(l2p, l2a, nvertices*sizeof(int));
  CHKERRQ(ierr);
  // Maps a set of integers in the application-defined ordering 
  // to the PETSc ordering
  // AO: the application ordering context and already has been
  // created before
  // l2p: the integer that will be replaced to the new mapping 
  ierr = AOApplicationToPetsc(ao, nvertices, l2p); CHKERRQ(ierr);
  
  // Renumber unit normals of dual face (from node1 to node2)
  // and the area of the dual mesh face
  FCALLOC(nedgeLocEst, &ftmp);
  FCALLOC(nedgeLoc, &ftmp1);
  //coordinates
  FCALLOC(4 * nedgeLoc, &grid->xyzn); // Normal to faces and lendth

  // ================================================================
  // ================================================================
  // Do the x-component
  i = 0; k = 0;
  remEdges = nedge;
  // Start timing this phase
  ierr = PetscTime(&time_ini); CHKERRQ(ierr);
  while(remEdges > 0)
  {
    // Start reading from the mesh file
    readEdges = PetscMin(remEdges, nedgeLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp, 
                                       readEdges, PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readEdges; j++)
    {
      // get edge index 
      if(edge_bit[k] == (i + j))
      {
        ftmp1[k] = ftmp[j];
        k++;
      }
    }
    i += readEdges;
    remEdges -= readEdges;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }
  for(i = 0; i < nedgeLoc; i++)
  {
#   ifdef INTERLACING
    grid->xyzn[4 * i] = ftmp1[eperm[i]]; 
#   else
    grid->xyzn[i] = ftmp1[eperm[i]];
#   endif
  }
  // ================================================================
  // Do the y-component
  i = 0; k = 0;
  remEdges = nedge;
  while(remEdges > 0)
  {
    // Start reading from the mesh file
    readEdges = PetscMin(remEdges, nedgeLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp, 
                                       readEdges, PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readEdges; j++)
    {
      // get edge index 
      if(edge_bit[k] == (i + j))
      {
        ftmp1[k] = ftmp[j];
        k++;
      }
    }
    i += readEdges;
    remEdges -= readEdges;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }
  for(i = 0; i < nedgeLoc; i++)
  {
#   ifdef INTERLACING
    grid->xyzn[4 * i + 1] = ftmp1[eperm[i]]; 
#   else
    grid->xyzn[nedgeLoc + i] = ftmp1[eperm[i]];
#   endif
  }
  // ================================================================
  // Do the z-component
  i = 0; k = 0;
  remEdges = nedge;
  while(remEdges > 0)
  {
    // Start reading from the mesh file
    readEdges = PetscMin(remEdges, nedgeLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp, 
                                       readEdges, PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readEdges; j++)
    {
      // get edge index 
      if(edge_bit[k] == (i + j))
      {
        ftmp1[k] = ftmp[j];
        k++;
      }
    }
    i += readEdges;
    remEdges -= readEdges;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }
  for(i = 0; i < nedgeLoc; i++)
  {
#   ifdef INTERLACING
    grid->xyzn[4 * i + 2] = ftmp1[eperm[i]]; 
#   else
    grid->xyzn[2 * nedgeLoc + i] = ftmp1[eperm[i]];
#   endif
  }
  // ================================================================
  // Do the length
  i = 0; k = 0;
  remEdges = nedge;
  while(remEdges > 0)
  {
    // Start reading from the mesh file
    readEdges = PetscMin(remEdges, nedgeLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp, 
                                       readEdges, PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readEdges; j++)
    {
      // get edge index 
      if(edge_bit[k] == (i + j))
      {
        ftmp1[k] = ftmp[j];
        k++;
      }
    }
    i += readEdges;
    remEdges -= readEdges;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }
  for(i = 0; i < nedgeLoc; i++)
  {
#   ifdef INTERLACING
    grid->xyzn[4 * i + 3] = ftmp1[eperm[i]]; 
#   else
    grid->xyzn[3 * nedgeLoc + i] = ftmp1[eperm[i]];
#   endif
  } 
  // ================================================================
  // ================================================================
  
  free(edge_bit);
  free(eperm);
  free(ftmp);
  free(ftmp1);

  ierr = PetscTime(&time_fin); CHKERRQ(ierr);
  time_fin -= time_ini;
  
  ierr = PetscPrintf(comm, "Edge normals partitioned\n");
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Time taken in this phase was %g\n", 
                     time_fin); CHKERRQ(ierr);

  // Use MeTiS to partition each subdomain of each process among
  // its threads
# if defined(_OPENMP)
  // Arrange for the division of work among threads
# if defined(HAVE_EDGE_COLORING)
# elif defined(HAVE_REDUNDANT_WORK)
  FCALLOC(4 * nnodesLoc, &grid->resd);
# else
  {
    PetscPrintf(comm, "MeTiS sub-domain partitioning\n");
    // Get the local adjacency structure of the graph
    // for partitioning the local graph into
    // max_threads pieces
    
    // Row pointers, and column index of non-zero
    // elements 
    int *ia, *ja; // stuffs for ILU preconditioner
    
    // ******** MiTiS v4.x variables

    // MeTiS v4.x : Array of 5 integers that is used
    // to pass parameters for the various phases of the 
    // algorithm .... If options[0] == 0 then default
    // values are used.
    // If options[0] == 1 then you have to specify the 
    // remaining four elements in the options array.
    int options[5]; 
    // Information about the weight of the vertices
    // and edges
    int *vwtg = 0, *adjwgt = 0; // 0 means no weights
    int numflag = 0; // Numbering scheme used for the
    // adjacency structure of the graph
    // (0 or 1) 0 means C-style numbering that is assumed
    // it starts from 1
    // Used to describe weather the graph is weighted or 
    // not (0, 1, 2, or 3)
    int wgtflag = 0; // No weights
    // Upon successful completion, this variable stores 
    // the number of edges that are cut by the partition
    int edgecut;
    
    int thr1, thr2, nedgeAllThreads, ned1, ned2;

    ICALLOC((nvertices + 1), &ia);
    ICALLOC((2 * nedgeLoc), &ja);  
    ia[0] = 0;
    // Get row pointer of the matrix A
    for(i = 1; i <= nvertices; i++)
      ia[i] = grid->ia[i] - i - 1;
    for(i = 0; i < nvertices; i++)
    {
      // Loop over the rows of the matrix A
      // of non-zero's
      int jstart = grid->ia[i] - 1;   // Start Row
      int jend = grid->ia[i + 1] - 1; // End Row 
      k = ia[i];
      // Loop over columns of the non-zero's in the matrix A
      for(j = jstart; j < jend; j++)
      {
        // Get the node index
        inode = grid->ja[j] - 1;
        if(inode != i)
          ja[k++] = inode;
      }
    }
    // grid->part_thr vector of size n that upon successful 
    // completion stores the partition vector of the graph. 
    // The numbering of this vector starts from either 0 | 1, 
    // depending on the value of numflag
    ICALLOC(nvertices, &grid->part_thr);
    ierr = PetscMemzero(grid->part_thr, 
                        nvertices * sizeof(int));
    CHKERRQ(ierr);
    options[0] = 0; // default values are used. 
    // call the pMeTiS library routine
    // Used max_threads as the number of parts to 
    // partition the graph
    if(max_threads > 1)
    {
      METIS_PartGraphRecursive(&nvertices, ia, ja, vwtg,
                               adjwgt, &wgtflag, &numflag,
                               &max_threads, options, 
                               &edgecut, grid->part_thr);
    }
 
    FILE *f = fopen("part32.in", "w");
    for(i = 0; i < nvertices; i++)
      fprintf(f, "%d\n", grid->part_thr[i]);

    
    ierr = PetscPrintf(MPI_COMM_WORLD, 
                       "The number of cut edges is %d\n", 
                       edgecut); CHKERRQ(ierr);
    // Write the partition vector to disk
    ierr = PetscOptionsHasName(0, "-omp_partitioning", &flg); 
    CHKERRQ(ierr);
    // Start writing
    if(flg)
    {
      int *loc2glo_glo;
      char part_file[P_LEN];
      FILE *fp;

      int *partv_loc, *partv_glo;
      ICALLOC(nnodesLoc, &partv_loc);
      ICALLOC(nnodes, &partv_glo);
      for(i = 0; i < nnodesLoc; i++)
        partv_loc[i] = grid->part_thr[i] + max_threads * rank;
      // Integer array (of length group size) containing the 
      // number of elements that are received from each process 
      // (significant only at root)
      int *counts; // stuff for MPI
      // Integer array (of length group size). Entry i specifies 
      // the displacement relative to recvbuf at which to place 
      // the incoming data from process i (significant only at root)
      int *disp; // stuff for MPI
      ICALLOC(size, &disp);
      ICALLOC(size, &counts);
      MPI_Allgather(&nnodesLoc, 1, MPI_INT, counts, 1, MPI_INT,
                    MPI_COMM_WORLD);
      disp[0] = 0;
      for(i = 1; i < size; i++)
        disp[i] = counts[i - 1] + disp[i - 1]; 
      ICALLOC(nnodes, &loc2glo_glo);
      MPI_Gatherv(grid->loc2glo, nnodesLoc, MPI_INT, loc2glo_glo,
                  counts, disp, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Gatherv(partv_loc, nnodesLoc, MPI_INT, partv_glo,
                  counts, disp, MPI_INT, 0, MPI_COMM_WORLD);
      if(rank == 0)
      {
        // Sorts an array of integers in place in increasing order; 
        // changes a second array to match the sorted first array
        ierr = PetscSortIntWithArray(nnodes, loc2glo_glo, partv_glo); 
        CHKERRQ(ierr);
        sprintf(part_file, "hyb_part_vec.%d", 2 * size);
        fp = fopen(part_file, "w");
        for(i = 0; i < nnodes; i++)
          fprintf(fp, "%d\n", partv_glo[i]);
        fclose(fp);
      }
      free(partv_loc);
      free(partv_glo);
      free(disp);
      free(counts);
      free(loc2glo_glo);
    }
    // Divide the work among threads
    k = 0;
    ICALLOC((max_threads + 1), &grid->nedge_thr);
    
    ierr = PetscMemzero(grid->nedge_thr, (max_threads + 1) * 
                        sizeof(int)); CHKERRQ(ierr);
    cross_edges = 0;
    for(i = 0; i < nedgeLoc; i++)
    {
      // Interlaced only -- NEED TO be CONFIRMED
      node1 = grid->eptr[k++] - 1;
      node2 = grid->eptr[k++] - 1;
      // get two nodes in the edges
      // and make sure whether those nodes are different
      // or not
      thr1 = grid->part_thr[node1];
      thr2 = grid->part_thr[node2];
      // assign the work to a thread
      grid->nedge_thr[thr1] += 1;
      // Two threads working on different components
      // cross edges concept
      if(thr1 != thr2)
      {
        // assign the work to another thread
        grid->nedge_thr[thr2] += 1;
        cross_edges++; // Edges with endpoints with diff comp.
      }

    }
    ierr = PetscPrintf(MPI_COMM_WORLD, "The number of cross "
                                "edges after MeTiS partitioning "
                                "is %d\n", cross_edges);
    CHKERRQ(ierr);
    ned1 = grid->nedge_thr[0]; // store first thread work
    grid->nedge_thr[0] = 1; // assign it to thread next thread
    // Start work assignment to the available threads
    for(i = 1; i <= max_threads; i++)
    {
      // Get the next thread work
      ned2 = grid->nedge_thr[i];
      // Assign the current one based on the previous one
      grid->nedge_thr[i] = grid->nedge_thr[i - 1] + ned1;
      // Increment to move to the next level
      ned1 = ned2;
    }

    // Allocate a shared edge array. Note that a cut edge 
    // is evaluated by both the threads but updates are 
    // done only for the locally owned node

    // Max thread work
    grid->nedgeAllThr = nedgeAllThreads = grid->nedge_thr[max_threads] - 1;
    ICALLOC(2 * nedgeAllThreads, &grid->edge_thr);
    ICALLOC(max_threads, &tmp);
    FCALLOC(4 * nedgeAllThreads, &grid->xyzn_thr);

    for(i = 0; i < max_threads; i++)
      tmp[i] = grid->nedge_thr[i] - 1;
    k = 0;
    for(i = 0; i < nedgeLoc; i++)
    {
      int ie1, ie2, ie3;
      // INTERLACED Data ..... NEEDS TO BE CONFIRMED
      node1 = grid->eptr[k++];
      node2 = grid->eptr[k++];
        
      thr1 = grid->part_thr[node1 - 1];
      thr2 = grid->part_thr[node2 - 1];

      ie1 = 2 * tmp[thr1];
      ie2 = 4 * tmp[thr1];
      ie3 = 4 * i;
      
      grid->edge_thr[ie1]     = node1;
      grid->edge_thr[ie1 + 1] = node2;

      grid->xyzn_thr[ie2]     = grid->xyzn[ie3];
      grid->xyzn_thr[ie2 + 1] = grid->xyzn[ie3 + 1];
      grid->xyzn_thr[ie2 + 2] = grid->xyzn[ie3 + 2];
      grid->xyzn_thr[ie2 + 3] = grid->xyzn[ie3 + 3];
      
      tmp[thr1] += 1;
      
      if(thr1 != thr2)
      {
        ie1 = 2 * tmp[thr2];
        ie2 = 4 * tmp[thr2];

        grid->edge_thr[ie1] = node1;
        grid->edge_thr[ie1 + 1] = node2;

        grid->xyzn_thr[ie2]     = grid->xyzn[ie3];
        grid->xyzn_thr[ie2 + 1] = grid->xyzn[ie3 + 1];
        grid->xyzn_thr[ie2 + 2] = grid->xyzn[ie3 + 2];
        grid->xyzn_thr[ie2 + 3] = grid->xyzn[ie3 + 3];
        
        tmp[thr2] += 1;
      }
      
    }  
  }
# endif
# endif

  // Remap coordinates
  nnodesLocEst = PetscMin(nnodes, 500000);
  FCALLOC(nnodesLocEst, &ftmp);
  FCALLOC(3 * nvertices, &grid->xyz);
  // =======================================================
  // =======================================================
  // x-component
  remNodes = nnodes;
  i = 0;
  ierr = PetscTime(&time_ini); CHKERRQ(ierr);
  while(remNodes > 0)
  {
    readNodes = PetscMin(remNodes, nnodesLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp,
                                       readNodes, 
                                       PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readNodes; j++)
    {
      if(a2l[i + j] >= 0)
      {
#       ifdef INTERLACING
        grid->xyz[3 * a2l[i + j]] = ftmp[j];
#       else
        grid->xyz[a2l[i + j]] = ftmp[j];
#       endif
      }
    }
    i += nnodesLocEst;
    remNodes -= nnodesLocEst;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }
  // =======================================================
  // y-component
  remNodes = nnodes;
  i = 0;
  while(remNodes > 0)
  {
    readNodes = PetscMin(remNodes, nnodesLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp,
                                       readNodes, 
                                       PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readNodes; j++)
    {
      if(a2l[i + j] >= 0)
      {
#       ifdef INTERLACING
        grid->xyz[3 * a2l[i + j] + 1] = ftmp[j];
#       else
        grid->xyz[nnodesLoc + a2l[i + j]] = ftmp[j];
#       endif
      }
    }
    i += nnodesLocEst;
    remNodes -= nnodesLocEst;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }
  // =======================================================
  // z-component
  remNodes = nnodes;
  i = 0;
  while(remNodes > 0)
  {
    readNodes = PetscMin(remNodes, nnodesLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp,
                                       readNodes, 
                                       PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readNodes; j++)
    {
      if(a2l[i + j] >= 0)
      {
#       ifdef INTERLACING
        grid->xyz[3 * a2l[i + j] + 2] = ftmp[j];
#       else
        grid->xyz[2 * nnodesLoc + a2l[i + j]] = ftmp[j];
#       endif
      }
    }
    i += nnodesLocEst;
    remNodes -= nnodesLocEst;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }
  // =======================================================
  // =======================================================

  // Renumber dual volume
  FCALLOC(nvertices, &grid->area);
  remNodes = nnodes;
  i = 0;
  while(remNodes > 0)
  {
    readNodes = PetscMin(remNodes, nnodesLocEst);
    ierr = PetscBinarySynchronizedRead(comm, fdes, ftmp,
                                       readNodes, 
                                       PETSC_SCALAR);
    CHKERRQ(ierr);
    for(j = 0; j < readNodes; j++)
    {
      if (a2l[i + j] >= 0)
        grid->area[a2l[i + j]] = ftmp[j];
    }
    i += nnodesLocEst;
    remNodes -= nnodesLocEst;
    ierr = MPI_Barrier(comm); CHKERRQ(ierr);
  }

  free(ftmp);
  ierr = PetscTime(&time_fin); CHKERRQ(ierr);
  time_fin -= time_ini;
  
  ierr = PetscPrintf(comm, "Coordinates remapped\n"); 
  CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Time take in this phase was %g\n",
                     time_fin); CHKERRQ(ierr);
  /* 
   * Now, handle all the solid boundaries - things to be done :
   * 1. Identify the nodes belonging to the solid  
   *    boundaries and count them.
   * 2. Put proper indices into f2ntn array,after making it
   *    of suitable size.
   * 3. Remap the normals and areas of solid faces (sxn,syn,szn,
   *    and sa arrays). 
   */
  
  ICALLOC(nnbound, &grid->nntet);
  ICALLOC(nnbound, &grid->nnpts);
  ICALLOC(4 * nnfacet, &grid->f2ntn);
  // Node number of solid nodes
  ICALLOC(nsnode, &grid->isnode);
  // Normals at solid nodes
  // *************************
  FCALLOC(nsnode, &grid->sxn);
  FCALLOC(nsnode, &grid->syn);
  FCALLOC(nsnode, &grid->szn);
  FCALLOC(nsnode, &grid->sa);
  // *************************
   
  // Read from the mesh file
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->nntet,
                                     nnbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->nnpts,
                                     nnbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->f2ntn,
                                     4 * nnfacet, PETSC_INT);
  CHKERRQ(ierr);
  // Node number of solid nodes from the mesh file 
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->isnode,
                                     nsnode, PETSC_INT);
  CHKERRQ(ierr);
  // Normal at solid nodes
  // *******************************************************
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->sxn,
                                     nsnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->syn,
                                     nsnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->szn,
                                     nsnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  // *******************************************************
  
//  for(i = 0; i < 10; i++)
//    PetscPrintf(comm, "%d\n", grid->nntet[i]);
  
  isurf = 0;
  nsnodeLoc = 0;
  nnfacetLoc = 0;
  nb = 0;
  nte = 0;
  // Working arrays
  ICALLOC(3 * nnfacet, &tmp);
  ICALLOC(nsnode, &tmp1);
  ICALLOC(nnodes, &tmp2);
  FCALLOC(4 * nsnode, &ftmp); 
  // Initialization to 0s
  ierr = PetscMemzero(tmp, 3 * nnfacet * sizeof(int));
  CHKERRQ(ierr);
  ierr = PetscMemzero(tmp1, nsnode * sizeof(int));
  CHKERRQ(ierr);
  ierr = PetscMemzero(tmp2, nnodes * sizeof(int));
  CHKERRQ(ierr);
  // Start identifying viscous boundaries  
  // Nodes
  j = 0;
  for (i = 0; i < nsnode; i++) 
  {
    // Get node number of slid node
    node1 = a2l[grid->isnode[i] - 1];
    if (node1 >= 0) 
    {
      // Store solid node value
      tmp1[nsnodeLoc] = node1;
      // Store solid node index
      tmp2[node1] = nsnodeLoc;
      // Store the node coordinates
      // Normals
      // x-coordinate 
      ftmp[j++] = grid->sxn[i];
      // y-coordinate
      ftmp[j++] = grid->syn[i];
      // z-coordinate
      ftmp[j++] = grid->szn[i];
      // Area
      // Solid area
      ftmp[j++] = grid->sa[i];
      // increment the number of local solid nodes
      nsnodeLoc++;
    }
  }
  // Facets
  for (i = 0; i < nnbound; i++) 
  {
    for (j = isurf; j < isurf + grid->nntet[i]; j++) 
    {
      // Facet nodes
      node1 = a2l[grid->isnode[
                  grid->f2ntn[j] - 1] - 1];
      node2 = a2l[grid->isnode[
                  grid->f2ntn[nnfacet + j] - 1] - 1];
      node3 = a2l[grid->isnode[
                  grid->f2ntn[2 * nnfacet + j] - 1] - 1]; 
      // Solid facet belongs to the current process
      if ((node1 >= 0) && 
          (node2 >= 0) && 
          (node3 >= 0))
      {
        // Increment the number of local faces
        nnfacetLoc++;
        nte++;
        // Store those nodes
        tmp[nb++] = tmp2[node1];
        tmp[nb++] = tmp2[node2];
        tmp[nb++] = tmp2[node3];
      }
    }
    isurf += grid->nntet[i];
    grid->nntet[i] = nte;
    nte = 0;
  }
  // Free the storage arrays to start copy the
  // data from the dummy arrays back to them  
  free(grid->f2ntn);
  free(grid->isnode);
  free(grid->sxn);
  free(grid->syn);
  free(grid->szn);
  free(grid->sa);
  // Allocate them again from permanent storage
  ICALLOC(4 * nnfacetLoc, &grid->f2ntn);
  ICALLOC(nsnodeLoc, &grid->isnode);
  FCALLOC(nsnodeLoc, &grid->sxn);
  FCALLOC(nsnodeLoc, &grid->syn);
  FCALLOC(nsnodeLoc, &grid->szn);
  FCALLOC(nsnodeLoc, &grid->sa);
  // Store solid nodes along with their
  // normals (coordinates)
  j = 0;
  for (i = 0; i < nsnodeLoc; i++) 
  {
    grid->isnode[i] = tmp1[i] + 1;
    grid->sxn[i]    = ftmp[j++];
    grid->syn[i]    = ftmp[j++];
    grid->szn[i]    = ftmp[j++];
    grid->sa[i]     = ftmp[j++];
  }
  // Store solid faces
  j = 0;
  for (i = 0; i < nnfacetLoc; i++) 
  {
    grid->f2ntn[i]                  = tmp[j++] + 1; 
    grid->f2ntn[nnfacetLoc + i]     = tmp[j++] + 1; 
    grid->f2ntn[2 * nnfacetLoc + i] = tmp[j++] + 1; 
  }
  // Free working arrays now for the next phases
  free(tmp);
  free(tmp1);
  free(tmp2);
  free(ftmp);
  
  // Now identify the triangles on which the current 
  // processor would perform force calculation
  // grid->sface_bit: solid face calculation
  ICALLOC(nnfacetLoc, &grid->sface_bit);
  // Initialize it to 0s
  PetscMemzero(grid->sface_bit, nnfacetLoc * sizeof(int));
  for (i = 0; i < nnfacetLoc; i++) 
  {
    // Get the triangle face
    // tetrahedral
    //          *
    //        * * *
    //       *  *  *
    //      *   *   *
    //     *    *    *
    //    *  *     *  *
    //   * * * * * * * *
    node1 = l2a[grid->isnode[
                grid->f2ntn[i] - 1] - 1];
    node2 = l2a[grid->isnode[
                grid->f2ntn[nnfacetLoc + i] - 1] - 1];
    node3 = l2a[grid->isnode[
                grid->f2ntn[2 * nnfacetLoc + i] - 1] - 1];
    if (((v2p[node1]  >= rank) && (v2p[node2]  >= rank) 
        && (v2p[node3]  >= rank)) && ((v2p[node1] == rank) 
        || (v2p[node2]  == rank) || (v2p[node3]  == rank))) 
    {
      grid->sface_bit[i] = 1;
    }
  }

  ierr = PetscPrintf(comm, "Solid boundaries partitioned\n");
  CHKERRQ(ierr);

  /* Now,handle all the viscous boundaries - things to be done :
   * 1. Identify the nodes belonging to the viscous
   *    boundaries and count them.
   * 2. Put proper indices into f2ntv array,after making it
   *    of suitable size
   * 3. Remap the normals and areas of viscous faces (vxn,vyn,vzn,
   *    and va arrays). 
  */

  ICALLOC(nvbound, &grid->nvtet);
  ICALLOC(nvbound, &grid->nvpts);
  ICALLOC(4 * nvfacet, &grid->f2ntv);
  // Node number of viscous nodes
  ICALLOC(nvnode, &grid->ivnode);
  // Normals at viscous nodes
  // *************************
  FCALLOC(nvnode, &grid->vxn);
  FCALLOC(nvnode, &grid->vyn);
  FCALLOC(nvnode, &grid->vzn);
  FCALLOC(nvnode, &grid->va);
  // *************************
  
  // Read from the mesh file
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->nvtet,
                                     nvbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->nvpts,
                                     nvbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->f2ntv,
                                     4 * nvfacet, PETSC_INT);
  CHKERRQ(ierr);
  // Node number of viscous nodes from the mesh file 
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->ivnode,
                                     nvnode, PETSC_INT);
  CHKERRQ(ierr);
  // Normal at viscous nodes
  // *******************************************************
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->vxn,
                                     nvnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->vyn,
                                     nvnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->vzn,
                                     nvnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  // *******************************************************

  isurf = 0;
  nvnodeLoc = 0;
  nvfacetLoc = 0;
  nb = 0;
  nte = 0;
  // Working arrays
  ICALLOC(3 * nvfacet, &tmp);
  ICALLOC(nvnode, &tmp1);
  ICALLOC(nvnode, &tmp2);
  FCALLOC(4 * nvnode, &ftmp); 
  // Initialization to 0s
  ierr = PetscMemzero(tmp, 3 * nvfacet * sizeof(int));
  CHKERRQ(ierr);
  ierr = PetscMemzero(tmp1, nvnode * sizeof(int));
  CHKERRQ(ierr);
  ierr = PetscMemzero(tmp2, nvnode * sizeof(int));
  CHKERRQ(ierr);
  // Start identifying the viscous boundaries 
  // Nodes
  j = 0;
  for (i = 0; i < nvnode; i++) 
  {
    // Get node number of viscous node
    node1 = a2l[grid->ivnode[i] - 1];
    if (node1 >= 0) 
    {
      // Store viscous node value
      tmp1[nvnodeLoc] = node1;
      // Store viscous node index
      tmp2[node1] = nvnodeLoc;
      // Store the node coordinates
      // Normals
      // x-coordinate 
      ftmp[j++] = grid->vxn[i];
      // y-coordinate
      ftmp[j++] = grid->vyn[i];
      // z-coordinate
      ftmp[j++] = grid->vzn[i];
      // Area
      // Viscous area
      ftmp[j++] = grid->va[i];
      // increment the number of local viscous nodes
      nvnodeLoc++;
    }
  }
  // Facets
  for (i = 0; i < nvbound; i++) 
  {
    for (j = isurf; j < isurf + grid->nvtet[i]; j++) 
    {
      // Face nodes
      node1 = a2l[grid->ivnode[
                  grid->f2ntv[j] - 1] - 1];
      node2 = a2l[grid->ivnode[
                  grid->f2ntv[nvfacet + j] - 1] - 1];
      node3 = a2l[grid->ivnode[
                  grid->f2ntv[2 * nvfacet + j] - 1] - 1]; 
      
      if ((node1 >= 0) && 
          (node2 >= 0) && 
          (node3 >= 0))
      {
        // Increment the number of local faces
        nvfacetLoc++;
        nte++;
        // Store those nodes
        tmp[nb++] = tmp2[node1];
        tmp[nb++] = tmp2[node2];
        tmp[nb++] = tmp2[node3];
      }
    }
    isurf += grid->nvtet[i];
    grid->nvtet[i] = nte;
    nte = 0;
  }
  // Free the storage arrays to start copy the
  // data from the dummy arrays back to them  
  free(grid->f2ntv);
  free(grid->ivnode);
  free(grid->vxn);
  free(grid->vyn);
  free(grid->vzn);
  free(grid->va);
  // Allocate them again from permanent storage
  ICALLOC(4 * nvfacetLoc, &grid->f2ntv);
  ICALLOC(nvnodeLoc, &grid->ivnode);
  FCALLOC(nvnodeLoc, &grid->vxn);
  FCALLOC(nvnodeLoc, &grid->vyn);
  FCALLOC(nvnodeLoc, &grid->vzn);
  FCALLOC(nvnodeLoc, &grid->va);
  // Store viscous nodes along with their
  // normals (coordinates)
  j = 0;
  for (i = 0; i < nvnodeLoc; i++) 
  {
    grid->ivnode[i] = tmp1[i] + 1;
    grid->vxn[i]    = ftmp[j++];
    grid->vyn[i]    = ftmp[j++];
    grid->vzn[i]    = ftmp[j++];
    grid->va[i]     = ftmp[j++];
  }
  // Store viscous faces
  j = 0;
  for (i = 0; i < nvfacetLoc; i++) 
  {
    grid->f2ntv[i]                  = tmp[j++] + 1; 
    grid->f2ntv[nvfacetLoc + i]     = tmp[j++] + 1; 
    grid->f2ntv[2 * nvfacetLoc + i] = tmp[j++] + 1; 
  }
  // Free working arrays now for the next phases
  free(tmp);
  free(tmp1);
  free(tmp2);
  free(ftmp);
  
  // Now identify the triangles on which the current 
  // processor would perform force calculation
  // grid->vface_bit: viscous face calculation
  ICALLOC(nvfacetLoc, &grid->vface_bit);
  // Initialize it to 0s
  PetscMemzero(grid->vface_bit, nvfacetLoc * sizeof(int));
  for (i = 0; i < nvfacetLoc; i++) 
  {
    // Get the triangle face
    // tetrahedral
    //          *
    //        * * *
    //       *  *  *
    //      *   *   *
    //     *    *    *
    //    *  *     *  *
    //   * * * * * * * *
    node1 = l2a[grid->ivnode[
                grid->f2ntv[i] - 1] - 1];
    node2 = l2a[grid->ivnode[
                grid->f2ntv[nvfacetLoc + i] - 1] - 1];
    node3 = l2a[grid->ivnode[
                grid->f2ntv[2 * nvfacetLoc + i] - 1] - 1];
    
    if (((v2p[node1]  >= rank) && (v2p[node2]  >= rank) 
        && (v2p[node3]  >= rank)) && ((v2p[node1] == rank) 
        || (v2p[node2]  == rank) || (v2p[node3]  == rank))) 
    {
      grid->vface_bit[i] = 1;
    }
  }
  free(v2p);
  ierr = PetscPrintf(comm, "Viscous boundaries partitioned\n");
  CHKERRQ(ierr);


  /* Now,handle all the free boundaries - things to be done :
   * 1. Identify the nodes belonging to the free
   *    boundaries and count them.
   * 2. Put proper indices into f2ntf array,after making it
   *    of suitable size
   * 3. Remap the normals and areas of free bound. faces (fxn,fyn,fzn,
   *    and fa arrays). 
  */
  
  ICALLOC(nfbound, &grid->nftet);
  ICALLOC(nfbound, &grid->nfpts);
  ICALLOC(4 * nffacet, &grid->f2ntf);
  // Node number of free nodes
  ICALLOC(nfnode, &grid->ifnode);
  // Normals at free nodes
  // *************************
  FCALLOC(nfnode, &grid->fxn);
  FCALLOC(nfnode, &grid->fyn);
  FCALLOC(nfnode, &grid->fzn);
  FCALLOC(nfnode, &grid->fa);
  // *************************
  
  // Read from the mesh file
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->nftet,
                                     nfbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->nfpts,
                                     nfbound, PETSC_INT);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->f2ntf,
                                     4 * nffacet, PETSC_INT);
  CHKERRQ(ierr);
  // Node number of free nodes from the mesh file 
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->ifnode,
                                     nfnode, PETSC_INT);
  CHKERRQ(ierr);
  // Normal at free nodes
  // *******************************************************
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->fxn,
                                     nfnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->fyn,
                                     nfnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  ierr = PetscBinarySynchronizedRead(comm, fdes, grid->fzn,
                                     nfnode, PETSC_SCALAR);
  CHKERRQ(ierr);
  // *******************************************************

  isurf = 0;
  nfnodeLoc = 0;
  nffacetLoc = 0;
  nb = 0;
  nte = 0;
  // Working arrays
  ICALLOC(3 * nffacet, &tmp);
  ICALLOC(nfnode, &tmp1);
  ICALLOC(nnodes, &tmp2);
  FCALLOC(4 * nfnode, &ftmp); 
  // Initialization to 0s
  ierr = PetscMemzero(tmp, 3 * nffacet * sizeof(int));
  CHKERRQ(ierr);
  ierr = PetscMemzero(tmp1, nfnode * sizeof(int));
  CHKERRQ(ierr);
  ierr = PetscMemzero(tmp2, nnodes * sizeof(int));
  CHKERRQ(ierr);
  // Start identifying the free boundaries 
  // Nodes
  j = 0;
  for (i = 0; i < nfnode; i++) 
  {
    // Get node number of free node
    node1 = a2l[grid->ifnode[i] - 1];
    if (node1 >= 0) 
    {
      // Store free node value
      tmp1[nfnodeLoc] = node1;
      // Store free node index
      tmp2[node1] = nfnodeLoc;
      // Store the node coordinates
      // Normals
      // x-coordinate 
      ftmp[j++] = grid->fxn[i];
      // y-coordinate
      ftmp[j++] = grid->fyn[i];
      // z-coordinate
      ftmp[j++] = grid->fzn[i];
      // Area
      // Viscous area
      ftmp[j++] = grid->fa[i];
      // increment the number of local viscous nodes
      nfnodeLoc++;
    }
  }
  // Facets
  for (i = 0; i < nfbound; i++) 
  {
    for (j = isurf; j < isurf + grid->nftet[i]; j++) 
    {
      // Face nodes
      node1 = a2l[grid->ifnode[
                  grid->f2ntf[j] - 1] - 1];
      node2 = a2l[grid->ifnode[
                  grid->f2ntf[nffacet + j] - 1] - 1];
      node3 = a2l[grid->ifnode[
                  grid->f2ntf[2 * nffacet + j] - 1] - 1]; 
      
      if ((node1 >= 0) && 
          (node2 >= 0) && 
          (node3 >= 0))
      {
        // Increment the number of local faces
        nffacetLoc++;
        nte++;
        // Store those nodes
        tmp[nb++] = tmp2[node1];
        tmp[nb++] = tmp2[node2];
        tmp[nb++] = tmp2[node3];
      }
    }
    isurf += grid->nftet[i];
    grid->nftet[i] = nte;
    nte = 0;
  }
  // Free the storage arrays to start copy the
  // data from the dummy arrays back to them  
  free(grid->f2ntf);
  free(grid->ifnode);
  free(grid->fxn);
  free(grid->fyn);
  free(grid->fzn);
  free(grid->fa);
  // Allocate them again from permanent storage
  ICALLOC(4 * nffacetLoc, &grid->f2ntf);
  ICALLOC(nfnodeLoc, &grid->ifnode);
  FCALLOC(nfnodeLoc, &grid->fxn);
  FCALLOC(nfnodeLoc, &grid->fyn);
  FCALLOC(nfnodeLoc, &grid->fzn);
  FCALLOC(nfnodeLoc, &grid->fa);
  // Store free nodes along with their
  // normals (coordinates)
  j = 0;
  for (i = 0; i < nfnodeLoc; i++) 
  {
    grid->ifnode[i] = tmp1[i] + 1;
    grid->fxn[i]    = ftmp[j++];
    grid->fyn[i]    = ftmp[j++];
    grid->fzn[i]    = ftmp[j++];
    grid->fa[i]     = ftmp[j++];
  }
  // Store free faces
  j = 0;
  for (i = 0; i < nffacetLoc; i++) 
  {
    grid->f2ntf[i]                  = tmp[j++] + 1; 
    grid->f2ntf[nffacetLoc + i]     = tmp[j++] + 1; 
    grid->f2ntf[2 * nffacetLoc + i] = tmp[j++] + 1; 
  }
  // Free working arrays now for the next phases
  free(tmp);
  free(tmp1);
  free(tmp2);
  free(ftmp);
  
  ierr = PetscPrintf(comm, "Free boundaries partitioned\n");
  CHKERRQ(ierr);
  
 
  // Put different mappings and other info into grid
  grid->nnodesLoc   = nnodesLoc;
  grid->nedgeLoc    = nedgeLoc;
  grid->nvertices   = nvertices;
  grid->nsnodeLoc   = nsnodeLoc;
  grid->nvnodeLoc   = nvnodeLoc;
  grid->nfnodeLoc   = nfnodeLoc;
  grid->nnfacetLoc  = nnfacetLoc;
  grid->nvfacetLoc  = nvfacetLoc;
  grid->nffacetLoc  = nffacetLoc;
  
  FCALLOC(nvertices, &grid->cdt);     // time step
  FCALLOC(nvertices * 4, &grid->phi);

  FCALLOC(7 * nnodesLoc, &grid->rxy);

  // Map the 'ja' array in petsc ordering
  for (i = 0; i < nnz; i++)
  {
    grid->ja[i] = l2a[grid->ja[i] - 1];
  }
  ierr = AOApplicationToPetsc(ao, nnz, grid->ja);
  CHKERRQ(ierr);
  ierr = AODestroy(&ao); CHKERRQ(ierr);

  // Put different mappings and other info into grid
  
  {
    int partLoc[7];
    int partMax[7], partMin[7], partSum[7];
    
    partLoc[0] = nnodesLoc;
    partLoc[1] = nvertices;
    partLoc[2] = nedgeLoc;
    partLoc[3] = nnfacetLoc;
    partLoc[4] = nffacetLoc;
    partLoc[5] = nsnodeLoc;
    partLoc[6] = nfnodeLoc;
    
    for (i = 0; i < 7; i++) 
    {
      partMin[i] = 0;
      partMax[i] = 0;
      partSum[i] = 0;
    }

    ierr = MPI_Allreduce(partLoc, partMax, 7, MPI_INT, 
                         MPI_MAX, comm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(partLoc, partMin, 7, MPI_INT, 
                         MPI_MIN, comm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(partLoc, partSum, 7, MPI_INT, 
                         MPI_SUM, comm); CHKERRQ(ierr);
    
    ierr = PetscPrintf(comm,"==============================\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Partitioning quality info ....\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"==============================\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"------------------------------------------------------------\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Item                    Min        Max    Average      Total\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"------------------------------------------------------------\n");
    CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local Nodes       %9d  %9d  %9d  %9d\n",
              partMin[0],partMax[0],partSum[0]/size,partSum[0]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local+Ghost Nodes %9d  %9d  %9d  %9d\n",
              partMin[1],partMax[1],partSum[1]/size,partSum[1]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local Edges       %9d  %9d  %9d  %9d\n",
              partMin[2],partMax[2],partSum[2]/size,partSum[2]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local solid faces %9d  %9d  %9d  %9d\n",
              partMin[3],partMax[3],partSum[3]/size,partSum[3]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local free faces  %9d  %9d  %9d  %9d\n",
              partMin[4],partMax[4],partSum[4]/size,partSum[4]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local solid nodes %9d  %9d  %9d  %9d\n",
              partMin[5],partMax[5],partSum[5]/size,partSum[5]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Local free nodes  %9d  %9d  %9d  %9d\n",
              partMin[6],partMax[6],partSum[6]/size,partSum[6]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"------------------------------------------------------------\n");
    CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(NULL, 0, "-partition_info", &flg);
  CHKERRQ(ierr);
  if (flg) 
  {
    char part_file[P_LEN];
    sprintf(part_file, "output.%d", rank);
    fptr1 = fopen(part_file, "w");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local and Global Grid Parameters are :\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\t\t\tGlobal\n");
    fprintf(fptr1,"nnodesLoc = %d\t\tnnodes = %d\n",nnodesLoc,nnodes);
    fprintf(fptr1,"nedgeLoc = %d\t\t\tnedge = %d\n",nedgeLoc,nedge);
    fprintf(fptr1,"nnfacetLoc = %d\t\tnnfacet = %d\n",nnfacetLoc,nnfacet);
    fprintf(fptr1,"nvfacetLoc = %d\t\t\tnvfacet = %d\n",nvfacetLoc,nvfacet);
    fprintf(fptr1,"nffacetLoc = %d\t\t\tnffacet = %d\n",nffacetLoc,nffacet);
    fprintf(fptr1,"nsnodeLoc = %d\t\t\tnsnode = %d\n",nsnodeLoc,nsnode);
    fprintf(fptr1,"nvnodeLoc = %d\t\t\tnvnode = %d\n",nvnodeLoc,nvnode);
    fprintf(fptr1,"nfnodeLoc = %d\t\t\tnfnode = %d\n",nfnodeLoc,nfnode);
    fprintf(fptr1,"\n");
    fprintf(fptr1,"nvertices = %d\n",nvertices);
    fprintf(fptr1,"nnbound = %d\n",nnbound);
    fprintf(fptr1,"nvbound = %d\n",nvbound);
    fprintf(fptr1,"nfbound = %d\n",nfbound);
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Different Orderings\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
  
    for (i = 0; i < nvertices; i++) 
    {
      fprintf(fptr1,"%d\t\t%d\t\t%d\n",i,grid->loc2pet[i],grid->loc2glo[i]);
    }
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Solid Boundary Nodes\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nsnodeLoc; i++) 
    {
      j = grid->isnode[i]-1;
      fprintf(fptr1,"%d\t\t%d\t\t%d\n",j,grid->loc2pet[j],grid->loc2glo[j]);
    }
    
    fprintf(fptr1,"\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"f2ntn array\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nnfacetLoc; i++) 
    {
      fprintf(fptr1,"%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntn[i],
              grid->f2ntn[nnfacetLoc+i],grid->f2ntn[2*nnfacetLoc+i]);
    }
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Viscous Boundary Nodes\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
  
    for (i = 0; i < nvnodeLoc; i++) 
    {
      j = grid->ivnode[i]-1;
      fprintf(fptr1,"%d\t\t%d\t\t%d\n",j,grid->loc2pet[j],grid->loc2glo[j]);
    }
    
    fprintf(fptr1,"\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"f2ntv array\n");
    fprintf(fptr1,"---------------------------------------------\n");
    
    for (i = 0; i < nvfacetLoc; i++) 
    {
      fprintf(fptr1,"%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntv[i],
              grid->f2ntv[nvfacetLoc+i],grid->f2ntv[2*nvfacetLoc+i]);
    }
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Free Boundary Nodes\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Local\t\tPETSc\t\tGlobal\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nfnodeLoc; i++) 
    {
      j = grid->ifnode[i]-1;
      fprintf(fptr1,"%d\t\t%d\t\t%d\n",j,grid->loc2pet[j],grid->loc2glo[j]);
    }
    fprintf(fptr1,"\n");
    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"f2ntf array\n");
    fprintf(fptr1,"---------------------------------------------\n");
    for (i = 0; i < nffacetLoc; i++) 
    {
      fprintf(fptr1,"%d\t\t%d\t\t%d\t\t%d\n",i,grid->f2ntf[i],
              grid->f2ntf[nffacetLoc+i],grid->f2ntf[2*nffacetLoc+i]);
    }
    fprintf(fptr1,"\n");

    fprintf(fptr1,"---------------------------------------------\n");
    fprintf(fptr1,"Neighborhood Info In Various Ordering\n");
    fprintf(fptr1,"---------------------------------------------\n");
    
    ICALLOC(nnodes,&p2l);
    for (i = 0; i < nvertices; i++)
      p2l[grid->loc2pet[i]] = i;
   
    for (i = 0; i < nnodesLoc; i++) 
    {
      jstart = grid->ia[grid->loc2glo[i]] - 1;
      jend = grid->ia[grid->loc2glo[i]+1] - 1;
      
      fprintf(fptr1,"Neighbors of Node %d in Local Ordering are :",i);
      
      for (j = jstart; j < jend; j++) 
      {
        fprintf(fptr1,"%d ",p2l[grid->ja[j]]);
      }
      fprintf(fptr1,"\n");

      fprintf(fptr1,"Neighbors of Node %d in PETSc ordering are :",grid->loc2pet[i]);
      
      for (j = jstart; j < jend; j++) 
      {
        fprintf(fptr1,"%d ",grid->ja[j]);
      }
      fprintf(fptr1,"\n");

      fprintf(fptr1,"Neighbors of Node %d in Global Ordering are :",grid->loc2glo[i]);
      
      for (j = jstart; j < jend; j++) 
      {
        fprintf(fptr1,"%d ",grid->loc2glo[p2l[grid->ja[j]]]);
      }
      fprintf(fptr1,"\n");
 
    }
   
    fprintf(fptr1,"\n");
    free(p2l);
    fclose(fptr1);
  }

  //Free the temporary arrays
  free(a2l);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);

  PetscFunctionReturn(0); 
}

// Color edges for multi-threading computation
#ifdef _OPENMP
#ifdef HAVE_EDGE_COLORING
int EdgeColoring(int nnodes, int nedge, int *e2n, int *eperm,
                 int *ncle, int *counte)
{
  int ierr;
  int ncolore = *ncle = 0;  // Number of colors
  int iedg    = 0;
  int ib      = 0;
  int ie      = nedge;      // Number of local edges (nedgeLoc)
  int tagcount;
  int i, n1, n2;
  int *tag;

  ICALLOC(nnodes, &tag); // Size of the number of local nodes + ghost nodes (nvertices)
 
  while(ib < ie) // Initially ib == 0 less than ie == number of local edges 
  {
    // Initialization
    for(i = 0; i < nnodes; i++) 
      tag[i] = 0;
      
    counte[ncolore] = 0; // Number of faces in color
    // loop over local edges
    for(i = ib; i < ie; i++)
    {
      // Read node 1 and node 2 (traverse over edges)
      // Using grid->eptr
      // Non-interlaced storage of data
      n1 = e2n[i];
      n2 = e2n[i + nedge];
      
      tagcount = tag[n1] + tag[n2];
      // If tagcount = 0 then this edge belongs in this color
      // This edge has been colored yet
      if(!tagcount) // tagcount == 0
      {
        // Color the edge by coloring the two edges of this edge
        tag[n1] = 1;
        tag[n2] = 1;
        // Reorder the nodes of the edge after coloring the edge
        e2n[i]          = e2n[iedg];
        e2n[i+nedge]    = e2n[iedg+nedge];
        e2n[iedg]       = n1;
        e2n[iedg+nedge] = n2;
        // Permutation array index of the edge     
        n1 = eperm[i];
        // Swap the node index in the permutation array
        eperm[i]    = eperm[iedg];
        eperm[iedg] = n1;
        // Add the edge to this tag
        iedg++;
        // Add one face to this color
        counte[ncolore] += 1;
      }
    }
    // Move to new edge
    ib = iedg;
    // Move to new color
    ncolore++;
  }
  // Total number of colors
  *ncle = ncolore;

  return 0;
}
#endif
#endif

// Allocates the memory for the fine grid
#undef  __FUNCT__
#define __FUNCT__ "set_up_grid"
int set_up_grid(GRID *grid)
{
  PetscErrorCode ierr;
  
  int nnodes;
  int tnode;
  int nedge;
  int nsface;
  int nvface;
  int nfface;
  int nbface; // Linearizing viscous
   
  PetscFunctionBegin;
  
   
  nnodes = grid->nnodes;
  tnode  = grid->nnodes;
  nedge  = grid->nedge;
  nsface = grid->nsface;
  nvface = grid->nvface;
  nfface = grid->nfface;

  nbface = nsface + nvface + nfface;
  // Now allocate memory for the other 
  // grid arrays
  // Face number of solid faces
  ICALLOC(nsface, &grid->isface);
  // Face number of viscous faces
  ICALLOC(nvface, &grid->ivface);
  // Face number of far field faces
  ICALLOC(nfface, &grid->ifface);
  // Stuffs for for linearizing viscous
  // *****************************
  FCALLOC(nbface * 15, &grid->us);
  FCALLOC(nbface * 15, &grid->vs);
  FCALLOC(nbface * 15, &grid->as);
  // ***************************** 
  
  // Allocate memory for viscous length
  // scale if turbulent
  // flow type (e.g. 0  means Euler)
  if(grid->jvisc >= 3)
  {
    FCALLOC(tnode,  &grid->slen);
    FCALLOC(nnodes, &grid->turbre);
    FCALLOC(nnodes, &grid->amut);
    FCALLOC(tnode,  &grid->turbres);
    FCALLOC(nedge,  &grid->dft1);
    FCALLOC(nedge,  &grid->dft2);
  }

  PetscFunctionReturn(0);
}
