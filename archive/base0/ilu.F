!
! =====================================================================72
!
! Get the IA array: Row starting position
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "GETIA"
      subroutine GETIA(nnodes, nedge, eptr, ideg, ia) 
      implicit none      
!
! Function parameters
!
      integer nnodes, nedge
      integer eptr(2, nedge)
      integer ideg(1), ia(1)
!
! Function arguments
!      
      integer i, node1, node2
!
! Initialize the dummy array
!
      do i = 1, nnodes
        ideg(i) = 0
      enddo
!
! First get the degree of each node using ideg as a dummy array
!
      do i = 1, nedge
!
! Read an edge
!        
        node1 = eptr(1, i)
        node2 = eptr(2, i)
!
! Counter the number of edges connect to that node
!
        ideg(node1) = ideg(node1) + 1
        ideg(node2) = ideg(node2) + 1
      enddo
!
! Now we can fill the ia array fairly easily
!
      ia(1) = 1
      do i = 1, nnodes
        ia(i + 1) = ia(i) + ideg(i) + 1
      enddo
!
! End of subroutine GETIA
!
      return
      end
!
! =====================================================================72
!
! Get the JA array: Column indices
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "GETJA"
      subroutine GETJA(nnodes, nedge, eptr, ia, iwork, ja)
      implicit none 
!
! Function parameters
!
      integer nnodes, nedge
      integer eptr(2, nedge)
      integer ia(1), iwork(1), ja(1)
!
! Function arguments
!
      integer i, index1, index2
      integer node1, node2 
      integer istart, iend
!
! Now we need to get the JA array
! First fill the diagonal places
!
      do i = 1, nnodes
        ja(ia(i)) = i
        iwork(i)  = 1
      enddo

      do i = 1, nedge
        node1         = eptr(1, i)
        node2         = eptr(2, i)

        index1        = ia(node1)     + iwork(node1) 
        iwork(node1)  = iwork(node1)  + 1
        ja(index1)    = node2
        
        index2        = ia(node2)     + iwork(node2)
        iwork(node2)  = iwork(node2)  + 1
        ja(index2)    = node1
      enddo

!
! Now lets sort all our "bins" and get the correct one on the diagonal
!

      do i = 1, nnodes
        istart = ia(i)
        iend   = ia(i + 1) - 1
                
        call SORTER(istart, iend, i, ja) 
      enddo
    

!
! End of subroutine GETJA
!
      return
      end
!
! =====================================================================72
!
! Sort each of our bins
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "SORTER"
      subroutine SORTER(istart, iend, inode, ja)
      implicit none      
!
! Function parameters
!
      integer istart, iend, inode
      integer ja(1)
!
! Function arguments
!      
      integer min, minsave, jsave, i, j
            
      do i = istart, iend
        min     = ja(i)
        minsave = ja(i)
        jsave   = i
                
        do j = i + 1, iend
          if(ja(j) .lt. min) then
            min = ja(j)
            jsave = j
          end if
        enddo

        ja(i)     = min
        ja(jsave) = minsave

      enddo    
!
! End of subroutine SORTER
!
      return
      end
