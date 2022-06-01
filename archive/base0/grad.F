!
! =============================== SUMGS ===============================72
! 
! Gets the weights for calculating gradients using least squares
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "SUMGS"
      subroutine SUMGS( nnodes, nedge, nedgeThreads, nThreads, eptr,    &
     &                  x, y, z, partitions, nedgeThreadLoc, r11, r12,  &
     &                  r13, r22, r23, r33, r44 )
      implicit none
!
! Function parameters
!
      integer           nnodes, nedge
      integer           nedgeThreads, nThreads
      integer           eptr(2, nedgeThreads)
      double precision  x(1), y(1), z(1)
      integer           partitions(nnodes)
      integer           nedgeThreadLoc(0: nThreads)
      double precision  r11(1), r12(1), r13(1), r22(1)
      double precision  r23(1), r33(1), r44(1)
!
! Function arguments
!      
      integer           i, node1, node2
      integer           firstEdgeLoc, lastEdgeLoc
      integer           tid, omp_get_thread_num
      double precision  x1, y1, z1
      double precision  x2, y2, z2 
      double precision  dx, dy, dz
      double precision  weight 
      double precision  w2, w11, w22, w33
      double precision  r12r11, r13r11, r23r22
      double precision  rmess, rmess1, rmess2
!
! Initialize all the rij to 0.0
!
      do i = 1, nnodes
        r11(i) = 0.0
        r12(i) = 0.0
        r13(i) = 0.0
        r22(i) = 0.0
        r23(i) = 0.0
        r33(i) = 0.0
        r44(i) = 0.0
      enddo
!
! Initializations
!
      weight        = 1.0d0
      w2            = weight * weight
!
! Accumulate the r 
!
      firstEdgeLoc  = 1
      lastEdgeLoc   = nedge
!
!$OMP   parallel  default(shared)                                       &
!$OMP&            private(firstEdgeLoc, lastEdgeLoc, tid, i, node1,     &
!$OMP&                    node2, x1, y1, z1, x2, y2, z2)
!
      tid           = omp_get_thread_num()
      firstEdgeLoc  = nedgeThreadLoc(tid)
      lastEdgeLoc   = nedgeThreadLoc(tid + 1) - 1
!
! Now loop over the edges and accumulate the r 
!
      do i = firstEdgeLoc, lastEdgeLoc
!
! Read an edge with two endpoints
!
        node1   = eptr(1, i)
        node2   = eptr(2, i)
!
! Read the xyz coordinates for the first node
!
        x1  = x(node1)
        y1  = y(node1)
        z1  = z(node1)
!
! Read the xyz coordinates for the second node          
!
        x2  = x(node2)
        y2  = y(node2)
        z2  = z(node2)
!
! Calculate the weights for the first node          
!
        if(partitions(node1) .eq. tid) then
              
          r11(node1) = r11(node1) + (x2 - x1) * (x2 - x1)! * w2
          r12(node1) = r12(node1) + (x2 - x1) * (y2 - y1)! * w2
          r13(node1) = r13(node1) + (x2 - x1) * (z2 - z1)! * w2

!          r11(node1) = sqrt(r11(node1))
!          r12(node1) = r12(node1) / r11(node1)
!          r13(node1) = r13(node1) / r11(node1)

        endif
!
! Calculate the weight for the second node
!
        if(partitions(node2) .eq. tid) then
            
          r11(node2) = r11(node2) + (x1 - x2) * (x1 - x2)! * w2
          r12(node2) = r12(node2) + (x1 - x2) * (y1 - y2)! * w2
          r13(node2) = r13(node2) + (x1 - x2) * (z1 - z2)! * w2

!          r11(node2) = sqrt(r11(node2))
!          r12(node2) = r12(node2) / r11(node2)
!          r13(node2) = r13(node2) / r11(node2)

        endif
      enddo
!
!$OMP end parallel
!
      
!
!/*
! Now calculate ||x(:)|| = r11 by taking the square root
! Also divide r12 and r13 by ||x(:)||
!*/
!
      do i = 1, nnodes
        r11(i) = sqrt(r11(i))
        r12(i) = r12(i) / r11(i)
        r13(i) = r13(i) / r11(i)
      Enddo
!
!/*
! Now calculate r22 and r23
!*/
!
      firstEdgeLoc = 1
      lastEdgeLoc  = nedge
!
!$OMP   parallel  default(shared)                                       &
!$OMP&            private(firstEdgeLoc, lastEdgeLoc, tid, i, node1,     &
!$OMP&                    node2, x1, y1, z1, x2, y2, z2, dx, dy, dz)
!
      tid           = omp_get_thread_num()
      firstEdgeLoc  = nedgeThreadLoc(tid)
      lastEdgeLoc   = nedgeThreadLoc(tid + 1) - 1

      do i = firstEdgeLoc, lastEdgeLoc
!
! Read an edge with two endpoints        
!
        node1 = eptr(1, i)
        node2 = eptr(2, i)
!
! Read the xyz coordinates for the first node
!
        x1 = x(node1)
        y1 = y(node1)
        z1 = z(node1)
!
! Read the xyz coordinates for the second node        
!
        x2 = x(node2)
        y2 = y(node2)
        z2 = z(node2)

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
         
!        dx = weight * dx
!        dy = weight * dy
!        dz = weight * dz

        if(partitions(node1) .eq. tid) then

          r22(node1) = r22(node1)+(dy-dx*r12(node1)/r11(node1))**2
          r23(node1) = r23(node1)+dz*(dy-dx*r12(node1)/r11(node1))

        endif

        if(partitions(node2) .eq. tid) then

          r22(node2) = r22(node2)+(-dy+dx*r12(node2)/r11(node2))**2
          r23(node2) = r23(node2)-dz*(-dy+dx*r12(node2)/r11(node2))
        endif

      enddo
!
!$OMP end parallel
!      
!
!/*
! Now finish getting r22 and r23
!*/
!
      do i = 1, nnodes
        r22(i) = sqrt(r22(i))
        r23(i) = r23(i)/r22(i)
      enddo
!
!/*
! Now all we have to do is get r33
!*/
!
      firstEdgeLoc = 1
      lastEdgeLoc  = nedge
!
!$OMP   parallel  default(shared)                                       &
!$OMP&            private(firstEdgeLoc, lastEdgeLoc, tid, i, node1,     &
!$OMP&                    node2, x1, y1, z1, x2, y2, z2, dx, dy, dz)
!
      tid           = omp_get_thread_num()
      firstEdgeLoc  = nedgeThreadLoc(tid)
      lastEdgeLoc   = nedgeThreadLoc(tid + 1) - 1

      do i = firstEdgeLoc, lastEdgeLoc
!
! Read an edge with two endpoints        
!
        node1 = eptr(1, i)
        node2 = eptr(2, i)

        x1 = x(node1)
        y1 = y(node1)
        z1 = z(node1)

        x2 = x(node2)
        y2 = y(node2)
        z2 = z(node2)

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

!        dx = weight * dx
!        dy = weight * dy
!        dz = weight * dz

        if(partitions(node1) .eq. tid) then

          r33(node1) = r33(node1) +                                     &
     &                  (dz-dx*r13(node1)/r11(node1)-                   &
     &                  r23(node1)/r22(node1)*                          &
     &                  (dy-dx*r12(node1)/                              &
     &                   r11(node1)))**2

        endif

        if(partitions(node2) .eq. tid) then

          r33(node2) = r33(node2) +                                     &
     &                  (-dz+dx*r13(node2)/r11(node2)-                  &
     &                  r23(node2)/r22(node2)*                          &
     &                  (-dy+dx*r12(node2)/                             &
     &                  r11(node2)))**2

        endif
      enddo
!
!$OMP end parallel
!
!/*
! Now just get the magnitude of r33
!*/
!
      do i = 1, nnodes
        r33(i) = sqrt(r33(i))
      enddo

      do i = 1, nnodes
        w11 = 1.d0/(r11(i)*r11(i))
        w22 = 1.d0/(r22(i)*r22(i))
        w33 = 1.d0/(r33(i)*r33(i))

        r12r11 = r12(i)/r11(i)
        r13r11 = r13(i)/r11(i)
        r23r22 = r23(i)/r22(i)
       
        rmess1 = r12(i)*r23(i)-r13(i)*r22(i)
        rmess2 = r11(i)*r22(i)*r33(i)*r33(i)
 
        rmess  =  rmess1/rmess2

        r11(i) = w11
        r22(i) = w22
        r33(i) = w33
        r12(i) = r12r11
        r13(i) = r13r11
        r23(i) = r23r22
        r44(i) = rmess
      enddo
!
! Finished with SUMGS
!
      return
      end
!
! ================================ LSTGS ==============================72
!
! Calculates the gradients at the nodes using weighted least squares
! This subroutine solves using Gram-Schmidt
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "LSTGS"
      subroutine LSTGS( nnodes, nedge, nedgeThreads, nThreads, eptr,    &
     &                  qnode, x, y, z, r11, r12, r13, r22, r23, r33,   &
     &                  r44, partitions, nedgeThreadLoc, grad )
      implicit none      

#include "fort.h"
!
! Function parameters
!
      integer           nnodes, nedge
      integer           nedgeThreads, nThreads
      integer           eptr(2, nedgeThreads) 
      double precision  qnode(4, nnodes)
      double precision  x(1), y(1), z(1)
      double precision  r11(1), r12(1), r13(1), r22(1)
      double precision  r23(1), r33(1), r44(1)
      integer           partitions(nnodes)
      integer           nedgeThreadLoc(0: nThreads)
      double precision  grad(3, 4, nnodes)
!
! Function arguments
!
      integer           i, node1, node2, ierr
      integer           firstEdgeLoc, lastEdgeLoc
      integer           tid, omp_get_thread_num
      double precision  flops
      double precision  dx1, dy1, dz1
      double precision  dx2, dy2, dz2
      double precision  dq1, dq2, dq3, dq4
      double precision  weight, w11, r12r11 
      double precision  r13r11, w22, r23r22
      double precision  w33
      double precision  rmess, coef1, coef2 
      double precision  termx, termy, termz
!
! Initializations
!
      flops         = 0.0
      weight        = 1.0d0
      firstEdgeLoc  = 1
      lastEdgeLoc   = nedge
!
! Zero out the gradients
!
      do i = 1, nnodes

        grad(1, 1, i) = 0.0
        grad(2, 1, i) = 0.0
        grad(3, 1, i) = 0.0

        grad(1, 2, i) = 0.0
        grad(2, 2, i) = 0.0
        grad(3, 2, i) = 0.0

        grad(1, 3, i) = 0.0
        grad(2, 3, i) = 0.0
        grad(3, 3, i) = 0.0

        grad(1, 4, i) = 0.0
        grad(2, 4, i) = 0.0
        grad(3, 4, i) = 0.0

      enddo
!
!$OMP   parallel  default(shared)                                       &
!$OMP&            private(firstEdgeLoc, lastEdgeLoc, tid, i, node1,     &
!$OMP&                    node2, dx1, dy1, dz1, dx2, dy2, dz2,          & 
!$OMP&                    dq1, dq2, dq3, dq4, w11, r12r11, r13r11, w22, &
!$OMP&                    r23r22, w33, rmess, coef1, coef2, termx,      &
!$OMP&                    termy, termz)                                 &
!$OMP&            reduction(+: flops)
!
      tid           = omp_get_thread_num()
      firstEdgeLoc  = nedgeThreadLoc(tid)
      lastEdgeLoc   = nedgeThreadLoc(tid + 1) - 1
!
! An edge-based loop
!
      do i = firstEdgeLoc, lastEdgeLoc
!
! Read an edge with two endpoints        
!
        node1 = eptr(1, i)
        node2 = eptr(2, i)

        dx1 = x(node2) - x(node1)
        dy1 = y(node2) - y(node1)
        dz1 = z(node2) - z(node1)

        dx1 = weight * dx1
        dy1 = weight * dy1
        dz1 = weight * dz1

        flops = flops + 6.0
!
! Update the first node
!
        if(partitions(node1) .eq. tid) then

          dq1 = weight * (qnode(1, node2) - qnode(1, node1))
          dq2 = weight * (qnode(2, node2) - qnode(2, node1))
          dq3 = weight * (qnode(3, node2) - qnode(3, node1))
          dq4 = weight * (qnode(4, node2) - qnode(4, node1))

          w11     = r11(node1)
          r12r11  = r12(node1)
          r13r11  = r13(node1)
          w22     = r22(node1)
          r23r22  = r23(node1)
          w33     = r33(node1)
          rmess   = r44(node1)

          coef1   = dy1 - dx1 * r12r11
          coef2   = dz1 - dx1 * r13r11 - r23r22 * coef1
          
          termx   = dx1 * w11   - w22 * r12r11 * coef1 + rmess * coef2
          termy   = w22 * coef1 - r23r22 * w33 * coef2
          termz   = w33 * coef2

          grad(1, 1, node1) = grad(1, 1, node1) + termx * dq1
          grad(2, 1, node1) = grad(2, 1, node1) + termy * dq1
          grad(3, 1, node1) = grad(3, 1, node1) + termz * dq1

          grad(1, 2, node1) = grad(1, 2, node1) + termx * dq2
          grad(2, 2, node1) = grad(2, 2, node1) + termy * dq2
          grad(3, 2, node1) = grad(3, 2, node1) + termz * dq2

          grad(1, 3, node1) = grad(1, 3, node1) + termx * dq3
          grad(2, 3, node1) = grad(2, 3, node1) + termy * dq3
          grad(3, 3, node1) = grad(3, 3, node1) + termz * dq3

          grad(1, 4, node1) = grad(1, 4, node1) + termx * dq4
          grad(2, 4, node1) = grad(2, 4, node1) + termy * dq4
          grad(3, 4, node1) = grad(3, 4, node1) + termz * dq4

          flops = flops + 49.0

        endif
!
! Now do the other node
!
        if(partitions(node2) .eq. tid) then

          dx2 = -dx1
          dy2 = -dy1
          dz2 = -dz1

          dq1 = weight * (qnode(1, node1) - qnode(1, node2))
          dq2 = weight * (qnode(2, node1) - qnode(2, node2))
          dq3 = weight * (qnode(3, node1) - qnode(3, node2))
          dq4 = weight * (qnode(4, node1) - qnode(4, node2))

          w11     = r11(node2)
          r12r11  = r12(node2)
          r13r11  = r13(node2)
          w22     = r22(node2)
          r23r22  = r23(node2)
          w33     = r33(node2)
          rmess   = r44(node2)

          coef1   = dy2 - dx2 * r12r11
          coef2   = dz2 - dx2 * r13r11 - r23r22 * coef1

          termx   = dx2 * w11   - w22 * r12r11 * coef1 + rmess * coef2
          termy   = w22 * coef1 - r23r22 * w33 * coef2
          termz   = w33 * coef2

          grad(1, 1, node2) = grad(1, 1, node2) + termx * dq1
          grad(2, 1, node2) = grad(2, 1, node2) + termy * dq1
          grad(3, 1, node2) = grad(3, 1, node2) + termz * dq1

          grad(1, 2, node2) = grad(1, 2, node2) + termx * dq2
          grad(2, 2, node2) = grad(2, 2, node2) + termy * dq2
          grad(3, 2, node2) = grad(3, 2, node2) + termz * dq2

          grad(1, 3, node2) = grad(1, 3, node2) + termx * dq3
          grad(2, 3, node2) = grad(2, 3, node2) + termy * dq3
          grad(3, 3, node2) = grad(3, 3, node2) + termz * dq3

          grad(1, 4, node2) = grad(1, 4, node2) + termx * dq4
          grad(2, 4, node2) = grad(2, 4, node2) + termy * dq4
          grad(3, 4, node2) = grad(3, 4, node2) + termz * dq4

          flops = flops + 52.0

        endif
      enddo 
!
!$OMP end parallel
!
      call PetscLogFlops(flops, ierr)
!
! End of LSTGS
!
      return
      end
