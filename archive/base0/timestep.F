!
! ================================ DELTAT2 ============================72
!
! Calculate a time step for each cell
! Note that this routine assumes conservative variables
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "DELTA2"
      subroutine DELTAT2( nnodes, nedge, nedgeThreads, nsnode, nfnode,  &
     &                    isLocal, nThreads, eptr, qnode, xn, yn, zn,   &
     &                    rl, vol, sxn, syn, szn, fxn, fyn, fzn,        &
     &                    isnode, ifnode, partitions, nedgeThreadLoc,   &
     &                    cdt )
      implicit none      

#include "fort.h"
!
! Function parameters
!
      integer           nnodes, nedge, nedgeThreads
      integer           nsnode, nfnode
      integer           isLocal, nThreads 
      integer           eptr(2, nedgeThreads)
      double precision  qnode(4, nnodes)
      double precision  xn(1), yn(1), zn(1), rl(1)
      double precision  vol(1)
      double precision  sxn(1), syn(1), szn(1)
      double precision  fxn(1), fyn(1), fzn(1)
      integer           isnode(1), ifnode(1)
      integer           partitions(nnodes)    
      integer           nedgeThreadLoc(0: nThreads)
      double precision  cdt(1) 
!
! Function arguments
!
      integer           i , node1, node2, inode, ierr
      integer           firstEdgeLoc, lastEdgeLoc
      integer           tid, omp_get_thread_num
      double precision  flops
      double precision  xnorm, ynorm, znorm, area
      double precision  u1, v1, w1 
      double precision  u2, v2, w2 
      double precision  u, v, c, w
      double precision  ubar, Vn, term
      double precision  beta, alpha
!
! Fortran global variables
!
      common /coeff/ beta, alpha
!
! Initialize the floating point counter to zero
!
      flops         = 0.0
!
!  If local time steping, loop over faces
!  and calculate time step as cdt = V/(sum(|u.n| +c.area)
!  This is time step for cfl=1. We will multiply by cfl number later
!     
      if (isLocal .gt. 0) then 
!
! First loop over nodes and zero out cdt
!
        do i = 1, nnodes
          cdt(i) = 0.0d0
        enddo
!
! Initialize the start and the end edges for each thread
!
        firstEdgeLoc  = 1
        lastEdgeLoc   = nedge
!
!$OMP   parallel default(shared)                                        &
!$OMP&          private(firstEdgeLoc, lastEdgeLoc, tid, i, node1,       &
!$OMP&                  node2, xnorm, ynorm, znorm, area, u1, v1, w1,   &
!$OMP&                  u2, v2, w2, u, v, w, ubar, c, term)             &
!$OMP&          reduction(+:flops)
!
        tid           = omp_get_thread_num()
        firstEdgeLoc  = nedgeThreadLoc(tid)
        lastEdgeLoc   = nedgeThreadLoc(tid + 1) - 1
!
! An edge-based loop
!
        do i = firstEdgeLoc, lastEdgeLoc
!
! Read an edge
!
          node1 = eptr(1, i)
          node2 = eptr(2, i)
!
! Get normal to face
!
          xnorm = xn(i)
          ynorm = yn(i)
          znorm = zn(i)
          area  = rl(i)

          xnorm = xnorm * area
          ynorm = ynorm * area
          znorm = znorm * area
!
!/*
! xnorm = xnormal * area of face  
! ynorm = ynormal * area of face  
! znorm = znormal * area of face  
!*/
!
          u1   = qnode(2, node1)
          v1   = qnode(3, node1)
          w1   = qnode(4, node1)
 
          u2   = qnode(2, node2)
          v2   = qnode(3, node2)
          w2   = qnode(4, node2)
!
! Get average values on face 
!
          u    = 0.5d0 * (u1 + u2)
          v    = 0.5d0 * (v1 + v2)
          w    = 0.5d0 * (w1 + w2)
          ubar = xn(i) * u + yn(i) * v + zn(i) * w
          c    = sqrt(ubar * ubar + beta)
 
          term = abs(u * xnorm + v * ynorm + w * znorm) + c * area
!
! Update first node
!
          if(partitions(node1) .eq. tid) then

            cdt(node1) = cdt(node1) + term
            flops = flops + 27.0

          endif
!
! Update second node
!
          if(partitions(node2) .eq. tid) then
            
            cdt(node2) = cdt(node2) + term
            flops = flops + 27.0 
          
          endif

        enddo
! 
!$OMP end parallel 
!
! Now loop over boundaries and close the contours
!
        do i = 1, nsnode
          inode = isnode(i)
!
! Get the normal
!
          xnorm = sxn(i)
          ynorm = syn(i)
          znorm = szn(i)
          area  = sqrt(xnorm * xnorm + ynorm * ynorm + znorm * znorm)

          u     = qnode(2, inode)
          v     = qnode(3, inode)
          w     = qnode(4, inode)

          ubar  = (u * xnorm + v * ynorm + w * znorm) / area

          c     = sqrt(ubar * ubar + beta)
          Vn    = abs(xnorm * u + ynorm * v + znorm * w) + c * area
          cdt(inode) = cdt(inode) + Vn
        enddo
        
        flops = flops + 24.0 * nsnode
!
! Now far field
!
        do i = 1, nfnode
!
          inode = ifnode(i)
!
! Get the normal
!
          xnorm = fxn(i)
          ynorm = fyn(i)
          znorm = fzn(i)
          area  = sqrt(xnorm * xnorm + ynorm * ynorm + znorm * znorm)

          u   = qnode(2, inode)
          v   = qnode(3, inode)
          w   = qnode(4, inode)
          ubar= (u * xnorm + v * ynorm + w * znorm) / area
          c   = sqrt(ubar * ubar + beta)
 
          Vn  = abs(xnorm * u + ynorm * v + znorm * w) + c * area
          cdt(inode) = cdt(inode) + Vn
        enddo
 
        flops = flops + 24.0 * nfnode
       
        do i = 1, nnodes
          cdt(i) = vol(i) / cdt(i)
        enddo   
 
         flops = flops + nnodes
 
      else
!
! If not doing local time stepping just set cdt=1
!
        do i = 1, nnodes
          cdt(i) = 1.0d0
        enddo

      end if

      call PetscLogFlops(flops, ierr)
!
! End of subroutine DELTAT2
! 
      return
      end
