!
! ================================= FLUX ==============================72
!
! Calculates the fluxes on the face and performs the flux balance
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "FLUX"
#ifdef HW_COUNTER
      subroutine FLUX(nnodes, nedge, nedgeThreads, nsnode, nfnode,      &
     &                nnfacet, nffacet, nThreads, eptr, qnode, x, y, z, &
     &                xn, yn, zn, ra, grad, fxn, fyn, fzn, isnode,      &
     &                ifnode, f2ntn, f2ntf, partitions, nedgeThreadLoc, &
     &                res, retTime, event_set, handler)
#else
      subroutine FLUX(nnodes, nedge, nedgeThreads, nsnode, nfnode,      &
     &                nnfacet, nffacet, nThreads, eptr, qnode, x, y, z, &
     &                xn, yn, zn, ra, grad, fxn, fyn, fzn, isnode,      &
     &                ifnode, f2ntn, f2ntf, partitions, nedgeThreadLoc, &
     &                res, retTime)
#endif
      implicit none

#include "fort.h"
!#include <fpapi.h>
!
! Function parameters
!
      integer           nnodes, nedge, nedgeThreads
      integer           nsnode, nfnode
      integer           nnfacet, nffacet
      integer           nThreads
      integer           eptr(2, nedgeThreads)
      double precision  qnode(4, nnodes) 
      double precision  x(1), y(1), z(1)
      double precision  xn(1), yn(1), zn(1), ra(1)
      double precision  grad(3, 4, nnodes) 
      double precision  fxn(1), fyn(1), fzn(1) 
      integer           isnode(1), ifnode(1)
      integer           f2ntn(nnfacet, 4)
      integer           f2ntf(nffacet, 4) 
      integer           partitions(nnodes) 
      integer           nedgeThreadLoc(0: nThreads)
      double precision  res(4, nnodes)
      double precision  retTime
#ifdef HW_COUNTER
      integer           event_set, handler
#endif
!
! Function arguments
!
      integer           inode, ierr, i
      integer           firstEdgeLoc, lastEdgeLoc
      integer           tid, omp_get_thread_num
      integer           node1, node2, node3
      double precision  omp_get_wtime
      double precision  flops
      double precision  p0, c0, u0, v0, w0 
      double precision  second 
      double precision  xmean, ymean, zmean
      double precision  xnorm, ynorm, znorm
      double precision  rlen, dot, size
      double precision  X1, Y1, Z1 
      double precision  X2, Y2, Z2 
      double precision  rx, ry, rz
      double precision  pL, uL, wL, vL
      double precision  pR, uR, vR, wR
      double precision  ubarL, ubarR     
      double precision  phi1, phi2, phi3, phi4, phi5
      double precision  phi6, phi7, phi8, phi9
      double precision  ubar, p, u, v, w, c, c2 
      double precision  eig1, eig2, eig3, eig4
      double precision  dp, du, dv, dw
      double precision  ti11, ti21, ti31, ti41
      double precision  dv1, dv2, dv3, dv4
      double precision  r11, r21, r31, r41
      double precision  r12, r22, r32, r42
      double precision  r13, r23, r33, r43
      double precision  r14, r24, r34, r44
      double precision  t1, t2, t3, t4
      double precision  fluxp1, fluxp2, fluxp3, fluxp4
      double precision  fluxm1, fluxm2, fluxm3, fluxm4
      double precision  res1, res2, res3, res4
      double precision  c68, c18
      double precision  ax, ay, az, bx, by, bz
      double precision  p1, p2, p3
      double precision  pa, pb, pc, ub, vb, wb
      double precision  area, ubar0, unorm
      double precision  t11, t21, t31, t41
      double precision  t12, t22, t32, t42
      double precision  t13, t23, t33, t43
      double precision  t14, t24, t34, t44
      double precision  c20
      double precision  rhs1, rhs2, rhs3, rhs4 
      double precision  pi, ui, vi, wi
      double precision  x3, y3, z3
      double precision  beta, alpha
      double precision  sTime, eTime
!
! Logging variables
!
      integer           fEvent, fClass, flag      
      character         *16 fLabel             
      data              flag/-1/
      data              fLabel/'FluxEvaluation  '/
      save              fEvent, fClass, flag, fLabel
!
! Fortran global variables
!
      common /coeff/  beta, alpha
      common /ivals/  p0, u0, v0, w0  
!
! Register flux evaluation in the PETSc profiler
!
      if(flag .eq. -1) then
        call PetscClassIdRegister('PetscFun3d', fClass, ierr)
        call PetscLogEventRegister(fLabel, fClass, FEvent, ierr)  
        flag = 1
      endif
!
! Start PETSc logging
!
      call PetscLogEventBegin(fEvent, ierr)
!
! Initialize the output residual vector to zero
!
      do i = 1, nnodes
        res(1, i) = 0.0d0
        res(2, i) = 0.0d0
        res(3, i) = 0.0d0
        res(4, i) = 0.0d0
      enddo   
!
! Initializations
!
      flops         = 0.0
      second        = 1.0d0
      firstEdgeLoc  = 1
      lastEdgeLoc   = nedge
      sTime         = 0.0
      eTime         = 0.0

      sTime =  omp_get_wtime()

#ifdef HW_COUNTER
      call PAPIF_start(event_set, ierr)
#endif
!
! Loop over all the faces and calculate the flux 
!
!$OMP   parallel  default(shared)                                       &
!$OMP&            private(i, node1, node2, xmean, ymean, zmean, xnorm,  &
!$OMP&                    ynorm, znorm, rlen, dot, X1, Y1, Z1, size,    &
!$OMP&                    X2, Y2, Z2, rx, ry, rz, pL, uL, vL, wL,       &
!$OMP&                    ubarL, pR, uR, vR, wR, ubarR, p, u, v, w,     &
!$OMP&                    ubar, phi1, phi2, phi3, phi4, phi5, phi6,     &
!$OMP&                    phi7, phi8, phi9, c2, c, eig1, eig2, eig3,    &
!$OMP&                    eig4, dp, du, dv, dw, ti11, ti21, ti31, ti41, &
!$OMP&                    dv1, dv2, dv3, dv4, r11, r21, r31, r41, r12,  &
!$OMP&                    r22, r32, r42, r13, r23, r33, r43, r14, r24,  &
!$OMP&                    r34, r44, t1, t2, t3, t4, fluxp1, fluxp2,     &
!$OMP&                    fluxp3, fluxp4, fluxm1, fluxm2, fluxm3,       &
!$OMP&                    fluxm4, res1, res2, res3, res4, c68, c18,     &
!$OMP&                    tid, firstEdgeLoc, lastEdgeLoc)               &
!$omp&            reduction(+: flops)
!
      tid             = omp_get_thread_num()
      firstEdgeLoc    = nedgeThreadLoc(tid)
      lastEdgeLoc     = nedgeThreadLoc(tid + 1) - 1
!
! An edge-based loop
!      
      do i = firstEdgeLoc, lastEdgeLoc

        node1 = eptr(1, i)
        node2 = eptr(2, i)        
!
! Calculate unit normal to face and length of face
!
        xmean   = .5d0 * (x(node1) + x(node2))
        ymean   = .5d0 * (y(node1) + y(node2))
        zmean   = .5d0 * (z(node1) + z(node2))        
        xnorm   = xn(i)
        ynorm   = yn(i)
        znorm   = zn(i)
        rlen    = ra(i)
!
! Now lets get our other 2 vectors
! For first vector, use {1,0,0} and subtract off the component
! in the direction of the face normal. If the inner product of
! {1,0,0} is close to unity, use {0,1,0}
!
        dot = xnorm
        if(abs(dot) .lt. 0.95d0) then
          X1 = 1. - dot * xnorm
          Y1 =    - dot * ynorm
          Z1 =    - dot * znorm
        else
          dot = ynorm
          X1  =       - dot * xnorm
          Y1  = 1.0d0 - dot * ynorm
          Z1  =       - dot * znorm
        end if
!
! Normalize the first vector
!
        size = sqrt(X1 * X1 + Y1 * Y1 + Z1 * Z1)
        X1   = X1 / size
        Y1   = Y1 / size
        Z1   = Z1 / size
!
! Take cross-product of normal and V1 to get V2
!
        X2 = ynorm * Z1 - znorm * Y1
        Y2 = znorm * X1 - xnorm * Z1
        Z2 = xnorm * Y1 - ynorm * X1
!
! Get variables on "left" and "right" side of face
!
        rx = second * (xmean - x(node1))
        ry = second * (ymean - y(node1))
        rz = second * (zmean - z(node1))

        pL = qnode(1, node1)  + grad(1, 1, node1) * rx                  &
     &                        + grad(2, 1, node1) * ry                  &
     &                        + grad(3, 1, node1) * rz

        uL = qnode(2, node1)  + grad(1, 2, node1) * rx                  &
     &                        + grad(2, 2, node1) * ry                  &
     &                        + grad(3, 2, node1) * rz 

        vL = qnode(3, node1)  + grad(1, 3, node1) * rx                  &
     &                        + grad(2, 3, node1) * ry                  &
     &                        + grad(3, 3, node1) * rz 

        wL = qnode(4, node1)  + grad(1, 4, node1) * rx                  &
     &                        + grad(2, 4, node1) * ry                  &
     &                        + grad(3, 4, node1) * rz 

        ubarL  = xnorm * uL + ynorm * vL + znorm * wL
         
        rx = second * (xmean - x(node2))
        ry = second * (ymean - y(node2))
        rz = second * (zmean - z(node2))
         
        pR   = qnode(1, node2)  + grad(1, 1, node2) * rx                &
     &                          + grad(2, 1, node2) * ry                &
     &                          + grad(3, 1, node2) * rz

        uR   = qnode(2, node2)  + grad(1, 2, node2) * rx                &
     &                          + grad(2, 2, node2) * ry                &
     &                          + grad(3, 2, node2) * rz

        vR   = qnode(3, node2)  + grad(1, 3, node2) * rx                &
     &                          + grad(2, 3, node2) * ry                &
     &                          + grad(3, 3, node2) * rz

        wR   = qnode(4, node2)  + grad(1, 4, node2) * rx                &
     &                          + grad(2, 4, node2) * ry                &
     &                          + grad(3, 4, node2) * rz 

        ubarR  = xnorm * uR + ynorm * vR + znorm * wR
!
! Compute averages
!
        p = .5d0 * (pL + pR)
        u = .5d0 * (uL + uR)
        v = .5d0 * (vL + vR)
        w = .5d0 * (wL + wR)

        ubar  = xnorm * u + ynorm * v + znorm * w

        phi1  = xnorm * beta + u * ubar
        phi2  = ynorm * beta + v * ubar
        phi3  = znorm * beta + w * ubar
        phi4  = Y2 * phi3 - Z2 * phi2
        phi5  = Z2 * phi1 - X2 * phi3
        phi6  = X2 * phi2 - Y2 * phi1
        phi7  = Z1 * phi2 - Y1 * phi3
        phi8  = X1 * phi3 - Z1 * phi1
        phi9  = Y1 * phi1 - X1 * phi2

        c2  = ubar * ubar + beta
        c   = sqrt(c2)
!
! Now compute eigenvalues, eigenvectors, and strengths
!
        eig1 = abs(ubar)
        eig2 = abs(ubar)
        eig3 = abs(ubar + c)
        eig4 = abs(ubar - c)

        dp = pr - pl
        du = ur - ul
        dv = vr - vl
        dw = wr - wl
!
! Components of T(inverse) (I will divide by c2 later)
!
        ti11 = -(u * phi4 + v * phi5 + w * phi6) / beta
        ti21 = -(u * phi7 + v * phi8 + w * phi9) / beta
        ti31 =  .5d0 * (c - ubar) / beta
        ti41 = -.5d0 * (c + ubar) / beta          
!
! jumps (T(inverse)*dq)
!
        dv1 = (ti11 * dp + phi4 * du + phi5 * dv + phi6 * dw) / c2
        dv2 = (ti21 * dp + phi7 * du + phi8 * dv + phi9 * dw) / c2
        dv3 = .5d0 * (2.d0*ti31*dp + xnorm*du + ynorm*dv + znorm*dw)/c2
        dv4 = .5d0 * (2.d0*ti41*dp + xnorm*du + ynorm*dv + znorm*dw)/c2
!
! Now get elements of T (call it r for now)
!
        r11 = 0.0d0
        r21 = X1
        r31 = Y1
        r41 = Z1
        r12 = 0.0d0
        r22 = X2
        r32 = Y2
        r42 = Z2
        r13 = c*beta
        r23 = xnorm*beta + u*(ubar + c)
        r33 = ynorm*beta + v*(ubar + c)
        r43 = znorm*beta + w*(ubar + c)
        r14 = -c*beta
        r24 = xnorm*beta + u*(ubar - c)
        r34 = ynorm*beta + v*(ubar - c)
        r44 = znorm*beta + w*(ubar - c)
!
! Calculate T* |lambda| *T(inverse)
!
        t1 = eig1*r11*dv1 + eig2*r12*dv2 + eig3*r13*dv3 + eig4*r14*dv4
        t2 = eig1*r21*dv1 + eig2*r22*dv2 + eig3*r23*dv3 + eig4*r24*dv4
        t3 = eig1*r31*dv1 + eig2*r32*dv2 + eig3*r33*dv3 + eig4*r34*dv4
        t4 = eig1*r41*dv1 + eig2*r42*dv2 + eig3*r43*dv3 + eig4*r44*dv4
!
! Modify to calculate .5(fl +fr) from nodes
! instead of extrapolated ones
!
        fluxp1 = rlen * beta * ubarL
        fluxp2 = rlen * (uL * ubarL + xnorm * pL)
        fluxp3 = rlen * (vL * ubarL + ynorm * pL)
        fluxp4 = rlen * (wL * ubarL + znorm * pL)
!
! Now the right side
!
        fluxm1 = rlen * beta * ubarR         
        fluxm2 = rlen * (uR * ubarR + xnorm * pR)
        fluxm3 = rlen * (vR * ubarR + ynorm * pR)
        fluxm4 = rlen * (wR * ubarR + znorm * pR)

        res1 = 0.5d0 * (fluxp1 + fluxm1 - rlen * t1)
        res2 = 0.5d0 * (fluxp2 + fluxm2 - rlen * t2)
        res3 = 0.5d0 * (fluxp3 + fluxm3 - rlen * t3)
        res4 = 0.5d0 * (fluxp4 + fluxm4 - rlen * t4)

        flops = flops + 318.0

        if(partitions(node1) .eq. tid) then

          res(1,node1) = res(1, node1) + res1
          res(2,node1) = res(2, node1) + res2
          res(3,node1) = res(3, node1) + res3
          res(4,node1) = res(4, node1) + res4
            
          flops = flops + 4.0

        endif

        if(partitions(node2) .eq. tid) then
            
          res(1, node2) = res(1, node2) - res1
          res(2, node2) = res(2, node2) - res2
          res(3, node2) = res(3, node2) - res3
          res(4, node2) = res(4, node2) - res4
           
          flops = flops + 4.0

        endif
      enddo
!
!$OMP end parallel
! 

#ifdef HW_COUNTER
      call PAPIF_stop(event_set, handler, ierr)
#endif

      eTime = omp_get_wtime()

      retTime = eTime - sTime

      call PetscLogFlops(flops, ierr)
      call PetscLogEventEnd(fEvent, ierr)
!
! Close contour over the boundaries 
!
      c68 = 0.75d0
      c18 = 0.125d0
!
! First do inviscid faces
!
      do i = 1, nnfacet
        node1 = isnode(f2ntn(i, 1))
        node2 = isnode(f2ntn(i, 2))
        node3 = isnode(f2ntn(i, 3)) 

        x1 = x(node1)
        y1 = y(node1)
        z1 = z(node1)
        p1 = qnode(1,node1)

        x2 = x(node2)
        y2 = y(node2)
        z2 = z(node2)
        p2 = qnode(1,node2)

        x3 = x(node3)
        y3 = y(node3)
        z3 = z(node3)
        p3 = qnode(1,node3)
               
        ax = x2 - x1
        ay = y2 - y1
        az = z2 - z1

        bx = x3 - x1
        by = y3 - y1
        bz = z3 - z1
!
! Normal points away from grid interior.
! Magnitude is 1/3 area of surface triangle.
!
        xnorm =-0.5d0*(ay*bz - az*by)/3.d0
        ynorm = 0.5d0*(ax*bz - az*bx)/3.d0
        znorm =-0.5d0*(ax*by - ay*bx)/3.d0

        pa = c68*p1 + c18*(p2 + p3)
        pb = c68*p2 + c18*(p3 + p1)
        pc = c68*p3 + c18*(p1 + p2)

        flops = flops + 35
        
        if (node1.le.nnodes) then
          res(2,node1) = res(2,node1) + xnorm*pa
          res(3,node1) = res(3,node1) + ynorm*pa
          res(4,node1) = res(4,node1) + znorm*pa
          flops = flops + 6
        endif

        if (node2.le.nnodes) then
          res(2,node2) = res(2,node2) + xnorm*pb
          res(3,node2) = res(3,node2) + ynorm*pb
          res(4,node2) = res(4,node2) + znorm*pb
          flops = flops + 6
        endif

        if (node3.le.nnodes) then
          res(2,node3) = res(2,node3) + xnorm*pc
          res(3,node3) = res(3,node3) + ynorm*pc
          res(4,node3) = res(4,node3) + znorm*pc
          flops = flops + 6
        endif
      enddo
!
! Now do far-field
!
      do i = 1, nfnode
        inode = ifnode(i)
!
!/*
! Get normal and "other" 2 vectors. Remember that fxn,fyn and fzn 
! has the magnitude of the face contained in it.
!*/
!
        xnorm   = fxn(i)
        ynorm   = fyn(i)
        znorm   = fzn(i)
        area    = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
        xnorm   = xnorm/area
        ynorm   = ynorm/area
        znorm   = znorm/area
!
! Now lets get our other 2 vectors
! For first vector, use {1,0,0} and subtract off the component
! in the direction of the face normal. If the inner product of
! {1,0,0} is close to unity, use {0,1,0}
!
        dot = xnorm
        if(abs(dot) .lt. 0.95d0) then
          X1 = 1.d0   - dot * xnorm
          Y1 =        - dot * ynorm
          Z1 =        - dot * znorm
        else
          dot = ynorm
          X1  =       - dot * xnorm
          Y1  = 1.d0  - dot * ynorm
          Z1  =       - dot * znorm
        end if
!
! Normalize the first vector (V1)
!
        size  = sqrt(X1 * X1 + Y1 * Y1 + Z1 * Z1)
        X1    = X1 / size
        Y1    = Y1 / size
        Z1    = Z1 / size
!
! Take cross-product of normal with V1 to get V2
!
        X2 = ynorm * Z1 - znorm * Y1
        Y2 = znorm * X1 - xnorm * Z1
        Z2 = xnorm * Y1 - ynorm * X1

!
! Calculate elements of T and T(inverse) evaluated at freestream
!
        ubar0 = xnorm*u0 + ynorm*v0 + znorm*w0
        c20   = ubar0*ubar0 + beta
        c0    = sqrt(c20)
        phi1  = xnorm*beta + u0*ubar0
        phi2  = ynorm*beta + v0*ubar0
        phi3  = znorm*beta + w0*ubar0
        phi4  = Y2*phi3 - Z2*phi2
        phi5  = Z2*phi1 - X2*phi3
        phi6  = X2*phi2 - Y2*phi1
        phi7  = Z1*phi2 - Y1*phi3
        phi8  = X1*phi3 - Z1*phi1
        phi9  = Y1*phi1 - X1*phi2

        t11 = 0.0d0
        t21 = X1
        t31 = Y1
        t41 = Z1

        t12 = 0.0d0
        t22 = X2
        t32 = Y2
        t42 = Z2

        t13 =  c0*beta
        t23 = xnorm*beta + u0*(ubar0 + c0)
        t33 = ynorm*beta + v0*(ubar0 + c0)
        t43 = znorm*beta + w0*(ubar0 + c0)

        t14 = -c0*beta
        t24 = xnorm*beta + u0*(ubar0 - c0)
        t34 = ynorm*beta + v0*(ubar0 - c0)
        t44 = znorm*beta + w0*(ubar0 - c0)

        ti11 = -(u0*phi4 + v0*phi5 + w0*phi6)/beta
        ti21 = -(u0*phi7 + v0*phi8 + w0*phi9)/beta
        ti31 =  .5d0*(c0 - ubar0)/beta
        ti41 = -.5d0*(c0 + ubar0)/beta
!
! Now, get the variables on the "inside"
!
        pi      = qnode(1,inode)
        ui      = qnode(2,inode)
        vi      = qnode(3,inode)
        wi      = qnode(4,inode)
        unorm   = xnorm*ui + ynorm*vi + znorm*wi
!
! If ubar is negative, take the reference condition from outside
!
        if(unorm.gt.0.0d0)then
          pr = pi
          ur = ui
          vr = vi
          wr = wi
        else
          pr = p0
          ur = u0
          vr = v0
          wr = w0
        end if
!
! Set rhs
!
        rhs1 = (ti11*pr + phi4*ur + phi5*vr + phi6*wr)/c20
        rhs2 = (ti21*pr + phi7*ur + phi8*vr + phi9*wr)/c20
        rhs3 = .5d0*(2.d0*ti31*pi + xnorm*ui + ynorm*vi + znorm*wi)/c20
        rhs4 = .5d0*(2.d0*ti41*p0 + xnorm*u0 + ynorm*v0 + znorm*w0)/c20
!
! Now do matrix multiplication to get values on boundary
!
        pb =                       t13*rhs3 + t14*rhs4
        ub = t21*rhs1 + t22*rhs2 + t23*rhs3 + t24*rhs4
        vb = t31*rhs1 + t32*rhs2 + t33*rhs3 + t34*rhs4
        wb = t41*rhs1 + t42*rhs2 + t43*rhs3 + t44*rhs4

        ubar = xnorm*ub + ynorm*vb + znorm*wb
        flops = flops + 180.0

        if (inode.le.nnodes) then
          res(1,inode) = res(1,inode)+area*beta*ubar
          res(2,inode) = res(2,inode)+area*(ub*ubar + xnorm*pb)
          res(3,inode) = res(3,inode)+area*(vb*ubar + ynorm*pb)
          res(4,inode) = res(4,inode)+area*(wb*ubar + znorm*pb)
          flops = flops + 18.0
        endif
       enddo

       call PetscLogFlops(flops,ierr)
!
! End of subroutine FLUX
!
      return
      end     
