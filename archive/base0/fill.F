!
! =================================== FILLA ===========================72
!
! Fill the nonzero term of the A matrix
! cfl is passed as a parameter
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "FILLA"
      subroutine FILLA( nnodes, nedge, nedgeThreads, nsnode, nfnode,    &
     &                  cfl, nThreads, eptr, qnode, xn, yn, zn, rl,     &
     &                  vol, cdt, sxn, syn, szn, fxn, fyn, fzn, isnode, &
     &                  ifnode, partitions, nedgeThreadLoc, A )
      implicit none     

#include "fort.h"
!
! Function parameters
!
      integer           nnodes, nedge, nedgeThreads
      integer           nsnode, nfnode
      double precision  cfl
      integer           nThreads
      integer           eptr(2, nedge)
      double precision  qnode(4, nnodes)
      double precision  xn(1), yn(1), zn(1), rl(1)
      double precision  vol(1)
      double precision  cdt(1)
      double precision  sxn(1), syn(1), szn(1)
      double precision  fxn(1), fyn(1), fzn(1)
      integer           isnode(1), ifnode(1)
      integer           partitions(nnodes)
      integer           nedgeThreadLoc(0: nThreads)
      Mat               A
!
! Function arguments
!
      integer           i, j, ir, ierr
      double precision  temp
      integer           firstEdgeLoc, lastEdgeLoc
      integer           tid, omp_get_thread_num
      integer           irow(3), icol(3)

      integer           rowIndx(3)
      integer           colIndx(2)
      
      integer           k, n, ic
      integer           node1, node2, inode
      double precision  val(32), val1(8,4)
      double precision  xnorm,ynorm,znorm,rlen,dot
      double precision  X1,Y1,Z1,X2,Y2,Z2,size,area
      double precision  pL,uL,vL,wL,ubarL,c2L,cL
      double precision  pR,uR,vR,wR,ubarR,c2R,cR
      double precision  pi,ui,vi,wi,unorm,c20,ubar0
      double precision  prp,uru,vrv,wrw
      double precision  p,u,v,w,ubar,c2,c
      double precision  eig1,eig2,eig3,eig4,eigeps
      double precision  phi1,phi2,phi3,phi4,phi5
      double precision  phi6,phi7,phi8,phi9
      double precision  rhs1,rhs1p,rhs1u,rhs1v,rhs1w
      double precision  rhs2,rhs2p,rhs2u,rhs2v,rhs2w
      double precision  rhs3,rhs3p,rhs3u,rhs3v,rhs3w
      double precision  rhs4,rhs4p,rhs4u,rhs4v,rhs4w
      double precision  c2inv
      double precision  y11,y21,y31,y41,y12,y22,y32,y42
      double precision  y13,y23,y33,y43,y14,y24,y34,y44
      double precision  t11,t21,t31,t41,t12,t22,t32,t42
      double precision  t13,t23,t33,t43,t14,t24,t34,t44
      double precision  ti11,ti21,ti31,ti41
      double precision  ti12,ti22,ti32,ti42
      double precision  ti13,ti23,ti33,ti43
      double precision  ti14,ti24,ti34,ti44
      double precision  a11,a21,a31,a41,a12,a22,a32,a42
      double precision  a13,a23,a33,a43,a14,a24,a34,a44
      double precision  pb,pbp,pbu,pbv,pbw
      double precision  ub,ubp,ubu,ubv,ubw
      double precision  vb,vbp,vbu,vbv,vbw
      double precision  wb,wbp,wbu,wbv,wbw
      double precision  unormb,unormbp,unormbu
      double precision  unormbv,unormbw
      double precision  p0,c0,u0,v0,w0
      double precision  beta, alpha
      double precision  flops
!
! Fortran global variables
!
      common /coeff/  beta, alpha
      common /ivals/  p0, u0, v0, w0
!
! Predefined variables for easy access
!
#define   dfp(a, b) val1(b, a)
#define   dfm(a, b) val1(b + 4, a)
!#define   xn(i)     xyzn(1, i)
!#define   yn(i)     xyzn(2, i)
!#define   zn(i)     xyzn(3, i)
!#define   rl(i)     xyzn(4, i)      
!
! Initializations
!
      flops         = 0.0
      firstEdgeLoc  = 1
      lastEdgeLoc   = nedge
!
! First just zero it out 
! Zeros all entries of a matrix. For sparse
! matrices this routine retains the old nonzero structure
!
      call MatZeroEntries(A, ierr)
!
! Loop over the nodes to compute the local indices of each row 
! and column to insert the values using PETSc routine 
!      
      do i = 1, nnodes
        temp = vol(i) / (cfl * cdt(i))
        do j = 1, 4
          ir = j - 1 + 4 * (i - 1)
!
! Inserts or adds values into certain locations of a matrix, 
! using a local ordering of the nodes
!          
          call MatSetValuesLocal( A, 1, ir,  1, ir, temp,               &
     &                            ADD_VALUES, ierr )
        enddo
      enddo

      flops = flops + 2.0 * nnodes
! 
! Loop over the edges and fill the matrix
! Now fill A from interior contributions 
!
! An edge-based loop
!
!!$OMP   parallel  default(shared)                                       &
!!$OMP&            private(i, node1, node2, xnorm,  ynorm, znorm, rlen,  &
!!$OMP&                    dot, X1, Y1, Z1, size, X2, Y2, Z2, pL, uL,    &
!!$OMP&                    vL, wL, ubarL, c2L, cL, pR, uR, vR, wR,       &
!!$OMP&                    ubarR, c2R, cR, p, u, v, w, ubar, c2, c,      &
!!$OMP&                    eig1, eig2, eig3, eig4, eigeps, phi1, phi2,   &
!!$OMP&                    phi3, phi4, phi5, phi6, phi7, phi8, phi9,     &
!!$OMP&                    c2inv, y11, y21, y31, y41, y12, y22, y32,     &
!!$OMP&                    y42, y13, y23, y33, y43, y14, y24, y34, y44,  &
!!$OMP&                    t11, t21, t31, t41, t12, t22, t32, t42, t13,  &
!!$OMP&                    t23, t33, t43, t14, t24, t34, t44, a11, a21,  &
!!$OMP&                    a31, a41, a12, a22, a32, a42, a13, a23, a33,  &
!!$OMP&                    a43, a14, a24, a34, a44, tid, firstEdgeLoc,   &
!!$OMP&                    lastEdgeLoc, val1, colIndx, rowIndx, j, k)    &
!!$omp&            reduction(+: flops)
!!
!      tid           = omp_get_thread_num()
!      firstEdgeLoc  = nedgeThreadLoc(tid)
!      lastEdgeLoc   = nedgeThreadLoc(tid + 1) - 1

      do i = firstEdgeLoc, lastEdgeLoc

        node1 = eptr(1, i)
        node2 = eptr(2, i)

        colIndx(1) = node1 - 1
        colIndx(2) = node2 - 1

        rowIndx(1) = node1 - 1
        rowIndx(2) = node2 - 1

        xnorm  = xn(i)
        ynorm  = yn(i)
        znorm  = zn(i)
        rlen   = rl(i)
!
! Now lets get our other 2 vectors
! For first vector, use {1,0,0} and subtract off the component
! in the direction of the face normal. If the inner product of
! {1,0,0} is close to unity, use {0,1,0}
!
        dot = xnorm
        if(abs(dot) .lt. 0.95d0) then
          X1  = 1.d0  - dot * xnorm
          Y1  =       - dot * ynorm
          Z1  =       - dot * znorm
        else
          dot = ynorm
          X1  =       - dot * xnorm
          Y1  = 1.d0  - dot * ynorm
          Z1  =       - dot * znorm
        end if
!
! Normalize the first vector
!
        size  = sqrt(X1 * X1 + Y1 * Y1 + Z1 * Z1)
        X1    = X1 / size
        Y1    = Y1 / size
        Z1    = Z1 / size
!
! Take cross-product of normal and V1 to get V2
!
        X2 = ynorm * Z1 - znorm * Y1
        Y2 = znorm * X1 - xnorm * Z1
        Z2 = xnorm * Y1 - ynorm * X1
!
! Variables on left
!
        pL    = qnode(1, node1)
        uL    = qnode(2, node1)
        vL    = qnode(3, node1)
        wL    = qnode(4, node1)
        ubarL = xnorm * uL + ynorm * vL + znorm * wL
        c2L   = ubarL * ubarL + beta
        cL    = sqrt(c2L)
!
! Variables on right
!
        pR    = qnode(1, node2)
        uR    = qnode(2, node2)
        vR    = qnode(3, node2)
        wR    = qnode(4, node2)
        ubarR = xnorm * uR + ynorm * vR + znorm * wR
        c2R   = ubarR * ubarR + beta
        cR    = sqrt(c2R)
!
! Regular Jacobians on left
!
        dfp(1, 1) = 0.0d0
        dfp(1, 2) = rlen * beta * xnorm
        dfp(1, 3) = rlen * beta * ynorm
        dfp(1, 4) = rlen * beta * znorm
!
        dfp(2, 1) = rlen * xnorm
        dfp(2, 2) = rlen * (ubarL + xnorm * uL)
        dfp(2, 3) = rlen * ynorm * uL
        dfp(2, 4) = rlen * znorm * uL
!
        dfp(3, 1) = rlen * ynorm
        dfp(3, 2) = rlen * xnorm * vL
        dfp(3, 3) = rlen * (ubarL + ynorm * vL)
        dfp(3, 4) = rlen * znorm * vL
!
        dfp(4, 1) = rlen * znorm
        dfp(4, 2) = rlen * xnorm * wL
        dfp(4, 3) = rlen * ynorm * wL
        dfp(4, 4) = rlen * (ubarL + znorm * wL)
!
! Now compute eigenvalues and |A| from averaged variables
! Avergage variables
!
        p     = .5d0 * (pL + pR)
        u     = .5d0 * (uL + uR)
        v     = .5d0 * (vL + vR)
        w     = .5d0 * (wL + wR)
        ubar  = xnorm * u + ynorm * v + znorm * w
        c2    = ubar * ubar + beta
        c     = sqrt(c2)

        eig1  = abs(ubar)
        eig2  = abs(ubar)
        eig3  = abs(ubar + c)
        eig4  = abs(ubar - c)
!
! Put in the eigenvalue smoothing stuff
!
        eigeps  = .1d0 * (abs(ubar) + abs(c))
           
        eig1 = rlen * eig1
        eig2 = rlen * eig2
        eig3 = rlen * eig3
        eig4 = rlen * eig4

        phi1  = xnorm * beta + u * ubar
        phi2  = ynorm * beta + v * ubar
        phi3  = znorm * beta + w * ubar
        phi4  = Y2 * phi3 - Z2 * phi2
        phi5  = Z2 * phi1 - X2 * phi3
        phi6  = X2 * phi2 - Y2 * phi1
        phi7  = Z1 * phi2 - Y1 * phi3
        phi8  = X1 * phi3 - Z1 * phi1
        phi9  = Y1 * phi1 - X1 * phi2
!
! Components of T(inverse) (call this y)
!
        c2inv = 1.d0/c2
        y11 = -c2inv*(u*phi4 + v*phi5 + w*phi6)/beta
        y21 = -c2inv*(u*phi7 + v*phi8 + w*phi9)/beta
        y31 =  .5d0*c2inv*(c-ubar)/beta
        y41 = -.5d0*c2inv*(c+ubar)/beta

        y12 =  c2inv*phi4
        y22 =  c2inv*phi7
        y32 =  c2inv*.5d0*xnorm
        y42 =  c2inv*.5d0*xnorm

        y13 =  c2inv*phi5
        y23 =  c2inv*phi8
        y33 =  c2inv*.5d0*ynorm
        y43 =  c2inv*.5d0*ynorm

        y14 =  c2inv*phi6
        y24 =  c2inv*phi9
        y34 =  c2inv*.5d0*znorm
        y44 =  c2inv*.5d0*znorm
!
! Now get elements of T 
!
        t11 = 0.0d0
        t21 = X1
        t31 = Y1
        t41 = Z1

        t12 = 0.0d0
        t22 = X2
        t32 = Y2
        t42 = Z2

        t13 = c*beta
        t23 = xnorm*beta + u*(ubar + c)
        t33 = ynorm*beta + v*(ubar + c)
        t43 = znorm*beta + w*(ubar + c)
 
        t14 = -c*beta
        t24 = xnorm*beta + u*(ubar - c)
        t34 = ynorm*beta + v*(ubar - c)
        t44 = znorm*beta + w*(ubar - c)
!
! Compute T*|lambda|*T(inv)
!
        a11=eig1*t11*y11 + eig2*t12*y21 + eig3*t13*y31 + eig4*t14*y41
        a12=eig1*t11*y12 + eig2*t12*y22 + eig3*t13*y32 + eig4*t14*y42
        a13=eig1*t11*y13 + eig2*t12*y23 + eig3*t13*y33 + eig4*t14*y43
        a14=eig1*t11*y14 + eig2*t12*y24 + eig3*t13*y34 + eig4*t14*y44

        a21=eig1*t21*y11 + eig2*t22*y21 + eig3*t23*y31 + eig4*t24*y41
        a22=eig1*t21*y12 + eig2*t22*y22 + eig3*t23*y32 + eig4*t24*y42
        a23=eig1*t21*y13 + eig2*t22*y23 + eig3*t23*y33 + eig4*t24*y43
        a24=eig1*t21*y14 + eig2*t22*y24 + eig3*t23*y34 + eig4*t24*y44

        a31=eig1*t31*y11 + eig2*t32*y21 + eig3*t33*y31 + eig4*t34*y41
        a32=eig1*t31*y12 + eig2*t32*y22 + eig3*t33*y32 + eig4*t34*y42
        a33=eig1*t31*y13 + eig2*t32*y23 + eig3*t33*y33 + eig4*t34*y43
        a34=eig1*t31*y14 + eig2*t32*y24 + eig3*t33*y34 + eig4*t34*y44

        a41=eig1*t41*y11 + eig2*t42*y21 + eig3*t43*y31 + eig4*t44*y41
        a42=eig1*t41*y12 + eig2*t42*y22 + eig3*t43*y32 + eig4*t44*y42
        a43=eig1*t41*y13 + eig2*t42*y23 + eig3*t43*y33 + eig4*t44*y43
        a44=eig1*t41*y14 + eig2*t42*y24 + eig3*t43*y34 + eig4*t44*y44
!
! Form .5*(A + |A|) 
!
        dfp(1,1) = .5d0*(dfp(1,1) + a11)
        dfp(1,2) = .5d0*(dfp(1,2) + a12)
        dfp(1,3) = .5d0*(dfp(1,3) + a13)
        dfp(1,4) = .5d0*(dfp(1,4) + a14)

        dfp(2,1) = .5d0*(dfp(2,1) + a21)
        dfp(2,2) = .5d0*(dfp(2,2) + a22)
        dfp(2,3) = .5d0*(dfp(2,3) + a23)
        dfp(2,4) = .5d0*(dfp(2,4) + a24)

        dfp(3,1) = .5d0*(dfp(3,1) + a31)
        dfp(3,2) = .5d0*(dfp(3,2) + a32)
        dfp(3,3) = .5d0*(dfp(3,3) + a33)
        dfp(3,4) = .5d0*(dfp(3,4) + a34)

        dfp(4,1) = .5d0*(dfp(4,1) + a41)
        dfp(4,2) = .5d0*(dfp(4,2) + a42)
        dfp(4,3) = .5d0*(dfp(4,3) + a43)
        dfp(4,4) = .5d0*(dfp(4,4) + a44)
!
! Now get regular Jacobians on right
!
        dfm(1,1) = 0.0d0
        dfm(1,2) = rlen*beta*xnorm
        dfm(1,3) = rlen*beta*ynorm
        dfm(1,4) = rlen*beta*znorm

        dfm(2,1) = rlen*xnorm
        dfm(2,2) = rlen*(ubarR + xnorm*uR)
        dfm(2,3) = rlen*ynorm*uR
        dfm(2,4) = rlen*znorm*uR

        dfm(3,1) = rlen*ynorm
        dfm(3,2) = rlen*xnorm*vR
        dfm(3,3) = rlen*(ubarR + ynorm*vR)
        dfm(3,4) = rlen*znorm*vR

        dfm(4,1) = rlen*znorm
        dfm(4,2) = rlen*xnorm*wR
        dfm(4,3) = rlen*ynorm*wR
        dfm(4,4) = rlen*(ubarR + znorm*wR)
!
! Form .5*(A - |A|) 
!
        dfm(1,1) = .5d0*(dfm(1,1) - a11)
        dfm(1,2) = .5d0*(dfm(1,2) - a12)
        dfm(1,3) = .5d0*(dfm(1,3) - a13)
        dfm(1,4) = .5d0*(dfm(1,4) - a14)

        dfm(2,1) = .5d0*(dfm(2,1) - a21)
        dfm(2,2) = .5d0*(dfm(2,2) - a22)
        dfm(2,3) = .5d0*(dfm(2,3) - a23)
        dfm(2,4) = .5d0*(dfm(2,4) - a24)

        dfm(3,1) = .5d0*(dfm(3,1) - a31)
        dfm(3,2) = .5d0*(dfm(3,2) - a32)
        dfm(3,3) = .5d0*(dfm(3,3) - a33)
        dfm(3,4) = .5d0*(dfm(3,4) - a34)

        dfm(4,1) = .5d0*(dfm(4,1) - a41)
        dfm(4,2) = .5d0*(dfm(4,2) - a42)
        dfm(4,3) = .5d0*(dfm(4,3) - a43)
        dfm(4,4) = .5d0*(dfm(4,4) - a44)

        flops = flops + 465.0
!
!        if(partitions(node1) .eq. tid) then
!
!!$OMP critical 
!
        call MatSetValuesBlockedLocal(A, 1, rowIndx(1), 2, colIndx,     &
     &                                    val1, ADD_VALUES, ierr)
!
!!$OMP end critical
!

!        endif

!        if(partitions(node2) .eq. tid) then
!
! Exchange elements in place
!
        do j = 1, 4
          do k = 1, 8
            val1(k, j) = -val1(k, j)
          enddo
        enddo
!
!!$OMP critical
!
        call MatSetValuesBlockedLocal(A, 1, rowIndx(2), 2, colIndx,     &
     &                                    val1, ADD_VALUES, ierr)
!
!!$OMP end critical
!
!        endif  
      enddo
!!
!!$OMP end parallel
!!
!
! Now loop over the boundaries
!
! For inviscid surface add contribution from pressure
!
      do 1070 i = 1, nsnode
        inode = isnode(i)
        if (inode .le. nnodes) then
          xnorm = sxn(i)
          ynorm = syn(i)
          znorm = szn(i)
          area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
          xnorm = xnorm/area
          ynorm = ynorm/area
          znorm = znorm/area
!
          val(1) = area*xnorm
          val(2) = area*ynorm
          val(3) = area*znorm
          irow(1) = 4*(inode-1) + 1
          irow(2) = 4*(inode-1) + 2
          irow(3) = 4*(inode-1) + 3
          ic = 4*(inode - 1)
!          
          call MatSetValuesLocal(A,3,irow,1,ic,val,ADD_VALUES,ierr)
!          
          flops = flops + 12.0
        endif
 1070 continue
!
!  Far-field boundary points
!
      do 1090 i = 1,nfnode
        inode  = ifnode(i)
        if (inode .le. nnodes) then
          xnorm = fxn(i)
          ynorm = fyn(i)
          znorm = fzn(i)
          area  = sqrt(xnorm*xnorm + ynorm*ynorm + znorm*znorm)
          xnorm = xnorm/area
          ynorm = ynorm/area
          znorm = znorm/area
!
! Now lets get our other 2 vectors
! For first vector, use {1,0,0} and subtract off the component
! in the direction of the face normal. If the inner product of
! {1,0,0} is close to unity, use {0,1,0}
!
          dot = xnorm
          if(abs(dot).lt.0.95d0)then
            X1 = 1.d0 - dot*xnorm
            Y1 =    - dot*ynorm
            Z1 =    - dot*znorm
          else
            dot = ynorm
            X1 =    - dot*xnorm
            Y1 = 1.d0 - dot*ynorm
            Z1 =    - dot*znorm
          end if
!
! Normalize the first vector (V1)
!
          size = sqrt(X1*X1 + Y1*Y1 + Z1*Z1)
          X1 = X1/size
          Y1 = Y1/size
          Z1 = Z1/size
!
! Take cross-product of normal with V1 to get V2
!
          X2 = ynorm*Z1 - znorm*Y1
          Y2 = znorm*X1 - xnorm*Z1
          Z2 = xnorm*Y1 - ynorm*X1
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

          ti11 = -(u0*phi4 + v0*phi5 + w0*phi6)/beta/c20
          ti21 = -(u0*phi7 + v0*phi8 + w0*phi9)/beta/c20
          ti31 = (c0 - ubar0)/(2.d0*beta*c20)
          ti41 = -(c0 + ubar0)/(2.d0*beta*c20)

          ti12 = phi4/c20
          ti22 = phi7/c20
          ti32 = .5d0*xnorm/c20
          ti42 = .5d0*xnorm/c20

          ti13 = phi5/c20
          ti23 = phi8/c20
          ti33 = .5d0*ynorm/c20
          ti43 = .5d0*ynorm/c20

          ti14 = phi6/c20
          ti24 = phi9/c20
          ti34 = .5d0*znorm/c20
          ti44 = .5d0*znorm/c20
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
!
          if(unorm.gt.0.0d0)then
            pr = pi
            prp = 1.0d0
            ur = ui
            uru = 1.0d0
            vr = vi
            vrv = 1.0d0
            wr = wi
            wrw = 1.0d0
          else
            pr = p0
            prp = 0.0d0
            ur = u0
            uru = 0.0d0
            v r = v0
            vrv = 0.0d0
            wr = w0
            wrw = 0.0d0
          end if
!
! Set rhs
!
          rhs1 = ti11*pr + ti12*ur + ti13*vr + ti14*wr
          rhs1p = ti11*prp
          rhs1u = ti12*uru
          rhs1v = ti13*vrv
          rhs1w = ti14*wrw
          rhs2 = ti21*pr + ti22*ur + ti23*vr + ti24*wr
          rhs2p = ti21*prp
          rhs2u = ti22*uru
          rhs2v = ti23*vrv
          rhs2w = ti24*wrw
          rhs3 = ti31*pi + ti32*ui + ti33*vi + ti34*wi
          rhs3p = ti31
          rhs3u = ti32
          rhs3v = ti33
          rhs3w = ti34
          rhs4 = ti41*p0 + ti42*u0 + ti43*v0 + ti44*w0
          rhs4p = 0.0d0
          rhs4u = 0.0d0
          rhs4v = 0.0d0
          rhs4w = 0.0d0
!
! Now do matrix multiplication to get values on boundary
!
          pb =                       t13*rhs3 + t14*rhs4
          pbp =                         t13*rhs3p !+ t14*rhs4p
          pbu =                         t13*rhs3u !+ t14*rhs4u
          pbv =                         t13*rhs3v !+ t14*rhs4v
          pbw =                         t13*rhs3w !+ t14*rhs4w
          ub = t21*rhs1 + t22*rhs2 + t23*rhs3 + t24*rhs4
          ubp = t21*rhs1p + t22*rhs2p + t23*rhs3p !+ t24*rhs4p
          ubu = t21*rhs1u + t22*rhs2u + t23*rhs3u !+ t24*rhs4u
          ubv = t21*rhs1v + t22*rhs2v + t23*rhs3v !+ t24*rhs4v
          ubw = t21*rhs1w + t22*rhs2w + t23*rhs3w !+ t24*rhs4w
          vb = t31*rhs1 + t32*rhs2 + t33*rhs3 + t34*rhs4
          vbp = t31*rhs1p + t32*rhs2p + t33*rhs3p !+ t34*rhs4p
          vbu = t31*rhs1u + t32*rhs2u + t33*rhs3u !+ t34*rhs4u
          vbv = t31*rhs1v + t32*rhs2v + t33*rhs3v !+ t34*rhs4v
          vbw = t31*rhs1w + t32*rhs2w + t33*rhs3w !+ t34*rhs4w
          wb = t41*rhs1 + t42*rhs2 + t43*rhs3 + t44*rhs4
          wbp = t41*rhs1p + t42*rhs2p + t43*rhs3p !+ t44*rhs4p
          wbu = t41*rhs1u + t42*rhs2u + t43*rhs3u !+ t44*rhs4u
          wbv = t41*rhs1v + t42*rhs2v + t43*rhs3v !+ t44*rhs4v
          wbw = t41*rhs1w + t42*rhs2w + t43*rhs3w !+ t44*rhs4w

          unormb = xnorm*ub + ynorm*vb + znorm*wb
          unormbp = xnorm*ubp + ynorm*vbp + znorm*wbp
          unormbu = xnorm*ubu + ynorm*vbu + znorm*wbu
          unormbv = xnorm*ubv + ynorm*vbv + znorm*wbv
          unormbw = xnorm*ubw + ynorm*vbw + znorm*wbw

!
! Now add contribution to lhs
!

          val(1) = area*beta*unormbp
          val(2) = area*beta*unormbu
          val(3) = area*beta*unormbv
          val(4) = area*beta*unormbw
!
          val(5) = area*(ub*unormbp + unormb*ubp + xnorm*pbp)
          val(6) = area*(ub*unormbu + unormb*ubu + xnorm*pbu)
          val(7) = area*(ub*unormbv + unormb*ubv + xnorm*pbv)
          val(8) = area*(ub*unormbw + unormb*ubw + xnorm*pbw)
!
          val(9) = area*(vb*unormbp + unormb*vbp + ynorm*pbp)
          val(10) = area*(vb*unormbu + unormb*vbu + ynorm*pbu)
          val(11) = area*(vb*unormbv + unormb*vbv + ynorm*pbv)
          val(12) = area*(vb*unormbw + unormb*vbw + ynorm*pbw)
!
          val(13) = area*(wb*unormbp + unormb*wbp + znorm*pbp)
          val(14) = area*(wb*unormbu + unormb*wbu + znorm*pbu)
          val(15) = area*(wb*unormbv + unormb*wbv + znorm*pbv)
          val(16) = area*(wb*unormbw + unormb*wbw + znorm*pbw)
!
          irow(1) = inode - 1
          call MatSetValuesBlockedLocal(A,1,irow,1,irow,val,            &
     &                                 ADD_VALUES,ierr)

          flops = flops + 337.0
        endif
 1090 continue
!
! Assemble matrix
!
      call PetscLogFlops(flops,ierr)
      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)
!      
! End of subroutine FILLA
!
      return
      end
