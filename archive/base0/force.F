!
! ================================ FORCE ==============================72
!
!  Calculates the forces (Drag FORCE, LIFT FORCE, and the momentum)
!  
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "FORCE"
      subroutine FORCE( nnodes, nsfacet, qnode, x, y, z, isnode, f2nts, &
     &                  clift, cdrag, cmom )
      implicit none      
!
! Function parameters
!       
      integer           nnodes, nsfacet
      double precision  qnode(4, nnodes)
      double precision  x(1), y(1), z(1)
      integer           isnode(1)
      integer           f2nts(nsfacet, 4)
      double precision  clift, cdrag, cmom
!
! Function arguments
!
      integer           i
      integer           node1, node2, node3
      double precision  p0, c0, u0, v0, w0
      double precision  x1, y1, z1
      double precision  x2, y2, z2
      double precision  x3, y3, z3
      double precision  ax, ay, az
      double precision  bx, by, bz
      double precision  xnorm, ynorm
      double precision  p1, p2, p3
      double precision  xmid, ymid      
      double precision  dcx, dcy, cp, press
      double precision  xr, yr
!
! Fortran Global variables
!
      common /ivals/  p0, u0, v0, w0
!
! Initializations
!
      clift = 0.0d0
      cdrag = 0.0d0
      cmom  = 0.0d0      
      xr    = 0.25d0
      yr    = 0.0d0
!
! Solid boundaries
!       
      do i = 1, nsfacet
!
! A solid facet
!        
        node1 = isnode(f2nts(i, 1))
        node2 = isnode(f2nts(i, 2))
        node3 = isnode(f2nts(i, 3))
!
! Facet nodes coordinates
!
        x1  = x(node1)
        y1  = y(node1)
        z1  = z(node1)

        x2  = x(node2)
        y2  = y(node2)
        z2  = z(node2)

        x3  = x(node3)
        y3  = y(node3)
        z3  = z(node3)
!
! Delta coordinates in all directions
!               
        ax = x2 - x1
        ay = y2 - y1
        az = z2 - z1

        bx = x3 - x1
        by = y3 - y1
        bz = z3 - z1
!
! Norm points outward, away from grid interior.
! Norm magnitude is area of surface triangle.
!
        xnorm = -0.5d0 * (ay * bz - az * by)
        ynorm =  0.5d0 * (ax * bz - az * bx)
!
! Pressure values store at every face node
!
        p1    = qnode(1, node1)
        p2    = qnode(1, node2)
        p3    = qnode(1, node3)
 
        press = (p1 + p2 + p3) / 3.0d0
        cp    = 2.d0 * (press - 1.0d0)        

        dcx   = cp * xnorm
        dcy   = cp * ynorm

        xmid  = (x1 + x2 + x3)
        ymid  = (y1 + y2 + y3)
!
! Update clift, cdrag, cmom
!
        clift = clift - dcx * v0 + dcy * u0
        cdrag = cdrag + dcx * u0 + dcy * v0
        cmom  = cmom  + (xmid - xr) * dcy - (ymid - yr) * dcx

      enddo
!
! End of subroutine FORCE
!
      return
      end
