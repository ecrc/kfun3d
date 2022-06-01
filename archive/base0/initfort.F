!
! ============================== INITCOMM =============================72
!
! Initializes FORTRAN common blocks
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "INITCOMM"
      subroutine INITCOMM()
      implicit none

#include "fort.h"
!
! Artificial compressibility parameter
!
      double precision  beta
!
! Angle of attach
!      
      double precision  alpha
!
! Flow field variables
!
      double precision  p0, u0, v0, w0
!
! Local variables
!
      integer           ierr
      double precision  pi, conv
!
! Fortran common blocks
!
      common /coeff/ beta, alpha
      common /ivals/ p0, u0, v0, w0
!
! Initializations
!            
      pi    = 4.0d0 * datan(1.0d0)
      conv  = 180.0d0 / pi
      alpha = 3.0
      beta  = 15.0
!
! Initialize global coeff variables from the database
!
      call PetscOptionsGetReal( PETSC_NULL_OBJECT,                      &
     &                          PETSC_NULL_CHARACTER, '-alpha', alpha,  &
     &                          PETSC_NULL_CHARACTER, ierr )
      call PetscOptionsGetReal( PETSC_NULL_OBJECT,                      &
     &                          PETSC_NULL_CHARACTER, '-beta', beta,    &
     &                          PETSC_NULL_CHARACTER, ierr )

!      call PetscOptionsGetReal( PETSC_NULL_CHARACTER, '-alpha', alpha,  &
!     &                          PETSC_NULL_CHARACTER, ierr )
!      call PetscOptionsGetReal( PETSC_NULL_CHARACTER, '-beta', beta,    &
!     &                          PETSC_NULL_CHARACTER, ierr )

!
! Initialize global ivals variables
!      
      p0    = 1.0d0
      u0    = cos(alpha/conv)
      v0    = sin(alpha/conv)
      w0    = 0.0d0
!
! End of subroutine INITCOMM
!
      return
      end
!
! ================================== INIT =============================72
!
! The initial guess: Initializes the flow field
! This routine is called by the formInitialGuess function
!
! =====================================================================72
!
#undef  __FUNCT__
#define __FUNCT__ "INIT"
      subroutine INIT(nnodes, qnode)
      implicit none
!
! Function parameters
!
      integer           nnodes, nvnode
      double precision  qnode(4, nnodes)
!
! Function arguments
!
      integer           i
      double precision  p0, c0, u0, v0, w0 
!
! Fortran global ivals
!
      common /ivals/  p0, u0, v0, w0

!
! Initialize the qnode vector
!
      do i = 1, nnodes
        qnode(1, i) = p0
        qnode(2, i) = u0
        qnode(3, i) = v0
        qnode(4, i) = w0
      enddo
!
! End of subroutine INIT
!      
      return
      end
