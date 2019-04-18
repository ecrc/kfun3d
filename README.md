# KFUN3D [![GitHub version](https://badge.fury.io/gh/ecrc%2FKFUN3D.svg)](https://badge.fury.io/gh/ecrc%2FKFUN3D) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) #
Unstructured Computations on Emerging Architectures

**KFUN3D** (https://ecrc.github.io/KFUN3D/) is an unstructured computational aerodynamics software system with irregular memory accesses, which is optimized and engineered on a wide variety of multi- and many-core emerging high performance computing scalable architectures, which are expected to be the building blocks of energy-austere exascale systems, and on which algorithmic- and architecture-oriented optimizations are essential for achieving worthy performance. KFUN3D investigates several state-of-the-practice shared-memory optimization techniques applied to key computational kernels for the important problem class of unstructured meshes, one of the seven Colella “dwarves,” which are essential for science and engineering. It illustrates for a broad-spectrum of emerging microprocessor architectures as representatives of the compute units in contemporary leading supercomputers, identifying and addressing performance challenges without compromising the floating-point numerics of the original code. While the linear algebraic kernels are bottlenecked by memory bandwidth for even modest numbers of hardware cores sharing a common address space, the edge-based loop kernels, which arise in the control volume discretization of the conservation law residuals and in the formation of the preconditioner for the Jacobian by finite-differencing the conservation law residuals, are compute-intensive and effectively exploit contemporary multi- and many-core processing hardware. KFUN3D therefore employs low- and high-level algorithmic- and architecture-specific code optimizations and tuning in light of thread- and data-level parallelism, with a focus on strong thread scaling at the node-level. The approaches are based upon novel multi-level hierarchical workload distribution mechanisms of data across different compute units (from the address space down to the registers) within every hardware core. These optimizations are expected to be of value for many other unstructured mesh partial differential equation-based scientific applications as multi- and many-core architecture evolves.

## The Original NASA FUN3D Code ##

KFUN3D is is closely related to the export-controlled state-of-the-practice FUN3D (https://fun3d.larc.nasa.gov/) code from NASA used to analyze the low-speed and high-lift behaviors of an aircraft in take-off and landing configurations. FUN3D is a tetrahedral, vertex-centered, unstructured mesh research code written in FORTRAN for solving the Euler and Navier-Stokes equations of fluid flow in incompressible and compressible forms. It was originally developed in the early 1990s under the direction of W. Kyle Anderson at NASA Langley Research Center.

## PETSc-FUN3D ##

PETSc-FUN3D is a research fork of the incompressible and compressible Euler subset of the original FUN3D code that was restructured to employ Portable, Extensible Toolkit for Scientific Computation (PETSc) solver framework (https://www.mcs.anl.gov/petsc/) for the study of distributed-memory scaling. PETSc-FUN3D performance is thoroughly discussed, analyzed, and modeled in [1], which culminated in the 1999 Gordon Bell Special Prize undertaken jointly by the primary architect of FUN3D and members of the PETSc development team, which ran on the world’s then most powerful supercomputer, the Intel ASCI Red machine at Sandia.

### Under Construction ###

## Contact ##

* mohammed.farhan@kaust.edu.sa

## License ###

MIT License

## Acknowledgments ##

Support in the form of computing resources was provided by KAUST Extreme Computing Research Center, KAUST Supercomputing Laboratory, KAUST Information Technology Research Division, Intel Parallel Computing Centers, Isambard Project at University of Bristol, CUDA Center of Excellence at KAUST, Blue Waters Supercomputer at University of Illinois at Urbana-Champaign, and Cray Center of Excellence at KAUST.

## Papers ##

[1] W. D. Gropp, D. K. Kaushik, D. E. Keyes, and B. F. Smith, *High-performance parallel implicit CFD*, Parallel Computing, vol. 27, no. 4, pp. 337–362, 2001, Parallel Computing in Aerospace.
