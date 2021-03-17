# SPH
An object oriented approach to Smoothed particle hydrodynamics. 
The code was written as part of the Computational Astrophysics course at the University of Cologne.

Requirements:
1.Numpy
2.Matplotlib
3.Numba

The Kernel used is the M4Spline Kernel.
The focus of the code was in 1D and the 2D and 3D Kernels while included, are not what the code was used or tested for.

A few test cases for the code included are
1. 1D Glass
2. Sod Shock Tube
3. Sedov Blast

The code supports both Perodic and Non-Periodic Boundary conditions 
and also supports Viscosity simulation. Although the simulation of Gravity is not supported.

A significant amount of the functions are optimized using the Numba Library.
This is done mainly to optimize the functions that scale as O(n^2) with the number of particles,
such that they scale better. 

A few functions are not optimized using Numba because in the context of the development of the program,
The overhead of the just in tme compiler negated any gains.

Authors:
1.Sagar Ramchandani 
2.Mykhailo Lobodin
