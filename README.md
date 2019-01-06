Python version of Miranda and Fackler's CompEcon toolbox for solving heterogenous agents models.
The main goal of the toolkit is to have low level basis matrix representation for functional approximation.
To use: import the toolkit in your folder and see the replication files to see how to generate the basis functions. 
Does not use any class structure.
Most functions are jitted. 
Functions included and their main functionality:
dprod - The direct sum (row-wise tensor) of two matrices
cheb_nodes - 1-d Chebyshev node
wealth_knot - 1-d Knots for wealth distributions
cheb_basex - 1-d  Cheb basis matrix
mono_basex - 1-d monomial basis matrix
spli_basex - 1-d spline basis matrix
cheb_diff - Differentiating matrix for Chebyshev polynomials
mono_diff - Differentiating matrix for monomial
funbas - Creates a multidimensional basis matrix
goldenx - Vectorized golden section search to maximize a collection of univariate functions simultaneously
equidistant_nonlin_grid - Creates a fine 1D grid, equidistant between the points of the original grid 
