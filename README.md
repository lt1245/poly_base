Python version of Miranda and Fackler's CompEcon toolbox for solving heterogenous agents models. <br />
The main goal of the toolkit is to have low level basis matrix representation for functional approximation. <br />
To use: import the toolkit in your folder and see the replication files to see how to generate the basis functions. <br />
Does not use any class structure. <br />
Most functions are jitted. <br />
Functions included and their main functionality: <br />
dprod - The direct sum (row-wise tensor) of two matrices <br />
cheb_nodes - 1-d Chebyshev node <br />
wealth_knot - 1-d Knots for wealth distributions <br />
cheb_basex - 1-d  Cheb basis matrix <br />
mono_basex - 1-d monomial basis matrix <br />
spli_basex - 1-d spline basis matrix <br />
cheb_diff - Differentiating matrix for Chebyshev polynomials <br />
mono_diff - Differentiating matrix for monomial <br />
funbas - Creates a multidimensional basis matrix <br />
goldenx - Vectorized golden section search to maximize a collection of univariate functions simultaneously <br />
equidistant_nonlin_grid - Creates a fine 1D grid, equidistant between the points of the original grid <br />
