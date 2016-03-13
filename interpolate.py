import numpy as np
from numba import jit
@jit
def dprod(A,B):
    """calculate tensor product of two matrices
    with the same number of rows
    
    Parameters
    ----------
    A : array_like
        In multidimensional approximation this is the 
        n - 1 dimensional basis matrix
    B: array_like
        In multidimensional approximation this is the 
        1 dimensional basis matrix

    Returns
    -------
    Res :  ndarray
        Matrix of shape (m,n), where m = no. rows in A and
        n = no. columns in A * no. columns in B

    Notes
    -----
    """

    nobsa , na = A.shape
    nobsb , nb = B.shape
    Res = np.empty((nobsa,nb*na))
    if nobsa != nobsb:
        return 'A and B must have same number of rows'
    for t in range(nobsa):
        for ia in range(na):
            for ib in range(nb):            
                Res[t,nb*(ia-1)+ib] = A[t,ia] * B[t, ib]
    return Res

@jit
def cheb_nodes(p, nodetype=0):

    """Chebyshev nodes - for 1 dimension.

    Returns Chebyshev nodes

    Parameters
    ----------
    p : array_like
        Parameter array containing:
         - the number of nodes
         - the lower bound of the approximation
         - the upper bound of the approximation
    nodetype : int
         - if 0 (default value) then use the usual nodes
         - if 1 then extend it to the endpoints
         - if 2 then us Lobatto nodes

    Returns
    -------
    x :  an array containing the Chebyshev nodes

    Notes
    -----
    """
    n , a , b = p[0] , p[1] , p[2]
    s = (b-a) / 2 
    m = (b+a) / 2  
    if (nodetype < 2):  # usual nodes
        k = np.pi*np.linspace(0.5,n-0.5,n)  
        x = m - np.cos(k[0:n]/n) * s  
        if (nodetype == 1):  # Extend nodes to endpoints
            aa = x[0]  
            bb = x[-1]  
            x = (bb*a - aa*b)/(bb-aa) + (b-a)/(bb-aa)*x
    else: # Lobatto nodes
        k = np.pi*np.linspace(0,n-1,n)
        x = m - np.cos(k[0:n]/(n-1)) * s
    return x
@jit
def cheb_basex(p, x): 
    """Cheb basis matrix - for 1 dimension.

    Returns a matrix whose columns are the values of the (first kind) Chebyshev 
    polynomial of maximum degree n-1 evaluated at the points `x`. Degree 0 is 
    the constant 0.

    Parameters
    ----------
    p : array_like
        Parameter array containing:
         - the order of approximation - the highest degree polynomial is n-1
         - the lower bound of the approximation
         - the upper bound of the approximation
    x : array_like
        Points at which to evaluate the b-splines.

    Returns
    -------
    bas : ndarray
        Matrix of shape (m,n), where ``m = len(x)`` and
        ``n - 1 = order(polynomial)``

    Notes
    -----
    Orthogonal polynomial
    """
    n , a , b = p[0] , p[1] , p[2]
    z = (2/(b-a)) * (x-(a+b)/2)
    m = z.shape[0]
    bas = np.empty((m, n));
    bas[:, 0] = 1.0
    bas[:, 1] = z[:]
    z = 2 * z
    for i in range(m):
        for j in range(2,n):
            bas[i, j] = z[i] * bas[i, j-1] - bas[i, j-2]
    return bas
@jit
def mono_basex(p, x): 
    """Monomials basis matrix- for 1 dimension.

    Returns a matrix whose columns are the values of the monomials of maximum 
    order n - 1 evaluated at the points `x`. Degree 0 is the constant 0.

    Parameters
    ----------
    p : array_like
        Parameter array containing:
         - the order of approximation - the highest degree polynomial is n-1
         - the lower bound of the approximation
         - the upper bound of the approximation
    x : array_like
        Points at which to evaluate the b-splines.

    Returns
    -------
    bas : ndarray
        Matrix of shape (m,n), where ``m = len(x)`` and
        ``n - 1 = order(polynomial)``

    Notes
    -----
    Also known as the Vandermonde matrix
    """
    n , a , b = p[0] , p[1] , p[2]
    z = (2/(b-a)) * (x-(a+b)/2)
    m = z.shape[0]
    bas = np.empty((m, n));
    bas[:, 0] = 1.0
    for i in range(m):
        for j in range(1,n):
            bas[i, j] = z[i] * bas[i, j-1]
    return bas
from scipy.interpolate import fitpack as spl
@jit
def spli_basex(p, x ,knots=None , deg = 3 , order = 0 ):
    """Vandermonde type matrix for splines.

    Returns a matrix whose columns are the values of the b-splines of deg
    `deg` associated with the knot sequence `knots` evaluated at the points
    `x`.

    Parameters
    ----------
    p : array_like
        Parameter array containing:
         - the number of knots
         - the lower bound of the approximation
         - the upper bound of the approximation
    x : array_like
        Points at which to evaluate the b-splines.
    deg : int
        Degree of the splines.
        Default: cubic splines
    knots : array_like
        List of knots. The convention here is that the interior knots have
        been extended at both ends by ``deg + 1`` extra knots - see augbreaks.
        If not given the default is equidistant grid
    order : int
        Evaluate the derivative of the spline
    Returns
    -------
    vander : ndarray
        Vandermonde like matrix of shape (m,n), where ``m = len(x)`` and
        ``n = len(augbreaks) - deg - 1``

    Notes
    -----
    The knots exending the interior points are usually taken to be the same
    as the endpoints of the interval on which the spline will be evaluated.

    """
    n , a , b = p[0] , p[1] , p[2]
    if knots is None:
        knots = np.linspace(a , b , n + deg + 1 - 2*deg)
    augbreaks = np.concatenate(( a * np.ones((deg)),knots, b * np.ones((deg))))
    m = len(augbreaks) - deg - 1
    v = np.empty((m, len(x)))
    d = np.eye(m, len(augbreaks))
    for i in range(m):
        v[i] = spl.splev(x, (augbreaks, d[i], deg),order)
    return v.T
@jit
def cheb_diff(p):
    """Differentiating matrix for Chebyshev polynomials

    Returns a matrix which multiplied from the right with the coefficients
    of the Chebyshev polynomial returns the derivative of the respective 
    Chebyshev polynomial. Can be used instead to evaluate the basis matrix of
    the derivative of a Chebyshev polynomial.

    Parameters
    ----------
    p : array_like
        Parameter array containing:
         - the number of knots = degree + 1 of the polynomial
         - the lower bound of the approximation
         - the upper bound of the approximation
    Returns
    -------
    D : ndarray
       Returns an upper triangular derivative operator matrix 

    Notes
    -----
    See usage in funbas
    """
    n , a , b = p[0] , p[1] , p[2]  
    D = np.zeros((n,n))
    for j in range(n):
        for i in range(int((n-j)/2)):
            D[j,j+1+2*i] = 4*((2*i+j+1))/(b-a)
    D[0,:] = D[0,:]/2
    return D
@jit
def mono_diff(p):  
    """Differentiating matrix for monomials

    Returns a matrix which multiplied from the right with the coefficients
    of the monomial returns the derivative of the respective 
    monomial. Can be used instead to evaluate the basis matrix of
    the derivative of a monomial.

    Parameters
    ----------
    p : array_like
        Parameter array containing:
         - the number of knots = degree + 1 of the polynomial
         - the lower bound of the approximation
         - the upper bound of the approximation
    Returns
    -------
    D : ndarray
       Returns an upper triangular derivative operator matrix 

    Notes
    -----
    See usage in funbas
    """
    n , a , b = p[0] , p[1] , p[2] 
    D = np.zeros((n,n))
    for j in range(n-1):
        D[j,j+1] = (j+1)/(b-a)*2
    return D
def funbas(p,x,order = None, polynomial = None):
    """Creating a multidimensional approximation basis matrix

    Returns a matrix which is a tensor product of the basis matrices 

    Parameters
    ----------
    p : array_like
        Parameter matrix where each row contains:
         - the number of knots(for splines) or  degree + 1
         of the polynomial (Chebyshev or Monomials)
         - the lower bound of the approximation
         - the upper bound of the approximation
    x : array_like
        Matrix of evaluation points, one column for each dimension
        (note the difference with p)
    order : array_like
        Specifies which derivatives should be evaluated - default is zero
        Can only contain natural numbers (integration is not done yet)
    polynomial : array_like
        Specifies the type of approximation to be used, one for each dimension:
        default is Chebyshev for all dimensions. 
        Can only contain the following strings:
        - 'cheb' for Chebyshev
        - 'mono' for Monomials
        - 'spli' for Cubic Splines - cannot pre-specify knots
        
    Returns
    -------
    Phi0 : ndarray
       Returns the n dimensional (n = no. of rows in p) basis matrix.

    Notes
    -----
    To perform a multidimensional approximation, first create a product of the evaluation
    points for each dimension - see the example
    """
    
        
    if x.ndim==1:
        if order is None:
            order = []
            order[0] = 0
        if polynomial is None:
            polynomial = []
            polynomial[0] = 'cheb'
        if polynomial=='cheb':
            Phi0=cheb_basex(p, x) @ np.linalg.matrix_power(cheb_diff(p),order[0])
        elif polynomial=='mono':
            Phi0=mono_basex(p, x) @ np.linalg.matrix_power(mono_diff(p),order[0])
        elif polynomial=='spli':
            Phi0=spli_basex(p, x, order = order[0]) 
    else:
        Phi0=np.ones((x.shape[0],1))  
        if order is None:
            order = np.zeros(x.shape[1], dtype=np.int)
        if polynomial is None:
            polynomial = ["cheb" for j in range(x.shape[1])]
        for j in range(x.shape[1]):
            if polynomial[j]=='cheb':
                Phi1=cheb_basex(p[j,:], x[:,j]) @ np.linalg.matrix_power(cheb_diff(p[j,:]),order[j])
            elif polynomial[j]=='mono':
                Phi1=mono_basex(p[j,:], x[:,j]) @ np.linalg.matrix_power(mono_diff(p[j,:]),order[j])
            elif polynomial[j]=='spli':
                Phi1=spli_basex(p[j,:], x[:,j], order = order[j]) 
            Phi0=dprod(Phi0,Phi1)
    return Phi0