
"""
Created on Mon Mar 14 10:12:58 2016
Replication of Kehoe, Midrigan, Pastorino: Debt Constraints and the Labor Wedge 
Method: Value Function Approximation/Collocation (following notes of Simon Mongey)
Authors: Don Jayamaha and Laszlo Tetenyi
"""
from poly_base import interpolate as ip #Imported from Laszlo's github directory
from Rowenhurst import Rowenhurst
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv
import scipy.linalg as slin
from numba import jit
from scipy.optimize import root

'Parameters'
rho_z = 0.79
sigma_z = 0.34
epsilon = 1. / 2.
nu = 1.0
eta = 0.32
theta = 0.8
beta = 0.88
sigma = 2.0
dampen_coeff_start = 0.1 # For both the Bellman iteration and Newton Iteration
# but only a strating value
fast_coeff = 10 # Number of iterations after the Newton ieration with 1.0 takes over
dampen_newton = 1.0 # For the aprime updating in newton method  

'Asset grid and z grid'
a_lower = 0.05
a_upper = 30.0
n_a = 10  #number of grid points for assets
n_d = 20 #number of grid points for assets in the stationary distribution
n_z = 5   #number of grid points for productivity
n_s = n_a * n_z #Overall number of gridpoints
n_ds = n_d * n_z #Overall number of gridpoints in the stationary distribution
#lin_stop = np.log(np.log(np.log(a_upper - a_lower + 1.)+1)+1) 
#agrid = np.exp(np.exp(np.exp(np.linspace(0, lin_stop, n_a)) - 1.) - 1.) -1. + a_lower
#agrid = np.reshape(agrid,(n_a,1))
agrid = np.linspace(a_lower,a_upper, n_a)
agrid = np.reshape(agrid,(n_a,1))
T,zgrid = Rowenhurst(rho_z, sigma_z**2, n_z )
zgrid = np.reshape(zgrid,(n_z,1))
s =  np.concatenate((np.kron(np.ones((n_z,1)),agrid),np.kron(zgrid,np.ones((n_a,1)))),1)

'Polynomial basis matrix' 
P = np.array(((n_a,a_lower,a_upper),(n_z,zgrid[0,0],zgrid[n_z-1,0])))
Polyname = ('spli','spli')
Order = (0,0)
Phi_s = ip.funbas(P,s,Order,Polyname)

'Starting guess for coefficients - start at the steady state value function'
cons_start =  ( (1.0 + eta **(1.0-epsilon) ) ** (-nu/(1.0-epsilon)) + 0.02 * (s[:,0,None] + (10.0- a_lower)) ) /(1.0 + eta)
house_start = eta * cons_start
V = 1.0 / (1.0 - beta) /(1.0-sigma) * ((cons_start ** ((epsilon-1.0)/epsilon) + eta **(1.0/epsilon) * house_start** ((epsilon-1.0)/epsilon) ) \
** ((epsilon-1.0)/epsilon) - (( (1.0 + eta **(1.0-epsilon) ) ** (-nu/(1.0-epsilon))))** (1.0+1.0/nu) / (1.0+1.0/nu)) ** (1.0 -sigma)

'Load initial guess for consumption and coefficients'
#c_guess=np.reshape(np.loadtxt("c_guess.csv"),(n_s,1))
#coeff_guess=np.reshape(np.loadtxt("coeff_guess.csv"),(n_s,1))
#coeff_e_guess=np.reshape(np.loadtxt("coeff_e_guess.csv"),(n_s,1))
#c_vec = 1.0 * np.ones((n_s,1)) 
#coeff= coeff_guess
#coeff_e= coeff_e_guess
#coeff =  slin.solve(Phi_s,V) 
#coeff_e = slin.solve(Phi_s , np.kron(T , np.eye(n_a)) @ Phi_s @ coeff)
c_guess = 1.0 * np.ones((n_s,1))
coeff_guess = slin.solve(Phi_s,V)
coeff_e_guess = slin.solve(Phi_s , np.kron(T , np.eye(n_a)) @ Phi_s @ coeff_guess)

'GHH preferences (Penalizing values for which g > disutility from working)'
@jit
def GHH(g,n):  
    Res = np.empty(g.shape)
    for i in range(len(g)):            
        if ((g[i,0] - (n[i,0] ** (1.0+ 1.0/nu))/(1.0+ 1.0/nu)) ) < 0:
            Res[i,0] = -10000000
        else:
            Res[i,0] =  1.0 / (1.0 -sigma) * ((g[i,0] - (n[i,0] ** (1.0+ 1.0/nu))/(1.0+ 1.0/nu)) ) ** (1.0 - sigma)    
    return Res

'Vectorized Newton method for maximization'
def newton_method(oldguess,first_deriv,second_deriv,dampen_newton):
    return oldguess - dampen_newton * np.multiply((1./second_deriv), first_deriv)

'Bellman iteration'
def bellman(coeff,coeff_e,dampen_coeff,s,c_vec,sprime,Phi_s,F):  
    Phi_xps = ip.funbas(P,sprime,(0,0),Polyname)
    coeff_next = slin.solve(Phi_s,( F(s, c_vec) + beta * Phi_xps @ coeff_e))
    coeff_e_next = slin.solve(Phi_s , np.kron(T , np.eye(n_a)) @ Phi_s @ coeff)
    coeff1 = (1.-dampen_coeff) * coeff+ dampen_coeff * coeff_next
    coeff_e1 = (1. -dampen_coeff) * coeff_e+ dampen_coeff *  coeff_e_next
    conv =  np.max( np.absolute (coeff_next - coeff))
    return conv, coeff1 , coeff_e1   
    
'Newton Iteration of the value function - it is unused now'
def newton_iter(coeff,coeff_e,dampen_coeff,s,c_vec,sprime,Phi_s,F):
    Phi_xps = ip.funbas(P,sprime,(0,0),Polyname)    
    g1 = Phi_s @ coeff - F(s, c_vec) -  beta * Phi_xps @ coeff_e 
    g2 = Phi_s @ coeff_e - np.kron(T,np.eye(n_a)) @ Phi_s @ coeff
    D = np.bmat([[Phi_s, - beta * Phi_xps], [ - np.kron(T,np.eye(n_a)) @ Phi_s, Phi_s]])
    res =np.concatenate((coeff,coeff_e)) - dampen_coeff * slin.inv(D) @ np.concatenate((g1,g2))
    coeff1 = res[0:int(res.shape[0]/2.0)]
    coeff_e1 = res[int(res.shape[0]/2.0):res.shape[0]]
    conv =  np.max( np.absolute (coeff1 - coeff))
    return  conv, coeff1 , coeff_e1 



'Functions to be used in loop'

def F(s , c):
    n_S = len(s)
    a = np.reshape(s[:,0],(n_S,1))
    z = np.reshape(s[:,1],(n_S,1))
    c = np.reshape(c,(n_S,1))
    
    def G(c,a,z):
        m_util_ch = eta * (u * q_t) ** (-epsilon) * c
        bo_co = a / ((1. - theta) * q_t1)
        h = np.min(np.concatenate((m_util_ch,bo_co),axis = 1),axis=1)
        h = np.reshape(h,(n_S,1))
        mu = 1.0 / q_t * (a / (c *(eta * (1.0 - theta)))) ** (-1. / epsilon) - u
        mu = np.max(np.concatenate((mu, np.zeros((n_S,1))),axis = 1),axis=1)
        mu = np.reshape(mu,(n_S,1))
        p = (1.0 + eta * ((u + mu) * q_t) ** (1.0 - epsilon)) ** (1.0 / (1.0- epsilon))
        p = np.reshape(p,(n_S,1))
        n = (w* (z / p)) **nu   
        return  h, mu , p , n
    
    h , mu , p , n = G(c , a , z )
    g = ( c**((epsilon-1.0)/epsilon) + (eta ** (1.0/epsilon)) * h **((epsilon-1.0)/epsilon) ) ** (epsilon/(epsilon-1.0))
    return GHH(g,n)
    
def aprimefunc(s,c):
    n_S = len(s)
    a = np.reshape(s[:,0],(n_S,1))
    z = np.reshape(s[:,1],(n_S,1))
    c = np.reshape(c,(n_S,1))
    def G(c,a,z):
        m_util_ch = eta * (u * q_t) ** (-epsilon) * c
        bo_co = a / ((1 - theta) * q_t1)
        h = np.min(np.concatenate((m_util_ch,bo_co),axis = 1),axis=1)
        h = np.reshape(h,(n_S,1))
        mu = 1.0 / q_t * (a / (c *(eta * (1.0 - theta)))) ** (-1 / epsilon) - u
        mu = np.max(np.concatenate((mu, np.zeros((n_S,1))),axis = 1),axis=1)
        mu = np.reshape(mu,(n_S,1))
        p = (1.0 + eta * ((u + mu) * q_t) ** (1.0 - epsilon)) ** (1.0 / (1.0- epsilon))
        p = np.reshape(p,(n_S,1))
        n = (w* (z / p)) **nu   
        return  h, mu , p , n
    
    h , mu , p , n = G(c , a , z)
    return  ( w * np.multiply(z , n)  + (1 + r) * a - c - u * q_t * h)
        
def aprimefunc_scal(a,z,c):
    s_scal = np.ones((1,2))
    c_scal = np.ones((1,1))
    s_scal[0,0] = a
    s_scal[0,1] = z
    c_scal[0,0] = c
    return aprimefunc(s_scal[0:1,:],c_scal[0:1,:])

def aprimefunc_x_scal(a,z,c):
    def aprimefunc_a(cpr):
        return aprimefunc_scal(a,z,cpr)
    return deriv(aprimefunc_a,c,dx= 1e-6,n=1)   
    
def aprimefunc_xx_scal(a,z,c):
    def aprimefunc_a(cpr):
        return aprimefunc_scal(a,z,cpr)
    return deriv(aprimefunc_a,c,dx= 1e-6,n=2)      
def aprimefunc_x(state , c_vec):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = aprimefunc_x_scal(state[i,0],state[i,1],c_vec[i,0])
    return Res 
def aprimefunc_xx(state , c_vec):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = aprimefunc_xx_scal(state[i,0],state[i,1],c_vec[i,0])
    return Res       
def F_scal(a,z,c):
    s_scal = np.ones((1,2))
    c_scal = np.ones((1,1))
    s_scal[0,0] = a
    s_scal[0,1] = z
    c_scal[0,0] = c
    return F(s_scal[0:1,:],c_scal[0:1,:])
   
def F_x_scal(a,z,c):
    def f_a(cpr):
        return F_scal(a,z,cpr)
    return deriv(f_a,c,dx= 1e-6,n=1)
    
def F_xx_scal(a,z,c):
    def f_a(cpr):
        return F_scal(a,z,cpr)
    return deriv(f_a,c,dx= 1e-6,n=2)   
def F_x(state , c_vec):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = F_x_scal(state[i,0],state[i,1],c_vec[i,0])
    return Res 
def F_xx(state , c_vec):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = F_xx_scal(state[i,0],state[i,1],c_vec[i,0])
    return Res 
'Inverting aprime_func at a_lower for each state variable'
def c_bounds(s,c_vec,aprime1 = None):
    n_S = len(s)
    if aprime1 == None:
        aprime1 = a_lower * np.ones((n_S,1))
    elif aprime1 == 'upper' :
        aprime1 = a_upper * np.ones((n_S,1))
    def c_solve(c):
        return (aprime1 - aprimefunc(s,c))[:,0]
    sol = root(c_solve,c_vec[:,0], method='hybr')   
    return np.reshape(sol.x,(n_S,1))
def main_loop(coeff,coeff_e,c_vec = c_guess,outer_loop = None,n_a1 = n_a,dampen_coeff = dampen_coeff_start):
    'Quantities'
    conv1 = 2.0
    iteration = 0
    agrid = np.linspace(a_lower,a_upper, n_a1)
    agrid = np.reshape(agrid,(n_a1,1))
    s =  np.concatenate((np.kron(np.ones((n_z,1)),agrid),np.kron(zgrid,np.ones((n_a1,1)))),1)
    c_vec = np.minimum(c_vec,c_bounds(s,c_vec))
    c_vec = np.maximum(c_vec,c_bounds(s,c_vec , 'upper'))
    aprime = aprimefunc(s,c_vec)
    while conv1 > 1e-3:
        conv2 = 10.0
        iteration = iteration + 1
        if iteration > fast_coeff:
            dampen_coeff = 1.0
        while conv2 > 1e-3:
            aprime_c = aprimefunc_x(s,c_vec)
            aprime_cc = aprimefunc_xx(s,c_vec)        
            sprime = np.concatenate((aprime,s[:,1,None]),axis = 1)
            Phi_xps1 = ip.funbas(P,sprime,(1,0),Polyname)
            Phi_xps2 = ip.funbas(P,sprime,(2,0),Polyname)
            
            
            Hessian = F_xx(s,c_vec) + beta *( np.multiply(Phi_xps1@coeff_e,aprime_cc ) + np.multiply(Phi_xps2@coeff_e, (aprime_c)**2 ))
            Jacobian = F_x(s,c_vec) + beta *( np.multiply(Phi_xps1@coeff_e,aprime_c )) 
            
            c_vec_next = np.minimum(newton_method(c_vec,Jacobian,Hessian,dampen_newton),c_bounds(s,c_vec))
            c_vec_next = np.maximum(c_vec_next,c_bounds(s,c_vec , 'upper'))
            #c_vec_next = newton_method(c_vec,Jacobian,Hessian,dampen_newton)
            conv2 = np.max( np.absolute (c_vec_next - c_vec ))
            c_vec = c_vec_next
            aprime = aprimefunc(s,c_vec)
            print(conv2)
        if outer_loop == 1:
            'Computing the stationary distribution'
            conv1 = 0
            Q_x = ip.spli_basex((n_a1,a_lower,a_upper),aprime[:,0],knots = None , deg= 1,order = 0)
            Q_z = np.kron(T,np.ones((n_a1,1)))
            Q = ip.dprod(Q_z,Q_x)
            w , v = slin.eig(Q.transpose())
            L = (v[:,0] / v[:,0].real.sum(0)).real
            agra = np.dot(L,aprime) # Aggregate asset level
            Res = (L,agra,c_vec)
        else:
            conv1, coeff , coeff_e = newton_iter(coeff,coeff_e,dampen_coeff,s,c_vec,sprime,Phi_s,F)
            Res = (coeff, coeff_e,c_vec)
        print((conv1,conv2))
    return Res
'Prices (fixed for now)'
q_t = 1.1
q_t1 = 1.1
w = 1.0
r = 0.01
u = (1. + r) * q_t1 / q_t - 1. 
coeff, coeff_e ,c_vec1 = main_loop(coeff_guess,coeff_e_guess)
c_guess1 = 1.0 * np.ones((n_ds,1))
L , agra, c_vec2 = main_loop(coeff,coeff_e,c_guess1,1,n_d)    
plt.plot(L)