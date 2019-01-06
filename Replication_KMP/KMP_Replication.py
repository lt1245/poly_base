
"""
Created on Mon Mar 14 10:12:58 2016
Replication of Kehoe, Midrigan, Pastorino: Debt Constraints and the Labor Wedge 
Method: Value Function Approximation/Collocation (following notes of Simon Mongey)
Authors: Don Jayamaha and Laszlo Tetenyi
"""
from poly_base import interpolate as ip #Imported from Laszlo's github directory
from Rouwenhorst import Rouwenhorst
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import derivative as deriv
import scipy.linalg as slin
import scipy as sp
from numba import jit
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import brentq
'Parameters'
rho_z = 0.79
sigma_z = 0.34
epsilon = 1. / 2.
nu = 1.0
eta = 0.32
theta = 0.8
beta = 0.88
sigma = 2.0
dampen_coeff_start = 0.5 # For both the Bellman iteration and Newton Iteration
# but only a strating value
fast_coeff = 10 # Number of iterations after the Newton ieration with 1.0 takes over
dampen_newton_start = 0.1 # For the aprime updating in newton method  
degree = 3 #Cubic splines for wealth
degree_z = 1 #linear splines for productivity

'Asset grid and z grid'
a_lower = 0.01
a_upper = 150.0
n_a = 31  #number of grid points for assets
n_d = 31 #number of grid points for assets in the stationary distribution
n_z = 10  #number of grid points for productivity
n_s = n_a * n_z #Overall number of gridpoints
n_ds = n_d * n_z #Overall number of gridpoints in the stationary distribution

'Rouwenhorst method for z'
T,zgrid = Rouwenhorst(rho_z, sigma_z**2, n_z )
zgrid = np.reshape(zgrid,(n_z,1))

'Polynomial basis matrix' 
P = np.array(((n_a,a_lower,a_upper),(n_z,zgrid[0,0],zgrid[-1,0])))
Polyname = ('spli','spli')
Order = (0,0)
#agrid = np.linspace(a_lower,a_upper, n_a)
agrid = ip.wealth_knot(P[0])
agrid = np.reshape(agrid,(n_a,1))

s =  np.concatenate((np.kron(np.ones((n_z,1)),agrid),np.kron(zgrid,np.ones((n_a,1)))),1) 
#agrid_sim = np.linspace(a_lower,a_upper, n_d)
agrid_sim = ip.wealth_knot((n_d,P[0,1],P[0,2]))
agrid_sim = np.reshape(agrid_sim,(n_d,1))
s_sim =  np.concatenate((np.kron(np.ones((n_z,1)),agrid_sim),np.kron(zgrid,np.ones((n_d,1)))),1)   

'Knots'
knots_z = zgrid[:,0] 
knots_x = ip.wealth_knot(P[0], degree=degree, curv=0.15)
#Phi_x = ip.spli_basex(P[0],s[:,0],deg = degree,order = 0)
Phi_x = ip.spli_basex(P[0],s[:,0],knots = knots_x ,deg = degree,order = 0)
Phi_z = ip.spli_basex(P[1],s[:,1],knots = knots_z ,deg = degree_z,order = 0)
Phi_s = ip.dprod(Phi_z,Phi_x)

#Phi_s = ip.funbas(P,s,Order,Polyname)
'Starting guess for coefficients - start at the steady state value function'
cons_start = 0.02 * s[:,0,None] + s[:,1,None] # ( (1.0 + eta **(1.0-epsilon) ) ** (-nu/(1.0-epsilon)) + 0.02 * (s[:,0,None] + (10.0- a_lower)) ) /(1.0 + eta)
house_start = eta * cons_start
V = 1.0 / (1.0 - beta) /(1.0-sigma) * ((cons_start ** ((epsilon-1.0)/epsilon) + eta **(1.0/epsilon) * house_start** ((epsilon-1.0)/epsilon) ) \
** ((epsilon-1.0)/epsilon) - (( (1.0 + eta **(1.0-epsilon) ) ** (-nu/(1.0-epsilon))))** (1.0+1.0/nu) / (1.0+1.0/nu)) ** (1.0 -sigma)
'Initial guess for consumption and coefficients'
c_guess =  0.02 * s[:,0,None] + s[:,1,None] 
c_guess1 = 0.02 * s_sim[:,0,None] + s_sim[:,1,None]
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
@jit
def m_util_ch(c,u,q_t):  
    Res = np.empty(c.shape)
    if u * q_t  < 1e-8:
        Res = 10000.0 * c
    else:
        Res = eta * (u * q_t) ** (-epsilon) * c
    return Res
    
'Vectorized Newton-Raphson method for maximization'
def newton_method(oldguess,first_deriv,second_deriv,dampen_newton,q_t,q_t1,r,w,u):
    return oldguess - dampen_newton * np.multiply((1./second_deriv), first_deriv)
    
'Vectorized BHHH method for maximization - unfinished'
def bhhh_method(oldguess,first_deriv,second_deriv,dampen_newton,q_t,q_t1,r,w,u):
    return oldguess - dampen_newton * np.multiply((1./second_deriv), first_deriv)

'Bellman iteration'
def bellman(coeff,coeff_e,dampen_coeff,s,c_vec,sprime,Phi_s,F,q_t,q_t1,r,w,u):
    #Phi_xprime = ip.spli_basex(P[0],sprime[:,0] , deg = degree,order = 0)
    Phi_xprime = ip.spli_basex(P[0],sprime[:,0],knots = knots_x , deg = degree,order = 0)
    Phi_xps = ip.dprod(Phi_z,Phi_xprime)
    coeff_next = slin.solve(Phi_s,( F(s, c_vec,q_t,q_t1,r,w,u) + beta * Phi_xps @ coeff_e))
    coeff_e_next = slin.solve(Phi_s , np.kron(T , np.eye(n_a)) @ Phi_s @ coeff)
    coeff1 = (1.-dampen_coeff) * coeff+ dampen_coeff * coeff_next
    coeff_e1 = (1. -dampen_coeff) * coeff_e+ dampen_coeff *  coeff_e_next
    conv =  np.max( np.absolute (coeff_next - coeff))
    return conv, coeff1 , coeff_e1   
    
'Newton Iteration of the value function'
def newton_iter(coeff,coeff_e,dampen_coeff,s,c_vec,sprime,Phi_s,F,q_t,q_t1,r,w,u):
    #Phi_xprime = ip.spli_basex(P[0],sprime[:,0] , deg = degree,order = 0)
    Phi_xprime = ip.spli_basex(P[0],sprime[:,0],knots = knots_x , deg = degree,order = 0)
    Phi_xps = ip.dprod(Phi_z,Phi_xprime)    
    g1 = Phi_s @ coeff - F(s, c_vec,q_t,q_t1,r,w,u) -  beta * Phi_xps @ coeff_e 
    g2 = Phi_s @ coeff_e - np.kron(T,np.eye(n_a)) @ Phi_s @ coeff
    D = np.bmat([[Phi_s, - beta * Phi_xps], [ - np.kron(T,np.eye(n_a)) @ Phi_s, Phi_s]])
    res =np.concatenate((coeff,coeff_e)) - dampen_coeff * slin.inv(D) @ np.concatenate((g1,g2))
    coeff1 = res[0:int(res.shape[0]/2.0)]
    coeff_e1 = res[int(res.shape[0]/2.0):res.shape[0]]
    conv =  np.max( np.absolute (coeff1 - coeff))
    return  conv, coeff1 , coeff_e1 



'Functions to be used in loop'

def F(s , c,q_t,q_t1,r,w,u):
    n_S = len(s)
    a = np.reshape(s[:,0],(n_S,1))
    z = np.reshape(s[:,1],(n_S,1))
    c = np.reshape(c,(n_S,1))
    
    def G(c,a,z):
        bo_co = a / ((1. - theta) * q_t1)
        h = np.min(np.concatenate((m_util_ch(c,u,q_t),bo_co),axis = 1),axis=1)
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
    
def aprimefunc(s,c,q_t,q_t1,r,w,u):
    n_S = len(s)
    a = np.reshape(s[:,0],(n_S,1))
    z = np.reshape(s[:,1],(n_S,1))
    c = np.reshape(c,(n_S,1))
    def G(c,a,z):
        bo_co = a / ((1 - theta) * q_t1)
        h = np.min(np.concatenate((m_util_ch(c,u,q_t),bo_co),axis = 1),axis=1)
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
        
def aprimefunc_scal(a,z,c,q_t,q_t1,r,w,u):
    s_scal = np.empty((1,2))
    c_scal = np.empty((1,1))
    s_scal[0,0] = a
    s_scal[0,1] = z
    c_scal[0,0] = c
    return aprimefunc(s_scal[0:1,:],c_scal[0:1,:],q_t,q_t1,r,w,u)

def aprimefunc_x_scal(a,z,c,q_t,q_t1,r,w,u):
    def aprimefunc_a(cpr):
        return aprimefunc_scal(a,z,cpr,q_t,q_t1,r,w,u)
    return deriv(aprimefunc_a,c,dx= 1e-6,n=1)   
    
def aprimefunc_xx_scal(a,z,c,q_t,q_t1,r,w,u):
    def aprimefunc_a(cpr):
        return aprimefunc_scal(a,z,cpr,q_t,q_t1,r,w,u)
    return deriv(aprimefunc_a,c,dx= 1e-6,n=2)      
def aprimefunc_x(state , c_vec,q_t,q_t1,r,w,u):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = aprimefunc_x_scal(state[i,0],state[i,1],c_vec[i,0],q_t,q_t1,r,w,u)
    return Res 
def aprimefunc_xx(state , c_vec,q_t,q_t1,r,w,u):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = aprimefunc_xx_scal(state[i,0],state[i,1],c_vec[i,0],q_t,q_t1,r,w,u)
    return Res       
def F_scal(a,z,c,q_t,q_t1,r,w,u):
    s_scal = np.empty((1,2))
    c_scal = np.empty((1,1))
    s_scal[0,0] = a
    s_scal[0,1] = z
    c_scal[0,0] = c
    return F(s_scal[0:1,:],c_scal[0:1,:],q_t,q_t1,r,w,u)
   
def F_x_scal(a,z,c,q_t,q_t1,r,w,u):
    def f_a(cpr):
        return F_scal(a,z,cpr,q_t,q_t1,r,w,u)
    return deriv(f_a,c,dx= 1e-6,n=1)
    
def F_xx_scal(a,z,c,q_t,q_t1,r,w,u):
    def f_a(cpr):
        return F_scal(a,z,cpr,q_t,q_t1,r,w,u)
    return deriv(f_a,c,dx= 1e-6,n=2)   
def F_x(state , c_vec,q_t,q_t1,r,w,u):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = F_x_scal(state[i,0],state[i,1],c_vec[i,0],q_t,q_t1,r,w,u)
    return Res 
def F_xx(state , c_vec,q_t,q_t1,r,w,u):
    n_S = len(state)
    Res = np.empty((n_S,1))
    for i in range(n_S):
        Res[i,0] = F_xx_scal(state[i,0],state[i,1],c_vec[i,0],q_t,q_t1,r,w,u)
    return Res 

'Inverting aprime_func at a_lower for each state variable'
def c_bounds(s,c_vec,q_t,q_t1,r,w,u,aprime1 = None):
    n_S = len(s)
    if aprime1 == None:
        aprime1 = a_lower * np.ones((n_S,1))
    elif aprime1 == 'upper' :
        aprime1 = a_upper * np.ones((n_S,1))
    def c_solve(c):
        return (aprime1 - aprimefunc(s,c,q_t,q_t1,r,w,u))[:,0]
    sol = root(c_solve,c_vec[:,0], method='hybr')   
    return np.reshape(sol.x,(n_S,1))

'Get housing and the risk free asset from wealth and labor supply'
def housefunc(s,c,q_t,q_t1,r,w,u):
    n_S = len(s)
    a = np.reshape(s[:,0],(n_S,1))
    z = np.reshape(s[:,1],(n_S,1))
    c = np.reshape(c,(n_S,1))
    def G(c,a,z):
        bo_co = a / ((1 - theta) * q_t1)
        h = np.min(np.concatenate((m_util_ch(c,u,q_t),bo_co),axis = 1),axis=1)
        h = np.reshape(h,(n_S,1))
        mu = 1.0 / q_t * (a / (c *(eta * (1.0 - theta)))) ** (-1 / epsilon) - u
        mu = np.max(np.concatenate((mu, np.zeros((n_S,1))),axis = 1),axis=1)
        mu = np.reshape(mu,(n_S,1))
        p = (1.0 + eta * ((u + mu) * q_t) ** (1.0 - epsilon)) ** (1.0 / (1.0- epsilon))
        p = np.reshape(p,(n_S,1))
        n = (w* (z / p)) **nu   
        return  h, mu , p , n
    h , mu , p , n = G(c , a , z) 
    aprime = w * np.multiply(z , n)  + (1.0 + r) * a - c - u * q_t * h
    bprime = aprime - h
    return h, n, bprime, aprime
    
def Perturb_Q(A):
    #Perturb Q
    eta =( A[A>0].min() )/(2.*n_ds)
    A = np.where(A<=0.0, A + eta , A)
    #Normalize Q
    for i in range(n_ds):
        A[i,:] = A[i,:]/(np.sum(A[i,:]))
    return A
    
def distribution(c_vec,n_a1,a_lower,a_upper,aprime,bprime,h,n):
    Q_x = ip.spli_basex((n_a1,a_lower,a_upper),aprime[:,0],knots = ip.wealth_knot((n_a1,P[0,1],P[0,2])), deg=1,order = 0) #knots = ip.wealth_knot((n_a1,P[0,1],P[0,2])),
    Q_z = np.kron(T,np.ones((n_a1,1)))
    Q = ip.dprod(Q_z,Q_x)
    Q = Perturb_Q(Q)
    Eigenval, Eigenvec = sp.linalg.eig(Q, left = True, right = False )
    Eigenvec=Eigenvec.real
    arg = np.argmin(np.abs(Eigenval - 1.))
    stationary_dist = Eigenvec[:,arg]
    stationary_dist =  (stationary_dist.real)/(stationary_dist.real.sum())
    L = stationary_dist
    agra = np.dot(L,aprime)
    agrb = np.dot(L,bprime)
    agrh = np.dot(L,h)
    agrn = np.dot(L,n)
    Res = (L,c_vec,bprime,h,n,agra,agrb,agrh,agrn,aprime)
    return Res

def main_loop(coeff,coeff_e,prices,c_vec = c_guess,outer_loop = None,n_a1 = n_a,dampen_coeff = dampen_coeff_start,dampen_newton = dampen_newton_start):
    q_t = prices[1]
    q_t1 = prices[1]
    r = prices[0]
    w = 1.0
    u = (1. + r) * q_t1 / q_t - 1.0     
    
    'Quantities'
    conv1 = 2.0
    iteration = 0
    #agrid = np.linspace(a_lower,a_upper, n_a1)
    agrid = ip.wealth_knot((n_a1,P[0,1],P[0,2]))
    agrid = np.reshape(agrid,(n_a1,1))
    s =  np.concatenate((np.kron(np.ones((n_z,1)),agrid),np.kron(zgrid,np.ones((n_a1,1)))),1)   
    #Phi_x = ip.spli_basex(P[0],s[:,0] , deg = degree,order = 0)    
    Phi_x = ip.spli_basex(P[0],s[:,0],knots = knots_x , deg = degree,order = 0)
    Phi_z = ip.spli_basex(P[1],s[:,1],knots = knots_z , deg = degree_z,order = 0)
    Phi_s = ip.dprod(Phi_z,Phi_x)
    c_max = c_bounds(s,c_vec,q_t,q_t1,r,w,u)    
    c_min = c_bounds(s,c_vec,q_t,q_t1,r,w,u , 'upper')
    c_vec = np.minimum(c_vec,c_max)
    c_vec = np.maximum(c_vec,c_min)
    aprime = aprimefunc(s,c_vec,q_t,q_t1,r,w,u)
    while conv1 > 1e-6:
        conv2 = 10.0
        iteration1 = 0
        iteration = iteration + 1
        while conv2 > 1e-7:
            iteration1 = iteration1 + 1
            if iteration1 > fast_coeff:# and outer_loop != 1:
                dampen_newton = 1.0
            aprime_c = aprimefunc_x(s,c_vec,q_t,q_t1,r,w,u)
            aprime_cc = aprimefunc_xx(s,c_vec,q_t,q_t1,r,w,u)        
            sprime = np.concatenate((aprime,s[:,1,None]),axis = 1)
            #Phi_xprime1 = ip.spli_basex(P[0],sprime[:,0], deg = degree,order = 1)
            #Phi_xprime2 = ip.spli_basex(P[0],sprime[:,0], deg = degree,order = 2)
            Phi_xprime1 = ip.spli_basex(P[0],sprime[:,0],knots = knots_x , deg = degree,order = 1)
            Phi_xprime2 = ip.spli_basex(P[0],sprime[:,0],knots = knots_x , deg = degree,order = 2)
            Phi_xps1 = ip.dprod(Phi_z,Phi_xprime1) 
            Phi_xps2 = ip.dprod(Phi_z,Phi_xprime2)            
            
            Hessian = F_xx(s,c_vec,q_t,q_t1,r,w,u) + beta *( np.multiply(Phi_xps1@coeff_e,aprime_cc ) + np.multiply(Phi_xps2@coeff_e, (aprime_c)**2 ))
            Jacobian = F_x(s,c_vec,q_t,q_t1,r,w,u) + beta *( np.multiply(Phi_xps1@coeff_e,aprime_c )) 
            
            c_vec_next = np.minimum(newton_method(c_vec,Jacobian,Hessian,dampen_newton,q_t,q_t1,r,w,u),c_max)
            c_vec_next = np.maximum(c_vec_next,c_min)
            conv2 = np.max( np.absolute (c_vec_next - c_vec ))
            np.copyto(c_vec,c_vec_next)
            h, n, bprime, aprime= housefunc(s,c_vec,q_t,q_t1,r,w,u)
            #aprime = aprimefunc(s,c_vec,q_t,q_t1,r,w,u)
            print(conv2)
        if outer_loop == 1:
            'Computing the stationary distribution'
            conv1 = 0
            Q_x = ip.spli_basex((n_a1,a_lower,a_upper),aprime[:,0],knots = ip.wealth_knot((n_a1,P[0,1],P[0,2])), deg=1,order = 0) #knots = ip.wealth_knot((n_a1,P[0,1],P[0,2])),
            Q_z = np.kron(T,np.ones((n_a1,1)))
            Q = ip.dprod(Q_z,Q_x)
            Q = Perturb_Q(Q)
            Eigenval, Eigenvec = sp.linalg.eig(Q, left = True, right = False )
            Eigenvec=Eigenvec.real
            arg = np.argmin(np.abs(Eigenval - 1.))
            stationary_dist = Eigenvec[:,arg]
            stationary_dist =  (stationary_dist.real)/(stationary_dist.real.sum())
            L = stationary_dist
            agra = np.dot(L,aprime)
            agrb = np.dot(L,bprime)
            agrh = np.dot(L,h)
            agrn = np.dot(L,n)
            Res = (L,c_vec,bprime,h,n,agra,agrb,agrh,agrn,aprime)
        else:
            if iteration > fast_coeff:
                dampen_coeff = 1.0
                conv1, coeff , coeff_e = newton_iter(coeff,coeff_e,dampen_coeff,s,c_vec,sprime,Phi_s,F,q_t,q_t1,r,w,u)            
            else:
                conv1, coeff , coeff_e = bellman(coeff,coeff_e,dampen_coeff,s,c_vec,sprime,Phi_s,F,q_t,q_t1,r,w,u)
            Res = (coeff, coeff_e,c_vec)
        print((conv1,conv2))
    return Res

'Get a much better initial guess'
prices_start = [0.02,1.0]
coeff_guess, coeff_guess_e ,c_guess = main_loop(coeff_guess,coeff_e_guess,prices_start)
#L,c_guess1,bprime,h,n,agra,agrb,agrh,agrn,aprime = main_loop(coeff_guess,coeff_e_guess,prices_start,c_guess1,1,n_d,dampen_coeff_start,dampen_newton_start)
#fast_coeff = 10
'Lets search for the stationary equilibrium'
#bounds = [(0.01,0.05),(0.6,1.4)]
c_vec = c_guess
prices = prices_start
q_t = prices[1]
q_t1 = prices[1]
r = prices[0]
w = 1.0
u = (1. + r) * q_t1 / q_t - 1.0    
coeff_e = coeff_guess_e
dampen_newton = dampen_newton_start
n_a1 = n_d
conv2 = 10.0
iteration1 = 0
while conv2 > 1e-7:
    iteration1 = iteration1 + 1
    if iteration1 > fast_coeff:# and outer_loop != 1:
        dampen_newton = 1.0
    aprime_c = aprimefunc_x(s,c_vec,q_t,q_t1,r,w,u)
    aprime_cc = aprimefunc_xx(s,c_vec,q_t,q_t1,r,w,u)        
    sprime = np.concatenate((aprime,s[:,1,None]),axis = 1)
    #Phi_xprime1 = ip.spli_basex(P[0],sprime[:,0], deg = degree,order = 1)
    #Phi_xprime2 = ip.spli_basex(P[0],sprime[:,0], deg = degree,order = 2)
    Phi_xprime1 = ip.spli_basex(P[0],sprime[:,0],knots = knots_x , deg = degree,order = 1)
    Phi_xprime2 = ip.spli_basex(P[0],sprime[:,0],knots = knots_x , deg = degree,order = 2)
    Phi_xps1 = ip.dprod(Phi_z,Phi_xprime1) 
    Phi_xps2 = ip.dprod(Phi_z,Phi_xprime2)            
    
    Hessian = F_xx(s,c_vec,q_t,q_t1,r,w,u) + beta *( np.multiply(Phi_xps1@coeff_e,aprime_cc ) + np.multiply(Phi_xps2@coeff_e, (aprime_c)**2 ))
    Jacobian = F_x(s,c_vec,q_t,q_t1,r,w,u) + beta *( np.multiply(Phi_xps1@coeff_e,aprime_c )) 
    
    c_vec_next = np.minimum(newton_method(c_vec,Jacobian,Hessian,dampen_newton,q_t,q_t1,r,w,u),c_max)
    c_vec_next = np.maximum(c_vec_next,c_min)
    conv2 = np.max( np.absolute (c_vec_next - c_vec ))
    np.copyto(c_vec,c_vec_next)
    h, n, bprime, aprime= housefunc(s,c_vec,q_t,q_t1,r,w,u)
    #aprime = aprimefunc(s,c_vec,q_t,q_t1,r,w,u)
    print(conv2)

#def iter_loop(prices):
#    if prices[0] > bounds[0][1] or prices[0] < bounds[0][0] or prices[1] > bounds[1][1] or prices[1] < bounds[1][0]:
#        Res = np.array([10.0,10.0])
#    else:
#        coeff1, coeff1_e ,c_vec11 = main_loop(coeff_guess,coeff_e_guess,prices,c_guess,None,n_a,dampen_coeff_start,dampen_newton_start)
#        L,c_vec2,bprime,h,n,agra,agrb,agrh,agrn,aprime = main_loop(coeff1,coeff1_e,prices,c_guess1,1,n_d,dampen_coeff_start,dampen_newton_start)
#        Res = np.array([agrb,agrh-1.0])
#        Res = Res[:,0]
#        #plt.plot(L)
#        print(L[0:50])
#    print("Convergence", Res)
#    return Res
#
#z = iter_loop([0.006,1.0])
#solution1 = root(iter_loop,prices_start,method = 'hybr')
#theta = 0.5
#solution2 = root(iter_loop,prices_start,method = 'broyden1')
def iter_loop1(prices):
    coeff1, coeff1_e ,c_vec11 = main_loop(coeff_guess,coeff_e_guess,prices,c_guess,None,n_a,dampen_coeff_start,dampen_newton_start)
    L,c_vec2,bprime,h,n,agra,agrb,agrh,agrn,aprime = main_loop(coeff1,coeff1_e,prices,c_guess1,1,n_d,dampen_coeff_start,dampen_newton_start)
    Res = np.array([agrb,agrh-3.5])
    print("Convergence", Res)
    Res = np.sum(Res[:,0]**2)
    #plt.plot(L)
    #print(L[0:50])    
    return Res
#z = iter_loop1([0.02,0.9])
#solution = minimize(iter_loop1,prices_start,method = 'L-BFGS-B',bounds = bounds)
#x =  np.array([ 0.0194985 ,  0.69187622])
#theta = 0.5
#solution2 = minimize(iter_loop1,prices_start,method = 'L-BFGS-B',bounds = bounds)
#'Solution of the initial steady state saved'
#sol1_x = np.array([ 0.02733261,  1.04685027])
#coeff_sol1, coeff_e_sol1 ,c_vec1_sol1 = main_loop(coeff_guess,coeff_e_guess,sol1_x)
#L_sol1,c_vec2_sol1,bprime_sol1,h_sol1,n_sol1,agra_sol1,agrb_sol1,agrh_sol1,agrn_sol1,aprime_sol1 = main_loop(coeff_sol1,coeff_e_sol1,sol1_x,c_guess1,1,n_d)
#'Optimality satisfied - make some plots'
#plt.plot(h)
#'Solve for the other stationary equilibrium'
#theta = 0.5
#dampen_coeff_start = 0.1 # For both the Bellman iteration and Newton Iteration
#dampen_newton_start = 0.01 # For the aprime updating in newton method  
#fast_coeff = 20
#solution2 = minimize(iter_loop,prices_start,method = 'L-BFGS-B', bounds = [(0.001,0.06),(0.2,3.0)])
#sol2_x = np.array([ 0.02690688,  0.4165023])
#houseprices = np.linspace(0.2,3.0,10)
#agrbs = np.empty((10,1))
#agrhs = np.empty((10,1))
#for i in range(10):
#    sol2_x = np.array([ 0.027, houseprices[i]])
#    coeff_sol2, coeff_e_sol2 ,c_vec1_sol2 = main_loop(coeff_guess,coeff_e_guess,sol2_x,c_guess,None,n_a,dampen_coeff_start,dampen_newton_start)
#    L_sol2,c_vec2_sol2,bprime_sol2,h_sol2,n_sol2,agra_sol2,agrb_sol2,agrh_sol2,agrn_sol2,aprime_sol2 = main_loop(coeff_sol2,coeff_e_sol2,sol2_x,c_guess1,1,n_d,dampen_coeff_start,dampen_newton_start)
#    agrbs[i,0] = agrb_sol2
#    agrhs[i,0] = agrh_sol2
#plt.plot(agrbs)
#plt.show()
#'Iterative root finding'
#def iter_loop1(q_t):
#    prices = np.empty((2,))
#    prices[1] = q_t
#    prices[0] = 0.02
#    coeff1, coeff1_e ,c_vec11 = main_loop(coeff_guess,coeff_e_guess,prices,c_guess,None,n_a,dampen_coeff_start,dampen_newton_start)
#    L,c_vec2,bprime,h,n,agra,agrb,agrh,agrn,aprime = main_loop(coeff1,coeff1_e,prices,c_guess1,1,n_d,dampen_coeff_start,dampen_newton_start)
#    Res = agrh-1.0
#    print("Convergence", Res)
#    return Res
#iter_loop1(2.5)
#brentq(iter_loop1,0.5,2.5)