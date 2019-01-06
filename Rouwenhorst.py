"""
Created on Mon Mar 14 12:16:13 2016
Rouwenhorst Algorithm
@author: donjayamaha
"""

import numpy as np
def Rouwenhorst(rho_logz, sigma_logz, zgrid_size ):

    def Transition_RM(p, q, n ):
        """
        Function to compute the transition matrix of descretized AR1.
        
        Inputs: p, q, n(= number of states).  
        
        Output: nXn transition matrix
        """
        T0 = np.array([[p, 1-p], [1-q, q]])
        for i in range(n-2):
            val1 = np.zeros([i+3,i+3])
            val2 = np.zeros([i+3,i+3])
            val3 = np.zeros([i+3,i+3])
            val4 = np.zeros([i+3,i+3])
            val1[0:i+2, 0:i+2] = T0*p
            val2[0:i+2, 1:i+3] = T0*(1-p)
            val3[1:i+3, 0:i+2] = T0*(1-q)
            val4[1:i+3, 1:i+3] = T0*q
            T0 = val1 + val2 + val3 + val4
            T0[1:-1] /= 2
        return T0
    
    #Save parameter values given in the question
    #note e here represents shocks
    E_logz = 0.
    var_logz = (sigma_logz**2)/(1- (rho_logz)**2)   #0.01 is sd of innovation , innovation here is not e but a shock to AR process of e
    
    #Grid size for z
    nz = zgrid_size
    
    #Use Rouwenhorst method to discretize income process
    p = (1.+rho_logz)/2
    q = (1.+rho_logz)/2
    psi = np.sqrt(var_logz*(nz-1.))
    
    zgrid = np.exp(np.linspace(-psi, psi, nz))
    T = Transition_RM(p, q, nz)
    
    
    S = np.linalg.matrix_power(T,100)
    S1 = (S[0,:]@(np.log(zgrid))**2)

    return T,zgrid
