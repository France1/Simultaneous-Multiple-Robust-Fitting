'''Python implementation of Tarel et al. Using Robust Estimation Algorithms for Tracking Explicit Curves. Computer 
Vision — ECCV 2002
'''
import numpy as np

def weight(X,a,y,scale=1):
    '''
    Auxiliary variable from Sec. 3.2
    '''
    return ((np.dot(X,a)-y)/scale)**2

def phi_prime(t,function='gauss'):
    '''
    Derivatives of noise function from Table 1
    '''
    if function == 'gauss':
        val = 1
    if function == 'laplace':
        val = 1/np.sqrt(1+t)
    if function == 'cauchy':
        val = 1/(1+t)
    if function == 'geman':
        val = 1/(1+t)**2
    return val

def update_params(X,y,a_init,function,scale):
    '''
    Steps 2,3 of algorithm in Sec. 3.2
    '''

    S = np.zeros((X.shape[1],X.shape[1]))
    t = np.zeros(X.shape[1])
    for i in range(len(X)):
        w_i = weight(X[i],a_init,y[i],scale)
        phi_prime_i = phi_prime(w_i,function)
        S += phi_prime_i*np.dot(X[i].reshape(-1,1),X[i].reshape(-1,1).T)
        t += phi_prime_i*X[i]*y[i]
    a_next = np.dot(np.linalg.inv(S),t)
    
    return a_next

def IRLS(X,y,a_0,function='gauss',scale=1):
    '''
    Algorithm of Sec 3.2
    '''
    tol = 0.001
    a_curr = a_0
    diff = np.linalg.norm(a_curr)
    
    while diff>tol:
        a_next = update_params(X,y,a_curr,function,scale)
        diff = np.linalg.norm(a_next-a_curr)
        # print(a_next, diff)
        a_curr = a_next
    
    return a_curr