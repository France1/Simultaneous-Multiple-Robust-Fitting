'''Python implementation of Tarel et al. Simultaneous robust fitting of multiple curves. VISAPP 2007
'''
import numpy as np

EPSILON = np.finfo(np.float).eps

def auxiliary_variable(x,a,y,scale):
    '''
    Auxiliary variable in Sec. 3
    '''
    return ((np.dot(x,a)-y)/scale)**2

def phi(t,function='gauss'):
    '''
    Derivatives of noise function in Table 1 of Tarel et al. IRLS paper
    '''
    if function == 'gauss':
        val = t
    if function == 'laplace':
        val = 2*(np.sqrt(1+t)-1)
    if function == 'cauchy':
        val = np.log(1+t)
    if function == 'geman':
        val = 1/(1+t)
    return val

def phi_prime(t,function='gauss'):
    '''
    Derivatives of noise function in Table 1 of Tarel et al. IRLS paper
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

def auxiliary_matrix(X,y,A,scale):
    '''
    Compute matrix of auxiliary variables w_ij
    '''
    
    n = len(X)
    m = len(A)
    W = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            W[i,j] = auxiliary_variable(X[i],A[j],y[i],scale)
            
    return W

def weight_matrix(W,function):
    '''
    Compute weight matrix from auxiliary variable matrix W
    '''

    L = np.zeros_like(W)
    n,m = W.shape
    for i in range(n):
        denominator_i = np.exp(-1/2*phi(W[i,:],function)).sum() + m*EPSILON
        for j in range(m):
#             print(i,j,denominator_i)
            L[i,j] = (EPSILON + np.exp(-1/2*phi(W[i,j],function)))/denominator_i*phi_prime(W[i,j],function)
            
    return L

def update_params(X,y,A,function,scale):
    '''
    Steps 2,3 of algorithm in Sec. 3
    '''
    W = auxiliary_matrix(X,y,A,scale)
    L = weight_matrix(W,function)
    n,m = L.shape
    A_next = np.zeros((m,X.shape[1]))

    for j in range(m):
        S = np.zeros((X.shape[1],X.shape[1]))
        t = np.zeros(X.shape[1])

        for i in range(n):
            l_i = L[i,j]
            S += l_i*np.dot(X[i].reshape(-1,1),X[i].reshape(-1,1).T)
            t += l_i*X[i]*y[i]
        a_j = np.dot(np.linalg.inv(S),t)
#         print(A_next,a_j)
        A_next[j] = a_j
    
    return A_next

def SMRF(X,y,A,function,scale):
    '''
    Algorithm of Sec 3.2
    '''
    tol = 0.001
    diff = 1
    A_curr = A
    
    while diff>tol:
        A_next = update_params(X,y,A_curr,function,scale)
        diff = np.linalg.norm(A_next.flatten()-A_curr.flatten())
        # print(A_next.flatten(),diff)
        A_curr = A_next
    
    return A_next