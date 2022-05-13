#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 22:15:39 2022

@author: mk
"""

import numpy as np
from scipy import integrate

def get_batch(y, t, knots, iteration, mode="random"):
    np.random.seed(iteration)
    
    if mode == "random":
        idx =np.random.choice(len(t),size=knots,replace=False)
        idx= np.sort(idx)
        tk = np.array(t[idx])
        return  tk
    
    elif mode == "midpoint":
        t_knots = np.array([t[t.shape[0]//2]])
        return t_knots

def get_tau(y, t, ti):
    F_indices = np.where(t <= ti)[0]
    B_indices = np.where(t > ti)[0]
    y_F = y[F_indices]
    y_B = y[B_indices]
    tau_F = t[F_indices]
    tau_B = t[B_indices]
    return y_F, y_B, tau_F, tau_B

def get_alpha_tau(ti, tau_F, tau_B, a, b, n, order):
    f = np.math.factorial
    
    #Calculate alpha forward
    delta_time_F = np.array((ti - tau_F)**(n-1))
    delta_tau_F = np.array((tau_F - a)**order)
    alpha_F = np.multiply(delta_time_F, delta_tau_F)
    
    #Calculate alpha backward
    delta_time_B = np.array((ti - tau_B)**(n-1))
    delta_tau_B = np.array((b - tau_B)**order)
    alpha_B = np.multiply(delta_time_B, delta_tau_B)
    
    alpha = np.concatenate((alpha_F, alpha_B), axis = 0)
    alpha = np.array(alpha/f(n-1))
    return alpha

def get_KFn_vector(ti, tau_F, a, n, order):
    if tau_F.size == 1:
        tau_F = np.array([tau_F])
    N_tau = tau_F.shape[0]
    f = np.math.factorial
    gamma = np.zeros((order, N_tau))
    zeta = np.zeros((order+1, N_tau))
    delta = np.array(tau_F - a)

    for i in range(0, order):
        gamma[i, :] = np.array((((ti - tau_F)**(n+i)))/(f(n+i)))
    for j in range(1, order+1):
        zeta[order,:]= zeta[order,:]+( ((-1)**(j+1)) * gamma[j-1, :] * \
                        ((delta**(order-j))/f(order-j))  * \
                            (f(order)/(f(j)*f(order-j))) * f(order))
    
    for i in range(0, order):
        if i==0:
            j=0
            zeta[i,:]= zeta[i,:]+(gamma[order-i+j-1, :] * \
                    ((delta**(order-j))/f(order-j)) * ((-1)**(j+1)) * \
                        (f(i)/(f(j)*f(i-j))) * f(order))
        else:
            for j in range(0, i+1):
                zeta[i,:]= zeta[i,:]+(gamma[order-i+j-1, :] * \
                            ((delta**(order-j))/f(order-j)) * \
                            ((-1)**(j+1)) * (f(i)/(f(j)*f(i-j))) * f(order))

    return zeta



def get_KBn_vector(ti, tau_B, b, n,order):
    if tau_B.size == 1:
        tau_B = np.array([tau_B])
    N_tau = tau_B.shape[0]
    f = np.math.factorial
    gamma = np.zeros((order, N_tau))
    zeta = np.zeros((order+1, N_tau))
    delta = np.array(b - tau_B)

    for i in range(0, order):
        gamma[i, :] = np.array((((ti - tau_B)**(n+i)))/(f(n+i)))

    for j in range(1, order+1):
        zeta[order,:] = zeta[order,:]+(gamma[j-1, :] * \
                        ((delta**(order-j))/f(order-j)) * \
                        (f(order)/(f(j)*f(order-j))) * f(order))
    
    for i in range(0, order):    
        if i==0:
            j=0
            zeta[i,:]= zeta[i,:]+ (gamma[order-i+j-1, :] * \
                    ((delta**(order-j))/f(order-j)) * \
                        (f(i)/(f(j)*f(i-j))) * f(order))
        else:
            for j in range(0, i+1):
                zeta[i,:]= zeta[i,:]+ (gamma[order-i+j-1, :] * \
                        ((delta**(order-j))/f(order-j)) * \
                        (f(i)/(f(j)*f(i-j))) * f(order))

    if n > 0:
        zeta = np.array(-1.0*zeta)
    return zeta

def get_KDSn(ti, tau_F, tau_B, a, b, n, ak, order):
    N = order+1
    beta = np.concatenate((ak, 1), axis = None)
    beta = beta.reshape(N,1)
    
    KFn_vector = get_KFn_vector(ti, tau_F, a, n, order)
    KBn_vector = get_KBn_vector(ti, tau_B, b, n, order)
    KDSn_vector = np.concatenate((KFn_vector, KBn_vector), axis = 1)
    
    KDSn = np.multiply(beta, KDSn_vector)
    KDSn = np.sum(KDSn, axis = 0)
    return KDSn

def get_KFn(ti, tau_F, a, n, ak, order):
    N = order+1
    beta = np.concatenate((ak, 1), axis = None)
    beta = beta.reshape(N,1)
    
    KFn_vector = get_KFn_vector(ti, tau_F, a, n, order)
    KFn = np.multiply(beta, KFn_vector)
    KFn = np.sum(KFn, axis = 0)
    return KFn

def get_KBn(ti, tau_B, b, n, ak, order):
    N = order+1
    beta = np.concatenate((ak, 1), axis = None)
    beta = beta.reshape(N,1)
    
    KBn_vector = get_KBn_vector(ti, tau_B, b, n, order)
    KBn = np.multiply(beta, KBn_vector)
    KBn = np.sum(KBn, axis = 0)
    return KBn

def get_LHS(y, t, tk, a, b, n, order):
    
    N = tk.shape[0]
    LHS_n = np.zeros(N)
    
    r = 0    
    for ti in tk:

        y_F, y_B, tau_F, tau_B = get_tau(y, t, ti)

        #Calculate alpha
        alpha = get_alpha_tau(ti, tau_F, tau_B, a, b, n, order)

        #Calculate Integrand:
        integrand = np.multiply(alpha, y)

        #Integrate
        integral = np.trapz(integrand, t)

        #Store value
        LHS_n[r] = np.array((integral))
        r += 1
    
    return LHS_n


def get_RHS(y, t, tk, a, b, n, order):
    
    Num = order+1
    N = tk.shape[0]
    RHS_n = np.zeros((N, Num))
    
    r = 0
    for ti in tk:
        y_F, y_B, tau_F, tau_B = get_tau(y, t, ti)

        #Calculate Kernel vector
        KFn_vector = get_KFn_vector(ti, tau_F, a, n, order)
        KBn_vector = get_KBn_vector(ti, tau_B, b, n, order)
        KDSn_vector = np.concatenate((KFn_vector, KBn_vector), axis = 1)

        #Calculate integrands
        integrand = np.multiply(KDSn_vector, y)

        #Integrate
        for k in range(0, Num):
            RHS_n[r, k] = np.trapz(integrand[k, :], t)

        r += 1

    return RHS_n


def get_P_Q(y, t, tk, a, b, n_low, n_max, order):
    
    n_vector = np.arange(n_low, n_max)
    
    for n in n_vector:
        LHS_n = get_LHS(y, t, tk, a, b, n, order)
        RHS_n = get_RHS(y, t, tk, a, b, n, order)
        
        if n == n_vector[0]:
            LHS = np.array(LHS_n)
            RHS = np.array(RHS_n)
        else:
            LHS = np.concatenate((LHS, LHS_n), axis = None)
            RHS = np.vstack((RHS, RHS_n))
    
    #Arrange P:
    P = np.array(RHS[:, 0:order])

    #Arrange Q:
    Q = np.subtract(LHS, RHS[:, order])
    
    return P, Q

def get_S(y, t, tk, a, b, ak, n_low, n_max, var, order): 
    
    N = tk.shape[0]
    
    n_vector = np.arange(n_low, n_max)
    
    S = np.zeros(((n_vector.shape[0])*N, (n_vector.shape[0])*N))
    
    m = 0
    for n in n_vector:
        
        Cov_inv = np.zeros((N, N))

        r1 = 0
        for ti in tk:
            y_F, y_B, tau_F, tau_B = get_tau(y, t, ti)

            KDSn = get_KDSn(ti, tau_F, tau_B, a, b, n, ak, order)
            alpha = get_alpha_tau(ti, tau_F, tau_B, a, b, n, order)
            integrand = np.array((alpha - KDSn)**2)

            Cov = np.array(var*(np.trapz(integrand, t)))
            Cov_inv[r1,r1] = np.array(1/Cov)

            r1 += 1
        
        S[(m)*N: (m+1)*N, (m)*N: (m+1)*N] = np.array(Cov_inv)
        m += 1
        
    return S

def get_uniform_batch(y, t, sample_points):

    idx = np.arange(0, t.shape[0], int(t.shape[0]/sample_points))
    idx = np.concatenate((idx, t.shape[0] - 1), axis = None) #Include last point
    idx = idx.astype(int)
    yk = np.array(y[idx])
    tk = np.array(t[idx])

    return yk, tk

def linearized_system(x, t, ak, order):
    
    x_dot = []
    x_last = -ak[0]*x[0]

    for idx in range (1, order):
        x_dot.append(x[idx])
        x_last += -ak[idx]*x[idx]

    x_dot.append(x_last)

    return x_dot

def get_fund_solutions(y, t, ak, order):
    basis = np.identity(order)
    fund_solution = np.zeros((order, t.shape[0]))
    for i in range(0, order):
        x0_i = basis[i, :]
        fund_solution[i, :] = integrate.odeint(linearized_system, x0_i, t, args =(ak,order))[:,0]
    return fund_solution

#Define L2 inner product
def L2p(f, g, t):
    integrand = np.multiply(f, g)
    integral = np.trapz(integrand, t)
    return integral

#Orthonormalize the fundamental equations
def orthonormalize_L2(fund_solution, t, order):
    N = t.shape[0]
    v = np.zeros((order, N))
    ortho_fund_solution = np.zeros((order, N))
    
    v[0, :] = np.array(fund_solution[0, :])
    
    for idx in range(1,order):
        v[idx, :] = np.array(fund_solution[idx, :])
        for jdx in range(0,idx):
            v[idx, :] += - ((L2p(fund_solution[idx, :], v[jdx, :], t)/L2p(v[jdx, :], v[jdx, :], t))*v[jdx, :])

    for i in range(0, order):
        ortho_fund_solution[i, :] = np.array(v[i, :]/np.sqrt(L2p(v[i, :], v[i, :], t)))

    return ortho_fund_solution

def reconstruct_state(y, t, ak, order):
    N = t.shape[0]
    yE = np.zeros((order, N))
    
    #Get orthonormal vectors
    fund_solution = get_fund_solutions(y, t, ak, order)
    ortho_fund_solution = orthonormalize_L2(fund_solution, t, order)
    c = np.zeros(order)
    for i in range(0, order):
        c[i] = L2p(y,ortho_fund_solution[i, :], t )
        yE[i, :] = np.array(c[i]*ortho_fund_solution[i, :])
        
    for i in range(0, order):
        for j in range(i+1, order):
            if i ==0 and j == 1:
                test_ortho = L2p(ortho_fund_solution[i, :], ortho_fund_solution[j, :], t)
            else:
                test_ortho = np.concatenate((test_ortho, L2p(ortho_fund_solution[i, :], ortho_fund_solution[j, :], t)), axis = None)
    
    yE = np.sum(yE, axis = 0)
    return yE

def get_OLS(y, t, a, b, n_low, n_max, order):
    idx_OLS = np.arange(0, t.shape[0], t.shape[0]/10)
    idx_OLS = idx_OLS.astype(int)
    t_OLS = np.array(t[idx_OLS])
    P_OLS, Q_OLS = get_P_Q(y, t, t_OLS, a, b, n_low, n_max, order)
    a_OLS = np.matmul(np.linalg.pinv(P_OLS), Q_OLS)
    return a_OLS

def rrls_solver (y, t, a, b, knots, tol, Stype,w_delta, order):
    
    iteration = 0
    eps = np.inf
    mode = "random"
    
    n_low = 1
    n_max = order
    
    #Get initial estimation
    a_initial = get_OLS(y, t, a, b, n_low, n_max, order)
    #a_initial = np.array([0.0, 0.0, 0.0, 0.0])
    ak = a_initial
    
    w_matrix = w_delta * (np.identity(order))

        
    #tk = get_batch(y, t, knots, iteration, mode)
    while tol<eps:
        
        tk = get_batch(y, t, knots, iteration, mode)
        P, Q = get_P_Q(y, t, tk, a, b, n_low, n_max, order)
        
        yE_1 = reconstruct_state(y, t, ak, order)
        var = np.var(y - yE_1) 
        
        
        S = get_S(y, t, tk, a, b, ak, n_low, n_max, var, order) #Use this for diagonal matrix
            
        #Algo starts
        
        one = np.matmul(w_matrix, P.T)
        two = np.matmul(np.matmul(P,w_matrix),P.T)
        three = np.add(two,np.linalg.inv(S))
        four = np.linalg.inv(three)
        
        K = np.matmul(one,four)
        
        five = np.matmul(K,P)
        w_matrix = np.matmul(np.subtract(np.identity(order), five), w_matrix)
        
        six = np.subtract(Q, np.matmul(P,ak))
        ak1 = np.add(ak,np.matmul(K,six))
        
        eps = np.linalg.norm(ak1 - ak, np.inf)
        
        ak = ak1
        iteration = iteration + 1
        print(iteration,ak)
        
    return ak
