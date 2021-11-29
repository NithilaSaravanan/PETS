#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math 
from scipy.integrate import odeint
from projection import *


def get_batch(y, t, knots, iteration, mode="random"):
	"""
	Getting batch of knots
	""" 

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
	"""
	Getting tau forward and backward
	"""

	F_indices = np.where(t <= ti)[0]
	B_indices = np.where(t > ti)[0]
	y_F = y[F_indices]
	y_B = y[B_indices]
	tau_F = t[F_indices]
	tau_B = t[B_indices]
	return y_F, y_B, tau_F, tau_B

def get_alpha_tau(ti, tau_F, tau_B, a, b, n, order):
	"""
	Calculating the Annhilator terms
	"""

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
	#alpha_ = np.array(alpha)
	alpha = np.array(alpha/f(n-1))
	return alpha


"""
Calculating kernels for P and Q matrices
"""
def get_KFn_vector(ti, tau_F, a, n, order):
	"""
	Forward kernels
	"""
	if tau_F.size == 1:
		tau_F = np.array([tau_F])
	N_tau = tau_F.shape[0]
	f = np.math.factorial
	gamma = np.zeros((order, N_tau))
	zeta = np.zeros((order+1, N_tau))
	delta = np.array(tau_F - a)

	for i in range(0, order):
		gamma[i, :] = np.array((((ti - tau_F)**(n+i))*f(n-1))/(f(n+i)))
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
	"""
	Backward kernel
	"""
	if tau_B.size == 1:
		tau_B = np.array([tau_B])
	N_tau = tau_B.shape[0]
	f = np.math.factorial
	gamma = np.zeros((order, N_tau))
	zeta = np.zeros((order+1, N_tau))
	delta = np.array(b - tau_B)

	for i in range(0, order):
		gamma[i, :] = np.array((((ti - tau_B)**(n+i))*f(n-1))/(f(n+i)))

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


"""
Calculating kernels for Covariance Matrix
"""

def get_KFn(ti, tau_F, a, n, ak, order):
	"""
	Forward kernels
	"""
	beta = np.concatenate((ak, 1), axis = None)
	beta = beta.reshape(order+1,1)
	
	KFn_vector = get_KFn_vector(ti, tau_F, a, n, order)
	KFn = np.multiply(beta, KFn_vector)
	KFn = np.sum(KFn, axis = 0)
	return KFn


def get_KBn(ti, tau_B, b, n, ak, order):
	"""
	Backward kernels
	"""
	beta = np.concatenate((ak, 1), axis = None)
	beta = beta.reshape(order+1,1)
	
	KBn_vector = get_KBn_vector(ti, tau_B, b, n, order)
	KBn = np.multiply(beta, KBn_vector)
	KBn = np.sum(KBn, axis = 0)
	return KBn


def get_KDSn(ti, tau_F, tau_B, a, b, n, ak, order):
	"""
	Summation
	"""
	beta = np.concatenate((ak, 1), axis = None)
	beta = beta.reshape(order+1,1)
	
	KFn_vector = get_KFn_vector(ti, tau_F, a, n, order)
	KBn_vector = get_KBn_vector(ti, tau_B, b, n, order)
	KDSn_vector = np.concatenate((KFn_vector, KBn_vector), axis = 1)
	
	KDSn = np.multiply(beta, KDSn_vector)
	KDSn = np.sum(KDSn, axis = 0)
	return KDSn


def get_LHS(y, t, tk, a, b, n, order):
	"""
	Get LHS
	"""
	
	f = np.math.factorial
	
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
	"""
	Get RHS
	"""

	f = np.math.factorial
	
	N = tk.shape[0]
	RHS_n = np.zeros((N, (order+1)))
	
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
		for k in range(0, (order+1)):
			RHS_n[r, k] = np.trapz(integrand[k, :], t)

		r += 1

	return RHS_n



def get_P_Q(y, t, tk, a, b, n_low, n_max, order):
	"""
	Calculate P and Q matrices
	"""

	f = np.math.factorial
	
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
	P = np.array(RHS[:, 0:(order)])

	#Arrange Q:
	Q = np.subtract(LHS, RHS[:, (order)])
	
	return P, Q


def get_S(y, t, tk, a, b, ak, n_low, n_max, var, order): 
	"""
	Get Diagonal Covariance Matrix
	"""

	f = np.math.factorial

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

			Cov = np.array((var**2)*(np.trapz(integrand, t)))
			Cov_inv[r1,r1] = np.array(1/Cov)

			r1 += 1
		
		S[(m)*N: (m+1)*N, (m)*N: (m+1)*N] = np.array(Cov_inv)
		m += 1
		
	return S

def get_OLS(y, t, a, b, n_low, n_max, order):
	"""
	Get OLS
	"""

	idx_OLS = np.arange(0, t.shape[0], t.shape[0]/10)
	idx_OLS = idx_OLS.astype(int)
	t_OLS = np.array(t[idx_OLS])
	P_OLS, Q_OLS = get_P_Q(y, t, t_OLS, a, b, n_low, n_max, order)
	a_OLS = np.matmul(np.linalg.pinv(P_OLS), Q_OLS)
	return a_OLS

def RLS (y, t, a, b, knots, tol, Stype,w_delta, order, verbose = False):
	"""
	RLS Algorithm
	"""    

	iteration = 0
	eps = np.inf
	mode = "random"
	
	n_low = 1
	n_max = order
	#Get initial estimation
	a_initial = get_OLS(y, t, a, b, n_low, n_max, order)
	ak = a_initial
	w_matrix = w_delta * (np.identity(order))

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
		seven = np.matmul(w_matrix, np.matmul(P.T,S))
		ak1 = np.add(ak,np.matmul(seven,six))
		
		eps = np.linalg.norm(ak1 - ak, np.inf)
		
		ak = ak1
		iteration = iteration + 1
		
		if verbose == True:
			print("In RLS Iteration : ", iteration,ak)

		#Break after a particular iteration
		if iteration > 100:
			break 
		
	return ak