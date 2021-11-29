#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
from scipy import integrate

"""
Reconstruct yE(t)
"""

def get_uniform_batch(y, t, sample_points):
	"""
	Uniform batch
	"""

	idx = np.arange(0, t.shape[0], int(t.shape[0]/sample_points))
	idx = np.concatenate((idx, t.shape[0] - 1), axis = None)
	idx = idx.astype(int)
	yk = np.array(y[idx])
	tk = np.array(t[idx])
	return yk, tk


def linearized_system(x, t, ak, order):
	"""
	Linearized system
	"""

	x_dot = []
	x_last = -ak[0]*x[0]

	for idx in range (1, order):
		x_dot.append(x[idx])
		x_last += -ak[idx]*x[idx]

	x_dot.append(x_last)
	return x_dot


def get_fund_solutions(y, t, ak, order):
	"""
	Fundamental solutions
	"""

	basis = np.identity(order)
	fund_solution = np.zeros((order, t.shape[0]))
	for i in range(0, order):
		x0_i = basis[i, :]
		fund_solution[i, :] = integrate.odeint(linearized_system, x0_i, t, args =(ak,order))[:,0]
	return fund_solution


def L2p(f, g, t):
	"""
	L2 Inner Product
	"""

	integrand = np.multiply(f, g)
	integral = np.trapz(integrand, t)
	return integral

def orthonormalize_L2(fund_solution, t, order):
	"""
	Orthonormalize fundamental solution
	"""

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
	"""
	Reconstruct state
	"""

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
