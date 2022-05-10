#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from scipy.linalg import expm

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF


#Function to return the approriate F matrix, based on the order

def fx1(x,dt):
	A = np.zeros((2,2))
	vals = [-x[1]]
	pos = [(0,0)]
	rows,cols = zip(*pos)
	A[rows,cols] = vals
	#multiplying by dt
	A =A*dt
	#finding e^(A)
	F = expm(A)
	return F @ x

def fx2(x,dt):
	A = np.zeros((4,4))
	vals = [1,-x[2], -x[3]]
	pos = [(0,1), (1,0), (1,1)]
	rows, cols = zip(*pos)
	A[rows,cols] =vals
	A=A*dt
	F=expm(A)
	return F @ x

def fx3(x,dt):
	A = np.zeros((6,6))
	vals = [1,1,-x[3], -x[4], -x[5]]
	pos = [(0,1), (1,2), (2,0), (2,1), (2,2)]
	rows, cols = zip(*pos)
	A[rows,cols] = vals
	A=A*dt
	F=expm(A)
	return F @ x

def fx4(x,dt):
	A = np.zeros((8,8))
	vals = [1,1,1,-x[4], -x[5], -x[6], -x[7]]
	pos = [(0,1), (1,2), (2,3), (3,0), (3,1), (3,2), (3,3)]
	rows, cols = zip(*pos)
	A=A*dt
	F=expm(A)
	return F @ x

def hx(x):
	return [x[0]]

#only requires the config file and the measured signal - no need for true derivatives
def kalmanukf_algo(config,yM):
	
	dimx = config['dim_x']
	w_length = config['dt_length']
	alp = config['alpha']
	bet = config['beta']
	kap = config['kappa']

	#sigma points calculation, based on alpha, beta and kappa
	sig_points = MerweScaledSigmaPoints(n=2*dimx,alpha=alp, beta=bet, kappa=kap)
	
	#setting up the UKF function based on the order of the system
	if dimx == 1:
		kf = UKF(dim_x=2*dimx, dim_z=1, dt=w_length, fx=fx1, hx=hx, points=sig_points)
	elif dimx == 2:
		kf = UKF(dim_x=2*dimx, dim_z=1, dt=w_length, fx=fx2, hx=hx, points=sig_points)
	elif dimx == 3:
		kf = UKF(dim_x=2*dimx, dim_z=1, dt=w_length, fx=fx3, hx=hx, points=sig_points)
	elif dimx == 4:
		kf = UKF(dim_x=2*dimx, dim_z=1, dt=w_length, fx=fx4, hx=hx, points=sig_points)
	else:
		print('\n Error in dimension, please retry with dimensions between 1 to 4')
		sys.exit(0)

	kf.Q[0:dimx, 0:dimx] = Q_discrete_white_noise(dim=dimx, dt=w_length, var=float(config['q_var']))
	kf.Q[dimx:2*dimx, dimx:2*dimx] = Q_discrete_white_noise(dim=dimx, dt=w_length, var=float(config['q_var']))
	
	kf.P *= float(config['p_val'])
	kf.R *= float(config['r_val'])
	kf.x = np.array(config['init_cond'])

	#print(kf.Q,kf.P, kf.R, kf.x)

	mu, cov = kf.batch_filter(yM)
	M,P,C = kf.rts_smoother(mu, cov)

	#printing the parameters to stdout only - not printing them to a txt file
	print("\n")
	if dimx ==1:
		print("The average parameter value of the system is estimated to be ",np.mean( M[:,1]))
		print("The actual true parameter value of the system is ",config['a_k'] )
	elif dimx == 2:
		print("The average parameter values of the system are estimated to be ",np.mean( M[:,2]), np.mean(M[:,3]))
		print("The actual true parameter values of the system are ",config['a_k'] )
	elif dimx == 3:
		print("The average parameter values of the system are estimated to be ",np.mean( M[:,3]), np.mean(M[:,4]),np.mean(M[:,5]))
		print("The actual true parameter values of the system are ",config['a_k'] )
	elif dimx == 4:
		print("The average parameter values of the system are estimated to be ",np.mean( M[:,4]), np.mean(M[:,5]), np.mean(M[:,6]), np.mean(M[:,7]))

		print("The actual true parameter values of the system are ",config['a_k'] )


	#returning the estimated parameters from the algorithm
	
	if dimx ==1:
		return(M[:,0])
	elif dimx == 2:
		return(M[:,0], M[:,1])
	elif dimx == 3:
		return(M[:,0], M[:,1], M[:,2])
	elif dimx == 4:
		return(M[:,0], M[:,1], M[:,2], M[:,3])	



