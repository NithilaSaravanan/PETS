#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from scipy.linalg import expm

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

#only requires the config file and the measured signal - no need for the true signal here
def kalman_algo(config,yM):
	dimx = config['dim_x']
	ak = config['a_k']
	
	kf = KalmanFilter(dim_x= dimx, dim_z = 1)
	
	w_length = config['dt_length']
	
	if dimx == 1:
		a_kal = ak
		
	elif dimx == 2:
		a_kal = np.zeros((2,2))
		vals = [1, -ak[0], -ak[1]]
		pos = [(0,1), (1,0),(1,1)]
		rows,cols = zip(*pos)
		a_kal[rows,cols] = vals
		
	elif dimx == 3:
		a_kal = np.zeros((3,3))
		vals = [1,1,-ak[0], -ak[1], -ak[2]]
		pos = [(0,1), (1,2), (2,0), (2,1), (2,2)]
		rows,cols = zip(*pos)
		a_kal[rows,cols] = vals
		
	elif dimx == 4:
		a_kal = np.zeros((4,4))
		vals = [1,1,1,-ak[0],-ak[1],-ak[2],-ak[3]]
		pos = [(0,1),(1,2),(2,3),(3,0),(3,1),(3,2),(3,3)]
		rows,cols = zip(*pos)
		a_kal[rows,cols] = vals
	
	else:
		print('\n Error in dimension, please retry with dimensions between 1 to 4')
		sys.exit(0)
		
	a_kal = a_kal*w_length
	kf.F = expm(a_kal)# Fixed F matrix for all windows
	
	#IMPORTANT: Keep the measured state as the first one, if not, manually change these according to your situation
	if dimx == 1:
		kf.H = np.array([[1]])
	elif dimx == 2:
		kf.H = np.array([[1,0]])
	elif dimx == 3:
		kf.H = np.array([[1,0,0]])
	elif dimx == 4:
		kf.H = np.array([[1,0,0,0]])
	else:
		print('\n Error in dimension, please retry with dimensions between 1 to 4')
		sys.exit(0)
		
	
	kf.Q = Q_discrete_white_noise(dim = dimx, dt = w_length, var= float(config['q_var']))
	kf.P = float(config['p_val'])* np.eye(dimx) # Initial condition for covariance
	kf.x = np.array(config['init_cond']) # initial conditions
	kf.R = float(config['r_val'])# Measurement noise


	#Make sure that yM is in the correct format i.e. LIST
	Fs = [kf.F for t in range(len(yM))]    
	Hs = [kf.H for t in range(len(yM))]
	Rs = [kf.R for t in range(len(yM))]
	Qs = [kf.Q for t in range(len(yM))]
	    
	mu, cov,bf1,bf2 = kf.batch_filter(yM, Rs =Rs, Fs = Fs, Hs = Hs, Qs = Qs)
	M,P,C, rts1 = kf.rts_smoother(mu, cov, Fs = Fs, Qs = Qs)

	if dimx ==1:
		return(M[:,0])
	elif dimx == 2:
		return(M[:,0], M[:,1])
	elif dimx == 3:
		return(M[:,0], M[:,1], M[:,2])
	elif dimx == 4:
		return(M[:,0], M[:,1], M[:,2], M[:,3])

