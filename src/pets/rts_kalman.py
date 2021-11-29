#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math 
import sys
import numpy as np

from scipy.integrate import odeint
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
sys.path.insert(0, '../../configs')
from kernelconfigs import *
from kalmanconfigs import *

def get_kalman_rts(A, C, x_0, order, y_noise, verbose=False):

	# Define Kalman filter
	window_dt = (b-a)/interval
	k_filter = KalmanFilter(dim_x=order, dim_z=1)

	# Initial state condition
	k_filter.x = x_0

	# State Transition Matrix F 
	F = expm(A*window_dt)
	k_filter.F = F

	# Output matrix
	k_filter.H = C

	# Parameters to tune
	k_filter.P *= kalman_P
	k_filter.R = kalman_R
	k_filter.Q = Q_discrete_white_noise(dim=order, \
										dt=window_dt, \
										var=kalman_Q_var)

	# Measurement signal to kalman filter
	y_input = y_noise

	# Batch filter and RTS smoother
	(mu, cov, _, _) = k_filter.batch_filter(y_input.tolist())
	(x_smooth, P, K, Pp) = k_filter.rts_smoother(mu, cov)

	# Return
	if verbose == True:
		return x_smooth, P, K, Pp
	else:
		return x_smooth