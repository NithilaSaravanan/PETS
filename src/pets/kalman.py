#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the code to execute the Kalman Filter algorithm.

@author : nithila
"""

import numpy as np
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

def kalman_algo(config, yM, ak):
    
    # Import the parameters from the config file
    a = config['a']
    b = config['b']
    interval = config['points']
    dim_x = config['dim_x']
    window_dt = config['dt_length']
    
    # Configure Kalman filter parameters
    k_filter = KalmanFilter(dim_x=dim_x, dim_z=1)
    
    # Initial state condition
    new_x_0 = np.array(config['init_cond'])
    k_filter.x = new_x_0
    
    # Reconstruct the A matrix in canonical form
    A = np.zeros((dim_x, dim_x))
    A[-1,:] = -(ak)
    for i in range(dim_x, 1, -1):
        A[-i, -i+1] = 1
        
    # Compute the State Transition Matrix F 
    F = expm(A*window_dt)
    k_filter.F = F
    
    # Output matrix
    C = np.zeros((dim_x))
    C[0] = 1
    k_filter.H = np.array([C])
    
    # Hyper-parameters to tune
    k_filter.P *= config['p_val']
    k_filter.R = config['r_val']
    k_filter.Q = Q_discrete_white_noise(dim=dim_x, dt=window_dt, var=config['q_var'])
    
    # Measurement signal to kalman filter
    y_input = yM
    
    # Batch filter and RTS smoother
    (mu, cov, _, _) = k_filter.batch_filter(y_input.tolist())
    (x_smooth, P, K, Pp) = k_filter.rts_smoother(mu, cov)
    
    return x_smooth