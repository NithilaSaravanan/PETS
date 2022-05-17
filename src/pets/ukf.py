#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains the code to execute the Unscented Kalman Filter algorithm.

@author : nithila
"""

import numpy as np
from scipy.linalg import expm

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF


def fx(x,dt):
    """
    This function returns the appropriate F matrix.
    """

    # Since the vector x is the augmented state and parameter vector
    # Dimension of the state will be the length-of-vector-x/2.  
    dim = int(len(x)/2)

    # Reconstruct the A matrix in canonical form
    A = np.zeros((dim*2, dim*2))
    A[(dim-1),0:dim] = -x[dim:]

    for i in range(0,(dim-1), 1):
        A[i, i+1] = 1
    
    # Compute the state transition matrix
    A = A*dt
    F = expm(A)
    
    # Matrix multiplication of F and x
    return F @ x
        
        
def hx(x):
    """
    Returns the measurement vector.
    """

    return [x[0]]


def kalmanukf_algo(config,yM):
    """
    Unscented Kalman filter algorithm
    """

    # Import the hyperparameters from the config file
    dimx = config['dim_x']
    w_length = config['dt_length']
    alp = config['alpha']
    bet = config['beta']
    kap = config['kappa']

    # Sigma points calculation, based on alpha, beta and kappa
    sig_points = MerweScaledSigmaPoints(n=2*dimx,alpha=alp, beta=bet, kappa=kap)
    
    # Setting up the UKF function based on the order of the system
    kf = UKF(dim_x=2*dimx, dim_z=1, dt=w_length, fx=fx, hx=hx, points=sig_points)
    kf.Q[0:dimx, 0:dimx] = Q_discrete_white_noise(dim=dimx, dt=w_length, var=float(config['q_var']))
    kf.Q[dimx:2*dimx, dimx:2*dimx] = Q_discrete_white_noise(dim=dimx, dt=w_length, var=float(config['q_var']))
    
    kf.P *= float(config['p_val'])
    kf.R *= float(config['r_val'])
    kf.x = np.array(config['init_cond'])

    mu, cov = kf.batch_filter(yM)
    M,P,C = kf.rts_smoother(mu, cov)

    # Calculate the mean of estimated parameters
    est_param = []
    for idx in range (dimx, dimx*2, 1):
        est_param.append(np.mean(M[:,idx]))

    # Returning the estimated parameters and states from the algorithm
    return est_param, M[:,0:dimx]



