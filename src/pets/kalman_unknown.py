#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import expm

from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF


#Function to return the approriate F matrix.

def fx(x,dt):
    dim = int(len(x)/2)
    A = np.zeros((dim*2, dim*2))
    A[(dim-1),0:dim] = -x[dim:]

    for i in range(0,(dim-1), 1):
        A[i, i+1] = 1
    
    A = A*dt
    F = expm(A)
    
    return F @ x
        
        
def hx(x):
    return [x[0]]

#only requires the config file and the measured signal
def kalmanukf_algo(config,yM):
    
    dimx = config['dim_x']
    w_length = config['dt_length']
    alp = config['alpha']
    bet = config['beta']
    kap = config['kappa']

    #sigma points calculation, based on alpha, beta and kappa
    sig_points = MerweScaledSigmaPoints(n=2*dimx,alpha=alp, beta=bet, kappa=kap)
    
    #setting up the UKF function based on the order of the system

    kf = UKF(dim_x=2*dimx, dim_z=1, dt=w_length, fx=fx, hx=hx, points=sig_points)

    kf.Q[0:dimx, 0:dimx] = Q_discrete_white_noise(dim=dimx, dt=w_length, var=float(config['q_var']))
    kf.Q[dimx:2*dimx, dimx:2*dimx] = Q_discrete_white_noise(dim=dimx, dt=w_length, var=float(config['q_var']))
    
    kf.P *= float(config['p_val'])
    kf.R *= float(config['r_val'])
    kf.x = np.array(config['init_cond'])

    mu, cov = kf.batch_filter(yM)
    M,P,C = kf.rts_smoother(mu, cov)

    #printing the parameters to stdout only - not printing them to a txt file
    print("\n")

    print("The average parameter values of the system are estimated to be :")
    for idx in range (dimx, dimx*2,1):
        print (np.mean(M[:,idx]))
    print("The actual true parameter values of the system are ",config['a_k'] )

    #returning the estimated parameters from the algorithm
    return M[:,0:dimx]



