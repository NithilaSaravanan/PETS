#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Please use the function noisy_signal below to geenrate/import/process your noisy (measured) signal.  

@author: nithila
"""

import numpy as np
from scipy.integrate import odeint

"""
Add White noise of choice
"""
def add_awgn(y, t, std):
    """
    Additive white gaussian noise to embed noise to the simulated signal
    """

    N = t.shape[0]    
    rng = np.random.default_rng()
    yM = np.array(y + std * rng.standard_normal(size=(t.shape[0])))
    return yM

def state_space(x,t,A, B):
    """State Space Model"""
    
    y_out = np.matmul(A,x)
    return y_out


def noisy_signal(a,b,points,ic,param):
    """
    This function will be called by the algorithm to obtain the true and noisy signal.
    User can modify this function to suit their requirements.
    """

    # Standard deviation of the gwn
    noise_std = 0

    t = np.linspace(a, b, points)
    ak = np.array((param))
    dim = len(ak)

    # Reconstruct the A matrix in canonical form
    A = np.zeros((dim, dim))
    A[-1,:] = -(ak)
    for i in range(dim, 1, -1):
        A[-i, -i+1] = 1

    B = np.array([])
    y_arr_true = odeint(state_space, ic, t, args=(A,B))
    y = y_arr_true[:,0]
    
    # Add gaussian white noise to the simulated signal
    y_noise = add_awgn(y, t, noise_std)

    return y_arr_true, y_noise, noise_std