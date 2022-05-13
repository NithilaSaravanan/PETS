#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:55:32 2022

Description: SRC file to generate the noisy signal that is to be estimated. Please use this function below to 
geenrate/import/process your noisy signal which will them be used by one of the estimation algorithms in the
package. 

@author: nithila
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint

"""
Add White noise of choice
"""
def add_awgn(y, t, std):
    N = t.shape[0]    
    #noise = np.random.randn()(0, std, size = N)
    rng = np.random.default_rng()
    yM = np.array(y + std * rng.standard_normal(size=(t.shape[0])))
    return yM

"""
System Modelling (4th order LTI)
"""
# Function declaration 

def state_space(x,t,A, B):
    """State Space Model"""
    
    y_out = np.matmul(A,x)
    return y_out


def noisy_signal(a,b,points,ic,param):
    #Configure this std to add AWGN noise of a set std dev
    noise_std = 1
    t = np.linspace(a, b, points)
    # Solve ODE
    ak = np.array((param))
    dim = len(ak)

    A = np.zeros((dim, dim))
    A[-1,:] = -(ak)
    for i in range(dim, 1, -1):
        A[-i, -i+1] = 1
    B = np.array([])
    y_arr_true = odeint(state_space, ic, t, args=(A,B))
    y = y_arr_true[:,0]
    y_noise = add_awgn(y, t, noise_std)
    return y_arr_true, y_noise, noise_std