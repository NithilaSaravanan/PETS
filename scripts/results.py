#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:48:05 2021

@author: mk
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

def plot_input_signal (y_true, y_input, t):
    fig_0 = plt.figure(figsize=(10, 4))
    plt.plot(t, y_input, 'b', label='y_noise')
    plt.plot(t, y_true, 'r', label='y_true')
    plt.xlabel('time(sec)')
    plt.legend(loc='best')
    return
    
def plot_y_estimates (y_true, y_est, t, derivatives=False):

    fig_0 = plt.figure(figsize=(10, 4))
    plt.plot(t, y_true[:,0], 'b', label='True y')
    plt.plot(t,y_est[:,0], 'r', label="Estimated y")
    plt.xlabel('time(seconds)')
    plt.legend(loc='best')
    plt.show()
    
    if derivatives == False:
        return
    
    fig_1 = plt.figure(figsize=(10, 4))
    plt.plot(t, y_true[:,1], 'b', label='True $\mathregular{y^{(1)}}$')
    plt.plot(t,y_est[:,1], 'r', label='Estimated $\mathregular{y^{(1)}}$')
    plt.xlabel('time(seconds)')
    plt.legend(loc='best')
    plt.show()

    fig_2 = plt.figure(figsize=(10, 4))
    plt.plot(t, y_true[:,2], 'b', label='True $\mathregular{y^{(2)}}$')
    plt.plot(t,y_est[:,2], 'r', label='Estimated $\mathregular{y^{(2)}}$')
    plt.xlabel('time(seconds)')
    plt.legend(loc='best')
    plt.show()

    fig_3 = plt.figure(figsize=(10, 4))
    plt.plot(t, y_true[:,3], 'b', label='True $\mathregular{y^{(3)}}$')
    plt.plot(t,y_est[:,3], 'r', label='Estimated $\mathregular{y^{(3)}}$')
    plt.xlabel('time(seconds)')
    plt.legend(loc='best')
    plt.show()
    return

def noise_errors (y_true, y_est, derivatives=False):
    lst_MAD = []
    lst_MSE = []

    mad_y = np.max(np.abs(y_true[:,0]-y_est[:,0]))
    mse_y = (math.sqrt(mean_squared_error(y_true[:,0], y_est[:,0])))
    lst_MAD.append(mad_y)
    lst_MSE.append(mse_y)
    
    if derivatives == False:
        return lst_MAD, lst_MSE
    
    mad_y = np.max(np.abs(y_true[:,1]-y_est[:,1]))
    mse_y = (math.sqrt(mean_squared_error(y_true[:,1], y_est[:,1])))
    lst_MAD.append(mad_y)
    lst_MSE.append(mse_y)
    
    mad_y = np.max(np.abs(y_true[:,1]-y_est[:,1]))
    mse_y = (math.sqrt(mean_squared_error(y_true[:,1], y_est[:,1])))
    lst_MAD.append(mad_y)
    lst_MSE.append(mse_y)

    mad_y = np.max(np.abs(y_true[:,1]-y_est[:,1]))
    mse_y = (math.sqrt(mean_squared_error(y_true[:,1], y_est[:,1])))
    lst_MAD.append(mad_y)
    lst_MSE.append(mse_y)    
    
    return lst_MAD, lst_MSE