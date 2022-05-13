#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 05:52:48 2022

@author: mk
"""

import numpy as np
np.set_printoptions(suppress=True)
import math 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100 #set to 500 or 600 for crisper, high-res images

#func to find RMSE
def find_mse(yc,yt):
	error_mse_y = (math.sqrt(mean_squared_error(yt, yc)))
	return error_mse_y

#func to find MAD
def find_mad(yc,yt):
	error_mad_y = np.max(np.abs(yt-yc))
	return error_mad_y

#func to find MAE
def find_mae(yc, yt):
    error_mae_y = mean_absolute_error(yt, yc)
    return error_mae_y

def plot_input_signal (y_true, y_measured, t, results_dir):
    r1_dir = results_dir + "noisyY_trueY.png"
    plt.plot(t, y_measured, 'b', label='Noisy y')
    plt.plot(t, y_true, 'r', label='True y')
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.legend(loc='best')
    plt.savefig(r1_dir)
    plt.show()
    return

def plot_y_estimates (y_true, y_est, t, results_dir):
    r0_dir = results_dir + "ytrue_est.png"
    plt.plot(t, y_true[:,0], 'r', label='True y')
    plt.plot(t,y_est[:,0], 'b', label="Estimated y")
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.legend(loc='best')
    plt.savefig(r0_dir)
    plt.show()
    
    for idx in range(1,len(y_true[0,:])):
        rn_dir = results_dir + "d"+str(idx)+"ytrue_est.png"
        plt.plot(t, y_true[:,idx], 'r', label= "True $\mathregular{y^{(%d)}}$" %idx)
        plt.plot(t,y_est[:,idx], 'b', label= "Estimated $\mathregular{y^{(%d)}}$" %idx)
        plt.xlabel('time')
        plt.ylabel("$\mathregular{y^{(%d)}}$t" %idx)
        plt.legend(loc='best')
        plt.savefig(rn_dir)
        plt.show()

    return

def plot_results(y_measured, y_true, y_est,t, results_dir):
    plot_input_signal(y_true[:,0], y_measured, t, results_dir)
    plot_y_estimates(y_true, y_est, t, results_dir)
    