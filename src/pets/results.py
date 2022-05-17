#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file generates the necessary results from the state estimation algorithm.


@author: mk
"""

import os
import numpy as np
np.set_printoptions(suppress=True)

import json
import math
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100 #set to 500 or 600 for crisper, high-resolution images


def find_mse(yc,yt):
    """
    Function to find RMSE
    """

    error_mse_y = (math.sqrt(mean_squared_error(yt, yc)))
    return error_mse_y


def find_mad(yc,yt):
    """
    Function to find MAD
    """

    error_mad_y = np.max(np.abs(yt-yc))
    return error_mad_y


def find_mae(yc, yt):
    """
    Function to find MAE
    """

    error_mae_y = mean_absolute_error(yt, yc)
    return error_mae_y


def plot_input_signal (y_true, y_measured, t, results_dir):
    """
    This function plots the measured and true value of the input states.
    """

    r1_dir = results_dir + "noisyY_trueY.png"
    plt.plot(t, y_measured, 'b', label='Noisy y')
    plt.plot(t, y_true, 'r', label='True y')
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.legend(loc='best')
    plt.savefig(os.path.abspath(r1_dir))
    plt.show()
    return

def plot_y_estimates (y_true, y_est, t, results_dir):
    """
    This function plots the estimated states and the true states for comparision.
    """

    r0_dir = results_dir + "ytrue_est.png"
    plt.plot(t, y_true[:,0], 'r', label='True y')
    plt.plot(t,y_est[:,0], 'b', label="Estimated y")
    plt.xlabel('time')
    plt.ylabel('y(t)')
    plt.legend(loc='best')
    plt.savefig(os.path.abspath(r0_dir))
    plt.show()
    
    for idx in range(1,len(y_true[0,:])):
        rn_dir = results_dir + "d"+str(idx)+"ytrue_est.png"
        plt.plot(t, y_true[:,idx], 'r', label= "True $\mathregular{y^{(%d)}}$" %idx)
        plt.plot(t,y_est[:,idx], 'b', label= "Estimated $\mathregular{y^{(%d)}}$" %idx)
        plt.xlabel('time')
        plt.ylabel("$\mathregular{y^{(%d)}}$t" %idx)
        plt.legend(loc='best')
        plt.savefig(os.path.abspath(rn_dir))
        plt.show()

    return

def plot_results(y_measured, y_true, y_est,t, results_dir):
    """
    Generate graphs to compare the estimated states and store the results.
    """

    plot_input_signal(y_true[:,0], y_measured, t, results_dir)
    plot_y_estimates(y_true, y_est, t, results_dir)
    

    rw2_dir = results_dir + "state_estimates.tsv"
    
    state_dict = {'y_measured' : y_measured,
                  'y_true'     : y_true[:,0],
                  'y_est'      : y_est[:,0]}
    
    for idx in range(1,len(y_true[0,:])):
        key_prefix = 'd'+str(idx)
        state_dict[key_prefix+'y_true'] = y_true[:,idx]
        state_dict[key_prefix+'y_est'] = y_est[:,idx]
        
    state_df = pd.DataFrame.from_dict(state_dict)
    state_df.to_csv(rw2_dir, sep='\t', encoding='utf-8', index = False)
        
    
def generate_error_metrics(y_true, y_est, results_dir):
    """
    Compute error metrics and store the results.
    """

    rw1_dir = results_dir + "rmse_mad_mae.txt"
    metric_dict = {'y_rmse':find_mse(y_est[:,0],y_true[:,0]),
                   'y_mad':find_mad(y_est[:,0],y_true[:,0]),
                   'y_mae':find_mae(y_est[:,0],y_true[:,0])}
    
    for idx in range(1,len(y_true[0,:])):
        key_prefix = 'd'+str(idx)
        metric_dict[key_prefix+'y_mse'] = find_mse(y_est[:,idx],y_true[:,idx])
        metric_dict[key_prefix+'y_mad'] = find_mad(y_est[:,idx],y_true[:,idx])
        metric_dict[key_prefix+'y_mae'] = find_mae(y_est[:,idx],y_true[:,idx])
        
    with open(os.path.abspath(rw1_dir),'w') as data: 
        json.dump(metric_dict, data, indent=2)