#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script executes the RRLS and the Projection algorithm. 
The configuration parameters are validated and the necessary functions are invoked.

@author : manoj
"""

import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), '..', 'src'))

# Importing the noisy function
from pets.noisy_input import noisy_signal

# Importing the RRLS and Projection functions
from pets.rrls import rrls_solver, projection_algo

# Importing the results functions from src/pets
from pets.results import plot_results, generate_error_metrics

import json
import numpy as np

# Navigating to the config files
this_dir = osp.dirname(__file__)
config_dir = osp.join(this_dir,'..','configs','config_kernel_projection.json')

def kernel_projection():
    """
    Function that executes RRLS algorithm for parameter estimation.
    Followed by Projection for state estimation.
    """
    
    # Loading the config parameters into the code
    with open(config_dir) as config_file:
        config = json.load(config_file)
    
    # Checking if the config parameters are correct
    if ((config['dim_x'] < 1) or (config['dim_x'] > 4)):
        print('\n Please enter a valid dimension \
              for the Kalman algorithm (1-4)')
        sys.exit(0)
    
    if (len(config['a_k']) != int(config['dim_x'])):
        print('\n Please enter the correct number of values \
              for the system parameters \n')
        sys.exit(0)
    
    if (len(config['init_cond']) != int(config['dim_x'])):
        print('\n Please enter the correct number of values \
              for the initial condition \n')
        sys.exit(0)
    
    if osp.isdir(config['res_dir']):
        pass
    else:
        print("Please enter a valid directory to store your results")
        sys.exit(0)
        
    # Retreive the parameters from the config file.
    a, b, points = [config['a'], config['b'], config['points']]
    t = np.linspace(a, b, points)
    ic = config['init_cond']
    param = config['a_k']
    w_mat = config['w_mat']
    order = config['dim_x']
    knots = config['knots']
    tol = config['tol']
    s_type = config['s_type']
    results_dir = config['res_dir']

    # Retreive the noisy signal, the true signal and the noise standard deviation values.
    y_arr_true, y_measured, awgn_std = noisy_signal(a, b, points, ic, param)

    # Estimate parameters using RRLS algorithm
    ak = rrls_solver(y_measured, t, a, b, knots, tol, s_type, w_mat , order)
    #ak = np.array(param)
    print("The estimated parameters are :", ak)
    
    # Estimate the states, using the Projection algorithm.
    y_arr_est = projection_algo(config, y_measured, t, ak)
    print("States have been reconstructed!")
    
    # Generate graphs comparing true and estimated state values
    plot_results(y_measured, y_arr_true, y_arr_est, t, results_dir)

    # Generate error metrics and store the results 
    generate_error_metrics(y_arr_true, y_arr_est, results_dir)

    print("Complete states information, graphs and error metrics are saved at", osp.abspath(results_dir))