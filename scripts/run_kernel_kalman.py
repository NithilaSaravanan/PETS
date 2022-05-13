#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file executes the RRLS algorithm to estimate the parameters followed
by the kalman algorithm to reconstruct the states and their derivatives.
"""

import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), '..', 'src'))
from pets.rrls import rrls_solver
from pets.noisy_input import noisy_signal
from pets.kalman import kalman_algo
from pets.gen_results import results4

import json
import numpy as np


def kernel_kalman():
    """
    Function that executes RRLS algorithm
    """

    this_dir = osp.dirname(__file__)
    config_dir = osp.join(this_dir, '..', 'configs',
                          'config_kernel_kalman.json')

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

    # Import data from config file
    a, b, points = [config['a'], config['b'], config['points']]
    t = np.linspace(a, b, points)
    ic = config['init_cond']
    param = config['a_k']
    w_mat = config['w_mat']
    order = config['dim_x']
    knots = config['knots']
    tol = config['tol']
    results_dir = config['res_dir']
    # Get noisy signal
    y_arr_true, y_noise, awgn_std = noisy_signal(a, b, points, ic, param)

    # Estimate parameters using RRLS algorithm
    #ak = rrls_solver(y_noise, t, a, b, knots, tol, "diagonal", w_mat , order)
    ak = np.array(param)
    print(ak)
    
    y_arr_est = kalman_algo(config, y_noise, ak)
    results4(y_noise, y_arr_true[:,0], y_arr_true[:,1], y_arr_true[:,2], y_arr_true[:,3], \
             y_arr_est[:,0], y_arr_est[:,1], y_arr_est[:,2], y_arr_est[:,3], t, results_dir)
    
    