#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script executes the Unscented Kalman Filter (UKF) algorithm. 
The configuration parameters are validated and the necessary functions are invoked.

@author : nithila
"""

import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), '..', 'src'))

#Importing the noisy function
from pets.noisy_input import noisy_signal

#Importing the kalman func from src/pets
from pets.ukf import kalmanukf_algo

#Importing the results functions from src/pets
from pets.results import plot_results, generate_error_metrics

import json
import numpy as np

# Navigating to the config files
this_dir = osp.dirname(__file__)
config_dir = osp.join(this_dir,'..','configs','config_kalman_unknown.json')


def kalmanukf_run():

	# Loading the config parameters into the code
	with open(config_dir) as config_file:
		config = json.load(config_file)

	#Checking if the config parameters are correct
	if ((config['dim_x'] < 1) or (config['dim_x'] > 4)):
		print('\n Please enter a valid dimension for the Kalman algorithm (1-4) \n')
		sys.exit(0)

	if (len(config['a_k']) !=int(config['dim_x'])):
		print('\n Please enter the correct number of values for the system parameters \n')
		sys.exit(0)
		
	if (len(config['init_cond']) !=(2* int(config['dim_x']))):
		print('\n Please enter the correct number of values for the initial condition \n')
		sys.exit(0)


	if osp.isdir(config['res_dir']):  
	    pass
	else:  
		print("Please enter the correct directory to store your results in")
		sys.exit(0)
	
	# Retreive the parameters from the config file.
	a, b, points = [config['a'], config['b'], config['points']]
	t = np.linspace(a, b, points)
	ic = config['init_cond']
	param = config['a_k']
	dim = config['dim_x']
	results_dir = config['res_dir']

	# Retreive the noisy signal, the true signal and the noise standard deviation values.
	y_arr_true, yM, awgn_std  =  noisy_signal(a,b,points,ic[:dim],param)
	
	# Estimate the parameters and the states of the system, using the Unscented Kalman Filter
	ak, y_arr_est = kalmanukf_algo(config,yM)
	print("The estimated parameters are :", ak)
	
	print("States have been reconstructed!")

	# Generate graphs comparing true and estimated state values
	plot_results(yM, y_arr_true, y_arr_est, t, results_dir)

	# Generate error metrics and store the results 
	generate_error_metrics(y_arr_true, y_arr_est, results_dir)

	print("Complete states information, graphs and error metrics are saved at", osp.abspath(results_dir))