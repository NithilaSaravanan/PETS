#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(__file__), '..', 'src'))

#Importing the noisy function
from pets.noisy_input import noisy_signal

#Importing the kalman func from src/pets
from pets.kalman_unknown import kalmanukf_algo

#Importing the results functions from src/pets
from pets.results import plot_results, generate_error_metrics

import json
import numpy as np

this_dir = osp.dirname(__file__)
#Navigating to the config files
config_dir = osp.join(this_dir,'..','configs','config_kalman_unknown.json')


def kalmanukf_run():

	#Loading the config parameters into the code
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
	elif osp.isfile(config['res_dir']):  
		print("Please enter the correct folder to store your results in")
		sys.exit(0)
	else:  
		print("Please enter the correct directory to store your results in")
		sys.exit(0)
	#If all this checks out, simply send the configs and the dirty signal to the kalman_known src and get
	#back a clean signal, which will then we sent out for processing, graphing, etc
	a,b,points = [config['a'], config['b'], config['points']]
	t = np.linspace(a, b, points)
	ic = config['init_cond']
	param = config['a_k']
	dim = config['dim_x']
	#Checking order to get the correct call
	y_arr_true, yM, awgn_std  =  noisy_signal(a,b,points,ic[:dim],param)
	
	
	#Getting clean states based on the order of the system
	y_arr_est = kalmanukf_algo(config,yM)

	print("States have been reconstructed!")

	results_dir = config['res_dir']

	#sending true and estimated signals for calculations and graphing
	plot_results(yM, y_arr_true, y_arr_est, t, results_dir)
	generate_error_metrics(y_arr_true, y_arr_est, results_dir)

	print("All plots, metrics and value dumps have been saved at ", results_dir)


