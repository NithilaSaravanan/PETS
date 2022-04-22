#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing the noisy function
from pets.noisy_input import noisy_signal

#Importing the kalman func from src/pets
from pets.kalman_known import kalman_algo

#Importing the results functions from src/pets
from pets.gen_results import results4

import json
import sys
import os.path as osp
import numpy as np

this_dir = osp.dirname(__file__)
#Navigating to the config files
config_dir = osp.join(this_dir,'..','configs','config_kalman_known.json')


def kalman_run():
	#Loading the config parameters into the code
	with open(config_dir) as config_file:
		config = json.load(config_file)

	#Checking if the config parameters are correct
	if ((config['dim_x'] < 1) or (config['dim_x'] > 4)):
		print('\n Please enter a valid dimension for the Kalman algorithm (1-4) \n')
		sys.exit(0)

	if (len(config['a_k']) != int(config['dim_x'])):
		print('\n Please enter the correct number of values for the system parameters \n')
		sys.exit(0)
		
	if (len(config['init_cond']) != int(config['dim_x'])):
		print('\n Please enter the correct number of values for the initial condition \n')
		sys.exit(0)


	if osp.isdir(config['res_dir']):  
	    pass  
	elif osp.isfile(config['res_dir']):  
		print("Please enter the correct directory to store your results in")
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


	#Checking order to get the correct call
	if config['dim_x']==1:
		yM,yT, awgn_std = noisy_signal(a,b,points,ic,param)
	elif config['dim_x'] == 2:
		yM, yT, dyT, awgn_std = noisy_signal(a,b,points,ic,param)
	elif config['dim_x'] == 3:
		yM, yT, dyT, ddyT, awgn_std = noisy_signal(a,b,points,ic,param)
	elif config['dim_x'] == 4:
		yM, yT, dyT, ddyT, dddyT, awgn_std  =  noisy_signal(a,b,points,ic,param)

	#Getting clean states based on the order of the system
	if config['dim_x']==1:
		yE = kalman_algo(config,yM)
	elif config['dim_x'] == 2:
		yE, dyE = kalman_algo(config,yM)
	elif config['dim_x'] == 3:
		yE, dyE, ddyE = kalman_algo(config,yM)
	elif config['dim_x'] == 4:
		yE, dyE, ddyE, dddyE = kalman_algo(config,yM)

	print("States have been reconstructed!")

	results_dir = config['res_dir']

	#sending true and estimated signals for calculations and graphing
	if config['dim_x']==1:
		results1(yM, yT, yE, t, results_dir, awgn_std)
	elif config['dim_x'] == 2:
		results2(yM, yT, dyT, yE, dyE, t, results_dir, awgn_std)
	elif config['dim_x'] == 3:
		results3(yM, yT, dyT, ddyT, yE, dyE, ddyE, t, results_dir, awgn_std)
	elif config['dim_x'] == 4:
		results4(yM, yT, dyT, ddyT, dddyT, yE, dyE, ddyE, dddyE, t, results_dir, awgn_std)

	print("All plots, metrics and value dumps have been saved at ", results_dir)

