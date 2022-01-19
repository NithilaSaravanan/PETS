#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 19:49:34 2022

Kalman script (Scripts folder) that takes config, input and all other realted parameters from the ru_estimation.py 
and transfer all of it to the src kalman code. Receives output from there and publishes results to the specified
directory. 

@author: nithila
"""

from pets.kalman_known import kalman_algo
from pets.noisy_input import noisy_signal
from pets.gen_results import results1,results2, results3,results4
import json
import sys
import os.path as osp
import numpy as np

this_dir = osp.dirname(__file__)

#Navigating to the config files
config_dir = osp.join(this_dir,'..','configs','config_kalman.json')

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
	
if (len(config['ini_cond']) != int(config['dim_x'])):
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
y_dirty, y_true = noisy_signal()
y_clean = kalman_algo(config, y_dirty) #y_clean has the clean signals

results_dir = config['res_dir']
#Break this into multiple signal, generate graph and values, and put it into the directory. The n represents that there is no derivative information
# associated with the dirty signal (something that should be part of the code for the kernels - if needed for kalman, this can be seperately
# generated)

if config['dim_x']==1:
	results1(y_clean, y_dirty,y_true,t,results_dir,'n')
elif config['dim_x'] == 2:
	results2(y_clean, y_dirty,y_true,t,results_dir,'n')
elif config['dim_x'] == 3:
	results3(y_clean, y_dirty,y_true,t,results_dir,'n')
elif config['dim_x'] == 4:
	results4(y_clean, y_dirty,y_true,t,results_dir,'n')

print(f'\n Your results have been printed to',results_dir)