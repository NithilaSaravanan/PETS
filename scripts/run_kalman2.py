#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Importing the noisy function
from pets.noisy_input import noisy_signal

#Importing the kalman func from src/pets
from pets.kalman_known2 import kalman_algo

import json
import sys
import os.path as osp
import numpy as np

this_dir = osp.dirname(__file__)
#Navigating to the config files
config_dir = osp.join(this_dir,'..','configs','config_kalman_known.json')

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

print(len(t),'\n')
print(a,b, points,'\n')
y_dirty, y_true = noisy_signal()

kalman_algo()
