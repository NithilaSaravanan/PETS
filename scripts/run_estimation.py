#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is the main script that is run to execute the contents of this library, Run this is the CLI with
appropriate arguments to get the intended output.
"""

import argparse
import sys
import os.path as osp
import run_kalman
#from run_kalman import kalman_func

this_dir = osp.dirname(__file__)

def select_algo(meth):
	if meth == 'kalman_known':
		run_kalman()
	

def main():
    
	parser = argparse.ArgumentParser()
	#parser.add_argument('-c', help = 'Enter the configuration file with all the tunable parameters for the method selected', default = 'nope')
	parser.add_argument('-m', help = 'Select the estimation method you want to use (ADD VALS HERE)', default = 'nope')
	#parser.add_argument('-g',help = 'Select whether you want graphs in the output directory (Y / N)',default = 'Y')
	try:
		args = parser.parse_args()
	except:
		print('\n One or more parameters are missing or incorrect. Please take a look at the help and provide correct arguments')
		parser.print_help()
		sys.exit(0)

	#config_file = args.c
	method_select = args.m
	#graph_choice = args.g
	
	"""
	if config_file == 'nope':
		print('\n Please provide the config file. You can paste the absolute file path after -c \n')
		parser.print_help()
		sys.exit()
	"""	
	if method_select == 'nope':
		print('\n Please select one of the estimations algorithms and try again \n')
		parser.print_help()
		sys.exit()
		
	print("\n \n Please make sure to have the config file for the corresponding algorithm correctly populated and have the true and the noisy signal functions ready in the noisy_signal.py script")
	choice = input("\n Press y/Y to continue and proceed with the script")
	
	if (str.upper(choice) == 'Y'):
		select_algo(method_select)
	else:
		sys.exit(0)
	
if __name__=='__main__':
    main()
