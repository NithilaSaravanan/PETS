#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The main script that must be run to execute the contents of this library.
Run this is the CLI with appropriate arguments to get the intended output.
"""

import argparse
import sys
import os.path as osp
from run_kalman import kalman_run
from run_kernel_kalman import kernel_kalman
from run_kalmanukf import kalmanukf_run

this_dir = osp.dirname(__file__)

def select_algo(meth):
	if meth == 'kalman_known':
		kalman_run()
	elif meth == 'kernel_kalman':
		kernel_kalman()
	elif meth == 'kalman_ukf':
		kalmanukf_run()
	else:
		print("Unrecognized estimation algorithm. Execution terminated. \n")
		sys.exit(0)	

def main():
    
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', help = 'Select the estimation method you want to use - \
					kalman_known, kalman_ukf, kernel_kalman  ', default = 'none')

	try:
		args = parser.parse_args()
	except:
		print('\n One or more parameters are missing or incorrect. \
			Please take a look at the help and provide correct arguments')
		parser.print_help()
		sys.exit(0)

	#config_file = args.c
	method_select = args.m
	
	if method_select == 'none':
		print('\n Please select one of the estimations algorithms and try again \n')
		parser.print_help()
		sys.exit()
		
	print("\nPlease make sure to have the config file for", method_select, 
		"and have the true and the noisy signal available in the noisy_signal.py script.",
		"\nPlease refer to help to know about all the options.")
	choice = input("\nPress y/Y to proceed :")
	
	if (str.upper(choice) == 'Y'):
		#Commenting just for Anaconda test
		select_algo(method_select)
	else:
		sys.exit(0)
	
if __name__=='__main__':
    main()
