#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The main script that executes the contents of this library.
Run this is the CLI with appropriate arguments to get the intended output.
"""

import argparse
import sys
import os.path as osp
from run_kalman import kalman_run
from run_kernel_kalman import kernel_kalman
from run_kalmanukf import kalmanukf_run
from run_kernel_projection import kernel_projection

this_dir = osp.dirname(__file__)

def run_algorithm(meth):
    """
    This method executes the estimation algorithm chosen by the user.
    """
    
    print('\n')
    if meth == 'kalman_statesonly':
        print ("Executing Kalman Filter algorithm.\n")
        kalman_run()

    elif meth == 'kernel_kalman':
        print ("Executing RRLS and Kalman Filter algorithm.\n")
        kernel_kalman()
    
    elif meth == 'kernel_projection':
        print ("Executing RRLS and Projection algorithm.\n")
        kernel_projection()
    
    elif meth == 'kalman_ukf':
        print ("Executing Unscented Kalman Filter algorithm.\n")
        kalmanukf_run()
    
    else:
        print("Unrecognized estimation algorithm. Execution terminated. \n")
        sys.exit(0)    


def main():
    """
    main function
    """
    
    #Parse input argument(s)
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help = 'Select the estimation method you want to use - \
                    kalman_statesonly, kalman_ukf, kernel_kalman  ', default = 'none')

    try:
        args = parser.parse_args()
    except:
        print('\n One or more parameters are missing or incorrect. \
            Please take a look at the help and provide correct arguments')
        parser.print_help()
        sys.exit(0)

    # Estimation algorithm selected by the user
    method_select = args.m
    
    if method_select == 'none':
        print('\n Please select one of the following estimation algorithms and try again. \n')
        parser.print_help()
        sys.exit()
        
    print("\nPlease make sure you have the config file available in PETS/configs for", method_select,
        "and the true and the noisy signal are available in the PETS/src/pets/noisy_signal.py script.")
    choice = input("\nPress y/Y to proceed :")
    
    if (str.upper(choice) != 'Y'):
        sys.exit(0)

    # Execute the selected algorithm.
    run_algorithm(method_select)
        
    
if __name__=='__main__':
    main()