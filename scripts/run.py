# !/usr/bin/env python3
# -*- coding: utf-8 -*-


import math 
import sys
import numpy as np
from scipy.integrate import odeint
sys.path.insert(0, '../src/pets')
sys.path.insert(0, '../configs')
from kernels import *
from rts_kalman import *
from results import *
# Configuration
estimate_parameters = True
include_derivatives = True


# Function declaration 

def state_space(x,t,A, B):
    """State Space Model"""
    
    y_out = np.matmul(A,x)
    return y_out


def add_gwn(y, std):
    """Add Gaussian white noise """
    
    noise = np.random.normal(0,std,size=y.shape[0])
    y_noise = np.array(y + noise)
    return y_noise


# Initialize System
a_i = np.array([1, 5, 5, 0])
#a_i = np.array([1.25, 3.5, 4.25, 3])

A = np.array((
    [0, 1, 0, 0], 
    [0, 0, 1, 0], 
    [0, 0, 0, 1], 
    (-a_i),
    ))

A1 = np.array((
    [0, 1, 0],
    [0, 0, 1],
    [1, -10, 0]
    ))

B = np.array([])

x_0 = np.array([0, 0, 0, 1])

x1_0 = np.array([1, 1, 0])
print("Canonical form: Matrix A\n", A)
print("\nInitial Condition:\n", x_0, "\n")

# Define interval from a to b 
a = 0
b = 6
interval = 60000
noise_std = 1
t = np.linspace(a, b, interval)

# Solve ODE
y_and_dys = odeint(state_space, x_0, t, args=(A,B))

# Get Original signal
y = y_and_dys[:,0]

# Add Gaussian White Noise
y_noise = add_gwn(y, noise_std)
y = y.reshape(y.shape[0],1)

# Calculating Signal to Noise Ratio
signal_power = np.mean(np.square(y))
noise_power = np.mean(np.square(y_noise))
SNR = signal_power/noise_power
SNR_dB = 10*np.log10(SNR)
print("SNR: ", SNR_dB)


#Applying RLS
if estimate_parameters:
    knots = 500
    tol = 0.01
    S_type = "diagonal"
    w_delta = 10**(6)
    order = 4
    print ("RLS Parameter Estimation in Progress ...")
    ak = RLS (y_noise, t, a, b, knots, tol, S_type, w_delta, order, verbose=True)
    A_EST = np.array((
    [0, 1, 0, 0], 
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    (-ak),
    ))

else:
    A_EST = A    

print("Estimated Parameters\n", A_EST, '\n')
C = np.array([[1., 0., 0., 0.]])
x_new = np.array([1., 1., 1., 1.])
x_predicted = get_kalman_rts(A_EST, C, x_new, 4, y_noise)

# Plot Results
plot_input_signal(y, y_noise, t)
plot_y_estimates(y_and_dys, x_predicted, t, include_derivatives)

# Error
error_mad, error_mse = noise_errors(y_and_dys, x_predicted, include_derivatives)