#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:55:32 2022

Description: SRC file to generate the noisy signal that is to be estimated. Please use this function below to 
geenrate/import/process your noisy signal which will them be used by one of the estimation algorithms in the
package. 

@author: nithila
"""

import pandas as pd
import numpy as np
from scipy.integrate import odeint

"""
Add White noise of choice
"""
def add_AWGN(y, t, std):
    N = t.shape[0]    
    #noise = np.random.randn()(0, std, size = N)
    rng = np.random.default_rng()
    yM = np.array(y + std * rng.standard_normal(size=(t.shape[0])))
    return yM

"""
System Modelling (4th order LTI)
"""
def states_system(a, b, points, x0, aT, std):    
    #Model 4th Order System
    def model(x,t):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        dx1dt = x2
        dx2dt = x3
        dx3dt = x4
        dx4dt = (-aT[0])*x1 + (-aT[1])*x2 + (-aT[2])*x3 + (-aT[3])*x4
        dxdt = [dx1dt, dx2dt, dx3dt, dx4dt]
        return dxdt

    #Solve ODE
    def solve_ODE(t, x0, T):
        yT = np.empty_like(t)
        dyT = np.empty_like(t)
        ddyT = np.empty_like(t)
        dddyT = np.empty_like(t)
        yT[0] = x0[0] #set initial condition 
        dyT[0] = x0[1]
        ddyT[0] = x0[2]
        dddyT[0] = x0[3]
        for i in range(1,T):
            t_span = [t[i-1], t[i]]
            x = odeint(model, x0, t_span)
            yT[i] = x[1][0] #Store solution
            dyT[i] = x[1][1]
            ddyT[i] = x[1][2]
            dddyT[i] = x[1][3]
            x0 = x[1] #Next initial condition
        return yT, dyT, ddyT, dddyT
    
    t = np.linspace(a, b, points)
    y, dy, ddy, dddy = solve_ODE(t,x0, points)
    yM = add_AWGN(y, t, std)
    return yM, y, dy, ddy, dddy, t


def noisy_signal(a,b,points,ic,param):
	#Configure this std to add AWGN noise of a set std dev
	std = 0
	yM, yT, dyT, ddyT, dddyT, t = states_system(a, b, points, ic, param, std)
	return(yM,yT,dyT,ddyT,dddyT, std)




