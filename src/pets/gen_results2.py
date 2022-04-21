#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 03:46:27 2022

This code can be used to generate outputs, store the values of the clean signal and the errors assocaited with them

@author: nithila
"""

import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)
import math 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100 #set to 500 or 600 for crisper, high-res images

#func to find RMSE
def find_mse(yc,yt):
	error_mse_y = (math.sqrt(mean_squared_error(yt, yc)))
	return error_mse_y

#func to find MAD
def find_mad(yc,yt):
	error_MAD_y = np.max(np.abs(yt-yc))
	return error_MAD_y


#generating output for a 4th order system -> dimx = 4
def results4(yM, yT, dyT, ddyT, dddyT, yE, dyE, ddyE, dddyE, t, results_dir, awgn_std):
	
	print('Im in!')
	
	#True vs noisy plot
	r1_dir = results_dir + "noisyY_trueY.png"
	plt.plot(t, yM, label = "Noisy y")
	plt.plot(t, yT, label = "True y")
	plt.title("True v/s Noisy Input Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.show()
	plt.savefig(r1_dir)
	
	r2_dir = results_dir + "ytrue_est.png"
	plt.plot(t, yT, label = "True y")
	plt.plot(t, yE, label = "Estimated y")
	plt.title("True v/s Estimated Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.show()
	plt.savefig(r2_dir)

	r3_dir =  results_dir + "dytrue_est.png"
	plt.plot(t, dyT, label = "True $\mathregular{y^{(1)}}$ ")
	plt.plot(t, dyE, label = "Estimated $\mathregular{y^{(1)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(1)}}$")
	plt.xlabel('time')
	plt.ylabel('dy(t)')
	plt.legend()
	plt.show()
	plt.savefig(r3_dir)

	r4_dir = results_dir + "ddytrue_est.png"
	plt.plot(t, ddyT, label = "True $\mathregular{y^{(2)}}$")
	plt.plot(t, ddyE, label = "Estimated $\mathregular{y^{(2)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(2)}}$")
	plt.xlabel('time')
	plt.ylabel('ddy(t)')
	plt.legend()
	plt.show()
	plt.savefig(r4_dir)

	r5_dir = results_dir + "dddytrue_est.png"
	plt.plot(t, dddyT, label = "True $\mathregular{y^{(3)}}$")
	plt.plot(t, dddyE, label = "Estimated $\mathregular{y^{(3)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(3)}}$")
	plt.xlabel('time')
	plt.ylabel('dddy(t)')
	plt.legend()
	plt.show()
	plt.savefig(r5_dir)


