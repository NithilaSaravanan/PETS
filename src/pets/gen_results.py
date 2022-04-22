#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 03:46:27 2022

This code can be used to generate outputs, store the values of the clean signal and the errors assocaited with them

@author: nithila
"""

import numpy as np
import pandas as pd
import json
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
	
	print('Graphs are being plotted!')
	
	#True vs noisy plot
	r1_dir = results_dir + "noisyY_trueY.png"
	plt.plot(t, yM, label = "Noisy y")
	plt.plot(t, yT, label = "True y")
	plt.title("True v/s Noisy Input Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r1_dir)
	plt.show()
	
	r2_dir = results_dir + "ytrue_est.png"
	plt.plot(t, yT, label = "True y")
	plt.plot(t, yE, label = "Estimated y")
	plt.title("True v/s Estimated Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r2_dir)
	plt.show()

	r3_dir =  results_dir + "dytrue_est.png"
	plt.plot(t, dyT, label = "True $\mathregular{y^{(1)}}$ ")
	plt.plot(t, dyE, label = "Estimated $\mathregular{y^{(1)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(1)}}$")
	plt.xlabel('time')
	plt.ylabel('dy(t)')
	plt.legend()
	plt.savefig(r3_dir)
	plt.show()

	r4_dir = results_dir + "ddytrue_est.png"
	plt.plot(t, ddyT, label = "True $\mathregular{y^{(2)}}$")
	plt.plot(t, ddyE, label = "Estimated $\mathregular{y^{(2)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(2)}}$")
	plt.xlabel('time')
	plt.ylabel('ddy(t)')
	plt.legend()
	plt.savefig(r4_dir)
	plt.show()

	r5_dir = results_dir + "dddytrue_est.png"
	plt.plot(t, dddyT, label = "True $\mathregular{y^{(3)}}$")
	plt.plot(t, dddyE, label = "Estimated $\mathregular{y^{(3)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(3)}}$")
	plt.xlabel('time')
	plt.ylabel('dddy(t)')
	plt.legend()
	plt.savefig(r5_dir)
	plt.show()

	#calculate rmse, mad and output them to a txt file
	rw1_dir = results_dir + "rmse_mad.txt"
	metric_dict = {'y_rmse':find_mse(yE,yT),'y_mad':find_mad(yE,yT),
'dy_rmse':find_mse(dyE,dyT),'dy_mad':find_mad(dyE,dyT),
'ddy_rmse':find_mse(ddyE,ddyT),'ddy_mad':find_mad(ddyE,ddyT),
'dddy_rmse':find_mse(dddyE,dddyT),'dddy_mad':find_mad(dddyE,dddyT)}
	#metric_df = pd.DataFrame(metric_dict)
	#metric_df.to_csv(rw1_dir, sep='\t', encoding='utf-8', index= True) 

	with open(rw1_dir,'w') as data: 
		json.dump(metric_dict, data, indent=2)


	#dump all values into a tsv file
	rw2_dir = results_dir + "state_estimates.tsv"
	state_dict = {'y_measured':yM, 'y_true': yT, 'y_est':yE, 
	'dy_true':dyT, 'dy_est':dyE, 
	'ddy_true':ddyT, 'ddy_est':ddyE,
	'dddy_true':dddyT, 'dddy_est':dddyE}
	
	state_df = pd.DataFrame.from_dict(state_dict)
	state_df.to_csv(rw2_dir, sep='\t', encoding='utf-8', index = False)

