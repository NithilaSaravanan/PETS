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



def find_mse(yc,yt):
	error_mse_y = (math.sqrt(mean_squared_error(yt, yc)))
	return error_mse_y

def find_mad(yc,yt):
	error_MAD_y = np.max(np.abs(yt-yc))
	return error_MAD_y



def results1(yc,yd,yt,t,r_dir,der_avail):
	r1_dir = r_dir + "NoisyY_TrueY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yd, label = "Noisy y")
	plt.plot(t, yt, label = "True y")
	plt.title("True v/s Noisy Input Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r1_dir)
	
	r2_dir = r_dir + "TrueY_EstY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yt, label = "True y")
	plt.plot(t, yc, label = "Estimated y")
	plt.title("True v/s Estimated Y")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r2_dir)
	
	mse = find_mse(yc,yt)
	mad = find_mad(yc,yt)
	rw1_dir = r_dir + "mse_mad.txt" 
	lines = ['MSE is ' + mse, 'MAD is ' + mad]
	with open(rw1_dir, 'w') as f:
	    for line in lines:
	        f.write(line)
	        f.write('\n')
	
	rw2_dir = r_dir + "Yvalues.tsv"
	data_dict = {'Ytrue': yt, 'Ynoisy':yd, 'Yest':yc}
	df = pd.DataFrame(data_dict)
	
	df.to_csv(rw2_dir, sep='\t', encoding='utf-8', index = False)

def results2(yc,yd,yt,t,r_dir,der_avail):
	#yc will have 2 components
	yc1 = yc[:,0]
	yc2 = yc[:,1]
	
	r1_dir = r_dir + "NoisyY_TrueY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yd, label = "Noisy y")
	plt.plot(t, yt, label = "True y")
	plt.title("True v/s Noisy Input Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r1_dir)
	
	r2_dir = r_dir + "TrueY_EstY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yt, label = "True y")
	plt.plot(t, yc1, label = "Estimated y")
	plt.title("True v/s Estimated Y")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r2_dir)
	
	r3_dir = r_dir + "EstdY.png"
	plt.figure(figsize=(10,5))
	#plt.plot(t, dyT, label = "True $\mathregular{y^{(1)}}$ ")
	plt.plot(t, yc2, label = "Estimated $\mathregular{y^{(1)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(1)}}$")
	plt.xlabel('time')
	plt.ylabel('dy(t)')
	plt.legend()
	plt.savefig(r3_dir)
	
	mse = find_mse(yc1,yt)
	mad = find_mad(yc1,yt)
	rw1_dir = r_dir + "mse_mad.txt" 
	lines = ['MSE is ' + mse, 'MAD is ' + mad]
	with open(rw1_dir, 'w') as f:
	    for line in lines:
	        f.write(line)
	        f.write('\n')
	
	rw2_dir = r_dir + "Yvalues.tsv"
	data_dict = {'Ytrue': yt, 'Ynoisy':yd, 'Yest':yc1, 'dYest':yc2}
	df = pd.DataFrame(data_dict)
	
	df.to_csv(rw2_dir, sep='\t', encoding='utf-8', index = False)

def results3(yc,yd,yt,t,r_dir,der_avail):
	#yc will have 3 components
	yc1 = yc[:,0]
	yc2 = yc[:,1]
	yc3 = yc[:,2]
	
	r1_dir = r_dir + "NoisyY_TrueY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yd, label = "Noisy y")
	plt.plot(t, yt, label = "True y")
	plt.title("True v/s Noisy Input Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r1_dir)
	
	r2_dir = r_dir + "TrueY_EstY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yt, label = "True y")
	plt.plot(t, yc1, label = "Estimated y")
	plt.title("True v/s Estimated Y")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r2_dir)
	
	r3_dir = r_dir + "EstdY.png"
	plt.figure(figsize=(10,5))
	#plt.plot(t, dyT, label = "True $\mathregular{y^{(1)}}$ ")
	plt.plot(t, yc2, label = "Estimated $\mathregular{y^{(1)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(1)}}$")
	plt.xlabel('time')
	plt.ylabel('dy(t)')
	plt.legend()
	plt.savefig(r3_dir)
	
	r4_dir = r_dir + "EstddY.png"
	#plt.plot(t, ddyT, label = "True $\mathregular{y^{(2)}}$")
	plt.plot(t, yc3, label = "Estimated $\mathregular{y^{(2)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(2)}}$")
	plt.xlabel('time')
	plt.ylabel('ddy(t)')
	plt.legend()
	plt.savefig(r4_dir)
	
	mse = find_mse(yc1,yt)
	mad = find_mad(yc1,yt)
	rw1_dir = r_dir + "mse_mad.txt" 
	lines = ['MSE is ' + mse, 'MAD is ' + mad]
	with open(rw1_dir, 'w') as f:
	    for line in lines:
	        f.write(line)
	        f.write('\n')
	
	rw2_dir = r_dir + "Yvalues.tsv"
	data_dict = {'Ytrue': yt, 'Ynoisy':yd, 'Yest':yc1, 'dYest':yc2, 'ddYest':yc3}
	df = pd.DataFrame(data_dict)
	
	df.to_csv(rw2_dir, sep='\t', encoding='utf-8', index = False)

def results4(yc,yd,yt,t,r_dir,der_avail):
	
	#yc will have 4 components, divide those components and plotting then seperately 
	
	yc1 = yc[:,0]
	yc2 = yc[:,1]
	yc3 = yc[:,2]
	yc4 = yc[:,3]
	
	r1_dir = r_dir + "NoisyY_TrueY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yd, label = "Noisy y")
	plt.plot(t, yt, label = "True y")
	plt.title("True v/s Noisy Input Signal")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r1_dir)
	
	r2_dir = r_dir + "TrueY_EstY.png"
	plt.figure(figsize=(10,5))
	plt.plot(t, yt, label = "True y")
	plt.plot(t, yc1, label = "Estimated y")
	plt.title("True v/s Estimated Y")
	plt.xlabel('time')
	plt.ylabel('y(t)')
	plt.legend()
	plt.savefig(r2_dir)
	
	r3_dir = r_dir + "EstdY.png"
	plt.figure(figsize=(10,5))
	#plt.plot(t, dyT, label = "True $\mathregular{y^{(1)}}$ ")
	plt.plot(t, yc2, label = "Estimated $\mathregular{y^{(1)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(1)}}$")
	plt.xlabel('time')
	plt.ylabel('dy(t)')
	plt.legend()
	plt.savefig(r3_dir)
	
	r4_dir = r_dir + "EstddY.png"
	#plt.plot(t, ddyT, label = "True $\mathregular{y^{(2)}}$")
	plt.plot(t, yc3, label = "Estimated $\mathregular{y^{(2)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(2)}}$")
	plt.xlabel('time')
	plt.ylabel('ddy(t)')
	plt.legend()
	plt.savefig(r4_dir)
	
	r5_dir = r_dir + "EstdddY.png"
	plt.figure(figsize=(10,5))
	#plt.plot(t, dddyT, label = "True $\mathregular{y^{(3)}}$")
	plt.plot(t, yc4, label = "Estimated $\mathregular{y^{(3)}}$")
	plt.title("Reconstruction of $\mathregular{y^{(3)}}$")
	plt.xlabel('time')
	plt.ylabel('dddy(t)')
	plt.legend()
	plt.savefig(r5_dir)
	
	mse = find_mse(yc1,yt)
	mad = find_mad(yc1,yt)
	rw1_dir = r_dir + "mse_mad.txt" 
	lines = ['MSE is ' + mse, 'MAD is ' + mad]
	with open(rw1_dir, 'w') as f:
	    for line in lines:
	        f.write(line)
	        f.write('\n')
	
	rw2_dir = r_dir + "Yvalues.tsv"
	data_dict = {'Ytrue': yt, 'Ynoisy':yd, 'Yest':yc1, 'dYest':yc2, 'ddYest':yc3, 'dddYest':yc4}
	df = pd.DataFrame(data_dict)
	
	df.to_csv(rw2_dir, sep='\t', encoding='utf-8', index = False)
	
	
	

