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

def noisy_signal():
    
    #Import/generate/simulate noisy signal here. DONOT generate derivatives of the noisy signal here.
    Ym = [1,2,3,4,5] # Dirty signal/ measured signal with noise
    Yt = [1,2,3,4,5] #True signal, without noise (for error calculations)
    
    return Ym, Yt

