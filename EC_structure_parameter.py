import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as nc
import math
from datetime import datetime, timedelta
import geopy.distance
from matplotlib import gridspec
from scipy.special import jv
from scipy.integrate import dblquad

######################################################################
#### Structure parameters from EC ####################################
######################################################################

def struct_def_t(X, Y, r, freq, avg_U):  
    # Determine a sample shift that corresponds best to the separation distance
    sample_shift = int(round(r * freq / avg_U))
    # Calculation the actual separation distance
    r_act = avg_U * sample_shift / freq
    
    # Calculate the structure parameter
    Cxy = (X[:len(X)-sample_shift] - X[sample_shift:]) * (Y[:len(Y)-sample_shift] - Y[sample_shift:])
    Cxy = np.nanmean(Cxy) / r_act ** (2/3)
    
    return Cxy

def struct_def_t_time(X, Y, t, freq, avg_U):  
    # Determine a sample shift that corresponds best to the separation distance
    sample_shift = int(round(t * freq))
    # Calculation the actual separation distance
    r_act = avg_U * t
    
    # Calculate the structure parameter
    Cxy = (X[:len(X)-sample_shift] - X[sample_shift:]) * (Y[:len(Y)-sample_shift] - Y[sample_shift:])
    Cxy = np.nanmean(Cxy) / r_act ** (2/3)
    
    return Cxy


