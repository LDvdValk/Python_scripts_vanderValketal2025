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


#############################################
######## attempt 2  for spectrum ############
#############################################

def aperture_average_Ward(K_turb,D,x,L):
    a = (2 * jv(1, (0.5*K_turb*D*x/L) ) )   /      (0.5*K_turb*D*x/L)
    b = (2 * jv(1, (0.5*K_turb*D*(1-(x/L) ) ) ) )    /     (0.5*K_turb*D*(1-(x/L) ) )
    return (a**2)*(b**2)

def wave_prop_new(K_turb,L,x,k_rad,u,f):
    a = (np.sin(((K_turb**2)*x*(L-x))/(2*k_rad*L)))**2
    b = ((K_turb*u)**2 - (2*np.pi*f)**2)**-0.5
    return a*b

def var_Wang_new(K_turb,L,x,k_rad,Cnn,D,u,f):
    var_comb = turb_intensity(K_turb,Cnn)*wave_prop_new(K_turb,L,x,k_rad,u,f)*aperture_average_Ward(K_turb,D,x,L)
    constant = 4*((np.pi)**2)*(k_rad**2)
    return constant*var_comb

def Theoretical_spectrum2(D,f_trans,L,u,Cnn): #Still needs to be scaled to Cnn
    c = 299792458 #m/s
    k_rad = 2*np.pi*f_trans/c
    freq = np.logspace(-4,3,num=100)
    S_array = np.zeros(shape = (len(freq)))
    S_error_array = np.zeros(shape = (len(freq)))
    int_x0 = 0.000
    int_x1 = 1   
    int_x1_new = L
    int_K1 = np.inf
    constant = 4*(np.pi**2)*(k_rad**2)
    # f_combined = lambda x,K_turb,L,k_rad,u,D,f,Cn2: S_combined_integral(K_turb,L,x,k_rad,u,D,f,Cnn) #Wrong!!
    f_combined = lambda x,K_turb,L,k_rad,Cnn,D,u,f: var_Wang_new(K_turb,L,x,k_rad,Cnn,D,u,f)

    for m in range(0,len(freq)):
        int_K0 = 2*np.pi*freq[m]/u
        integral,error = dblquad(f_combined,int_K0,int_K1,int_x0,int_x1_new,args = (L,k_rad,Cnn,D,u,freq[m])) #used to be (L,k_rad,u,D,freq[m],Cn2)
        S_array[m] = constant*integral
    fS = freq*S_array[:].ravel()
    return freq,S_array,fS


#############################################
######## Var computation ####################
#############################################

def Theoretical_spectrum_Ward(f_trans,L,Cnn,D):
    c = 299792458 #m/s
    k_rad = 2*np.pi*f_trans/c
    int_x0 = 0.0
    int_x1 = L
    int_K0 = 0
    int_K1 = np.inf
    
    f_combined = lambda x,K_turb,L,k_rad,Cnn,D: var_Wang_ext(K_turb,L,x,k_rad,Cnn,D)
    integral,error = dblquad(f_combined,int_K0,int_K1,int_x0,int_x1,args = (L,k_rad,Cnn,D))
    return integral

def var_Wang_ext(K_turb,L,x,k_rad,Cnn,D):
    var_comb = turb_intensity(K_turb,Cnn)*wave_prop_Wang(K_turb,L,x,k_rad)*aperture_average_Ward(K_turb,D,x,L)
    constant = 4*((np.pi)**2)*(k_rad**2)
    return constant*var_comb

def wave_prop_Wang(K_turb,L,x,k_rad):
    a = (np.sin(((K_turb**2)*x*(L-x))/(2*k_rad*L)))**2
    return a

#################################
#### Without considering aperture averaging
#################################

def var_Wang(K_turb,L,x,k_rad,Cnn): #without considering aperture averaging
    var_comb = turb_intensity(K_turb,Cnn)*wave_prop_Wang(K_turb,L,x,k_rad)
    constant = 4*(np.pi)**2*(k_rad**2)
    return constant*var_comb

def Theoretical_spectrum_Wang(f_trans,L,Cnn):
    c = 299792458 #m/s
    k_rad = 2*np.pi*f_trans/c
    int_x0 = 0.0
    int_x1 = L
    int_K0 = 0
    int_K1 = np.inf
    
    f_combined = lambda x,K_turb,L,k_rad,Cnn: var_Wang(K_turb,L,x,k_rad,Cnn)
    integral,error = dblquad(f_combined,int_K0,int_K1,int_x0,int_x1,args = (L,k_rad,Cnn))
    return integral


######################################################
####### Captured amount of variance ##################
######################################################

def var_capt(f_boundary,D,f_trans,L,u,Cnn):
    freq,S_array,fS = Theoretical_spectrum2(D,f_trans,L,u,Cnn)
    var = np.trapz(y = S_array, x=freq)
    # var = np.trapz(y = fS, x = np.log10(freq_array))
    index_cut = abs(f_boundary-freq).argmin()
    var_cut = np.trapz(y = S_array[:index_cut+1], x = freq[:index_cut+1]) #+1 because the :index is excluded
    return var,var_cut

def var_frac_capt_double(f0,f1,D,f_trans,L,u,Cnn):
    freq,S_array,fS = Theoretical_spectrum2(D,f_trans,L,u,Cnn)
    
    var = np.trapz(y = S_array, x = freq)
    index_cut_0 = abs(f0 - freq).argmin()
    var_cut_0 = np.trapz(y = S_array[:index_cut_0+1], x = freq[:index_cut_0+1])
    index_cut_1 = abs(f1 - freq).argmin()
    var_cut_1 = np.trapz(y = S_array[index_cut_1:], x = freq[index_cut_1:])
    var_cut = var_cut_0+var_cut_1
    var_capt = var-var_cut
    return var_capt/var