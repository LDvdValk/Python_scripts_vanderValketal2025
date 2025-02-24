import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as nc
import math
import statsmodels.api as sm
from datetime import datetime, timedelta
import geopy.distance
from matplotlib import gridspec
from scipy.special import jv
from scipy.integrate import dblquad


######################################################################
#### Spectrum from signal ###########################################
######################################################################

def spectrum_calc(timeseries,fs):
    freq = np.fft.rfftfreq(n=len(timeseries),d=1/fs)
    freq_full = np.fft.fftfreq(n=len(timeseries),d=1/fs)
    X = np.fft.fft(timeseries)
    X_div = X/len(timeseries)
    X_2 = np.abs(X_div)**2
    if len(timeseries)%2 == 0:
        E = 2*X_2[1:len(freq)-1]
        E = np.append(E,X_2[len(freq)-1])
    else:
        E = 2*X_2[1:len(freq)]
    S = E/(fs/len(timeseries))
    return X,X_2,E,S,freq[1:],freq_full

def spectrum_calc_split(timeseries,fs):
    freq = np.fft.rfftfreq(n=len(timeseries),d=1/fs)
    freq_full = np.fft.fftfreq(n=len(timeseries),d=1/fs)
    X = np.fft.fft(timeseries)
    X_div = X/len(timeseries)
    X_2_real = np.real(X_div)**2
    X_2_imag = np.imag(X_div)**2
    X_2 = np.imag(X_div)**2
    if len(timeseries)%2 == 0:
        E_real = 2*X_2_real[1:len(freq)-1]
        E_real = np.append(E_real,X_2_real[len(freq)-1])
        E_imag = 2*X_2_imag[1:len(freq)-1]
        E_imag = np.append(E_imag,X_2_imag[len(freq)-1])
        E = 2*X_2[1:len(freq)-1]
        E = np.append(E,X_2[len(freq)-1])
    else:
        E_real = 2*X_2_real[1:len(freq)]
        E_imag = 2*X_2_imag[1:len(freq)]
        E_imag = 2*X_2[1:len(freq)]
    S_real = E_real/(fs/len(timeseries))
    S_imag = E_imag/(fs/len(timeseries))
    S = E/(fs/len(timeseries))
    return S,S_real,S_imag,freq,freq_full

def filtering(timeseries):
    filt = np.ones(len(timeseries))
    taper = (1-np.linspace(-1,1,2*math.floor(0.03*len(timeseries)))**2)**2
    filt[0:int(0.5*len(taper))] = taper[0:int(0.5*len(taper))]
    filt[len(timeseries)-int(0.5*len(taper)):] = taper[int(0.5*len(taper)):]
    return filt

def detrend(timeseries):
    if np.isnan(timeseries).all() == True:
        timeseries_dt = timeseries
    else:
        y = timeseries
        x = np.arange(0,len(timeseries),1)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        x = sm.add_constant(x)
        model = sm.OLS(y,x)
        results = model.fit()
        res_coef = results.params
        timeseries_dt = timeseries - (res_coef[0]+res_coef[1]*np.arange(0,len(timeseries),1))
    return timeseries_dt
    #Use timesteps as "x-axis", so nans can be throuwn out

def spectrum_snip_inverse(X,freq,f_snip):
    index_f_snip = abs(freq-f_snip).argmin()
    length_X = len(X)
    X[index_f_snip:length_X//2] = 0
    X[length_X//2+1:-index_f_snip] = 0
    timeseries = np.fft.ifft(X)
    return timeseries 

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(timeseries, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    #print('b and a',b,a)
    #print(np.count_nonzero(np.isnan(timeseries)))
    y = signal.filtfilt(b, a, timeseries)
    return y

def spectrum_welch(timeseries,samp_rate,chunks):
    split_series = np.array_split(timeseries,chunks)
    spectrum_X = np.zeros((chunks,int(len(timeseries)/chunks)))
    for i in range(0,chunks):
        X,freq = spectrum_calc(split_series[i],samp_rate)
        print(X)
        print(abs(X))
        spectrum_X[i,:] = X
    #spectrum_X [:,0] = 0
    X_mean = np.mean(spectrum_X,axis=0)
    return X_mean,freq

def specsmooth(f, S, smooth_width=0.2, smooth_step=0.2):
    F = f
    pos_FFT = 0;
    pos_FFTsm = 0;
    F = pd.DataFrame(F)
    S = np.real(S) #--> Only taking the real part of S to prevent exponential explosion of specsmooth workings
    S = pd.DataFrame(S)
    S_sm = S.copy()*np.nan
    F_sm = F.copy()*np.nan
    while pos_FFT < len(F):
        Nsm_half = round(0.5*smooth_width*pos_FFT);
        
        if Nsm_half <= 1:
            S_sm.loc[pos_FFTsm,:] = S.loc[pos_FFT,:];
            F_sm.loc[pos_FFTsm,:] = F.loc[pos_FFT,:];
            Nsm_half = 1;
        else:
            # Nsm: the number of datapoints used for smoothing at pos_FFT
            Nsm = 2*Nsm_half + 1;
            
            # Wfunc: bell-shaped "weighting_function" over Nsm points
            Wfunc = 0.5*(1-np.cos(2*np.pi*np.arange(0,Nsm)/(Nsm-1)));
    
            # Truncate the weighting_function if at pos_FFT, pos_FFT+Nsm_half
            # is larger than the length of the spectrum
            if (pos_FFT + Nsm_half) > len(F):
                Wfunc = Wfunc[1:(Nsm-(pos_FFT+Nsm_half-len(F)))]; # possible -1 missing due to python indexing
            
            Wfunc = Wfunc/sum(Wfunc)
            Wfunc = pd.DataFrame(Wfunc)
            # Perform smoothing using Wfunc
            Spec_weight = F*0;
            # print(Wfunc)
            # print(Spec_weight.loc[(pos_FFT-Nsm_half):(pos_FFT-Nsm_half+len(Wfunc)),:])
            Spec_weight[(pos_FFT-Nsm_half):(pos_FFT-Nsm_half+len(Wfunc))] = Wfunc;
            if len(S.shape) < 2: size = 1
            else: size=S.shape[1]
            for i in np.arange(0,size): #np.arange(1,1) returns an empty array
                S_sm.loc[pos_FFTsm,i] = sum(S.loc[:,i]*pd.Series(Spec_weight[0]));
            F_sm.loc[pos_FFTsm,:] = F.loc[pos_FFT,0];
        
        # Go to next point in spectrum to be smoothed.
        # Position of this point is defined by smooth_step (see header).
        step_size = round(smooth_step*Nsm_half);
        if  step_size < 1:
            step_size = 1
        pos_FFT = pos_FFT + step_size;
        pos_FFTsm = pos_FFTsm + 1;
        
    return(F_sm.values, S_sm.values)