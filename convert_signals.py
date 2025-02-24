import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as nc
import math
from datetime import datetime, timedelta
import geopy.distance
from matplotlib import gridspec

######################################################################
##### General ########################################################
######################################################################



def convert_mV_dB(df, mode, device = None):
    """
    Convert raw detector voltages to power (dBm). Values obtained from van Leth et al. (2018)
    """
    if device == None:
        col_name_suffix = ['Ral_38_V','Ral_38_H','Nokia_H','Ral_26_H']
    else:
        col_name_suffix = [device]
    for i in col_name_suffix:
        if mode == 'raw' or mode == 'raw variance':
            volt = df[i]
        else:
            volt = df[mode+'_'+i]
        if mode == 'mean' or mode == 'min' or mode == 'max' or mode == 'raw':
            if i == 'Ral_38_V':
                power = -99.755 + volt * 10**-3 * 20.595
            elif i == 'Ral_38_H':
                power = -99.871 + volt * 10**-3 * 20.596
            elif i == 'Nokia_H':
                power = 22.433 + volt * 10**-3 * -34.228
            elif i == 'Ral_26_H':
                power = -99.871 + volt * 10**-3 * 20.596 
        elif mode == 'variance' or mode == 'raw variance':
            if i == 'Ral_38_V':
                power = ((volt**0.5) * 10**-3 * 20.595)**2 #first volt_var to sd --> change unit 
            elif i == 'Ral_38_H':
                power = ((volt**0.5) * 10**-3 * 20.596)**2 
            elif i == 'Nokia_H':
                power = ((volt**0.5) * 10**-3 * -34.228)**2
            elif i == 'Ral_26_H':
                power = ((volt**0.5) * 10**-3 * 20.596)**2
    power = power.to_frame() #make it a dataframe, useful for following steps
    return power


def convert_dB_mW(df):
    """
    Convert power in dBm to power in mW. Values obtained from van Leth et al. (2018)
    """ 
    mW = 10**(df/10) #formally this is a fraction, but since the fraction has a denominator of 1 mW, this immediately also is the power in mW
    return mW 


def linklength(site_a_latitude = 51.968657,site_a_longitude = 5.68273,site_a_altitude = 51,
                site_b_latitude = 51.985230,site_b_longitude = 5.664312,site_b_altitude = 62):
    coords_a = (site_a_latitude, site_a_longitude)
    coords_b = (site_b_latitude, site_b_longitude)
    length = geopy.distance.distance(coords_a, coords_b).km
    return(length)

def return_stats(df,stepsize = '30s',mean = True, max = True, min = True, variance = True, max_min = True):
    col_name = df.columns.values[0]
    df_stats = pd.DataFrame()
    if mean == True:
        mean_res = df.resample(stepsize,label='right',closed='right').mean()
        mean_res.columns = ['mean_'+col_name]
        df_stats = pd.concat([df_stats,mean_res],axis=1)
    if max == True:
        max_res = df.resample(stepsize,label='right',closed='right').max()
        max_res.columns = ['max_'+col_name]
        df_stats = pd.concat([df_stats,max_res],axis=1)
    if min == True:
        min_res = df.resample(stepsize,label='right',closed='right').min()
        min_res.columns = ['min_'+col_name]
        df_stats = pd.concat([df_stats,min_res],axis=1)
    if variance == True:
        variance_res = df.resample(stepsize,label='right',closed='right').var()
        variance_res.columns = ['variance_'+col_name]
        df_stats = pd.concat([df_stats,variance_res],axis=1)
    if max_min == True:
        df_stats['max_min_'+col_name] = df_stats['max_'+col_name]-df_stats['min_'+col_name]
    return df_stats

def seperate_dry_wet_disdro(df_mw,df_prec):
    f_prec = df_prec.index.freq.delta.seconds**-1 #Hz
    f_stats = df_mw.index.freq.delta.seconds**-1 #Hz
    if f_prec > f_stats: #so if frequency (stepsize) of the precipitation measurements is larger (smaller) than the frequency (stepsize) of the resampled mw link data
        df_prec = df_prec.resample(df_mw.index.freq,label='right',closed = 'right').median()
    elif f_prec > f_stats:
        df_prec = df_prec.resample(df_mw.index.freq).backfill()
    df_mw_prec = pd.concat([df_mw,df_prec],join = 'inner',axis=1)
    df_mw_wet = df_mw_prec[df_mw_prec['prec_num']>0]
    df_mw_dry = df_mw_prec[df_mw_prec['prec_num'] == 0]
    return df_mw_dry,df_mw_wet

def resample_stats(I,stepsize):
    device=I.columns[0]
    print(device)
    mean = I.resample(stepsize,label='right',closed = 'right').mean()
    median = I.resample(stepsize,label='right',closed = 'right').median()
    min_val = I.resample(stepsize,label='right',closed = 'right').min()
    max_val = I.resample(stepsize,label='right',closed = 'right').max()
    #var = I.resample(stepsize,label='right',closed = 'right').var()
    ln_I = np.log(I)
    ln_var = ln_I.resample(stepsize,label='right',closed = 'right').var()
    index = ln_var.index
    print(mean[device].values)
    df = pd.DataFrame(data={device+'_mean':mean[device].values,device+'_median':median[device].values,device+'_min':min_val[device].values,device+'_max':max_val[device].values,device+'_ln_var':ln_var[device].values},index = index)
    print(df)
    df.to_pickle('stats_'+device+'_'+stepsize+'.pkl')
    return min_val,max_val,mean,median,ln_var

def resample_stats_monthly(I,stepsize,month):
    device=I.columns[0]
    mean = I.resample(stepsize,label='right',closed = 'right').mean()
    median = I.resample(stepsize,label='right',closed = 'right').median()
    min_val = I.resample(stepsize,label='right',closed = 'right').min()
    max_val = I.resample(stepsize,label='right',closed = 'right').max()
    instant = I.resample(stepsize,label='right',closed = 'right').apply(lambda x : x.iloc[-1]) #last does not return nan
    #var = I.resample(stepsize,label='right',closed = 'right').var()
    ln_I = np.log(I)
    ln_var = ln_I.resample(stepsize,label='right',closed = 'right').var()
    index = ln_var.index
    df = pd.DataFrame(data={device+'_mean':mean[device].values,device+'_median':median[device].values,device+'_min':min_val[device].values,device+'_max':max_val[device].values,device+'_ln_var':ln_var[device].values,device+'_instant':instant[device].values},index = index)
    return df
    

######################################################################
##### Rain specific ##################################################
######################################################################

def baseline(df,method, period = 24,min_periods = 12):
    '''
    Determine baseline in receiver signal.
    '''
    if method == 'max':  #assumption that it doesn't rain for certain period (e.g. 24 h)
        ref = df.rolling(str(period)+'H',center=True,min_periods = min_periods).max()
    elif method == 'Overeem11': # requires additional data wheather it rains or not
        ref = df.rolling(str(period)+'H',center=False,min_periods = min_periods).median()
        #So it depends on the previous 24 H
    elif method == 'Chwala12':
        pass
    elif method == 'Wang12':
        pass
    return ref

def wet_dry(df,method, period):
    if method == 'Schleiss10':
        #SD
        pass
    if method == 'Chwala2019':
        #FFT
        pass

def attenuation(power,ref,L,Tx):
    """
    Convert Rx and Tx to specific attenuation.
    """
    name = power.name
    k = (ref - power) / L
    k.loc[k<0]=0
    k.rename(name,inplace = True)
    return k

def rain_rate(k,n_dev=1):
    """
    Convert specific attenuation to average rain rate. Values obtained from van Leth et al. (2018)
    """
    if n_dev > 1:
        df_rain = pd.DataFrame()
        for i in range(0,len(k.columns)):
            device = k.columns[i]
            k_dev = k[device]
            if 'Ral_38_V' in device:
                rain = 4.164*k_dev**1.073
            elif 'Ral_38_H' in device:
                rain = 3.828*k_dev**1.049
            elif 'Nokia_H' in device:
                rain = 3.828*k_dev**1.049
            elif 'Ral_26_H' in device:
                rain = 7.704*k_dev**0.931
            df_rain = pd.concat([df_rain,rain],axis=1)
        df_rain.set_index(pd.DatetimeIndex(df_rain.index),inplace = True)
    if n_dev == 1:
        device = k.name
        #print(device)
        if 'Ral_38_V' in device:
            df_rain = 4.164*k**1.073
        elif 'Ral_38_H' in device:
            df_rain = 3.828*k**1.049
        elif 'Nokia_H' in device:
            df_rain = 3.828*k**1.049
        elif 'Ral_26_H' in device:
            df_rain = 7.704*k**0.931
    return df_rain

def prec_type_to_num(df):
    df['prec_num'] = 0
    df.loc[df['prec_type'] == 'Rain','prec_num'] = 1
    df.loc[df['prec_type'] == 'Snow','prec_num'] = 2
    df.loc[df['prec_type'] == 'Hail','prec_num'] = 3
    df.loc[df['prec_type'] == 'NA','prec_num'] = np.nan
    return df

def create_series_type(df,precipitation_type):
    """
    Cut up large df into seperate series during which it rains, snows or hails
    """
    df_prec = df.copy()
    df_prec.loc[df_prec['prec_type'] != precipitation_type,'prec_type'] = 'NA'
    df_prec["isStatusChanged"] = df_prec['prec_type'].shift(1) != df_prec['prec_type']
    if precipitation_type == 'Snow':
        df_prec['month'] = df_prec.index.month
    series = np.split(df_prec, np.where((df_prec['isStatusChanged']))[0])
    series = [i for i in series if (len(i)>10)] #equals 5 minutes
    series = [i for i in series if i['prec_type'][0]==precipitation_type]
    if precipitation_type == 'Snow':
        series = [i for i in series if (i['month'][0]<6) or (i['month'][0]>8)]
    return series

def create_series_intensity(df,n_agree,min_time,wet_dry):
    """
    Cut up large df into seperate series during which it rains, snows or hails
    """
    min_length = min_time/0.5 # min_time in minutes
    df_prec = df.copy()
    df_dry = df_prec.loc[(df_prec==0).all(axis=1)]#.asfreq(freq='30s') #All disdros dry
    df_prec['count']= df_prec[df_prec>0].count(axis=1)
    df_wet = df_prec.loc[(df_prec['count']>=n_agree)]#.asfreq(freq='30s') #All disdros wet
    df_wet['time'] = df_wet.index
    df_wet['subsequent'] = (df_wet['time'].shift(periods = -1)-df_wet['time']) != pd.Timedelta(30,'s')
    df_wet['subsequent'] = df_wet['subsequent'].shift(periods = 1)
    series_wet = np.split(df_wet, np.where((df_wet['subsequent']))[0])
    series_wet = [i for i in series_wet if (len(i)>=min_length)]
    df_dry['time'] = df_dry.index
    df_dry['subsequent'] = (df_dry['time'].shift(periods = -1)-df_dry['time']) != pd.Timedelta(30,'s')
    df_dry['subsequent'] = df_dry['subsequent'].shift(periods = 1)
    series_dry = np.split(df_dry, np.where((df_dry['subsequent']))[0])
    series_dry = [i for i in series_dry if (len(i)>=min_length)]
    if wet_dry == 'Both':
        return series_wet,series_dry
    elif wet_dry == 'Wet':
        return series_wet
    elif wet_dry == 'Dry':
        return series_dry

######################################################################
##### Evaporation specific ###########################################
######################################################################

def struc_refr_index(I,L,stepsize):
    """
    Variance of the natural logarithm of the intensity to structure parameter of refractive index of air. Based on Tatarski (1971) (and Leijnse et al. (2007))
    """
    #print(I)
    c_wave = 299702547 #Is this correct?
    lambda_26, lambda_38 = c_wave/(26*10**9), c_wave/(38*10**9)
    k_26, k_38 = 2*np.pi/lambda_26, 2*np.pi/lambda_38 
    constant = (2**(14/3)*math.gamma(7/3)*np.cos(np.pi/12))/(np.pi*(3*np.pi)**0.5*math.gamma(8/3))
    device = I.name
    ln_I = np.log(I)
    var_ln_I = ln_I.resample(stepsize,label='right',closed='right').var()
    if 'Ral_26_H' in device:
        Cn2 = constant* k_26**(-7/6) * L**(-11/6) * var_ln_I
    else:
        Cn2 = constant* k_38**(-7/6) * L**(-11/6) * var_ln_I
    return Cn2

def custom_resampler(array_like):
    return np.var(array_like/np.mean(array_like))

def AT_AQ(T,P,Q):
    """
    dimensionless sensitivity coefficients of the refractive index AT and AQ at radio wavelengths longer than 3 mm are given by Andreas [1989]
    """
    b = 0.776*10**-6 # K Pa-1
    c = 1.723 #K m3 kg-1
    AT = -b*P/T - c*Q/T
    AQ = c*Q/T
    #AT.rename('AT',axis='columns',inplace=True)
    #AQ.rename('AQ',axis='columns',inplace=True)
    return AT,AQ

def Obukhov_length(u_star,P,T,RH,H,LvE):
    """
    Calculating the Obukhov length in unstable conditions
    """
    kappa = 0.4
    g = 9.81 #m s-2
    c_p = 1005 #J kg-1 K-1
    C = 2.16679 #gK/J
    e = RH/100*0.6113*np.exp(5423*(1/273.15 - 1/T)) #kPa
    Q = C*e/T #kg m-3
    rho = P/(287.04*T) - 0.61*Q
    Lv = 1000 * (2501 - 2.361*(T - 273.15))
    L_Ob = - (rho*u_star**3)/(kappa*g*((H/(c_p*T))+(0.61*(LvE)/Lv)))
    #L_Ob.rename('L_Ob',axis='columns',inplace=True)
    return L_Ob

def calc_u_star(u,z_u,d_0,z_0,L_Ob):
    """
    Calculating friction velocity
    """
    kappa = 0.4
    u_star = (kappa*u)/(np.log((z_u-d_0)/z_0)-Busi_Dyer((z_u-d_0)/L_Ob)+Busi_Dyer(z_0/L_Ob))
    #u_star.rename('u_star',axis='columns',inplace=True)
    return u_star

def Busi_Dyer(y):
    """
    Businger Dyer equation
    """
    x = (1-16*y)**(1/4)
    psi = 2*np.log((1+x)/2) + np.log((1+x**2)/2) - 2*np.arctan(x)+0.5*np.pi
    return psi

def H_LvE(Bowen,Rnet,G):
    H = Bowen/(1+Bowen)*(Rnet-G)
    LvE = 1/(1+Bowen)*(Rnet-G)
    #H.rename('H',axis='columns',inplace=True)
    #LvE.rename('LvE',axis='columns',inplace=True)
    return H,LvE

def Cn2_Ct2_Cq2(H,LvE,P,T,RH,u_star,z_s,d_0,L_Ob,r_tq):
    """
    Calculating the Structure parameters
    """
    c_p = 1005 #J kg-1 K-1
    C = 2.16679 #gK/J
    e = RH/100*0.6113*np.exp(5423*(1/273.15 - 1/T)) #kPa
    Q = C*e/T #kg m-3
    rho = P/(287.04*T) - 0.61*Q
    f_Ob = 4.9 * (1 - 6.1*((z_s-d_0)/L_Ob))**(-2/3)
    Lv = 1000 * (2501 - 2.361*(T - 273.15))
    AT, AQ = AT_AQ(T,P,Q)
    Ct2 = (H**2/(rho**2*c_p**2))*(1/(u_star**2 * (z_s-d_0)**(2/3)))*f_Ob
    Cq2 = (LvE**2/Lv**2)*(1/(u_star**2 * (z_s-d_0)**(2/3)))*f_Ob
    Cn2 = AT**2 * (Ct2/T**2) + AQ**2 * (Cq2/Q**2) + 2*AT*AQ*((r_tq*(Ct2**0.5)*(Cq2**0.5))/(T*Q))
    return Cn2,Ct2,Cq2,AT,AQ,Q

def power_spectrum(df):
    sampling_rate = 20 #Hz
    data = df['Nokia']
    ps = np.abs(np.fft.fft(data)*(1/sampling_rate))**2 / (len(data)/sampling_rate)
    #new line
    freq = np.linspace(0,sampling_rate/2,len(ps))
    return ps,freq

def iterations_Cn2(u_diff_guess,bowen_guess,Rnet,G,
                           u_star_guess,z_s,P,T,RH,u,z_u,
                           d_0,z_0,r_tq):
    """
    Iteratively solving the friction velocity and obukhov length and computing Cn2 subsequently
    """
    H,LvE = H_LvE(bowen_guess,Rnet,G)
    print('H : '+str(H)+', LvE : '+str(LvE))
    iteration = 0
    list_L_Ob = []
    list_u_star = []
    u_diff = u_diff_guess
    while (u_diff>10**-6):
      #Calculating friction velocity and obukhov length, iteratively
      if (iteration == 0):
         u_star_st = u_star_guess
         Obukhov = Obukhov_length(u_star_st,P,T,RH,H,LvE) 
         u_star = calc_u_star(u,z_u,d_0,z_0,Obukhov)
      else:
         u_star_st = u_star
         Obukhov = Obukhov_length(u_star_st,P,T,RH,H,LvE)
         u_star = calc_u_star(u,z_u,d_0,z_0,Obukhov)
      iteration+=1         
      u_diff = abs(u_star_st - u_star)
      Cn2_bowen, Ct2_bowen, Cq2_bowen, AT_bowen, AQ_bowen, Q = Cn2_Ct2_Cq2(H,LvE,P,T,RH,u_star,z_s,d_0,Obukhov,r_tq)
    return u_star,Obukhov,H,LvE, Cn2_bowen, Ct2_bowen, Cq2_bowen, AT_bowen, AQ_bowen, Q

def Bowen_solver(bowen_guess,Cn2_mw,u_diff_guess,Rnet,G, u_star_guess,z_s,
                    P,T,RH,u,z_u,d_0,z_0,r_tq,index,filename): 
    """
    Iteratively solving the correct Bowen ratio by comparing the "theoretical" Cn2 (based on meteorological measurements) with the observed Cn2 (scintillometer).
    """             
    u_star,Obukhov,H,LvE,Cn2_bowen,Ct2_bowen,Cq2_bowen,AT_bowen, AQ_bowen, Q = iterations_Cn2(u_diff_guess,bowen_guess,
                                                                        Rnet,G,u_star_guess,z_s,
                                                                        P,T,RH,u,z_u,
                                                                        d_0,z_0,r_tq)  
    print('u*: '+str(u_star)+', L_Ob: '+str(Obukhov)+', Bowen: '+str(bowen_guess))
    Cn2_diff = abs(Cn2_bowen-Cn2_mw)
    df = pd.read_pickle(filename)
    print(df.loc[index])
    df.loc[index,'u_star'] = u_star
    df.loc[index,'L_Ob'] = Obukhov
    df.loc[index,'Bowen'] = bowen_guess
    df.loc[index,'H'] = H
    df.loc[index,'LvE'] = LvE
    df.loc[index,'Cn2'] = Cn2_bowen
    df.loc[index,'Ct2'] = Ct2_bowen
    df.loc[index,'Cq2'] = Cq2_bowen
    df.loc[index,'AT'] = AT_bowen
    df.loc[index,'AQ'] = AQ_bowen
    df.loc[index,'Q'] = Q
    df.to_pickle(filename)
    return Cn2_diff



######################################################################
##### Veenkampen specific ###########################################
######################################################################

def potential_T(T,P,ref_P = 1000):
    Theta = T * (ref_P/P)**0.286
    return Theta

def vap_pres(RH,T):
    e = RH/100*0.6113*np.exp(5423*(1/273.15 - 1/T)) #kPa
    return e

def e_sat(T):
    e_s = 0.6113*np.exp(5423*(1/273.15 - 1/T)) #kPa
    return e_s

def w_sat(P,e_s):
    w_s = 0.622*e_s/(p-e_s)
    return w_s

def w(RH,w_s):
    w = RH/100*w_s
    return w

def virtual_T(T,e,P):
    Tv = T/(1-(e/P)*(1-0.622))
    return Tv

def Ri_num(T_top,T_low,u_top,u_low,dz_T,dz_u,P,RH):
    T_av = (T_top+T_low)/2
    Th_top = potential_T(T_top,P,ref_P=100)
    Th_low = potential_T(T_low,P,ref_P=100)
    e = vap_pres(RH,T_top) #kPa --> so same unit as P
    Tv_av = virtual_T(T_av,e,P)
    Thv_top = virtual_T(Th_top,e,P)
    Thv_low = virtual_T(Th_low,e,P)
    dThv_dz = (Thv_top - Thv_low)/dz_T
    du_dz = (u_top - u_low)/dz_u
    Ri = (9.81/Tv_av)*dThv_dz/(du_dz**2)
    return Ri