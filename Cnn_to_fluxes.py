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


def CTT_Cqq_to_Cnn(T,q,r_Tq,A_T,A_q,f_TT,f_qq,T_star,q_star):
    C_TT = f_TT*T_star**2/z**2/3
    C_qq = f_qq*q_star**2/z**2/3
    C_Tq = r_Tq*(C_TT*C_qq)**0.5
    Cnn = A_T**2/T**2*C_TT+2*A_T*A_q/(T*q)*C_Tq+A_q**2/q**2*C_qq
    return Cnn,C_TT,C_qq,C_Tq

def AT_Aq(f_trans,p,abs_q,T): #f_trans in Hz, p in hPa,abs_q in kg m-3, T in K
    c = 299792458 #m s-1
    lambda_trans = c/f_trans
    #Computation after Andreas 1988
    lambda_trans_mm = lambda_trans*1000
    A_T = -(77.6*10**-6*p + 1.73*abs_q)/T #Note: for 38 GHz and 160 GHz minor differences between 
    A_q = 1.73*abs_q/T
    if lambda_trans_mm < 3:
        a1 = [1.382221,-0.2135129, -0.1485997, -0.1088790]*10**3
        a2 = [1.65,0.1619430, 0.1782352, 0.1918662]
        a3 = [0.1993324, 3.353494, 3.100942, 3.004944]
        A_T_mw1 = 0
        A_q_mw1 = 0
        for i in range(1,5):
            A_T_mw1 += a1[i-1]*(296/T)**a2[i-1]*(0.303/lambda_trans_mm)**(2*i)*(-a2[i-1]+a3[i-1]*(296/T)*(1+a2[i-1]))
            A_q_mw1 += a1[i-1]*(296/T)**a2[i-1]*(1-a3[i-1]*(296/T))*(0.303/lambda_trans_mm)**(2*i)

        A_T_mw1 = A_T_mw1 *10**-6 * (abs_q/T)
        A_q_mw1 = A_q_mw1 *10**-6
        A_T = A_T + A_T_mw1 #Note: for 38 GHz and 160 GHz A_T_mw1 is relatively small --> A_T&A_q roughly equal for 38 and 160 GHz
        A_q = A_q + A_q_mw1
    return A_T,A_q

def AT_Aq_Ward(p,T,q): #p in Pa, T in K, q in kg kg-1:
    bt1 = 0.776*10**-6
    bt2 = (7500/T - 0.056)*10**-6
    Rd = 287.052874 # J kg-1 K-1
    Rv = 461.5 #J kg-1 K-1
    R = Rd + q*(Rv-Rd)
    A_T = - p/T * (bt1 + bt2*Rv/R*q)
    bq2 = (3750/T - 0.056)*10**-6
    A_q = (p/T)*(Rv/R)*q*bq2*(1-(q/R)*(Rv-Rd))
    return A_T, A_q


def AT_Aq_LAS_Ward(p,T,q,lambda_LAS): #p in Pa, T in K, q in kg kg-1:
    bt1 = (0.237134 + (68.39397/(130-lambda_LAS**(-2))) + (0.45473/(38.9-lambda_LAS**(-2))))*10**(-6)
    bt2 = (0.648731 + 0.0058058*lambda_LAS**(-2) - 0.000071150 * lambda_LAS**(-4) + 0.000008851*lambda_LAS**(-6))*10**(-6) - bt1
    bq2 = bt2
    Rd = 287.052874 # J kg-1 K-1
    Rv = 461.5 #J kg-1 K-1
    R = Rd + q*(Rv-Rd)
    A_T = - p/T * (bt1 + bt2*Rv/R*q)
    A_q = (p/T)*(Rv/R)*q*bq2*(1-(q/R)*(Rv-Rd))
    return A_T, A_q

def C_TT_C_qq_2beam(Cnn_MWS,Cnn_LAS,AT_LAS,Aq_LAS,AT_MWS,Aq_MWS,r_Tq,T,q,Bowen):
    if Bowen == 'high':
        S = -1
    elif Bowen == 'low':
        S = 1
    C_TT = ((Aq_MWS**2 * Cnn_LAS) + (Aq_LAS**2 * Cnn_MWS) + (2*r_Tq*Aq_LAS*Aq_MWS*S*(Cnn_LAS*Cnn_MWS)**0.5))/((AT_LAS*Aq_MWS - AT_MWS*Aq_LAS)**2 * T**(-2))
    C_qq = ((AT_MWS**2 * Cnn_LAS) + (AT_LAS**2 * Cnn_MWS) + (2*r_Tq*AT_LAS*AT_MWS*S*(Cnn_LAS*Cnn_MWS)**0.5))/((AT_LAS*Aq_MWS - AT_MWS*Aq_LAS)**2 * q**(-2))
    return C_TT, C_qq

def iterations_H(C_tt,u,T,q,rho,z,z_u,z_0):
    """
    Iteratively solving for the dual beam method so that u* and L_ob are also computed.
    """
    Karman = 0.4
    c_p = 1006 #J kg-1 K-1
    h_0 = z_0*8
    d_0 = 2*h_0/3
    H_diff = 1
    
    u_star_guess = Karman*u/(np.log(z_u/z_0)) #By assuming L_Ob is almost infinite
    L_guess = 10**5
    iteration = 0
    while (H_diff>10**-2):
        #Calculating friction velocity and obukhov length, iteratively
        if (iteration == 0):
            f_TT,f_qq = f_TT_f_qq(10,L_guess)
            T_star = np.sqrt((C_tt*z**(2/3))/f_TT)
            H = T_star*rho*c_p*u_star_guess
            LvE = H*2 #only for the function below, not used further
            _,_,L_Ob = flux_T_q_star_L_Ob(H,LvE,rho,u_star_guess,T,q)
            u_star = calc_u_star(u,z_u,d_0,z_0,L_Ob)
        else: 
            H_st = H
            f_TT,f_qq = f_TT_f_qq(10,L_Ob)
            T_star = np.sqrt((C_tt*z**(2/3))/f_TT)
            H = T_star*rho*c_p*u_star
            LvE = H*2 #only for the function below, not used further
            _,_,L_Ob = flux_T_q_star_L_Ob(H,LvE,rho,u_star,T,q)
            u_star = calc_u_star(u,z_u,d_0,z_0,L_Ob)
            H_diff = abs(H_st - H)
        iteration+=1  
    return H,u_star,L_Ob,f_TT,f_qq

def f_TT_f_qq(z,L_Ob):
    c1_T = 5.6 #After Kooijmans and Hartogensis
    c2_T = 6.5
    c1_q = 4.5
    c2_q = 7.3
    f_TT = c1_T*(1-c2_T*z/L_Ob)**(-2/3)
    f_qq = c1_q*(1-c2_q*z/L_Ob)**(-2/3)
    return f_TT,f_qq  

def K_Ctt_Cqq(u_star,q,f_TT,f_qq):
    K_Ctt = u_star*f_TT**(-1/2)
    K_Cqq = u_star*(1-q)**(-1/2)*f_qq**(-1/2)
    return K_Ctt,K_Cqq

def flux_T_q_star_L_Ob(SH,LvE,rho,u_star,T,q):
    cp = 1006 #J kg-1 K-1
    Lv = 2.3*10**6 #J kg-1
    g = 9.81 #m s-2
    k = 0.4
    T_star = - SH/(rho*cp*u_star)
    q_star = - (1-q)*LvE/(rho*Lv*u_star)
    L_Ob = - rho*cp*u_star**3/((g/T)*k*SH)
    return T_star,q_star,L_Ob

def CTT_Cqq_to_T_q_star(C_TT,C_qq,f_TT,f_qq,z):
    T_star = (C_TT*(z**(2/3))/f_TT)**0.5
    q_star = (C_qq*(z**(2/3))/f_qq)**0.5
    return T_star,q_star

#################################################
###### Method Leijnse et al., 2007 ##############
#################################################

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
    return H,LvE

def Cnn_Ctt_Cqq(z_s,d_0,L_Ob,T,p,q,H,rho,u_star,LvE,Lv,r_Tq):
    """
    Calculating the Structure parameters
    """
    c_p = 1005 #J kg-1 K-1
    f_TT,f_qq = f_TT_f_qq(z_s,L_Ob)
    AT, Aq = AT_Aq_Ward(p,T,q)
    Ctt = (H**2/(rho**2*c_p**2))*(1/(u_star**2 * (z_s-d_0)**(2/3)))*f_TT
    Cqq = (LvE**2/Lv**2)*(1/(u_star**2 * (z_s-d_0)**(2/3)))*f_qq
    Cnn = AT**2 * (Ctt/T**2) + Aq**2 * (Cqq/q**2) + 2*AT*Aq*((r_Tq*(Ctt**0.5)*(Cqq**0.5))/(T*q))
    return Cnn,Ctt,Cqq,AT,Aq


def iterations_Cnn(bowen_guess,Rnet,G,u,z_u,z_0,rho,T,q,d_0,z_s,p,Lv,r_Tq):
    """
    Iteratively solving the friction velocity and obukhov length and computing Cn2 subsequently
    """
    Karman = 0.4
    H,LvE = H_LvE(bowen_guess,Rnet,G)
    iteration = 0
    u_diff = 1
    u_star_guess = Karman*u/(np.log(z_u/z_0))
    while (u_diff>10**-6):
      #Calculating friction velocity and obukhov length, iteratively
      if (iteration == 0):
         u_star_st = u_star_guess
         _,_,L_Ob = flux_T_q_star_L_Ob(H,LvE,rho,u_star_st,T,q)
         u_star = calc_u_star(u,z_u,d_0,z_0,L_Ob)
      else:
         u_star_st = u_star
         _,_,L_Ob = flux_T_q_star_L_Ob(H,LvE,rho,u_star_st,T,q)
         u_star = calc_u_star(u,z_u,d_0,z_0,L_Ob)
      iteration+=1         
      u_diff = abs(u_star_st - u_star)
      Cnn_bowen, Ctt_bowen, Cqq_bowen, AT_bowen, Aq_bowen = Cnn_Ctt_Cqq(z_s,d_0,L_Ob,T,p,q,H,rho,u_star,LvE,Lv,r_Tq)
    return u_star,L_Ob,H,LvE, Cnn_bowen, Ctt_bowen, Cqq_bowen, AT_bowen, Aq_bowen

def Bowen_solver(bowen_guess,var_name,Cnn_mw,Rnet,G,u,z_u,z_0,rho,T,q,d_0,z_s,p,Lv,r_Tq,index,filename): 
    """
    Iteratively solving the correct Bowen ratio by comparing the "theoretical" Cn2 (based on meteorological measurements) with the observed Cn2 (scintillometer).
    """             
    u_star,L_Ob,H,LvE,Cnn_bowen,Ctt_bowen,Cqq_bowen,AT_bowen, Aq_bowen = iterations_Cnn(bowen_guess,Rnet,G,u,z_u,z_0,rho,T,q,d_0,z_s,p,Lv,r_Tq)
    
    Cnn_diff = abs(Cnn_bowen-Cnn_mw)
    df = pd.read_pickle(filename)
    df.loc[index,var_name[0]+':u_star_'+var_name[1]] = u_star
    df.loc[index,var_name[0]+':L_Ob_'+var_name[1]] = L_Ob
    df.loc[index,var_name[0]+':H_'+var_name[1]] = H
    df.loc[index,var_name[0]+':LvE_'+var_name[1]] = LvE
    df.loc[index,var_name[0]+':Cnn_'+var_name[1]] = Cnn_bowen
    df.loc[index,var_name[0]+':Ctt_'+var_name[1]] = Ctt_bowen
    df.loc[index,var_name[0]+':Cqq_'+var_name[1]] = Cqq_bowen
    #df.loc[index,'AT_EBM_'+var_name] = AT_bowen
    #df.loc[index,'Aq_EBM_'+var_name] = Aq_bowen
    df.to_pickle(filename)
    return Cnn_diff

def free_convection(rho,Lv,z,T,Ctt,Cqq):
    a = 0.44
    b = 0.51
    cp = 1006 #J kg-1 K-1
    g = 9.81
    H = a*rho*cp*z*((g/T)**0.5)*(Ctt**(3/4))
    LvE = b*rho*Lv*z*((g/T)**0.5)*(Ctt**(1/4))*(Cqq**(1/2))
    return H, LvE

def Ctt_Cqq_FC_EBM(H,LvE,rho,Lv,z,T):
    a = 0.44
    b = 0.51
    cp = 1006 #J kg-1 K-1
    g = 9.81
    Ctt = (H/(a*rho*cp*z*((g/T)**0.5)))**(4/3)
    Cqq = (LvE/(b*rho*Lv*z*((g/T)**0.5)*(Ctt**(1/4))))**2
    return Ctt,Cqq

def free_convection_EBM(bowen_guess,Rnet,G,rho,Lv,z,T,q,p,r_Tq):
    """
    Computing Cnn based on guessed Bowen ratio and net radiation using free convection.
    """
    Karman = 0.4
    c_p = 1006 #J kg-1 K-1
    AT, Aq = AT_Aq_Ward(p,T,q)
    H,LvE = H_LvE(bowen_guess,Rnet,G)
    Ctt, Cqq = Ctt_Cqq_FC_EBM(H,LvE,rho,Lv,z,T)
    Ctq = r_Tq*(Ctt*Cqq)**0.5
    Cnn = ((AT**2)/(T**2))*Ctt + ((2*AT*Aq)/(T*q))*Ctq+ ((Aq**2)/(q**2))*Cqq
    return Ctt,Cqq,Cnn,H,LvE

def Bowen_solver_FC(bowen_guess,var_name,Cnn_mw,Rnet,G,rho,Lv,z,T,q,p,r_Tq,index,filename): 
    """
    Iteratively solving the correct Bowen ratio by comparing the "theoretical" Cn2 (based on meteorological measurements) with the observed Cn2 (scintillometer) using Free convection.
    """             
    Ctt_FC,Cqq_FC,Cnn_FC,H_FC,LvE_FC = free_convection_EBM(bowen_guess,Rnet,G,rho,Lv,z,T,q,p,r_Tq)
    
    Cnn_diff = abs(Cnn_FC-Cnn_mw)
    df = pd.read_pickle(filename)
    df.loc[index,var_name[0]+':H_'+var_name[1]] = H_FC
    df.loc[index,var_name[0]+':LvE_'+var_name[1]] = LvE_FC
    df.loc[index,var_name[0]+':Cnn_'+var_name[1]] = Cnn_FC
    df.loc[index,var_name[0]+':Ctt_'+var_name[1]] = Ctt_FC
    df.loc[index,var_name[0]+':Cqq_'+var_name[1]] = Cqq_FC
    df.to_pickle(filename)
    return Cnn_diff

