# OS
import os, signal

# Stuff
import numpy as np
import pandas as pd

# Astropy
from astropy.table import Table, vstack, hstack
from astropy.io import ascii

# Scipy
from scipy.optimize import minimize
from scipy import stats

# Subprocess
import subprocess

# Warnings
import warnings
warnings.filterwarnings("ignore")

# matplotlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Seaborn
import seaborn as sns

# Time
import time 
                        
# Set plotting style 
sns.set(style="ticks", palette="muted", 
        rc={"xtick.bottom" : True, "ytick.left" : True})
plt.style.use(r'./matplotlibrc.txt')

observed_muHer_frequencies = pd.read_csv('./updated_muHer_freqs.txt', skiprows=2, sep='\s+')
v_avcr_obs = [766.228, 824.458, 903.838]
v_avcr_obs_unc = [0.122, 0.129, 0.131]

def get_history(profile_number):
    i=0
    model_number = None
    while (model_number == None):
        try:  
            index = pd.read_csv('profiles.index', names=['model_number', 'priority', 'profile_number'], skiprows=1, sep='\s+', engine ='python')
            model_number = index[index.profile_number == profile_number].model_number.values[0]                   
        except:
            print('Retrying Index File')
            i += 1
            if i == 10:
                print('model crashed at get history in get params')
                break
            elif i < 10:
                time.sleep(1)
                pass

    teff = None
    i = 0
    while (teff == None):
       try:
           DF = pd.read_csv('history.data', skiprows=5, sep='\s+')
           hist = DF[DF.model_number == model_number]
           teff = hist['log_Teff'].values[0]
       except:
           print('Retrying History File')
           i += 1
           if i == 10:
               print('model crashed at history file in get params')
               break
           elif i < 10:
               time.sleep(1)
               pass
            
    return hist

def get_params(profile_number):
    hist = get_history(profile_number)
    
    teff = 10**hist['log_Teff'].values[0]
    logg = hist['log_g'].values[0]
    lum = 10**hist['log_L'].values[0]
            
    return teff, logg, lum
    
def get_table_params(profile_number):
    hist = get_history(profile_number)
    
    teff = 10**hist['log_Teff'].values[0]
    logg = hist['log_g'].values[0]
    lum = 10**hist['log_L'].values[0]
    age = hist['star_age'].values[0]/1e9
    radius = 10**hist['log_R'].values[0]
    
    return teff, logg, lum, age, radius

def get_frequencies(profile_number):
    path = os.path.join('profile' + str(profile_number) + '-freqs.dat')
    return pd.read_csv(path, skiprows=5, sep='\s+')
# --------------------------------- CALC a0 ------------------------------- #    
def calc_a0(profile_number):
    frequencies = get_frequencies(profile_number)
    teff, logg, lum = get_params(profile_number)
    v_ac = (logg/4.44)*(teff/5777)**(-1/2)*5000
    
    obs_l0_freqs = observed_muHer_frequencies['nu'][observed_muHer_frequencies['l'] == 0].values
    obs_l0_errors = observed_muHer_frequencies['error'][observed_muHer_frequencies['l'] == 0].values
    
    model_l0_freqs = frequencies['Re(freq)'][frequencies['l'] == 0].values
    model_l0_Es = frequencies['E_norm'][frequencies['l'] == 0].values
    
    obs_index = []
    for model_l0_freq in model_l0_freqs:
        diffs = abs(obs_l0_freqs - np.ones(len(obs_l0_freqs))*model_l0_freq)
        min_index = np.argmin(diffs)
        if (diffs[min_index] < 15):
            index = np.where( obs_l0_freqs == obs_l0_freqs[min_index] )[0]
            obs_index.append(index[0])
        else:
            continue
    obs_l0_freqs = obs_l0_freqs[obs_index]
    obs_l0_errors = obs_l0_errors[obs_index]
     
    model_index = []
    for obs_l0_freq in obs_l0_freqs:
        diffs = abs(model_l0_freqs - np.ones(len(model_l0_freqs))*obs_l0_freq)   
        min_index = np.argmin(diffs)
        if (diffs[min_index] < 15):
            index = np.where( model_l0_freqs == model_l0_freqs[min_index] )[0]
            if len(index) == 1:
                model_index.append(index[0])
        else:
            continue
    model_l0_freqs = model_l0_freqs[model_index]
    model_l0_Es = model_l0_Es[model_index]
    
    best = None
    a0 = None; a3 = None
    for ntrials in range(50):
        corr_model_freqs = lambda a: (a[0]*model_l0_freqs) + (a[1]*(model_l0_freqs/v_ac)**3)/model_l0_Es
        func = lambda a: sum(((corr_model_freqs(a)-obs_l0_freqs)/obs_l0_errors)**2)
        now = minimize(func, (1, -2E-7))
        if len(now.x) == 2 and (best is None or best.fun > now.fun):
            best = now
            a0 = best.x[0]; a3 = best.x[1]
            
    return a0, a3
# --------------------------------- VAVCRs -------------------------------- #
def calc_v_avcr(profile_number):
    frequencies = get_frequencies(profile_number)
    
    teff, logg, lum = get_params(profile_number)
    v_ac = (logg/4.44)*(teff/5777)**(-1/2)*5000
            
    obs_l0_freqs = observed_muHer_frequencies['nu'][observed_muHer_frequencies['l'] == 0].values
    obs_l0_errors = observed_muHer_frequencies['error'][observed_muHer_frequencies['l'] == 0].values

    model_l0_freqs = frequencies['Re(freq)'][frequencies['l'] == 0].values
    model_l0_Es = frequencies['E_norm'][frequencies['l'] == 0].values
    
    model_l1_freqs = frequencies['Re(freq)'][frequencies['l'] == 1].values
    model_l1_Es = frequencies['E_norm'][frequencies['l'] == 1].values
    
    obs_l1_freqs = observed_muHer_frequencies['nu'][observed_muHer_frequencies['l'] == 1].values
    
    corrected_l1_freqs = model_l1_freqs + (best_a3*(model_l1_freqs/v_ac)**3)/model_l1_Es

    v_avcr = 9999999
    v_avcr_indexes = []
    for corrected_l1_freq in corrected_l1_freqs:
        diffs = abs(v_avcr_obs - np.ones(len(v_avcr_obs))*corrected_l1_freq)
        min_index = np.argmin(diffs)
        if (diffs[min_index] < 15):
            v_avcr_index = np.where( (corrected_l1_freqs == corrected_l1_freq) )[0]
            v_avcr_indexes.append(v_avcr_index[0])
    
    if len(v_avcr_obs) == len(corrected_l1_freqs[v_avcr_indexes]):
        v_avcr = sum(((v_avcr_obs-corrected_l1_freqs[v_avcr_indexes])/v_avcr_obs_unc)**2)
    else:
        v_avcr = 9999999

    return v_avcr
# ---------------------------------------------------------------------- #    
teff_obs = 5580; teff_obs_unc = 80 # 22
logg_obs = 4.010; logg_obs_unc = 0.01 # 0.025
mass_obs = 1.11; mass_obs_unc = 0.06
lum_obs = 2.6; lum_obs_unc = 0.1 # 0.026
radius_obs = 1.704; radius_obs_unc = 0.016
# --------------------------PLOT HR-DIAGRAM----------------------------- #
def plot_hr(profile_number, ax_i, ax_k):          
    start = 5
    
    i = 0
    teff_length = None
    while (teff_length == None):
        try:
            history_file = pd.read_csv('history.data', skiprows=5, sep='\s+') 
            teff = 10**history_file['log_Teff'][start:]
            lum = 10**history_file['log_L'][start:]
            teff_length = len(teff)
        except:
            print('Retrying HR plot')
            i += 1
            if i == 10:
                print('model crashed at HR plot')
                break
            elif i < 10:
                time.sleep(1)
                pass
    
    best_age = None
    for prof_num in appended_profile_numbers:
        hist = get_history(prof_num)
        ax[ax_i, ax_k].plot(10**hist['log_Teff'], 10**hist['log_L'], '|', color='red', markersize=12)
        if prof_num == best_profile:
            ax[ax_i, ax_k].plot(10**hist['log_Teff'], 10**hist['log_L'], '|', color='red', markersize=16)
            best_age = hist['star_age'].values[0]/1e9; best_age = round(best_age, 3)
            
        
    ax[ax_i, ax_k].errorbar(x=teff_obs, y=lum_obs, 
            xerr=teff_obs_unc, yerr=lum_obs_unc, 
            marker='.', markersize=8, color='k',
            capsize=7, capthick=2, zorder=99)
    
    ax[ax_i, ax_k].errorbar(x=teff_obs, y=lum_obs, 
            xerr=teff_obs_unc+20, yerr=lum_obs_unc+0.1, 
            marker='.', markersize=8, color='k',
            capsize=7, capthick=2, zorder=99)
           
    ax[ax_i, ax_k].plot(teff, lum, lw=2, color='k')
    
    ax[ax_i, ax_k].set_xlabel(r'Effective Temperature $T_{\rm{eff}}/\rm{K}$')
    ax[ax_i, ax_k].set_ylabel(r'Luminosity $L/\rm{L}_\odot$')
    ax[ax_i, ax_k].set_title('HR Diagram', size=16, loc='left')
    ax[ax_i, ax_k].set_xlim(5400, 5950)
    ax[ax_i, ax_k].set_ylim(0.9, 2.85)
    
    ax[ax_i, ax_k].text(5550, 1.90, r'pp-rate = '+str(rate), fontsize=12)
    ax[ax_i, ax_k].text(5550, 1.75, r'$Z_{ini}$ = '+str(Zini), fontsize=12)
    ax[ax_i, ax_k].text(5550, 1.6, r'$Y_{ini}$ = '+str(Yini), fontsize=12)
    ax[ax_i, ax_k].text(5550, 1.45, r'$\alpha_{mlt}$ = '+str(alpha_mlt), fontsize=12)
    ax[ax_i, ax_k].text(5550, 1.3, r'$f/f_0$ = '+str(f_ov)+'/0.05', fontsize=12)
    ax[ax_i, ax_k].text(5550, 1.15, r'$M_{ini}$ = '+str(M), fontsize=12)
    ax[ax_i, ax_k].text(5550, 1.0, 'Age = '+str(best_age), fontsize=12)
    
    ax[ax_i, ax_k].invert_xaxis()
# ----------------------- PLOT a0 EVOLUTION ------------------------- #
def plot_a0_evolution(ax_i, ax_k):
    for prof_num in appended_profile_numbers:
        hist = get_history(prof_num)
        age = hist['star_age'].values[0]/1e9

        index = np.where( (np.array(appended_profile_numbers) == prof_num) )[0]
        a0 = a0s[index[0]]

        ax[ax_i, ax_k].plot(age, a0, '.', color='red',)
            
        ax[ax_i, ax_k].set_xlabel(r'star age [Gyr]')
        ax[ax_i, ax_k].set_ylabel(r'$a_0$')
        ax[ax_i, ax_k].set_title('a0 evolution', size=16, loc='left')
# ----------------------- SMALL HR DIAGRAM -------------------------- #
def plot_small_hr(ax_i, ax_k):        
    start = 5
    
    i = 0
    teff_length = None
    while (teff_length == None):
        try:
            history_file = pd.read_csv('history.data', skiprows=5, sep='\s+') 
            teff = 10**history_file['log_Teff'][start:]
            lum = 10**history_file['log_L'][start:]
            teff_length = len(teff)
        except:
            print('Retrying small HR plot')
            i += 1
            if i == 10:
                print('model crashed at small HR plot')
                break
            elif i < 10:
                time.sleep(1)
                pass
    
    best_age = None
    lower_xlim = None
    lower_ylim = None
    upper_xlim = None
    upper_ylim = None
    for prof_num in appended_profile_numbers:
        hist = get_history(prof_num)
        ax[ax_i, ax_k].plot(10**hist['log_Teff'], 10**hist['log_L'], '|', color='red', markersize=12)
        if prof_num == best_profile:
            ax[ax_i, ax_k].plot(10**hist['log_Teff'], 10**hist['log_L'], '|', color='red', markersize=20)
            best_age = hist['star_age']/1e9; best_age = round(best_age, 3)
        elif prof_num == appended_profile_numbers[0]:
            lower_xlim = 10**hist['log_Teff'].values[0]-2
            lower_ylim = 10**hist['log_L'].values[0]-0.005
        elif prof_num == appended_profile_numbers[-1]:
            upper_xlim = 10**hist['log_Teff'].values[0]+2
            upper_ylim = 10**hist['log_L'].values[0]+0.005
            
        
    ax[ax_i, ax_k].errorbar(x=teff_obs, y=lum_obs, 
            xerr=teff_obs_unc, yerr=lum_obs_unc, 
            marker='.', markersize=8, color='k',
            capsize=7, capthick=2, zorder=99)
    
    ax[ax_i, ax_k].errorbar(x=teff_obs, y=lum_obs, 
            xerr=teff_obs_unc+20, yerr=lum_obs_unc+0.1, 
            marker='.', markersize=8, color='k',
            capsize=7, capthick=2, zorder=99)
           
    ax[ax_i, ax_k].plot(teff, lum, lw=4, color='k', zorder=-99)
 
    ax[ax_i, ax_k].set_xlabel(r'Effective Temperature $T_{\rm{eff}}/\rm{K}$')
    ax[ax_i, ax_k].set_ylabel(r'Luminosity $L/\rm{L}_\odot$')
    ax[ax_i, ax_k].set_title('HR Diagram zoom', size=16, loc='left')
    ax[ax_i, ax_k].set_xlim(lower_xlim, upper_xlim)
    ax[ax_i, ax_k].set_ylim(lower_ylim, upper_ylim)
    ax[ax_i, ax_k].invert_xaxis()
# ------------------------CHI^2 FORMULA------------------------------ #
def chi_squared(obs, exp, exp_unc):
    return ((obs - exp)/exp_unc)**2
# ------------------------SURFACE CORRECTION ------------------------------- #
def surf_correction(profile_number):
    Teff, logg, lum = get_params(profile_number)
    
    muHer_obs = pd.read_csv('../updated_muHer_freqs.txt', skiprows=2, sep='\s+')
    model = get_frequencies(profile_number)
    
    sorted_frequency_table = Table( names=('n', 'l', 'sorted_obs_freqs', 'sorted_obs_errors', 'sorted_model_freqs', 'sorted_E_norms') )
    a0 = None; a3 = None
    for ell in np.unique(model['l'].values):
        model_freqs = model['Re(freq)'][model['l'] == ell].values
        model_ls = model['l'][model['l'] == ell].values
        model_ns = model['n_p'][model['l'] == ell].values
        model_Es = model['E_norm'][model['l'] == ell].values
        
        obs_freqs = muHer_obs['nu'][muHer_obs['l'] == ell].values
        obs_errors = muHer_obs['error'][muHer_obs['l'] == ell].values
        obs_ls = muHer_obs['l'][muHer_obs['l'] == ell].values
        
        obs_index = []
        for model_freq in model_freqs:
            diffs = abs(obs_freqs - np.ones(len(obs_freqs))*model_freq)
            min_index = np.argmin(diffs)
            if (diffs[min_index] < 10):
                index = np.where( obs_freqs == obs_freqs[min_index] )[0]
                if len(index) == 1:
                    obs_index.append(index[0])
            else:
                continue
        obs_freqs = obs_freqs[obs_index]
        obs_errors = obs_errors[obs_index]
         
        model_index = []
        for obs_freq in obs_freqs:
            diffs = abs(model_freqs - np.ones(len(model_freqs))*obs_freq)   
            min_index = np.argmin(diffs)
            if (diffs[min_index] < 10):
                index = np.where( model_freqs == model_freqs[min_index] )[0]
                if len(index) == 1:
                    model_index.append(index[0])
            else:
                continue      
        model_freqs = model_freqs[model_index]
        model_Es = model_Es[model_index]
        model_ls = model_ls[model_index]
                   
        table = Table()
        table['sorted_obs_freqs'] = obs_freqs
        table['l'] = model_ls
        table['sorted_obs_errors'] = obs_errors
        table['sorted_model_freqs'] = model_freqs        
        table['sorted_E_norms'] = model_Es
        sorted_frequency_table = vstack([sorted_frequency_table, table])
    
    length_sorted_table = len(sorted_frequency_table['sorted_obs_freqs'])
        
    obs_freqs = sorted_frequency_table['sorted_obs_freqs'][sorted_frequency_table['l'] != 1]
    obs_errors = sorted_frequency_table['sorted_obs_errors'][sorted_frequency_table['l'] != 1]
    model_freqs = sorted_frequency_table['sorted_model_freqs'][sorted_frequency_table['l'] != 1]
    model_Es = sorted_frequency_table['sorted_E_norms'][sorted_frequency_table['l'] != 1]
    
    v_ac = (logg/4.44)*(Teff/5777)**(-1/2)*5000
    best = None
    a0 = None; a3 = None
    for ntrials in range(10):
        corr_model_freqs = lambda a: (a[0]*model_freqs) + (a[1]*(model_freqs/v_ac)**3)/model_Es
        func = lambda a: sum(((corr_model_freqs(a)-obs_freqs)/obs_errors)**2)
        now = minimize(func, (1, -2E-7))
        if len(now.x) == 2 and (best is None or best.fun > now.fun):
            print(ntrials)
            best = now
            a0 = best.x[0]; a3=best.x[1]
    print('Surface correction succesful!')
    
    corrected_frequency_table = Table( names=('l', 'obs_freqs', 'obs_errors', 'model_freqs', 'E_norms', 'corrected_freqs') )
    model_freqs = model['Re(freq)'].values
    model_Es = model['E_norm'].values
    model_ls = model['l'].values
       
    for ell in (0, 1, 2, 3):
        model_freqs = model['Re(freq)'][model['l'] == ell].values
        model_Es = model['E_norm'][model['l'] == ell].values
        model_ls = model['l'][model['l'] == ell].values
        obs_freqs = muHer_obs['nu'][muHer_obs['l'] == ell].values
        obs_errors = muHer_obs['error'][muHer_obs['l'] == ell].values
        obs_ls = muHer_obs['l'][muHer_obs['l'] == ell].values
        corrected_freqs = model_freqs + (a3*(model_freqs/v_ac)**3)/model_Es

        obs_index = []
        for corrected_freq in corrected_freqs:
            diffs = abs(obs_freqs - np.ones(len(obs_freqs))*corrected_freq)
            min_index = np.argmin(diffs)
            if (diffs[min_index] < 10):
                index = np.where( obs_freqs == obs_freqs[min_index] )[0]
                if len(index) == 1:
                    obs_index.append(index[0])
            else:
                continue
        obs_freqs = obs_freqs[obs_index]
        obs_errors = obs_errors[obs_index]
         
        model_index = []
        for obs_freq in obs_freqs:
            diffs = abs(corrected_freqs - np.ones(len(corrected_freqs))*obs_freq)   
            min_index = np.argmin(diffs)
            if (diffs[min_index] < 10):
                index = np.where( corrected_freqs == corrected_freqs[min_index] )[0]
                if len(index) == 1:
                    model_index.append(index[0])
            else:
                continue 
        model_freqs = model_freqs[model_index]
        model_Es = model_Es[model_index]
        model_ls = model_ls[model_index]
        corrected_freqs = corrected_freqs[model_index]
        
        table = Table()
        table['l'] = model_ls
        table['obs_freqs'] = obs_freqs
        table['obs_errors'] = obs_errors
        table['model_freqs'] = model_freqs        
        table['E_norms'] = model_Es
        table['corrected_freqs'] = corrected_freqs
        corrected_frequency_table = vstack([corrected_frequency_table, table])
    
    weights = [0.95, 0.05]
    chi2_spec  = weights[0]*(chi_squared(lum_obs, lum, lum_obs_unc) + chi_squared(teff_obs, Teff, teff_obs_unc) + chi_squared(logg_obs, logg, logg_obs_unc))
    chi2_seism = weights[1]*sum(chi_squared(corrected_frequency_table['obs_freqs'], corrected_frequency_table['corrected_freqs'], corrected_frequency_table['obs_errors']))
    chi2_total = chi2_spec + chi2_seism
    
    return corrected_frequency_table, chi2_total, chi2_spec, chi2_seism, a0, a3, v_ac, length_sorted_table
# ------------------------------------------------------------------------- #
def plot_echelle(profile_number, ax_i, ax_k):
    symbols = ('o', '^', 's', 'D')
    muHer_obs = pd.read_csv('../updated_muHer_freqs.txt',
                        skiprows=2, sep='\s+')
    model = get_frequencies(profile_number)
    for ell in (0, 1, 2, 3):
        old_nus = model['Re(freq)'][model['l'] == ell].values
        model_Es = model['E_norm'][model['l'] == ell].values
        nus = old_nus+(a3*(old_nus/v_ac)**3)/model_Es
        obs_nus = muHer_obs['nu'][muHer_obs['l'] == ell]
        
        shift = 18  
        Dnu = 64.2
        ax[ax_i, ax_k].plot(((nus-np.ones(len(nus))*shift) % Dnu),
                        nus,
                            symbols[ell], color='red', ms=6, markerfacecolor="None", label='corr', zorder=99)
        # Observed
        ax[ax_i, ax_k].plot(((obs_nus-np.ones(len(obs_nus))*shift) % Dnu),
                        obs_nus,
                            symbols[ell], color='k', ms=6, label='obs')
        # Uncorrected
        ax[ax_i, ax_k].plot(((old_nus-np.ones(len(old_nus))*shift) % Dnu),
                        old_nus,
                            symbols[ell], color='darkgrey', markerfacecolor="None", ms=6, label='model')
                            
        ax[ax_i, ax_k].set_xlim([-10, 64.2*1.4])
        ax[ax_i, ax_k].set_ylabel(r'frequency $\nu/\mu\rm{Hz}$')
        ax[ax_i, ax_k].set_xlabel(r'$(\nu-{\nu_{0}})\; \rm{mod}\; \Delta\nu/\mu\rm{Hz}$')
    ax[ax_i, ax_k].yaxis.set_label_position("right")
    ax[ax_i, ax_k].yaxis.tick_right()
    ax[ax_i, ax_k].set_title('Echelle diagram')
# -------------------------------- PLOT O-C ----------------------------------- #
def plot_oc(ax_i, ax_k):
    symbols = ('o', '^', 's', 'D')
    for ell in (0, 1, 2, 3):
        nus = corrected_frequency_table['corrected_freqs'][corrected_frequency_table['l'] == ell]
        old_nus = corrected_frequency_table['model_freqs'][corrected_frequency_table['l'] == ell]
        obs_nus = corrected_frequency_table['obs_freqs'][corrected_frequency_table['l'] == ell]
        
        diff = obs_nus-old_nus
        corrected_diff = obs_nus-nus
        
        ax[ax_i, ax_k].axhline(0, ls='--', c='darkgray', zorder=-99)
        ax[ax_i, ax_k].plot(obs_nus, diff, symbols[ell], color='black', ms=6, markerfacecolor="None")
        ax[ax_i, ax_k].plot(obs_nus, corrected_diff, symbols[ell], color='red', ms=6, markerfacecolor="None")        
        
        ax[ax_i, ax_k].set_ylim(-20, 10)
        ax[ax_i, ax_k].set_xlim(600, 1800)
        
        ax[ax_i, ax_k].set_xlabel(r'frequency $\nu/\mu\rm{Hz}$')
        ax[ax_i, ax_k].set_ylabel(r'O-C $\nu/\mu\rm{Hz}$')
        ax[ax_i, ax_k].yaxis.set_label_position("right")
        ax[ax_i, ax_k].yaxis.tick_right()
        ax[ax_i, ax_k].set_title('O-C plot')
        
    model_freqs = corrected_frequency_table['model_freqs'][corrected_frequency_table['l'] != 1]
    model_Es = corrected_frequency_table['E_norms'][corrected_frequency_table['l'] != 1]
    
    corr = (a3*(model_freqs/v_ac)**3)/model_Es
    ax[ax_i, ax_k].plot(model_freqs, corr, '.', color='blue', zorder=-99, ms=4)
    ax[ax_i, ax_k].text(625, -17, r'$\rm{a}_0$ = '+str(a0))
    ax[ax_i, ax_k].text(625, -19, r'$\rm{a}_3$ = '+str(a3))
# ----------------------------------------------------------------------------- #

delta_nu_obs = 64.06732857142856
delta_nu_obs_unc = 0.2
sigma_limits = [5, 3, 2, 1, 10]
prof_num_ints = [10, 5, 3, 2, 1, 20]


good_models_table = Table()
good_models_table['Y'] = [0.29]#[0.27, 0.27, 0.28, 0.28, 0.29, 0.29, 0.29, 0.29, 0.30]
good_models_table['Z'] = [0.032]#[0.026, 0.028, 0.030, 0.034, 0.030, 0.032, 0.034, 0.034, 0.032]
good_models_table['alpha_mlt'] = [1.9]#[2.0, 2.0, 2.0, 1.9, 2.0, 1.9, 1.9, 2.0, 2.0]

print(len(good_models_table))


#for row_number in range(len(good_models_table)):
row_number = 0
row = good_models_table[row_number]
Yini = row['Y']; Zini = row['Z']; M = 1.15; alpha_mlt = row['alpha_mlt']; rate = 1.0
for Yini in (0.28, 0.285, 0.29, 0.295, 0.30):  
    for Zini in (0.030, 0.031, 0.032, 0.033, 0.034):
        for alpha_mlt in (1.8, 1.9, 2.0):
            for M in (1.1, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20): 
                f_ov = 0.1
                #LOGS_directory = 'Y:'+str(Yini)+'_Z:'+str(Zini)+'_amlt:'+str(alpha_mlt)+'_fov:'+str(f_ov)+'_M:'+str(M)
                LOGS_directory = 'Y:'+str(Yini)+'_Z:'+str(Zini)+'_amlt:'+str(alpha_mlt)+'_M:'+str(M)+'_fov:'+str(f_ov)+'_rate:'+str(rate)
                print(LOGS_directory)
                #max_num_profile_models = -1
                
                if os.path.exists(os.path.join(os.getcwd(), LOGS_directory)):
                    print('folder already exists')
                    continue
                    
                FNULL = open(os.devnull, 'w')
                #subprocess.call('bash mesa-source.sh '+str(Yini)+' '+str(Zini)+' '+str(Zini)+' '+str(alpha_mlt)+' '+str(M)+' '+str(LOGS_directory)+' '+str(f_ov)+' '+str(max_num_profile_models), shell=True)
                subprocess.call('bash mesa-source.sh '+str(Yini)+' '+str(Zini)+' '+str(Zini)+' '+str(alpha_mlt)+' '+str(M)+' '+str(LOGS_directory)+' '+str(f_ov)+' '+str(rate), shell=True)
                #process = subprocess.Popen('./star >/dev/null &', shell=True)
                #process = subprocess.Popen('./star ', shell=True)
                process = subprocess.Popen('./star', stdout=FNULL, shell=False)
                
                print(process.pid)
                
                model_running = False
                while model_running == False:
                    is_model_running = process.poll()
                    #print(is_model_running)
                    if is_model_running != None:
                        print('MESA model hasent started yet')
                        model_running = False
                        time.sleep(2)
                    elif is_model_running == None:
                        model_running = True
                    
                #is_model_running = True
                #while is_model_running != None:
                  #  try:
                 #       is_model_running = process.poll
                        #pid = process.pid
                        #os.kill(pid, 0)
                 #  except:
                  #     print('MESA model hasent started yet')
                   #     time.sleep(2)
                   #     pass
                
                
                pid = process.pid
                
                mother_dir = os.getcwd()
                logs_dir_full_path = os.path.join(mother_dir, LOGS_directory)
                
                
                best_profile = -1
                
                number_of_trials = 50
                appended_profile_numbers = []
               
                v_avcr_diffs = []
                v_avcrs = []
                
                a0s = []
                a3s = []
                
                 
                profile_number = 5 
                profile_number_check = 20
                #while profile_number_check == None:
                #    try:
                #        DF = pd.read_csv(os.path.join(logs_dir_full_path, 'history.data'), skiprows=5, sep='\s+')
                #        min_center_h1 = min(DF['center_h1'])
                #    except:
                #        print('history messed up')
                #        time.sleep(5)
                #        continue
                #        
                #    if min_center_h1 < 0.6:
                #        i=0
                #        try:  
                #            print(min_center_h1)
                #            index_file = pd.read_csv(os.path.join(logs_dir_full_path, 'profiles.index'), names=['model_number', 'priority', 'profile_number'], skiprows=1, sep='\s+', engine ='python')
                 #           profile_number_check = len(index_file.profile_number)+20
                 #           profile_number = len(index_file.profile_number)+1     
                 #           print('now past early MS')             
                 #       except:
                 #           print('index messed up')
                 #           time.sleep(5)
                 #           continue
                 #   elif min_center_h1 > 0.6:
                 #       print(min_center_h1)
                 #       print('still on early MS')
                 #       time.sleep(20)
            
                                
                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number_check) + '.data.FGONG')
                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number_check) + '.data')        
                
                
                check = False
                best_profile_number = None
                while check == False:
                    if os.path.exists(profile_path) and os.path.exists(FGONG_path):
                        os.chdir(logs_dir_full_path)
                        #print('dir changed')
            
                        FGONG_file = os.path.join('profile' + str(profile_number) + '.data.FGONG')
                                              
                        print('calculating l=0 freqs')  
                        subprocess.call('../gyre6freqs_l0.sh -i ' + FGONG_file + ' -f', shell=True)
            
                        logs_dir = os.getcwd()
                        dat_file = os.path.join(logs_dir, 'profile' + str(profile_number) + '-freqs.dat')
                                              
                        if os.path.exists(dat_file):
                            a0, a3 = calc_a0(profile_number)
                            print('a0 and a3:', a0, a3)
                            a0s.append(a0)
                            a3s.append(a3)
                              
                            a0s_diff = abs(a0s-np.ones(len(a0s)))                                
                            
                            if (a0 == 1.0) and (best_profile_number == None):
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[1]
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data') 
                                continue
                            elif ( abs(a0-1) > 1.05*min(a0s_diff[a0s_diff != 0]) ) and (best_profile_number == None) and (a0 > 1):
                                best_profile_number = profile_number
                                print('best a0 found --> |a0-1| has been minimized')
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[4]
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')                                
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data') 
                                continue
                            elif ( abs(a0-1) > 1.05*min(a0s_diff[a0s_diff != 0]) ) and (best_profile_number == None) and (a0s_diff[-2] <= 0.0001):
                                best_profile_number = profile_number
                                print('best a0 found --> |a0-1| has been minimized')
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[4]
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')                                
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data')
                                continue
                            elif (abs(a0-1) >= 0.01) and (best_profile_number == None):
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[1]
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data') 
                                continue
                            elif (abs(a0-1) >= 0.001) and (abs(a0-1) < 0.01) and (best_profile_number == None):
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[2]
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data') 
                                continue
                            elif (abs(a0-1) >= 0.0001) and (abs(a0-1) < 0.001) and (best_profile_number == None):
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[3]
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data') 
                                continue
                            elif (best_profile_number != None):
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[4]
                                final_profile = None
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data')
                                i = 0
                                while (final_profile == None):
                                    try:
                                        index_file = pd.read_csv('profiles.index', names=['model_number', 'priority', 'profile_number'], skiprows=1, sep='\s+', engine ='python')
                                        final_profile = len(index_file.profile_number)
                                    except:
                                        print('retrying index file')
                                        i += 1
                                        if i == 10:
                                            print('model crashed at a0 optimization')
                                            break
                                        elif i < 10:
                                            time.sleep(1)
                                            pass   
                                if (profile_number == best_profile_number+10) or (profile_number == final_profile):
                                    check = True
                            else:   
                                appended_profile_numbers.append(profile_number)
                                profile_number += prof_num_ints[4]   
                                FGONG_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data.FGONG')
                                profile_path = os.path.join(logs_dir_full_path, 'profile' + str(profile_number) + '.data')                     
                                continue
                        
                    else:
                        is_model_running = process.poll()
                        if is_model_running == None:
                            print('proces is running: waiting')
                            time.sleep(10)
                        elif is_model_running != None:
                            print('proces is not running: model complete')
                            if (best_profile_number != None):
                                break
                            elif (best_profile_number == None):
                                os.chdir(mother_dir)
                                break
                                
                        #try:
                        #   os.kill(pid, 0)
                         ##   print('proces is running: waiting')
                        #    time.sleep(10)
                        #except:
                        #    print('proces is not running')
                        #    print('model complete')
                        #    os.chdir(mother_dir)
                        #    break
                                                       
                                                                                                             
                if best_profile_number != None:
                
                    a0s_diff = abs(a0s-np.ones(len(a0s)))        
                    best_a0_index = np.where( a0s_diff == min(a0s_diff[a0s_diff > 0]) )[0]
                    best_a0_profile = appended_profile_numbers[best_a0_index[0]]
                
                    
                    best_a0_profile = best_profile_number
                    #print(process.pid)
                    #print(pid)
                    #print('Killing Star Model >:)')
                    #os.kill(pid, signal.SIGKILL)
                    
                    try:
                        print('Killing Star Model >:)')
                        os.kill(pid, signal.SIGKILL)
                    except:
                        print('model is already done')
                        
                    #subprocess.call('kill '+ str(pid), shell=True)
                    
                    print('Now scanning profiles around best_a0_profile:', best_a0_profile)
                    i = 0
                    final_profile = None
                    while (final_profile == None):
                        try:
                            index_file = pd.read_csv('profiles.index', names=['model_number', 'priority', 'profile_number'], skiprows=1, sep='\s+', engine ='python')
                            final_profile = len(index_file.profile_number)
                        except:
                            print('retrying index file for a0 scan')
                            i += 1
                            if i == 10:
                                print('model crashed at a0 scan')
                                break
                            elif i < 10:
                                time.sleep(1)
                                pass
            
                    profile_numbers = []
            
                    for profile_number in appended_profile_numbers:
                        if (profile_number in range(best_a0_profile-10, best_a0_profile+11)) and (profile_number <= len(index_file.profile_number)+1):
            
                            logs_dir = os.getcwd()      
                            profile_file_name = os.path.join(logs_dir, 'profile' + str(profile_number) + '.data')  
                            
                            if os.path.exists(profile_file_name):
                                   
                                best_a3_index = np.where( np.array(appended_profile_numbers) == profile_number )[0] 
                                best_a3 = a3s[best_a3_index[0]]
            
                                print('a3:', best_a3)
                                
                                FGONG_file = os.path.join('profile' + str(profile_number) + '.data.FGONG')
                                print('calculating l=0 and l=1 freqs')                            
                                subprocess.call('../gyre6freqs_l0_l1.sh -i ' + FGONG_file + ' -f', shell=True)
                                
                                v_avcr = calc_v_avcr(profile_number)
                                
                                v_avcrs.append(v_avcr)
                                print('v_avcr:', v_avcr)
            
                                v_avcr_diff = v_avcr
                                v_avcr_diffs.append(v_avcr_diff)
                                
                                profile_numbers.append(profile_number)
            
                    if (min(v_avcr_diffs) != 9999999) and (len(v_avcr_diffs) > 0):         
                        best_profile_index = np.argmin(v_avcr_diffs)
                        best_profile = profile_numbers[best_profile_index]       
            
                        print('Now calculating all frequencies for best a0 profile:', best_a0_profile)
                        best_FGONG_file = os.path.join('profile' + str(best_a0_profile) + '.data.FGONG')
                        subprocess.call('../gyre6freqs.sh -i ' + best_FGONG_file + ' -f', shell=True)
            
                        
                        corrected_frequency_table, chi2_total, chi2_spec, chi2_seism, a0, a3, v_ac, length_sorted_table = surf_correction(best_a0_profile)
                        
                        if (length_sorted_table > 40) and (chi2_total < 200):
                            
                            fig, ax = plt.subplots(2, 2, figsize=(18,12), dpi = 120)
                            fig.suptitle(r'Best a0 model: $\chi^2$ = '+str(round(chi2_total, 2)), fontsize=22, fontweight='bold')
                            plot_hr(best_a0_profile, 0, 0)
                            plot_echelle(best_a0_profile, 0, 1)
                            plot_a0_evolution(1, 0)
                            plot_oc(1, 1)
                            fig.savefig('../figurer_til_Earl/med_overshoot/a0/'+str(LOGS_directory)+'_BEST_a0.png')
                            
                            Teff, logg, lum, age, radius = get_table_params(best_a0_profile)
                            
                            base_table = Table.read('../parameters_a0.ascii', format='ascii.fixed_width')
                            table = Table()
                            table['M'] = [M]
                            table['Y'] = [Yini]
                            table['Z'] = [Zini]
                            table['alpha_mlt'] = [alpha_mlt]
                            table['f_ov'] = [alpha_mlt]
                            table['age'] = [round(age, 3)]
                            table['Teff'] = [round(Teff,2)]
                            table['L'] = [round(lum,2)]
                            table['R'] = [round(radius,2)]
                            table['logg'] = [round(logg, 2)]
                            table['chi2_total'] = [chi2_total]
                            table['chi2_spec'] = [chi2_spec]
                            table['chi2_seism'] = [chi2_seism]
                            
                            combined_table = vstack([base_table, table])
                            combined_table.write('../parameters_a0.ascii', overwrite=True, format='ascii.fixed_width')
                            
                            print('Now calculating all frequencies for best MM profile:', best_profile)
                            best_FGONG_file = os.path.join('profile' + str(best_profile) + '.data.FGONG')
                            subprocess.call('../gyre6freqs.sh -i ' + best_FGONG_file + ' -f', shell=True)
                            
                            corrected_frequency_table, chi2_total, chi2_spec, chi2_seism, a0, a3, v_ac, length_sorted_table = surf_correction(best_profile)
                            
                            fig, ax = plt.subplots(2, 2, figsize=(18,12), dpi = 120)
                            fig.suptitle(r'Best MM model: $\chi^2$ = '+str(round(chi2_total, 2)), fontsize=22, fontweight='bold')
                            plot_hr(best_profile, 0, 0)
                            plot_echelle(best_profile, 0, 1)
                            plot_a0_evolution(1, 0)
                            plot_oc(1, 1)
                            fig.savefig('../figurer_til_Earl/med_overshoot/MM/'+str(LOGS_directory)+'_BEST_MM.png')
                            
                            Teff, logg, lum, age, radius = get_table_params(best_profile)
                            
                            base_table = Table.read('../parameters_MM.ascii', format='ascii.fixed_width')
                            table = Table()
                            table['M'] = [M]
                            table['Y'] = [Yini]
                            table['Z'] = [Zini]
                            table['alpha_mlt'] = [alpha_mlt]
                            table['f_ov'] = [f_ov]
                            table['age'] = [round(age, 3)]
                            table['Teff'] = [round(Teff,2)]
                            table['L'] = [round(lum,2)]
                            table['R'] = [round(radius,2)]
                            table['logg'] = [round(logg, 2)]
                            table['chi2_total'] = [chi2_total]
                            table['chi2_spec'] = [chi2_spec]
                            table['chi2_seism'] = [chi2_seism]
                            
                            combined_table = vstack([base_table, table])
                            combined_table.write('../parameters_MM.ascii', overwrite=True, format='ascii.fixed_width')
                                                       
                            os.chdir(mother_dir)
                            continue
                            
                        
                        else:
                            print('model is shit!')
                            os.chdir(mother_dir)
                            continue
                    else:
                        print('model is shit!')
                        os.chdir(mother_dir)
                        continue
                            
                
        

    

    
    
    
    