# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 12:08:42 2025

@author: javier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from scipy import stats


#%% FUNCTIONS

dB10 = lambda x: 10*np.log10(x)
dB20 = lambda x: 20*np.log10(x)

def butter_LPF(data, order, f_cutoff ,f_sampling):
 
    ''' butterworth low-pass filter '''
    
    sos = signal.butter(order, f_cutoff, fs = f_sampling, output = 'sos')
    # data_filtered = f_sampling*signal.sosfilt(sos, data)
    data_filtered = signal.sosfilt(sos, data)
    
    return data_filtered


def calculate_PSD(data, f_sampling, t_skip, Navg = 1, ymin = -120, ymax = 20, label = '', debug = False):
    
    # Skip the first t_skip seconds
    samples_skip = int(f_sampling*t_skip)
    data_crop = data[samples_skip:]
    
    # Calculate PSD:    
    freqs, Sxx = signal.welch(data_crop, f_sampling, nperseg = len(data_crop)/Navg)
    
    if (debug == True):
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel('f [Hz]') 
        ax.set_ylabel('Power Spectral Density [$\mathrm{dBV_\mathrm{rms}/\sqrt{Hz}}$]') 
        ax.set_ylim(ymin,ymax)
        ax.semilogx(freqs, dB10(Sxx), label = label)
        ax.grid()    
        ax.legend()
    
    return freqs, Sxx

#%% CONFIG SCRIPT

extension     = '.dataframe'
filename_root = 'C:\\Users\\MErtl\\Desktop\\Digital ISFET Detection\\SMARTLOC2\\PYTHON\\measurements\\'

filename      = 'ionimager0.2without_filter'

pixels_to_plot = [79,158,238,178,161,177,193,86,51,90,131]

pixels_bead = [79,158,174,238,107,106,227,130,178,161,177,193,176]
pixels_no_bead = [200,136,36,48,185,152]

# Filter-Einstellungen
f_LPF = 0.25
do_LPF = True

do_PSD = False
n_PSD_averages = 2



#%% RUN


df_read = pd.read_pickle(filename_root + filename + extension)

timestamps          = df_read.Timestamps[0]
measurement_config  = df_read.Measurement_configuration[0]

f_sampling = measurement_config['f_sampling']
T_sampling_ms = measurement_config['sampling_period_ms']
time_max_s = measurement_config['stop_time_s']

print(f'Sampling frequency: {f_sampling} Hz')






title = filename_root+filename

# prepare plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)

if(do_PSD):
    ax.set_xlabel('f [Hz]') 
    ax.set_ylabel('Power Spectral Density [$\mathrm{dBV_\mathrm{rms}/\sqrt{Hz}}$]') 
else:
    ax.set_xlabel('time (s)') 
    ax.set_ylabel('eps') 
    ax.set_ylabel('mV') 
    
ax.grid() 
colormap = plt.get_cmap('tab20')
number_of_plots = len(pixels_to_plot)
ax.title.set_text(title)

xmax = time_max_s
if (xmax >= 14400):
    ax.set_xticks(np.arange(0,xmax+3600,3600))  
elif (xmax >= 7200):
    ax.set_xticks(np.arange(0,xmax+600,600))    
elif (xmax >= 1200):
    ax.set_xticks(np.arange(0,xmax+300,300))
elif (xmax >= 600):
    ax.set_xticks(np.arange(0,xmax+60,60)) 
    


t_equalize = 405

for n_pixel in range(256):
    
    data_mV = df_read.Milivolts[n_pixel]
    data_eps = df_read.Eps[n_pixel]
    data_events = df_read.Events[n_pixel]
    
    data_eps_equalized = data_eps-data_eps[timestamps == t_equalize]
    data_mV_equalized = data_mV-data_mV[timestamps == t_equalize]

            
    if (n_pixel in pixels_to_plot):
        
        # if n_pixel in pixels_bead:
        #     color = 'red'
        # if n_pixel in pixels_no_bead:
        #     color = 'blue'
        
        if(do_PSD):
            freqs, Sxx = calculate_PSD(data_eps, f_sampling, t_skip = 0, Navg = n_PSD_averages, ymin = -120, ymax = 0, debug = False)
            ax.semilogx(freqs, dB10(Sxx), label = f'pixel{n_pixel}')
        else:
            data_eps_filtered = butter_LPF(data_eps, order = 1, f_cutoff = f_LPF ,f_sampling = f_sampling)   
            data_mV_filtered = butter_LPF(data_mV, order = 1, f_cutoff = f_LPF ,f_sampling = f_sampling)   
            data_mV_filtered_equalized = data_mV_filtered-data_mV_filtered[timestamps == t_equalize]
            # ax.plot(timestamps, data_events, label = f'pixel{n_pixel}')    
            
            if(do_LPF):
                # ax.plot(timestamps, data_eps_filtered, label = f'pixel{n_pixel}') 
                # ax.plot(timestamps, data_mV_filtered, label = f'pixel{n_pixel}')    
                ax.plot(timestamps, data_mV_filtered_equalized, label = f'pixel{n_pixel}')    
                
            else:
                # ax.plot(timestamps, data_eps, label = f'pixel{n_pixel}', color = color)    
                # ax.plot(timestamps, data_eps_equalized, label = f'pixel{n_pixel}') 
                # ax.plot(timestamps, data_mV, label = f'pixel{n_pixel}') 
                ax.plot(timestamps, data_mV_equalized, label = f'pixel{n_pixel}') 
                

        ax.legend(ncols = 2)
        
    
# %%