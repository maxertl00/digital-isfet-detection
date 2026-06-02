#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:35:41 2025

@author: javier & max
"""

# Section: Module overview
# Purpose: Play 2D time-series data from an ISFET array stored in a pandas DataFrame.
# This module loads measurement data, applies optional preprocessing (low-pass filter,
# equalization, drift compensation), and provides an interactive 2D video player.

import time
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from scipy import signal
from scipy import stats

from Plot_functions import (start_dynamic_plot_2D_plt,update_dynamic_plot_2D_plt)
from smartloc2_functions import (reshape_data_smartloc2)


def butter_LPF(data, order, f_cutoff ,f_sampling):
 
    ''' Butterworth low-pass filter '''
    
    sos = signal.butter(order, f_cutoff, fs = f_sampling, output = 'sos')
    # data_filtered = f_sampling*signal.sosfilt(sos, data)
    data_filtered = signal.sosfilt(sos, data)
    
    return data_filtered

def run_2Dvideo(filename, configuration, window = None):
    # Section: Video runner
    # Purpose: Load measurement data, preprocess signals, and run interactive playback.
    
    
    # Section: Configuration extraction
    # Purpose: Read playback and processing parameters from the configuration dict.
    speed               = configuration['speed']
    val_max             = configuration['val_max']
    val_min             = configuration['val_min']
    t_start_video       = configuration['t_start_video']
    t_end_video         = configuration['t_end_video']
    do_compensate_drift = configuration['do_compensate_drift']
    t_start_drift       = configuration['t_start_drift']
    t_end_drift         = configuration['t_end_drift']    
    do_LPF              = configuration['do_LPF']
    freq_LPF            = configuration['freq_LPF']
    
    
    # Section: Data loading
    # Purpose: Load the saved pandas DataFrame containing timestamps and per-pixel signals.
    df_read = pd.read_pickle(filename)
    
    measurement_configuration = df_read.Measurement_configuration[0]
    sampling_period_ms  = measurement_configuration['sampling_period_ms']
    f_sampling = measurement_configuration['f_sampling'] 
    
    timestamps = df_read.Timestamps[0]

    array_milivolts = df_read.Eps.to_numpy()
    array_milivolts_mod = df_read.Eps.to_numpy()
    
    # Section: Time cropping
    # Purpose: Determine the time range used for video playback and for drift compensation.
    # crop for video length
    video_crop_array = (timestamps >= t_start_video) & (timestamps < t_end_video)
    timestamps_video = timestamps[video_crop_array]
    
    # crop for drift compensation
    drift_crop_array = (timestamps >= t_start_drift) & (timestamps < t_end_drift)
    timestamps_crop = timestamps[drift_crop_array]

    array_milivolts_cropped = [pixel_milivolts[video_crop_array] for pixel_milivolts in array_milivolts]

    # Section: Preprocessing
    # Purpose: Optionally apply filtering to the per-pixel time series.
    # Apply LPF if required
    if(do_LPF == True):
        array_milivolts_mod = np.array([butter_LPF(milivolts_mod, 4, freq_LPF, f_sampling) for milivolts_mod in array_milivolts_mod])
           
          
    # equalization code removed

        
    
    # Section: Per-pixel processing
    # Purpose: Optionally fit and remove linear drift per pixel, then crop for playback.
    ## Loop over pixels and modify the values
    for n_pixel in range(64):
          
        if(do_compensate_drift == True):
            mV_crop = array_milivolts[n_pixel][drift_crop_array]
            res = stats.linregress(timestamps_crop, mV_crop)
            # scipy.stats.linregress returns an object with attribute 'slope'
            drift = getattr(res, 'slope', None)
            if drift is None:
                drift = 0.0
            array_milivolts_mod[n_pixel] = array_milivolts[n_pixel] - drift*timestamps
            
        # if(do_equalize == True):
      
        array_milivolts_cropped[n_pixel] = array_milivolts_mod[n_pixel][video_crop_array]
    
    # Section: Plot setup
    # Purpose: Initialize the 2D plot and define the window title and labels.
    # Start plot
    title = 'ISFET Array 2D Video Player - ENTER: Pause | ←/→: Jump 3 sec | ↑/↓: Speed | Q: Quit'
    fig, image, ax = start_dynamic_plot_2D_plt(val_max = val_max, val_min = val_min, title = title, zlabel = 'eps')

    # Section: Playback state
    # Purpose: Keep track of playback index, speed, pause and quit state.
    # State variables for control
    state = {
        'index': 0,
        'speed': speed,
        'paused': False,
        'quit': False
    }
    
    # Speed steps: discrete speed multiplier options
    speed_steps = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    # Section: Interaction handlers
    # Purpose: Define keyboard controls for playback (pause, seek, speed, quit).
    # Keyboard event handler
    def on_key(event):
        print(f"Key pressed: {event.key}")  # Debug output
        if event.key == 'enter':
            state['paused'] = not state['paused']
            print(f"Paused: {state['paused']}")
        elif event.key == 'left':
            # jump back by 3 seconds worth of frames
            frames_jump = int(round(3 * f_sampling))
            state['index'] = max(0, state['index'] - frames_jump)
        elif event.key == 'right':
            # jump forward by 3 seconds worth of frames
            frames_jump = int(round(3 * f_sampling))
            state['index'] = min(len(timestamps_video) - 1, state['index'] + frames_jump)
        elif event.key == 'up':
            # Increase speed
            current_idx = speed_steps.index(min(speed_steps, key=lambda x: abs(x - state['speed'])))
            if current_idx < len(speed_steps) - 1:
                state['speed'] = speed_steps[current_idx + 1]
        elif event.key == 'down':
            # Decrease speed
            current_idx = speed_steps.index(min(speed_steps, key=lambda x: abs(x - state['speed'])))
            if current_idx > 0:
                state['speed'] = speed_steps[current_idx - 1]
        elif event.key == 'q':
            state['quit'] = True
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Print instructions
    print("\n=== VIDEO CONTROLS ===")
    print("ENTER       : Pause/Play")
    print("←  (Left)   : Rewind 3 seconds")
    print("→  (Right)  : Forward 3 seconds")
    print("↑  (Up)     : Increase speed")
    print("↓  (Down)   : Decrease speed")
    print("               Steps: 0.1 → 0.25 → 0.5 → 0.75 → 1 → 1.5 → 2 → 3 → ... → 10")
    print("Q            : Quit")
    print("=" * 70 + "\n")

    # Section: Playback loop
    # Purpose: Iterate through frames, update plot, and handle events.
    # Start 2D video
    while state['index'] < len(timestamps_video) and not state['quit']:
        
        if not state['paused']:
            start_time = time.time()
            t = timestamps_video[state['index']]
            
            array_mV_frame = np.array([pixel_mV[state['index']] for pixel_mV in array_milivolts_cropped])      
        
            data_2d = reshape_data_smartloc2(array_mV_frame)
            status_line = f'Time: {round(t,2)}s  |  Fs: {round(f_sampling,2)} Hz  |  Speed: {round(state["speed"],2)}x  |  Frame: {state["index"]}/{len(timestamps_video)-1}  |  {"PAUSED" if state["paused"] else "PLAYING"}'
            update_dynamic_plot_2D_plt(fig, image, ax, data_2d, 
                                       title = title + '\n' + status_line)
            
            fig.canvas.draw_idle()
            
            # Framerate control with aggressive event polling
            frame_duration = sampling_period_ms*1e-3/state['speed']
            poll_interval = 0.01  # Poll every 10ms
            while (time.time() - start_time) < frame_duration:
                fig.canvas.flush_events()  # Process all pending events
                time.sleep(poll_interval)
                if state['quit']:
                    break
            
            if not state['paused']:  # Only advance if not paused
                state['index'] += 1
        else:
            # While paused: process events and display status
            t = timestamps_video[state['index']]
            array_mV_frame = np.array([pixel_mV[state['index']] for pixel_mV in array_milivolts_cropped])
            data_2d = reshape_data_smartloc2(array_mV_frame)
            status_line = f'Time: {round(t,2)}s  |  Fs: {round(f_sampling,2)} Hz  |  Speed: {round(state["speed"],2)}x  |  Frame: {state["index"]}/{len(timestamps_video)-1}  |  {"PAUSED" if state["paused"] else "PLAYING"}'
            update_dynamic_plot_2D_plt(fig, image, ax, data_2d,
                                       title = title + '\n' + status_line)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(0.05)  # Polling while paused
    
    plt.show(block=True)
    return



#%% RUN CODE FOR TESTS
if __name__ == "__main__":
    extension     = '.dataframe'
    filename_root = r'C:\Users\MErtl\Desktop\Digital ISFET Detection\SMARTLOC2\PYTHON\measurements\\'

    # Open a GUI file dialog at the measurements folder for user to choose a .dataframe
    measurements_dir = os.path.normpath(filename_root)
    initial_dir = measurements_dir if os.path.isdir(measurements_dir) else os.getcwd()
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(initialdir=initial_dir,
                                          title='Select .dataframe file',
                                          filetypes=(('DataFrame files', '*.dataframe'), ('All files', '*.*')))
    root.destroy()
    if not selected:
        print('No file selected. Exiting.')
        raise SystemExit(1)
    video_2D_file = selected
    
    config_2Dvideo = dict(
        speed               = 1,
        val_max             = 100,
        val_min             = -100,
        t_start_video       = 0,
        t_end_video         = np.inf,  # Bis zum Ende der Daten
        do_compensate_drift = False,
        t_start_drift       = 90,
        t_end_drift         = 120,
        do_equalize         = True,
        t_equalize          = 10,  # Früher Punkt zum Normalisieren
        do_LPF      =       False,
        freq_LPF    =       0.05
        )            
    
    
    run_2Dvideo(video_2D_file, config_2Dvideo)