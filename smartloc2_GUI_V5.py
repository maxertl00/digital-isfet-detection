import tkinter as tk
from tkinter import ttk,filedialog,Toplevel, Frame

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import threading
import time

import queue
import os
from datetime import datetime
import numpy as np
import platform
import pandas as pd
import serial
import serial.tools.list_ports
from scipy import stats

from pydwf import DwfLibrary, DwfEnumConfigInfo, DwfAnalogOutNode, DwfAnalogOutFunction, DwfAnalogOutIdle, DwfDeviceID
from pydwf.utilities import openDwfDevice

from smartloc2_functions import (serial_connect,do_chip_reset,do_array_config,create_dataframe,reshape_data_smartloc2,
                                do_acq_config,do_acq_loop,read_mem,read_overflow)


#%% - App metadata -

APP_NAME = "SmartLoC2 GUI"
APP_VERSION = "v0.1.0"
APP_AUTHOR = "Javier Cuenca Michans [javier.cuenca@csic.es]"


#%% PYDWF FUNCTIONS

def analog_out_custom_waveform(analogOut, waveform, t_final):
    """Demonstrate simple analog output control.

    Configure output channel for square wave output, with 'idle' behavior set to
    drive the initial signal value. But note that we never actually start this waveform generation!

    We just set the signal amplitude in Volt, which changes the output of the idle level of the signal.
    This is the simplest way to directly drive the output of the analog output channels that is also at
    least somewhat performant; setting a single channel's output level in this way takes roughly 0.5 … 1 ms.
    """

    CH1 = 0
    node = DwfAnalogOutNode.Carrier

    # Reset the channel.
    analogOut.reset(CH1)

    # Enable the node
    analogOut.nodeEnableSet(CH1, node, True)
    
    # Set custom waveform function
    analogOut.nodeFunctionSet(CH1, node, DwfAnalogOutFunction.Custom)    

    # Set amplitude and offset
    analogOut.nodeAmplitudeSet(CH1, node, 1)
    analogOut.nodeOffsetSet(CH1, node, 0)
    
    # Set the desired waveform
    analogOut.nodeDataSet(CH1, node, waveform)

    # Wait duration before each waveform emission.
    analogOut.waitSet(CH1, 0)
  
    # The frequency of a custom waveform is (1 / waveform_duration).
    analogOut.nodeFrequencySet(CH1, node, 1/t_final)
  
    # Run for indefinite length
    analogOut.runSet(CH1, 0)
    
    # Repeat the Wait/Running states indefinitely.
    # analogOut.repeatSet(CH1, 0)

    # Idle value is the initial value
    analogOut.idleSet(CH1, DwfAnalogOutIdle.Initial)    

    # input("Press Enter to start the waveform...")

    analogOut.configure(False, True)    # Start
    print('waveform started')

def maximize_analog_out_buffer_size(configuration_parameters):
    """Select the configuration with the highest possible analog out buffer size."""
    return configuration_parameters[DwfEnumConfigInfo.AnalogOutBufferSize] 


#%% SMARTLOC2 FUNCTIONS

def linear_function_inverse(x, m_pos, m_neg):

    '''
    counts -> milivolts
    inverse of a linear function with m_pos positive slope for x >= 0
    and m_neg negative slope for x < 0
    '''
    
    # Scalar mode
    if(isinstance(x, float)):  
        # Apply the function
        if (x < 0):
            y = x / m_neg
        else:
            y = x / m_pos
    
    # Array mode
    else:             
        y = np.zeros(len(x))
        
        # Apply the function
        y[x < 0]  = x[x < 0] / m_neg
        y[x >= 0] = x[x >= 0] / m_pos
    
    return y


def gain_analysis(file_to_save, amplitudes, initial_delay, pulse_period, plot_configuration):

    file_to_save_gains = file_to_save + '_gains'
    extension = '.dataframe'
    
    df_gains = pd.DataFrame({
        "Pixel_index": range(256),
        "Gains_positive": None,
        "Gains_negative": None
        })
    
    amplitudes_pos, amplitudes_neg = np.split(1e3*amplitudes,2)
    
    # obtain the positive gains
    t_start = initial_delay
    array_gains_pos, eps_out_pos, array_offsets_pos = gain_regression_calc(file_to_save+extension, amplitudes_pos, pulse_period, t_start, 
                                               pixels_to_plot = plot_configuration['pixels_to_plot'], debug_level = 0, label = 'Positive')
    
    # obtain the negative gains
    t_start = initial_delay + pulse_period*len(amplitudes)/2 
    array_gains_neg, eps_out_neg, array_offsets_neg = gain_regression_calc(file_to_save+extension, amplitudes_neg, pulse_period, t_start, 
                                               pixels_to_plot = plot_configuration['pixels_to_plot'], debug_level = 0, label = 'Negative')
    
    ## - PLOTS -
    
    plt.ioff()
    
    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_subplot(111)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title(f'SmartLoC2 gain calibration ({file_to_save_gains})', fontsize=11)        
    
    ax1.set_xlabel('Input voltage [mV]')
    ax1.set_ylabel('Event frequency [eps]')    
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = iter(prop_cycle.by_key()['color'])
    
    amplitudes_pos_z = np.insert(amplitudes_pos, 0, 0) 
    amplitudes_neg_z = np.insert(amplitudes_neg, 0, 0) 
    # eps_out_pos = np.insert(eps_out_pos, 0, 0, axis = 1) 
    # eps_out_neg = np.insert(eps_out_neg, 0, 0, axis = 1) 
    
    for pixel_index in plot_configuration['pixels_to_plot']:
        
        gain_pos = array_gains_pos[pixel_index]
        offset_pos = array_offsets_pos[pixel_index]
        
        gain_neg = array_gains_neg[pixel_index]
        offset_neg = array_offsets_neg[pixel_index]    
    
        color = next(colors)
        ax1.plot(amplitudes_pos, eps_out_pos[pixel_index], 'o', color = color, label = f'pix {pixel_index}') 
        ax1.plot(amplitudes_neg, eps_out_neg[pixel_index], 'o', color = color) 
        
        ax1.plot(amplitudes_pos_z, gain_pos*amplitudes_pos_z + offset_pos, '--', color = color) # label = f'pix{pixel_index} R2 = {rsquare}', color = color)   
        ax1.plot(amplitudes_neg_z, gain_neg*amplitudes_neg_z + offset_neg, '--', color = color) # label = f'pix{pixel_index} R2 = {rsquare}', color = color)
        
        ax1.legend(ncols = 2)    

      
    array_gains_pos_2D = reshape_data_smartloc2(array_gains_pos) 
    array_gains_neg_2D = reshape_data_smartloc2(array_gains_neg) 
    
    fig2 = plot_2D_plt(array_gains_pos_2D, title = f'{file_to_save_gains} positive gains [eps/mV]', annot = True, fmt =  '.1f', annot_size = 10,
                       minval = 1, maxval = 4)
    fig3 = plot_2D_plt(array_gains_neg_2D, title = f'{file_to_save_gains} negative gains [eps/mV]', annot = True, fmt =  '.1f', annot_size = 10,
                       minval = 1, maxval = 4)

    fig1.savefig(f'{file_to_save_gains}.png', dpi = 250)
    fig1.savefig(f'{file_to_save_gains}.svg')     
    plt.close(fig1)
    
    fig2.savefig(f'{file_to_save_gains}_map_pos.png', dpi = 250)
    fig2.savefig(f'{file_to_save_gains}_map_pos.svg')   
    plt.close(fig2)
    
    fig3.savefig(f'{file_to_save_gains}_map_neg.png', dpi = 250)
    fig3.savefig(f'{file_to_save_gains}_map_neg.svg')     
    plt.close(fig3)   
    
    plt.ion()
    

    ## - SAVE RESULTS -
    
    df_gains['Gains_positive'] = array_gains_pos
    df_gains['Gains_negative'] = array_gains_neg
    
    # check if the file exists
    if(os.path.isfile(file_to_save_gains+'.dataframe')):
        file_to_save_gains = file_to_save_gains + '_' + str(int(time.time()))    
        print(f'File exists! Saving as {file_to_save_gains}')
    
    df_gains.to_pickle(file_to_save_gains+extension)    
    print(f'{file_to_save_gains+extension} stored in disk')


def gain_regression_calc(filename, amplitudes_in, period, t_start, pixels_to_plot = [0], debug_level = 0, label = '-'):
        
    # Source - https://stackoverflow.com/a
    # Posted by Egal, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-11-24, License - CC BY-SA 4.0
    
    def reject_outliers_2(data, m=2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.)
        return data[s < m]
    
       
    array_size = 256
    
    df_read = pd.read_pickle(filename)
    
    timestamps = df_read.Timestamps[0]

    pulse_width = period/2
    
    timestamps_rising = [t_start + n*period for n in range(len(amplitudes_in))]
    print(f'Rising timestamps: {timestamps_rising}')
    
    eps_out = np.zeros((array_size,len(amplitudes_in)))
    array_gains = np.zeros(array_size)
    array_offsets = np.zeros(array_size)
        
    for pixel_index in range(array_size):
        
        eps = df_read.Eps[pixel_index]
        
        for n,t_rising in enumerate(timestamps_rising):
        
            # take the values between 1/4 and 3/4 of the pulse HIGH duration
            eps_pulse = eps[(timestamps > (t_rising + pulse_width/4)) & (timestamps < (t_rising + 3*pulse_width/4))]
            eps_pulse_filt = reject_outliers_2(eps_pulse)
            
            # take the values between 1/4 and 3/4 of the pulse LOW duration
            eps_base = eps[(timestamps > (t_rising + pulse_width/4 + pulse_width)) & (timestamps < (t_rising + 3*pulse_width/4 + pulse_width))]
            eps_base_filt = reject_outliers_2(eps_base)
            
            eps_out[pixel_index, n] = np.average(eps_pulse_filt) - np.average(eps_base_filt) 
            
        # linear regression
        res = stats.linregress(amplitudes_in, eps_out[pixel_index])
        gain = round(res.slope,4)
        offset = round(res.intercept,4)
        rsquare = round(res.rvalue**2,6)
        
        array_gains[pixel_index] = gain
        array_offsets[pixel_index] = offset
                
    return array_gains, eps_out, array_offsets  

def custom_pulsed_waveform(timestamps, amplitudes, period, initial_delay = 0, DC_value = 0):
    
    ''' Create a custom pulsed waveform with pulses of a certain period and different amplitudes '''
    
    waveform = np.zeros(len(timestamps))
    
    for index,amp in enumerate(amplitudes):
        
        t_rising = index*period + initial_delay
        t_falling = index*period + period/2 + initial_delay
        
        waveform[(timestamps >= t_rising) & (timestamps < t_falling)] = amp
        
    return waveform + DC_value

def perform_measurement(ser, df_in, measurement_configuration, plot_configuration, app = None, cancel_event = None):
    
    # finish_measurement = 0
    
    # ''' This function will be called when pressing ctrl+C '''
    # def signal_handler(signal, frame):
    #     nonlocal finish_measurement
    #     finish_measurement = 1
    
    # signal.signal(signal.SIGINT, signal_handler)    # Assign signal_handler to a keyboard interrupt (SIGINT)    
    
    
    df_out = df_in
    
    # Retrieve measurement configuration
    measurement_time_s  =   measurement_configuration['measurement_time_s']
    sampling_period_ms  =   measurement_configuration['sampling_period_ms']
    f_sampling  =           measurement_configuration['f_sampling']    
    AZERO_sample =          measurement_configuration['AZERO_sample']
    disFB_sample =          measurement_configuration['disFB_sample']
    apply_gain_calib =      measurement_configuration['apply_gain_calib']
    gain_calib_name =       measurement_configuration['gain_calib_name']
    
    # Retrieve plot configuration 
    enable_plot             = plot_configuration['enable_plot']
    pixels_to_plot          = plot_configuration['pixels_to_plot']
    plot_refresh_time_ms    = plot_configuration['plot_refresh_time_ms']   
    plot_type               = plot_configuration['plot_type']
    line_ylim               = plot_configuration['line_ylim']
    map2D_maxval            = plot_configuration['map2D_maxval']
    map2D_minval            = plot_configuration['map2D_minval']    
    
    
    # Retrieve gain galibration
    if(apply_gain_calib):
        if(os.path.isfile(gain_calib_name)):
            log_message(f'Applying gain calibration from: "{gain_calib_name}"')
            df_gains = pd.read_pickle(gain_calib_name)
            gain_positive = df_gains.Gains_positive.to_numpy()
            gain_negative = df_gains.Gains_negative.to_numpy()
        else:
            log_message(f'File "{gain_calib_name}" does not exist!!')
            measurement_configuration['apply_gain_calib'] = 0
            apply_gain_calib = 0
    

    plot_Nlines = len(pixels_to_plot)+1
    
    # Start plot(s)
    app.after(0, lambda: start_plot(measurement_configuration,plot_configuration))


    array_size = 256
    number_of_samples = int(measurement_time_s / (sampling_period_ms*1e-3))

    array_events        = np.zeros((number_of_samples, array_size))
    array_eps           = np.zeros((number_of_samples, array_size))
    array_milivolts     = np.zeros((number_of_samples, array_size))    
    array_OF            = np.zeros((number_of_samples))
    
    # estimated timestamps
    timestamps            = np.arange(0, measurement_time_s, sampling_period_ms*1e-3)    
    stop_time_s           = measurement_time_s    
    
    
    outlier_threshold_events = 512    
    
    
    # acquisition configuration
    s_L = sampling_period_ms & 0xFF
    s_H = (sampling_period_ms & 0xFF00) >> 8

    do_acq_config(ser)
    
    ser.write(bytes([s_L]))    # sampling_period_ms_L
    ser.write(bytes([s_H]))    # sampling_period_ms_H        
    
    nsample = 0
    
    print('=== START ACQUISITION ===')
    do_acq_loop(ser)
    
    while(True):
    # for nsample in range(number_of_samples):
               
        data_overflow = read_overflow(ser)
        events_8b, array_events[nsample] = read_mem(ser, size = array_size, acq = True, debug = False)
        
        if(nsample == AZERO_sample):
            ser.write(bytes([ord('A')]))    # do AZERO
            print("do AZERO")
        elif(nsample == disFB_sample):
            ser.write(bytes([ord('F')]))    # disable FB
            print("disable FB")
        else:
            ser.write(bytes([ord('C')]))    # continue acquisition
        
            
        # Remove outliers (have to find out why they are produced)
        # array_events[nsample][array_events[nsample] < -outlier_threshold_events] = array_events[nsample-1][array_events[nsample] < -outlier_threshold_events]
        # array_events[nsample][array_events[nsample] > outlier_threshold_events] = array_events[nsample-1][array_events[nsample] > outlier_threshold_events]
            
        
        array_eps[nsample] = array_events[nsample]*f_sampling
        
        if(apply_gain_calib):
                    array_milivolts[nsample] = np.array([linear_function_inverse(data,gain_pos,gain_neg) for data,gain_pos,gain_neg in zip(array_eps[nsample],gain_positive,gain_negative)])        
                
        
        if(enable_plot and ((timestamps[nsample]*1e3 % plot_refresh_time_ms) == 0)):
            
            if(apply_gain_calib):
                data_to_plot = array_milivolts[nsample]            
            else:
                data_to_plot = array_eps[nsample] 
                
            if(plot_type in ['line','both']): 
                timestamps_to_plot = np.repeat(timestamps[nsample], plot_Nlines)   
                root.after(0, lambda: update_plot(timestamps_to_plot, np.append(data_to_plot[pixels_to_plot],array_OF[nsample])) )

            if(plot_type in ['2D map','both']):
                data_to_plot_2D = reshape_data_smartloc2(data_to_plot)
                root.after(0, lambda: update_plot_2D(data_to_plot_2D, f'SmartLoC2 (t = {str(timestamps[nsample])} s)'))
        

        # if (finish_measurement == 1):
        #     stop_time_s = timestamps[nsample]
        #     ser.write(bytes([ord('S')]))    # stop acquisition
        #     time.sleep(1e-3)          
        #     break         
        
        if (cancel_event != None):
            if cancel_event.is_set():
                log_message("Measurement canceled by user")
                break

        if(nsample == number_of_samples-1):
            break
            
        nsample +=1
            
        
            
    # acquisition finished
    ser.write(bytes([ord('S')]))    # stop acquisition

    log_message('Measurement finished')
    

    measurement_configuration['stop_time_s'] = round(stop_time_s,3)

    if(np.sum(array_OF)):
        log_message('OJO! Overflow detected!')        

    # store the data to the dataframe
    # memsize_read dependent?
    
    df_out.at[0, 'Timestamps'] = timestamps
    df_out.at[0, 'Measurement_configuration'] = measurement_configuration
    
    for pix_index in range(array_size):
        events  = np.transpose(array_events)[pix_index]
        eps     = np.transpose(array_eps)[pix_index]
        mV      = np.transpose(array_milivolts)[pix_index]
        
        df_out.at[pix_index, 'Events']  = events.astype(int)
        df_out.at[pix_index, 'Eps']     = eps.astype(int)        
        if(apply_gain_calib == True):
            df_out.at[pix_index, 'Milivolts'] = mV.astype(float)
        
    print('Data stored to the dataframe')            
    
    return df_out


#%% PLOT FUNCTIONS


def StartDynamicPlot(xmax, Nlines = 1, xlabel = '', ylabel = '', title = '', labels = ''):
    
    # plt.ion()
    
    ##################### NEW ################

    win_line = Toplevel(root)
    win_line.title("Dynamic Line Plot")
    win_line.geometry("740x560")    

    # --- container frame for toolbar + canvas ---
    container = Frame(win_line)
    container.pack(fill="both", expand=True)

    # Matplotlib figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))    


    # Canvas (store globally)
    canvas_line = FigureCanvasTkAgg(fig, master=container)
    canvas_widget = canvas_line.get_tk_widget()

    # Toolbar in the same container
    toolbar = NavigationToolbar2Tk(canvas_line, container)
    toolbar.pack(side="top", fill="x")
    toolbar.update()

    # Canvas under toolbar
    canvas_widget.pack(side="top", fill="both", expand=True)
    canvas_line.draw()
    ##################### --- ################

    # fig, ax = plt.subplots()
    # fig.set_size_inches(10, 8)
    
    lines = list()
    
    for n in range(Nlines):
        lines.append(ax.plot([],[])[0])
    
    ax.set_autoscaley_on(True)
    ax.set_xlim(0, xmax)
    if (xmax >= 14400):
        ax.set_xticks(np.arange(0,xmax+3600,3600))     
    elif (xmax >= 7200):
        ax.set_xticks(np.arange(0,xmax+600,600))    
    elif (xmax >= 1200):
        ax.set_xticks(np.arange(0,xmax+300,300))
    elif (xmax >= 600):
        ax.set_xticks(np.arange(0,xmax+60,60))        
    elif (xmax >= 120):
        ax.set_xticks(np.arange(0,xmax+30,30))
    elif (xmax >= 60):
        ax.set_xticks(np.arange(0,xmax+5,5))        
    elif (xmax != 0):
        ax.set_xticks(np.arange(0,xmax+2,2))
    else:
        ax.set_autoscalex_on(True)
    ax.grid()
    
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.title.set_text(title)
    
    if(len(lines) == len(labels)):
        for line, label in zip(lines,labels):
            line.set_label(label)        
        ax.legend(ncols = 2)
    
    return fig, ax, lines, canvas_line

def start_dynamic_plot_2D_plt(val_max, val_min, annotate = False, xlabel = '', ylabel = '', title = '', zlabel = ''):
    
    # plt.ion()
    
    ##################### NEW ################
    
    win_2d = Toplevel(root)
    win_2d.title("Dynamic 2D Map")
    win_2d.geometry("740x560")

    container = Frame(win_2d)
    container.pack(fill="both", expand=True)    
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)    
    
    
    ##################### --- ################
    
    
    # fig, ax = plt.subplots()
    # fig.set_size_inches(10,8)
    
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.title.set_text(title)
    
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    
    data2d = np.zeros((16,16))
    data2d[0][0] = val_max
    data2d[0][1] = val_min
    
    # image = ax.imshow(data2d, cmap = 'coolwarm')
    image = ax.imshow(data2d, cmap = 'RdBu')
    
    if(annotate == True):
        for y in range(16):
            for x in range(16):
                plt.text(x, y, f'{int(16*x+y)}',
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize = 8,
                     )  
            
    fig.colorbar(image, label = zlabel)
    
    
    ##################### NEW ################
    
    # Canvas (store globally)
    canvas_2d = FigureCanvasTkAgg(fig, master=container)
    canvas_widget = canvas_2d.get_tk_widget()

    toolbar = NavigationToolbar2Tk(canvas_2d, container)
    toolbar.pack(side="top", fill="x")
    toolbar.update()

    canvas_widget.pack(side="top", fill="both", expand=True)
    canvas_2d.draw()    
    
    ##################### --- ################    
      
        
    return fig, image, ax, canvas_2d

def UpdateDynamicPlot(fig, ax, lines, xdata, ydata, canvas = None):
    
    if (len(lines) == len(xdata) == len(ydata)):
    
        for line,x,y in zip(lines,xdata,ydata):     
            line.set_xdata(np.append(line.get_xdata(), x))
            line.set_ydata(np.append(line.get_ydata(), y))
    
        ax.relim()
        ax.autoscale_view()
        
        canvas.draw_idle()
        
    else:
        print('Cannot plot. Array sizes do not match.')            

    return

def update_dynamic_plot_2D_plt(fig, image, ax, data2d, title, canvas_2d):
    
    # plt.ion()
    
    ax.title.set_text(title)
    image.set_data(data2d)
    canvas_2d.draw_idle()
  
def plot_2D_plt(data_2d, title = '', minval = None, maxval = None, annot = False, fmt =  None, annot_size = 10, barlabel = None):
    
    plt.ion()
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)
    
    ax.title.set_text(title)
    
    data2d = np.ones((16,16))
    data2d[0][0] = maxval
    data2d[0][1] = minval
    
    image = ax.imshow(data2d, cmap = 'coolwarm')
    # image = ax.imshow(data2d, cmap = 'RdBu')    

    image.set_data(data_2d)
    
    if(annot == True):
        for y in range(16):
            for x in range(16):
                plt.text(x, y, f'{data_2d[y,x]:{fmt}}',
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize = annot_size,
                     )  
            
    plt.colorbar(image, label = barlabel)    
    
    # fig.canvas.draw()
    # fig.canvas.flush_events()      
    
    return fig 


#%% - GUI -        
        

# ---------------- App ----------------
root = tk.Tk()
root.title(APP_NAME)
root.geometry("800x800")




# Forward declarations of state vars (defined later)
running = False
cancel_event = threading.Event()
active_plot = True

fig = None
ax = None
lines = None
canvas_lines = None

fig_2D = None
image_2D = None
ax_2D = None
canvas_2D = None

def log_message(msg: str):
    if not msg.endswith("\n"):
        msg = msg + "\n"
    ts = datetime.now().strftime("%H:%M:%S")
    log_queue.put(f"[{ts}] {msg}")


def on_exit():
    """Exit app; if a measurement is running, cancel first."""
    if running and not cancel_event.is_set():
        log_message("Exit requested: canceling measurement…")
        cancel_event.set()
        # Give the worker a moment to wrap up, then destroy
        root.after(150, root.destroy)
    else:
        root.destroy()


def show_about(parent):
    
    # Gather version info
    py_ver_str = platform.python_version()      # e.g., '3.12.5'
    tk_ver = parent.tk.call('info', 'patchlevel')  # e.g., '8.6.14'    
    
    # Modal Toplevel with selectable info and copy button
    win = tk.Toplevel(parent)
    win.title(f"About {APP_NAME}")
    win.transient(parent)     # stay on top of parent
    win.grab_set()            # modal
    win.resizable(False, False)

    # Center on parent
    win.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() // 2) - 200
    y = parent.winfo_rooty() + (parent.winfo_height() // 2) - 120
    win.geometry(f"400x160+{x}+{y}")

    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    title_lbl = ttk.Label(frm, text=APP_NAME, font=(default_font[0], 12, "bold"))
    title_lbl.pack(anchor="w")

    ver_lbl = ttk.Label(frm, text=f"Version: {APP_VERSION}")
    ver_lbl.pack(anchor="w", pady=(2, 0))

    dev_lbl = ttk.Label(frm, text=f"Developer: {APP_AUTHOR}")
    dev_lbl.pack(anchor="w")
    
    # Versions section
    versions_frame = ttk.Frame(frm)
    versions_frame.pack(fill="x", pady=(4, 8))
    ttk.Label(versions_frame, text=f"Python: {py_ver_str}").grid(row=0, column=0, sticky="w")
    ttk.Label(versions_frame, text=f"Tk/Tkinter: {tk_ver}").grid(row=1, column=0, sticky="w")

    # Close with Esc / Enter
    win.bind("<Escape>", lambda e: win.destroy())
    win.bind("<Return>", lambda e: win.destroy())
    
    
    
# ----- Grid weights on root -----
root.columnconfigure(0, weight=1)
# Row 0: top button strip (fixed height)
root.rowconfigure(0, weight=0)
# Row 1: plot (expands)
root.rowconfigure(1, weight=0)
# Row 1: plot (expands)
root.rowconfigure(2, weight=0)
# Row 2: log (expands some, you can set to 0 if you want it fixed)
root.rowconfigure(3, weight=1)
    

# ---------------- Menu bar ----------------
menubar = tk.Menu(root)

file_menu = tk.Menu(menubar, tearoff=False)

file_menu.add_command(label="Exit", command=on_exit)
menubar.add_cascade(label="File", menu=file_menu)
root.protocol("WM_DELETE_WINDOW", on_exit)  # Make window X behave like File→Exit


help_menu = tk.Menu(menubar, tearoff=False)
help_menu.add_command(label="About", command=lambda: show_about(root))
menubar.add_cascade(label="Help", menu=help_menu)

root.config(menu=menubar)



# -------------------- Styling (pure ttk) --------------------
style = ttk.Style()
# Robust cross-platform option:
style.theme_use('clam')

# Base font by OS
default_font = ("Segoe UI", 10) if os.name == "nt" else ("DejaVu Sans", 10)
style.configure(".", font=default_font)

style.configure("TLabel", padding=(2, 2))
style.configure("TEntry", padding=(4, 4))
style.configure("TButton", padding=(6, 4))
style.configure("TRadiobutton", padding=(2, 2))
style.configure("TCheckbutton", padding=(2, 2))
style.configure("TLabelframe", padding=8)
style.configure("TLabelframe.Label", padding=(4, 0))


# -------------------- Top panels container --------------------
row_config = 0

top = ttk.Frame(root, padding=10)
top.grid(row=row_config, column=0, sticky="ew")
top.columnconfigure(0, weight=1)
top.columnconfigure(1, weight=1)


# -------------------- Left: Measurement configuration ------------------
lf_meas = ttk.LabelFrame(top, text="MEASUREMENT CONFIGURATION", padding=10)
lf_meas.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
# lf_meas.columnconfigure(0, weight=1)
lf_meas.columnconfigure(1, weight=1)

meas_param1 = tk.StringVar(value="4")
meas_param2 = tk.StringVar(value="60")
meas_param3 = tk.StringVar(value="500")

ttk.Label(lf_meas, text="Measurement time [s]:").grid(row=0, column=0, sticky="w", pady=3)
ttk.Entry(lf_meas, textvariable=meas_param1).grid(row=0, column=1, sticky="ew", pady=3)

ttk.Label(lf_meas, text="Disable high-pass filter at [s]:").grid(row=1, column=0, sticky="w", pady=3)
ttk.Entry(lf_meas, textvariable=meas_param2).grid(row=1, column=1, sticky="ew", pady=3)

ttk.Label(lf_meas, text="Sampling period [ms]:").grid(row=2, column=0, sticky="w", pady=3)
ttk.Entry(lf_meas, textvariable=meas_param3).grid(row=2, column=1, sticky="ew", pady=3)

ttk.Label(lf_meas, text="File to store the data:").grid(row=3, column=0, sticky="w", padx=(0, 8))

default_name = "filename"
filename_var = tk.StringVar(value=default_name)
filename_entry = ttk.Entry(lf_meas, textvariable=filename_var)
filename_entry.grid(row=3, column=1, sticky="ew")

do_gain_calib = tk.IntVar()
ttk.Checkbutton(lf_meas, text="Run gain calibration", variable=do_gain_calib).grid(row=4, column=0, columnspan=2, sticky="w", pady=3)


apply_gain_calib = tk.IntVar()
ttk.Checkbutton(lf_meas, text="Apply gain calibration", variable=apply_gain_calib).grid(row=5, column=0, columnspan=2, sticky="w", pady=3)


ttk.Label(lf_meas, text="Gain calibration file:").grid(row=6, column=0, sticky="w", padx=(0, 8))


# a single row with an inner frame
row_frame = ttk.Frame(lf_meas)
row_frame.grid(row=7, column=0, columnspan = 2, sticky="ew")
row_frame.columnconfigure(0, weight=1)  # entry expands
row_frame.columnconfigure(1, weight=0)  # button fixed

default_name = "gain_calib_file"
filename_gain_var = tk.StringVar(value=default_name)
filename_entry = ttk.Entry(row_frame, textvariable=filename_gain_var).grid(row=7, column=0, sticky="ew")

def browse_file():
    path = filedialog.askopenfilename(
        title="Select gain calibration file",
        defaultextension=".dataframe",
        filetypes=[("Dataframe (pickle) files", "*.dataframe")]
    )
    if path:
        filename_gain_var.set(path)

ttk.Button(row_frame, text="...", command=browse_file).grid(row=7, column=1, sticky='e')


# -------------------- Right: Plot configuration ------------------------
lf_plot = ttk.LabelFrame(top, text="PLOT CONFIGURATION", padding=10)
lf_plot.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
lf_plot.columnconfigure(1, weight=1)

plot_type = tk.StringVar(value="line")
ttk.Label(lf_plot, text="Plot type:").grid(row=0, column=0, sticky="w", pady=(0, 3))

nrow = 0
rb_frame = ttk.Frame(lf_plot)
rb_frame.grid(row=nrow, column=1, sticky="w", pady=(0, 3))
ttk.Radiobutton(rb_frame, text="line",     variable=plot_type, value="line").pack(side="left", padx=(0, 10))
ttk.Radiobutton(rb_frame, text="2D map", variable=plot_type, value="2D map").pack(side="left", padx=(0, 10))
ttk.Radiobutton(rb_frame, text="both",       variable=plot_type, value="both").pack(side="left")

plot_param1 = tk.StringVar(value="1,2,3,4,251,252,253,254")
plot_param2 = tk.StringVar(value="500")
plot_param3 = tk.StringVar(value="250")
plot_param4 = tk.StringVar(value="250")

nrow += 1
ttk.Label(lf_plot, text="Pixels to plot:").grid(row=nrow, column=0, sticky="w", pady=3)
ttk.Entry(lf_plot, textvariable=plot_param1).grid(row=nrow, column=1, sticky="ew", pady=3)

nrow += 1
ttk.Label(lf_plot, text="Plot update period [ms]:").grid(row=nrow, column=0, sticky="w", pady=3)
ttk.Entry(lf_plot, textvariable=plot_param2).grid(row=nrow, column=1, sticky="ew", pady=3)

nrow += 1
ttk.Label(lf_plot, text="Y axis max (line):").grid(row=nrow, column=0, sticky="w", pady=3)
ttk.Entry(lf_plot, textvariable=plot_param3).grid(row=nrow, column=1, sticky="ew", pady=3)

nrow += 1
ttk.Label(lf_plot, text="Y axis max (2D map):").grid(row=nrow, column=0, sticky="w", pady=3)
ttk.Entry(lf_plot, textvariable=plot_param4).grid(row=nrow, column=1, sticky="ew", pady=3)

nrow += 1
plot_invyaxis_opt = tk.BooleanVar(value=False)
ttk.Checkbutton(lf_plot, text="Invert y axis", variable=plot_invyaxis_opt).grid(row=nrow, column=0, columnspan=2, sticky="w", pady=3)


# ---------------- Plot area ----------------
# row_plot = 2

# plot_frame = ttk.Frame(root, padding=(10, 0, 10, 0))
# plot_frame.grid(row=row_plot, column=0, sticky="nsew")
# plot_frame.columnconfigure(0, weight=1)
# plot_frame.rowconfigure(0, weight=1)

# fig,ax,lines = StartDynamicPlot(0,1,'x label', 'y label', 'title')


# # fig = Figure(figsize=(6, 4), dpi=100)
# # ax = fig.add_subplot(111)
# # ax.set_title("Random Data Measurement")
# # ax.set_xlabel("Sample")
# # ax.set_ylabel("Value")
# # line, = ax.plot([], [], 'b-')

# canvas = FigureCanvasTkAgg(fig, master=plot_frame)
# canvas_widget = canvas.get_tk_widget()
# canvas_widget.grid(row=0, column=0, sticky="nsew")
# canvas.draw()

# ---------------- Serial port  ----------------
row_serial_port = 1

serial_frame = ttk.Frame(root, padding=(10,6))
serial_frame.grid(row=row_serial_port, column=0, sticky="ew")
serial_frame.columnconfigure(0, weight=0)
serial_frame.columnconfigure(1, weight=0)
serial_frame.columnconfigure(2, weight=0)

list_btn = tk.Button(serial_frame, text="list PORTS", fg="black", bg="#C2BAB8", activebackground="#ABA4A2", height = 2)
list_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

input_serial_port = tk.StringVar(value="COMx")

ttk.Label(serial_frame, text="Serial port:").grid(row = 0, column=1, sticky="ew")
ttk.Entry(serial_frame, textvariable=input_serial_port).grid(row=0, column=2, sticky="ew")

# ---------------- Buttons ----------------
row_buttons = 2

btn_frame = ttk.Frame(root, padding=(10,6))
btn_frame.grid(row=row_buttons, column=0, sticky="ew")
btn_frame.columnconfigure(0, weight=1)
btn_frame.columnconfigure(1, weight=1)
btn_frame.columnconfigure(2, weight=1)
btn_frame.columnconfigure(3, weight=1)

col_buttons = 0

run_btn = tk.Button(btn_frame, text="RUN", fg="white", bg="#28a745", activebackground="#218838", height = 2)
run_btn.grid(row=0, column=col_buttons, sticky="ew", padx=(0, 8))
col_buttons+= 1

autozero_btn = tk.Button(btn_frame, text="AUTOZERO", fg="black", bg="#ffc107", activebackground="#e0a800", height = 2)
autozero_btn.grid(row=0, column=col_buttons, sticky="ew", padx=(0, 8))
col_buttons+= 1

finish_btn = tk.Button(btn_frame, text="FINISH", fg="white", bg="#dc3545", activebackground="#c82333", height = 2)
finish_btn.grid(row=0, column=col_buttons, sticky="ew", padx=(0, 8))
col_buttons+= 1

plot_btn = tk.Button(btn_frame, text="stop PLOT", fg="black", bg="#C2BAB8", activebackground="#ABA4A2", height = 2)
plot_btn.grid(row=0, column=col_buttons, sticky="ew", padx=(0, 8))
col_buttons+= 1



# ---------------- Log area ----------------
row_log = 3

log_frame = ttk.LabelFrame(root, text="LOG", padding=(10, 6))
log_frame.grid(row=row_log, column=0, sticky="nsew", padx=10, pady=10)
log_frame.columnconfigure(0, weight=1)
log_frame.rowconfigure(0, weight=1)

log_text = tk.Text(log_frame, height=6, wrap="word", state="disabled")
log_text.grid(row=0, column=0, sticky="nsew")

log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
log_scroll.grid(row=0, column=1, sticky="ns")
log_text.configure(yscrollcommand=log_scroll.set)

# ---------------- Thread-safe logging ----------------
log_queue = queue.Queue()

def flush_log_queue():
    try:
        while True:
            chunk = log_queue.get_nowait()
            log_text.configure(state="normal")
            log_text.insert("end", chunk)
            log_text.see("end")
            log_text.configure(state="disabled")
    except queue.Empty:
        pass
    finally:
        root.after(75, flush_log_queue)

flush_log_queue()

# ---------------- Measurement logic (background thread) ----------------
data_x = []
data_y = []
running = False  # reassign here explicitly to be in module scope
cancel_event = threading.Event()

    

def measurement_worker(ser, measurement_configuration,plot_configuration,file_to_save):
    """Runs in a background thread; never touch Tk widgets here."""
    try:
        log_message("===================")
        log_message("Measurement started")
        
        df_in = create_dataframe()        
        
        if(measurement_configuration['do_gain_calib']):
            log_message('Gain calibration measurement')
                        
            ## GAIN CALIBRATION
            amplitudes = np.array([20e-3, 40e-3, 60e-3, 80e-3, -20e-3, -40e-3, -60e-3, -80e-3])
            # amplitudes = np.array([20e-3, 40e-3, 60e-3, -20e-3, -40e-3, -60e-3])
            # amplitudes = np.array([20e-3, 40e-3, -20e-3, -40e-3])
            pulse_period = 10
            sampling_period = 0.5
            initial_delay = measurement_configuration['disFB_time_s'] + 5
            measurement_time_s = initial_delay + pulse_period*len(amplitudes)
            measurement_configuration['measurement_time_s'] = measurement_time_s
            ###################            
            
            dwf = DwfLibrary()       
            
            # Enumerate all Digilent Waveforms devices and return the count.
            device_count = dwf.deviceEnum.enumerateDevices()
            log_message(f"Number of Digilent Waveforms devices found: {device_count}")    
            for n in range(device_count):
                log_message(f' - {dwf.deviceEnum.deviceName(n)}')
                  
            device = openDwfDevice(dwf, enum_filter= DwfDeviceID(3), score_func=maximize_analog_out_buffer_size)
            
            log_message("Connected to Analog Discovery 2")    
            
            t_final = pulse_period*len(amplitudes) + initial_delay
            timestamps = np.arange(start = 0, stop = t_final, step = sampling_period)
     
            # create the waveform
            waveform = custom_pulsed_waveform(timestamps, amplitudes, pulse_period, initial_delay = initial_delay)            
     
            # configure the AD2 and output the waveform
            analog_out_custom_waveform(device.analogOut, waveform, t_final)
                        
        
        df_out = perform_measurement(ser, df_in, measurement_configuration, plot_configuration, root, cancel_event) 
        
    except Exception as e:
        log_message(f"ERROR: {e}")
        
    finally:
        
        
        
        extension = '.dataframe'
        
        if(os.path.isfile(file_to_save+extension)):
            file_to_save = file_to_save + '_' + str(int(time.time()))    
            log_message(f'File exists! Saving as "{file_to_save}" ...')    
        
        df_out.to_pickle(file_to_save+extension) 
        log_message(f'"{file_to_save+extension}" stored in disk')
        
        # CLOSE ANALOG DISCOVERY
        if(measurement_configuration['do_gain_calib']):
            device.close()
            print('Analog discovery closed')  
        
        # CLOSE SERIAL PORT
        ser.close()
        print('Serial port closed')        
        
        
        if(measurement_configuration['do_gain_calib']):
            root.after(0, lambda: gain_analysis(file_to_save, amplitudes, initial_delay, pulse_period, plot_configuration))
        
        
        # Reset UI from main thread
        def _cleanup():
            global running
            running = False
            run_btn.config(state="normal")
            finish_btn.config(state="disabled")
        root.after(0, _cleanup)


## - Button handlers -

def on_list_ports():
    
    ports = list(serial.tools.list_ports.comports())
    log_message("===================")
    [log_message(f'{p.name} : {p.product}') for p in ports]  


def on_run():
    global running, data_x, data_y
    if running:
        log_message("RUN ignored: already running.")
        return
    running = True
    
    cancel_event.clear()
    run_btn.config(state="disabled")
    finish_btn.config(state="normal")  # enable cancel    
    
    cfg = {
        "meas": {
            "param1": meas_param1.get(),
            "param2": meas_param2.get(),
            "param3": meas_param3.get(),
        },
        "plot": {
            "type": plot_type.get(),
            "invert_yaxis": plot_invyaxis_opt.get(),
            "param1": plot_param1.get(),
            "param2": plot_param2.get(),
            "param3": plot_param3.get(),
            "param4": plot_param4.get(),
        },
        "file": filename_var.get(),
    }
    
    print(cfg['meas'])    
    print(cfg['plot'])
    
    measurement_time_s = int(cfg['meas']['param1'])    
    disFB_time_s = int(cfg['meas']['param2'])
    sampling_period_ms = int(cfg['meas']['param3'])
    AZERO_time_s = 5
    
    f_sampling = 1/(sampling_period_ms*1e-3)
    AZERO_sample = int(AZERO_time_s*f_sampling)
    disFB_sample = int(disFB_time_s*f_sampling)

    global do_gain_calib
    
    measurement_configuration = dict(
        sampling_period_ms      = sampling_period_ms,
        measurement_time_s      = measurement_time_s,
        f_sampling              = f_sampling,
        AZERO_time_s            = AZERO_time_s,
        AZERO_sample            = AZERO_sample,
        disFB_time_s            = disFB_time_s,
        disFB_sample            = disFB_sample,
        do_gain_calib           = do_gain_calib.get(),
        apply_gain_calib        = apply_gain_calib.get(),
        gain_calib_name         = filename_gain_var.get()
        )  


    plot_configuration = dict(
        pixels_to_plot          = np.fromstring(cfg['plot']['param1'], dtype = int, sep = ','),
        plot_refresh_time_ms    = int(cfg['plot']['param2']),
        plot_type               = cfg['plot']['type'],
        line_ylim               = int(cfg['plot']['param3']),
        map2D_maxval            = int(cfg['plot']['param4']),
        map2D_minval            = -1*int(cfg['plot']['param4']),
        enable_plot             = True
        )    
    
    
    filepath = './measurements/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)    
    
    filename = cfg['file']
    file_to_save = filepath + filename
    
        
    # OPEN SERIAL PORT
    
    # ports = list(serial.tools.list_ports.comports())
    # [print(f'{p.name} : {p.product}') for p in ports]        
    
    global input_serial_port

    serial_port = input_serial_port.get()
    
    # baudrate = 115200
    baudrate = 230400
    
    if(serial_port[:3] == 'COM'):
        ser = serial_connect(port = serial_port, baudrate = baudrate)  
    else:
        ser = serial_connect(port = '/dev/'+serial_port, baudrate = baudrate)
    
    do_chip_reset(ser)
    
    do_array_config(ser)
    
    t = threading.Thread(target=measurement_worker, args=(ser,measurement_configuration,plot_configuration,file_to_save), daemon=True)
    t.start()


def on_autozero():
    log_message("empty function...")


def on_finish_cancel():
    """FINISH acts as Cancel: signal worker to stop."""
    if running and not cancel_event.is_set():
        # log_message("Cancel requested by user. Stopping measurement…")
        cancel_event.set()
    else:
        log_message("No measurement to stop.")


def on_plot():

    global active_plot
    if(active_plot):
        active_plot = False
        log_message("set plot non interactive")
        plot_btn.config(text = 'edit PLOT')
    else:
        active_plot = True
        log_message("set plot interactive")
        plot_btn.config(text = 'stop PLOT')


def start_plot(measurement_configuration,plot_configuration):
    
    print('STARTING PLOT IN MAIN THREAD')
    
    # Retrieve measurement configuration
    measurement_time_s  =   measurement_configuration['measurement_time_s']    
    apply_gain_calib =      measurement_configuration['apply_gain_calib']
    
    # Retrieve plot configuration 
    pixels_to_plot          = plot_configuration['pixels_to_plot']
    plot_type               = plot_configuration['plot_type']
    line_ylim               = plot_configuration['line_ylim']
    map2D_maxval            = plot_configuration['map2D_maxval']
    map2D_minval            = plot_configuration['map2D_minval']    
    
    global fig,ax,lines,canvas_lines,fig_2D, image_2D, ax_2D, canvas_2D
    
    # Prepare dynamic plot   
    if(apply_gain_calib):
        ylabel = 'mV'     
    else:
        ylabel = 'Eps'  
    
    
    if(plot_type in ['line','both']):         
        plot_Nlines = len(pixels_to_plot)+1
        fig, ax, lines, canvas_lines = StartDynamicPlot(measurement_time_s, Nlines = plot_Nlines, xlabel = 'time [s]', ylabel = ylabel,
                                                  title = 'SmartLoC2', labels = [f'pix{n}' for n in pixels_to_plot] + ['OF']) 
        if(line_ylim != None):
            ax.set_ylim(-line_ylim, line_ylim)

    if(plot_type in ['2D map','both']):
        fig_2D, image_2D, ax_2D, canvas_2D = start_dynamic_plot_2D_plt(val_max = map2D_maxval, val_min = map2D_minval, title = 'SmartLoC2', 
                                                            zlabel = ylabel)


def update_plot(xdata, ydata):
    
    global fig,ax,lines,canvas_lines
    UpdateDynamicPlot(fig, ax, lines, xdata, ydata, canvas_lines)


def update_plot_2D(data2d, title):
    
    global fig_2D, image_2D, ax_2D,canvas_2D
    update_dynamic_plot_2D_plt(fig_2D, image_2D, ax_2D, data2d, title, canvas_2D)    


# def keep_matplotlib_alive():
    
#     global active_plot, fig, fig_2D
#     if(active_plot):
#         plt.pause(0.001)  # process Matplotlib events to keep plot always responsive 

#     root.after(10, keep_matplotlib_alive)  # schedule again after 10 ms


list_btn.config(command=on_list_ports)
run_btn.config(command=on_run)
autozero_btn.config(command=on_autozero)
finish_btn.config(command=on_finish_cancel, state="disabled")
plot_btn.config(command=on_plot)

# ---------------- Mainloop ----------------
if __name__ == "__main__":
    
    # Call this once before mainloop starts:
    # keep_matplotlib_alive()    
    
    log_message('Welcome to SmartLoC2 GUI :)')
    
    root.mainloop()