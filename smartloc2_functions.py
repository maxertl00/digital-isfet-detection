#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:25:37 2025

@author: javier
"""

import os
import time
import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import signal

# import smartloc2_classes as SLOC2


#%% PLOT FUNCTIONS





#%% FUNCTIONS

def mode_fill(ser):
    ser.write(bytes([ord('F')]))   # fill mode    
    
def mode_fill_1col(ser):
    ser.write(bytes([ord('X')]))   # fill mode 1 col       

def mode_idle(ser):
    ser.write(bytes([ord('I')]))    # idle mode

def mode_write(ser):
    ser.write(bytes([ord('E')]))    # write mode
    
def mode_resetmem(ser):
    ser.write(bytes([ord('H')]))    # reset mode
    
def do_nrst_sl(ser):
    ser.write(bytes([ord('C')]))    # nrst_sl
    
def do_set_vref(ser):
    ser.write(bytes([ord('D')]))    # vref
    
def do_acq_config(ser):
    ser.write(bytes([ord('M')]))
    
def do_acq_loop(ser):
    ser.write(bytes([ord('L')])) 


def do_chip_reset(ser):
    
    memsize_read = 32

    # RESET CHIP
    ser.write(bytes([ord('A')]))
    time.sleep(1e-3)
    ser.write(bytes([ord('B')]))
    time.sleep(1e-3)    
    
    # check mem
    print('after chip reset:')
    read_mem(ser, size = memsize_read)      
    
    # RESET ARRAY (DIGITAL)
    do_nrst_sl(ser)
    time.sleep(1e-3)
    
    # SET VREF
    do_set_vref(ser)
    time.sleep(1e-3)
    
    mode_idle(ser)
    time.sleep(1e-3)        
    
    # RESET MEM
    mode_resetmem(ser)
    time.sleep(1e-3)    
    
    mode_idle(ser)
    time.sleep(1e-3)
    
    # check mem
    print('after resetmem:')
    read_mem(ser, size = memsize_read)      
    
    return


def do_array_config(ser):
    
    memsize_read = 32
    
    mode_fill_1col(ser)
    # mode_fill(ser)
    time.sleep(1e-3)
    
    mode_idle(ser)
    time.sleep(1e-3)
    
    mode_write(ser)
    time.sleep(500e-3)    # pixels enable time
    
    # check mem
    print('after fill and write:')
    read_mem(ser, size = memsize_read)    
    time.sleep(1e-3)
    
    mode_idle(ser)
    time.sleep(1e-3)    
    
    # RESET MEM
    mode_resetmem(ser)
    time.sleep(1e-3)    
    
    mode_idle(ser)
    time.sleep(1e-3)
    
    # check mem
    print('after resetmem:')
    read_mem(ser, size = memsize_read)       
    
    return


   
   


def reshape_data_smartloc2(data_in):
    
    ''' This function reshapes a 1D array of 256 elements into a 16x16 2D array
    according to the pixel distribution in the SmartLoC1 ASIC '''
    
    # copy the array so we do not modify the original
    data_in_1d = np.copy(data_in)
    
    # turn 1-dimensional 1x256 array into a 2-dimensional 16x16 array
    data_in_2d = np.reshape(data_in_1d, (-1,16))

    # transpose the 2-d array
    data_out_2d = np.transpose(data_in_2d)
    
    # flip every column
    data_in_2d[0::1, :] = data_in_2d[0::1, ::-1]      
    
    return data_out_2d

def StartDynamicPlot(xmax, Nlines = 1, xlabel = '', ylabel = '', title = '', labels = ''):
    
    plt.ion()

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    
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
    else:
        ax.set_xticks(np.arange(0,xmax+2,2))
    ax.grid()
    
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.title.set_text(title)
    
    if(len(lines) == len(labels)):
        for line, label in zip(lines,labels):
            line.set_label(label)        
        ax.legend(ncols = 2)
    
    return fig, ax, lines

def UpdateDynamicPlot(fig, ax, lines, xdata, ydata):
    
    if (len(lines) == len(xdata) == len(ydata)):
    
        for line,x,y in zip(lines,xdata,ydata):     
            line.set_xdata(np.append(line.get_xdata(), x))
            line.set_ydata(np.append(line.get_ydata(), y))
    
        ax.relim()
        ax.autoscale_view()
        
        fig.canvas.draw()
        fig.canvas.flush_events()     
        
    else:
        print('Cannot plot. Array sizes do not match.')            

    return

def start_dynamic_plot_2D_plt(val_max, val_min, annotate = False, xlabel = '', ylabel = '', title = '', zlabel = ''):
    
    plt.ion()
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,8)
    
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
            
    plt.colorbar(image, label = zlabel)
      
        
    return fig, image, ax

def update_dynamic_plot_2D_plt(fig, image, ax, data2d, title):
    
    plt.ion()
    
    ax.title.set_text(title)
    image.set_data(data2d)
    
    fig.canvas.draw()
    fig.canvas.flush_events() 

def serial_connect(port = '', baudrate = 115200, timeout = None):
           
    # Configure serial port
    ser = serial.Serial()
    ser.baudrate = baudrate
    ser.port = port
    ser.timeout = timeout
    
    # Open serial port
    ser.open()
    print('Serial port opened')     
    
    return ser

def convert_config_value(config_value='1011101110'):

    '''
    convert from configuration value to array of ints
    SAME configuration for all pixels
    '''

    config_value = config_value[::-1]  # reverse the string for correct bit order

    data_first_pixel_bits = [config_value[int(n / 2)] for n in range(20)]  # take the bits from the config value

    data_first_pixel = [255 if (data == '1') else 0 for data in
                        data_first_pixel_bits]  # convert to int8 (0x0 or 0xFF)

    data_array = np.tile(data_first_pixel, 16)

    return data_array    

def read_mem(ser, size, acq = False, debug = True):

    if (acq == False):
        ser.write(bytes([ord('G')]))    # read mem
    
    rx_bytes = [ser.readline() for n in range(size*2)]
    time.sleep(1e-3)
    
    read_data_8b = [int(a,16) for a in rx_bytes]
    read_data_16b = np.array(read_data_8b, np.uint8).view('>H')     
    
    if (debug == True):
        # print(read_data_8b)
        print(read_data_16b)
    
    return read_data_8b, read_data_16b.astype(np.int32)-512    

def read_overflow(ser):

    # ser.write(bytes([ord('G')]))    # read mem
    
    rx_bytes = [ser.readline() for n in range(2)]
    time.sleep(1e-3)
    
    read_data_8b = [int(a,16) for a in rx_bytes]
    read_data_16b = np.array(read_data_8b, np.uint8).view('>H')     
    
    # print(read_data_16b)
    
    return read_data_16b   

def mode_fill(ser):
    ser.write(bytes([ord('F')]))   # fill mode    
    
def mode_fill_1col(ser):
    ser.write(bytes([ord('X')]))   # fill mode 1 col       

def mode_idle(ser):
    ser.write(bytes([ord('I')]))    # idle mode

def mode_write(ser):
    ser.write(bytes([ord('E')]))    # write mode
    
def mode_resetmem(ser):
    ser.write(bytes([ord('H')]))    # reset mode

# def mode_acq(ser):
#     ser.write(bytes([ord('J')]))    # acq mode 
    
def do_nrst_sl(ser):
    ser.write(bytes([ord('C')]))    # nrst_sl
    
def do_set_vref(ser):
    ser.write(bytes([ord('D')]))    # vref
    
# def do_acq_sl(ser):
#     ser.write(bytes([ord('K')]))    # acq_sl 
    
def do_acq_config(ser):
    ser.write(bytes([ord('M')]))
    
def do_acq_loop(ser):
    ser.write(bytes([ord('L')]))    
    
def create_dataframe():
    
    # Dataframe to represent and store the measurement data
    df = pd.DataFrame(
        {
            "Col": np.repeat(np.arange(16),16),
            "Row": np.tile(np.arange(16),16),
            "Events": None,
            # "Outflow": None,
            "Eps": None,            # Events per second
            "Milivolts": None,              
            "Timestamps": None,            
            # "Clock_ticks" : None,     
            "Measurement_configuration":None,
            "Plot_configuration": None,
            # "Data_ATEST_mV": None,
            # "Events_TEST": None,
            # "AZERO_samples": None
        }
    )    

    return df    

def perform_measurement(ser, df_in, measurement_configuration, plot_configuration):
    
    finish_measurement = 0
    
    # ''' This function will be called when pressing ctrl+C '''
    # def signal_handler(signal, frame):
    #     nonlocal finish_measurement
    #     finish_measurement = 1
    
    # signal.signal(signal.SIGINT, signal_handler)    # Assign signal_handler to a keyboard interrupt (SIGINT)    
    
    
    df_out = df_in
    
    # Retrieve measurement configuration
    measurement_time_s  =   measurement_configuration['measurement_time_s']
    sampling_period_ms  =   measurement_configuration['sampling_period_ms']
    number_of_samples  =    measurement_configuration['number_of_samples']    
    f_sampling  =           measurement_configuration['f_sampling']    
    AZERO_sample =          measurement_configuration['AZERO_sample']
    disFB_sample =          measurement_configuration['disFB_sample']
    
    # Retrieve plot configuration 
    enable_plot             = plot_configuration['enable_plot']
    pixels_to_plot          = plot_configuration['pixels_to_plot']
    plot_refresh_time_ms    = plot_configuration['plot_refresh_time_ms']   
    plot_type               = plot_configuration['plot_type']
    line_ylim               = plot_configuration['line_ylim']
    map2D_maxval            = plot_configuration['map2D_maxval']
    map2D_minval            = plot_configuration['map2D_minval']    
    
    

    # Prepare dynamic plot    
    if(enable_plot):
        ylabel = 'Eps'     
        if(plot_type in ['line','both']):         
            plot_Nlines = len(pixels_to_plot)+1
            fig, ax, lines = StartDynamicPlot(measurement_time_s, Nlines = plot_Nlines, xlabel = 'time [s]', ylabel = ylabel,
                                                      title = 'SmartLoC2', labels = [f'pix{n}' for n in pixels_to_plot] + ['OF']) 
            if(line_ylim != None):
                ax.set_ylim(-line_ylim, line_ylim)

        if(plot_type in ['2D map','both']):
            fig_2D, image_2D, ax_2D = start_dynamic_plot_2D_plt(val_max = map2D_maxval, val_min = map2D_minval, title = 'SmartLoC2', 
                                                                zlabel = ylabel)

    array_size = 256

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
    
    
    print('=== START ACQUISITION ===')
    do_acq_loop(ser)
    
    for nsample in range(number_of_samples):
        
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
        

        if(enable_plot and ((timestamps[nsample]*1e3 % plot_refresh_time_ms) == 0)):
            data_to_plot = array_eps[nsample]
            if(plot_type in ['line','both']): 
                timestamps_to_plot = np.repeat(timestamps[nsample], plot_Nlines)   
                UpdateDynamicPlot(fig, ax, lines, timestamps_to_plot, np.append(data_to_plot[pixels_to_plot],array_OF[nsample]))     
            if(plot_type in ['2D map','both']):
                data_to_plot_2D = reshape_data_smartloc2(data_to_plot)
                update_dynamic_plot_2D_plt(fig_2D, image_2D, ax_2D, data_to_plot_2D, title = f'SmartLoC2 (t = {str(timestamps[nsample])} s)')
        

        
            
        if (finish_measurement == 1):
            stop_time_s = timestamps[nsample]
            ser.write(bytes([ord('S')]))    # stop acquisition
            time.sleep(1e-3)          
            break         
        
            
    # acquisition finished
    if(finish_measurement == 0):
        ser.write(bytes([ord('S')]))    # stop acquisition if not interrupted


    measurement_configuration['stop_time_s'] = round(stop_time_s,3)

    if(np.sum(array_OF)):
        print('OJO! Overflow detected!!')        

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
        
    print('Data stored to the dataframe')            
    
    
    
    
    return df_out     


def pack_bitplanes_loop(x, nbits=10, *, msb_first=True):
    x = np.asarray(x)
    x = x & ((1 << nbits) - 1)
    if x.shape != (16,):
        raise ValueError("Expected (16,) input")

    out = np.empty(nbits * 2, dtype=np.uint8)
    for k in range(nbits):
        b = ((x >> k) & 1).astype(np.uint8)  # 16 bits for this plane

        # First byte from elements 0..7
        if msb_first:
            byte0 = np.sum(b[:8] << (7 - np.arange(8, dtype=np.uint8)))
            byte1 = np.sum(b[8:] << (7 - np.arange(8, dtype=np.uint8)))
        else:
            byte0 = np.sum(b[:8] << np.arange(8, dtype=np.uint8))
            byte1 = np.sum(b[8:] << np.arange(8, dtype=np.uint8))

        out[2*k] = byte0
        out[2*k + 1] = byte1

    return out