#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartLoC2 GUI V8
================

Digital ISFET detection dashboard for the SmartLoC2 chip.

What is new in V8 (compared to V7)
----------------------------------
* The four workspace components -- **Configuration**, **Line Plot**,
  **2D Map** and **Console** -- now live inside a freely re-arrangeable
  docking dashboard (see :mod:`smartloc2_docking`):
    - drag a panel by its title bar to move it onto another zone
      (left / right / top / bottom / swap), with a live drop preview,
    - drag the splitters between panels to resize them however you like,
    - float any panel out into its own window and dock it back,
    - the layout (sizes, arrangement, floating windows) is remembered
      automatically and restored on the next launch.
* A refreshed, light scientific theme with status badges per panel,
  colour-coded action buttons and a status bar.

The measurement / hardware logic (pydwf + serial acquisition, gain
calibration, plotting) is unchanged from V7 -- only the user interface and
some house-keeping were reworked.

Original GUI author: Javier Cuenca Michans [javier.cuenca@csic.es]
"""

from __future__ import annotations

import os
import queue
import platform
import threading
import time
from datetime import datetime
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from scipy import stats

import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.patches import Rectangle

import serial
import serial.tools.list_ports

from pydwf import (
    DwfLibrary,
    DwfEnumConfigInfo,
    DwfAnalogOutNode,
    DwfAnalogOutFunction,
    DwfAnalogOutIdle,
    DwfDeviceID,
)
from pydwf.utilities import openDwfDevice

from smartloc2_functions import (
    serial_connect,
    do_chip_reset,
    do_array_config,
    create_dataframe,
    reshape_data_smartloc2,
    do_acq_config,
    do_acq_loop,
    read_mem,
    read_overflow,
)

from smartloc2_docking import DockManager, Palette, ScrollableFrame, apply_light_scientific_theme


# =========================================================================== #
#  App metadata & constants                                                   #
# =========================================================================== #

APP_NAME = "SmartLoC2 GUI V8"
APP_VERSION = "v8.0.0"
APP_AUTHOR = "Javier Cuenca Michans [javier.cuenca@csic.es]"

ARRAY_SIZE = 256                 # number of pixels in the 16x16 SmartLoC2 array
GRID_SIDE = 16                   # array side length
MAX_PLOTTED_PIXELS = 8           # maximum number of pixel traces in the line plot
DEFAULT_BAUDRATE = 230400

LAYOUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "smartloc2_layout.json")

# Default pixel selection shown in the line plot.
DEFAULT_SELECTED_PIXELS = [1, 2, 3, 4, 5, 6, 7, 8]


# =========================================================================== #
#  PYDWF (Analog Discovery) helpers -- unchanged behaviour                     #
# =========================================================================== #

def analog_out_custom_waveform(analogOut, waveform, t_final):
    """Drive the analog-output channel of the Analog Discovery with a custom
    waveform (used for gain calibration)."""
    CH1 = 0
    node = DwfAnalogOutNode.Carrier

    analogOut.reset(CH1)
    analogOut.nodeEnableSet(CH1, node, True)
    analogOut.nodeFunctionSet(CH1, node, DwfAnalogOutFunction.Custom)
    analogOut.nodeAmplitudeSet(CH1, node, 1)
    analogOut.nodeOffsetSet(CH1, node, 0)
    analogOut.nodeDataSet(CH1, node, waveform)
    analogOut.waitSet(CH1, 0)

    # The frequency of a custom waveform is (1 / waveform_duration).
    analogOut.nodeFrequencySet(CH1, node, 1 / t_final)
    analogOut.runSet(CH1, 0)
    analogOut.idleSet(CH1, DwfAnalogOutIdle.Initial)
    analogOut.configure(False, True)  # start
    print("waveform started")


def maximize_analog_out_buffer_size(configuration_parameters):
    """Pick the device configuration with the largest analog-out buffer."""
    return configuration_parameters[DwfEnumConfigInfo.AnalogOutBufferSize]


# =========================================================================== #
#  SmartLoC2 signal-processing helpers -- unchanged behaviour                  #
# =========================================================================== #

def linear_function_inverse(x, m_pos, m_neg):
    """counts -> millivolts.

    Inverse of a piecewise-linear function with positive slope ``m_pos`` for
    ``x >= 0`` and negative slope ``m_neg`` for ``x < 0``.
    """
    if isinstance(x, float):  # scalar mode
        return x / m_neg if x < 0 else x / m_pos

    # array mode
    y = np.zeros(len(x))
    y[x < 0] = x[x < 0] / m_neg
    y[x >= 0] = x[x >= 0] / m_pos
    return y


def custom_pulsed_waveform(timestamps, amplitudes, period, initial_delay=0, DC_value=0):
    """Create a pulsed waveform with one pulse per amplitude, each of length
    ``period``."""
    waveform = np.zeros(len(timestamps))
    for index, amp in enumerate(amplitudes):
        t_rising = index * period + initial_delay
        t_falling = index * period + period / 2 + initial_delay
        waveform[(timestamps >= t_rising) & (timestamps < t_falling)] = amp
    return waveform + DC_value


def gain_regression_calc(filename, amplitudes_in, period, t_start,
                         pixels_to_plot=None, debug_level=0, label="-"):
    """Estimate per-pixel gains by linear regression of event frequency vs.
    input amplitude."""
    if pixels_to_plot is None:
        pixels_to_plot = [0]

    # Outlier rejection helper.
    # Source - https://stackoverflow.com/a  (Egal, modified by community)
    # Retrieved 2025-11-24, License - CC BY-SA 4.0
    def reject_outliers_2(data, m=2.0):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / (mdev if mdev else 1.0)
        return data[s < m]

    df_read = pd.read_pickle(filename)
    timestamps = df_read.Timestamps[0]
    pulse_width = period / 2

    timestamps_rising = [t_start + n * period for n in range(len(amplitudes_in))]
    print(f"Rising timestamps: {timestamps_rising}")

    eps_out = np.zeros((ARRAY_SIZE, len(amplitudes_in)))
    array_gains = np.zeros(ARRAY_SIZE)
    array_offsets = np.zeros(ARRAY_SIZE)

    for pixel_index in range(ARRAY_SIZE):
        eps = df_read.Eps[pixel_index]

        for n, t_rising in enumerate(timestamps_rising):
            # values between 1/4 and 3/4 of the pulse HIGH duration
            eps_pulse = eps[(timestamps > (t_rising + pulse_width / 4)) &
                            (timestamps < (t_rising + 3 * pulse_width / 4))]
            eps_pulse_filt = reject_outliers_2(eps_pulse)

            # values between 1/4 and 3/4 of the pulse LOW duration
            eps_base = eps[(timestamps > (t_rising + pulse_width / 4 + pulse_width)) &
                           (timestamps < (t_rising + 3 * pulse_width / 4 + pulse_width))]
            eps_base_filt = reject_outliers_2(eps_base)

            eps_out[pixel_index, n] = np.average(eps_pulse_filt) - np.average(eps_base_filt)

        result = cast(Any, stats.linregress(amplitudes_in, eps_out[pixel_index]))
        array_gains[pixel_index] = round(float(result.slope), 4)
        array_offsets[pixel_index] = round(float(result.intercept), 4)

    return array_gains, eps_out, array_offsets


def gain_analysis(file_to_save, amplitudes, initial_delay, pulse_period, plot_configuration):
    """Run the full gain-calibration analysis, generate plots and store the
    resulting gains dataframe to disk."""
    file_to_save_gains = file_to_save + "_gains"
    extension = ".dataframe"

    df_gains = pd.DataFrame({
        "Pixel_index": range(ARRAY_SIZE),
        "Gains_positive": None,
        "Gains_negative": None,
    })

    amplitudes_pos, amplitudes_neg = np.split(1e3 * amplitudes, 2)

    # positive gains
    t_start = initial_delay
    array_gains_pos, eps_out_pos, array_offsets_pos = gain_regression_calc(
        file_to_save + extension, amplitudes_pos, pulse_period, t_start,
        pixels_to_plot=plot_configuration["pixels_to_plot"], debug_level=0, label="Positive")

    # negative gains
    t_start = initial_delay + pulse_period * len(amplitudes) / 2
    array_gains_neg, eps_out_neg, array_offsets_neg = gain_regression_calc(
        file_to_save + extension, amplitudes_neg, pulse_period, t_start,
        pixels_to_plot=plot_configuration["pixels_to_plot"], debug_level=0, label="Negative")

    # ----- plots -----
    plt.ioff()

    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.title(f"SmartLoC2 gain calibration ({file_to_save_gains})", fontsize=11)
    ax1.set_xlabel("Input voltage [mV]")
    ax1.set_ylabel("Event frequency [eps]")

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = iter(prop_cycle.by_key()["color"])

    amplitudes_pos_z = np.insert(amplitudes_pos, 0, 0)
    amplitudes_neg_z = np.insert(amplitudes_neg, 0, 0)

    for pixel_index in plot_configuration["pixels_to_plot"]:
        gain_pos = array_gains_pos[pixel_index]
        offset_pos = array_offsets_pos[pixel_index]
        gain_neg = array_gains_neg[pixel_index]
        offset_neg = array_offsets_neg[pixel_index]

        color = next(colors)
        ax1.plot(amplitudes_pos, eps_out_pos[pixel_index], "o", color=color, label=f"pix {pixel_index}")
        ax1.plot(amplitudes_neg, eps_out_neg[pixel_index], "o", color=color)
        ax1.plot(amplitudes_pos_z, gain_pos * amplitudes_pos_z + offset_pos, "--", color=color)
        ax1.plot(amplitudes_neg_z, gain_neg * amplitudes_neg_z + offset_neg, "--", color=color)
        ax1.legend(ncols=2)

    array_gains_pos_2D = reshape_data_smartloc2(array_gains_pos)
    array_gains_neg_2D = reshape_data_smartloc2(array_gains_neg)

    fig2 = plot_2D_plt(array_gains_pos_2D, title=f"{file_to_save_gains} positive gains [eps/mV]",
                       annot=True, fmt=".1f", annot_size=10, minval=1, maxval=4)
    fig3 = plot_2D_plt(array_gains_neg_2D, title=f"{file_to_save_gains} negative gains [eps/mV]",
                       annot=True, fmt=".1f", annot_size=10, minval=1, maxval=4)

    fig1.savefig(f"{file_to_save_gains}.png", dpi=250)
    fig1.savefig(f"{file_to_save_gains}.svg")
    plt.close(fig1)

    fig2.savefig(f"{file_to_save_gains}_map_pos.png", dpi=250)
    fig2.savefig(f"{file_to_save_gains}_map_pos.svg")
    plt.close(fig2)

    fig3.savefig(f"{file_to_save_gains}_map_neg.png", dpi=250)
    fig3.savefig(f"{file_to_save_gains}_map_neg.svg")
    plt.close(fig3)

    plt.ion()

    # ----- save results -----
    df_gains["Gains_positive"] = array_gains_pos
    df_gains["Gains_negative"] = array_gains_neg

    if os.path.isfile(file_to_save_gains + ".dataframe"):
        file_to_save_gains = file_to_save_gains + "_" + str(int(time.time()))
        print(f"File exists! Saving as {file_to_save_gains}")

    df_gains.to_pickle(file_to_save_gains + extension)
    print(f"{file_to_save_gains + extension} stored in disk")


# =========================================================================== #
#  The application                                                            #
# =========================================================================== #

class SmartLoC2App:
    """Main application object: owns the Tk root, the docking dashboard and all
    measurement state."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_NAME)
        self.root.geometry("1500x920")
        self.root.minsize(1100, 720)

        # --- base font (comfortable density) ---
        self.base_font = ("Segoe UI", 11) if os.name == "nt" else ("DejaVu Sans", 11)
        self.style = apply_light_scientific_theme(self.root, self.base_font)

        # --- measurement / plot state ---
        self.running = False
        self.active_plot = True
        self.cancel_event = threading.Event()
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        self.selected_pixels = list(DEFAULT_SELECTED_PIXELS)
        self.ground_truth_pixels: list[int] = []

        # matplotlib handles (line plot)
        self.fig = self.ax = self.lines = self.canvas_lines = None
        # matplotlib handles (2D map)
        self.fig_2D = self.image_2D = self.ax_2D = self.canvas_2D = None
        self.map_click_cid = None
        self.map_hover_cid = None
        self.selected_overlay_patches: list[Rectangle] = []
        self.ground_truth_artist = None
        self.map_hover_annotation = None
        self.map_hover_marker = None
        self.map_index_labels = []
        self.map_data_current = None

        # --- Tk variables ---
        self.save_measurement_var = tk.BooleanVar(value=True)
        self.selection_status_var = tk.StringVar()
        self.ground_truth_var = tk.StringVar(value="")
        self.input_serial_port = tk.StringVar(value="")

        self.meas_time_var = tk.StringVar(value="900")
        self.disfb_time_var = tk.StringVar(value="60")
        self.sampling_period_var = tk.StringVar(value="500")
        self.filename_var = tk.StringVar(value="filename")
        self.filename_gain_var = tk.StringVar(value="gain_calib_file")
        self.do_gain_calib = tk.IntVar()
        self.apply_gain_calib = tk.IntVar()

        self.plot_refresh_var = tk.StringVar(value="500")
        self.line_ylim_var = tk.StringVar(value="250")
        self.map_maxval_var = tk.StringVar(value="250")
        self.plot_invyaxis_opt = tk.BooleanVar(value=False)

        self.status_conn_var = tk.StringVar(value="Disconnected")
        self.status_state_var = tk.StringVar(value="Idle")
        self.status_port_var = tk.StringVar(value="Port: -")

        # --- build UI ---
        self._build_menu()
        self._build_dashboard()
        self._build_status_bar()
        self._wire_events()

        self.refresh_selected_pixels_label()
        self.refresh_serial_ports()
        self.root.after(50, self._initialize_dashboard_preview)
        self._flush_log_queue()

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    # --------------------------------------------------------------------- #
    #  Logging                                                              #
    # --------------------------------------------------------------------- #
    def log_message(self, msg: str):
        if not msg.endswith("\n"):
            msg = msg + "\n"
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {msg}")

    def _flush_log_queue(self):
        try:
            while True:
                chunk = self.log_queue.get_nowait()
                self.console_text.configure(state="normal")
                self.console_text.insert("end", chunk)
                self.console_text.see("end")
                self.console_text.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.root.after(75, self._flush_log_queue)

    # --------------------------------------------------------------------- #
    #  Menu bar                                                             #
    # --------------------------------------------------------------------- #
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Exit", command=self.on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(label="Reset layout", command=self._reset_layout)
        menubar.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=lambda: self.show_about(self.root))
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def show_about(self, parent):
        py_ver_str = platform.python_version()
        tk_ver = parent.tk.call("info", "patchlevel")

        win = tk.Toplevel(parent)
        win.title(f"About {APP_NAME}")
        win.transient(parent)
        win.grab_set()
        win.resizable(False, False)
        win.configure(bg=Palette.PANEL_BG)

        win.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() // 2) - 200
        y = parent.winfo_rooty() + (parent.winfo_height() // 2) - 120
        win.geometry(f"420x180+{x}+{y}")

        frm = ttk.Frame(win, padding=16, style="DockBody.TFrame")
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text=APP_NAME, style="DockTitle.TLabel",
                  font=(self.base_font[0], 13, "bold")).pack(anchor="w")
        ttk.Label(frm, text=f"Version: {APP_VERSION}", style="Muted.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Label(frm, text=f"Developer: {APP_AUTHOR}", style="Muted.TLabel").pack(anchor="w")

        versions = ttk.Frame(frm, style="DockBody.TFrame")
        versions.pack(fill="x", pady=(8, 8))
        ttk.Label(versions, text=f"Python: {py_ver_str}", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(versions, text=f"Tk/Tkinter: {tk_ver}", style="Muted.TLabel").grid(row=1, column=0, sticky="w")

        win.bind("<Escape>", lambda _e: win.destroy())
        win.bind("<Return>", lambda _e: win.destroy())

    # --------------------------------------------------------------------- #
    #  Dashboard (dockable panels)                                          #
    # --------------------------------------------------------------------- #
    def _build_dashboard(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)  # status bar

        host = ttk.Frame(self.root, padding=Palette.GAP)
        host.grid(row=0, column=0, sticky="nsew")

        self.dock = DockManager(self.root, host, LAYOUT_FILE)

        # Register the four panels with icons and default slots.
        config_panel = self.dock.add_panel("config", "CONFIGURATION", "left-top", icon="⚙")
        console_panel = self.dock.add_panel("console", "CONSOLE", "left-bottom", icon="⌨")
        line_panel = self.dock.add_panel("line", "LINE PLOT", "right-top", icon="📈")
        map_panel = self.dock.add_panel("map", "2D MAP", "right-bottom", icon="▦")

        self._build_config_panel(config_panel.body)
        self._build_console_panel(console_panel.body)

        # Plot containers (filled later by start_plot)
        self.line_plot_container = line_panel.body
        self.map_plot_container = map_panel.body

        self.panel_config = config_panel
        self.panel_console = console_panel
        self.panel_line = line_panel
        self.panel_map = map_panel

        self.dock.finalize()

    def _build_config_panel(self, container):
        # The configuration form is taller than the panel can be, so host it in
        # a scrollable frame -- every control stays reachable at any panel size.
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        scroller = ScrollableFrame(container)
        scroller.grid(row=0, column=0, sticky="nsew")
        parent = scroller.inner
        parent.columnconfigure(0, weight=1)

        # ----- Measurement section -----
        lf_meas = ttk.LabelFrame(parent, text="MEASUREMENT", style="Section.TLabelframe", padding=10)
        lf_meas.grid(row=0, column=0, sticky="ew")
        lf_meas.columnconfigure(1, weight=1)

        ttk.Label(lf_meas, text="Measurement time [s]:", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(lf_meas, textvariable=self.meas_time_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Checkbutton(lf_meas, text="Store measurement to file",
                        variable=self.save_measurement_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(lf_meas, text="Disable high-pass filter at [s]:", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(lf_meas, textvariable=self.disfb_time_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(lf_meas, text="Sampling period [ms]:", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(lf_meas, textvariable=self.sampling_period_var).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Label(lf_meas, text="File to store the data:", style="Muted.TLabel").grid(row=4, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(lf_meas, textvariable=self.filename_var).grid(row=4, column=1, sticky="ew")

        ttk.Checkbutton(lf_meas, text="Run gain calibration",
                        variable=self.do_gain_calib).grid(row=5, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(lf_meas, text="Apply gain calibration",
                        variable=self.apply_gain_calib).grid(row=6, column=0, columnspan=2, sticky="w", pady=3)

        ttk.Label(lf_meas, text="Gain calibration file:", style="Muted.TLabel").grid(row=7, column=0, sticky="w", padx=(0, 8))
        row_frame = ttk.Frame(lf_meas, style="DockBody.TFrame")
        row_frame.grid(row=8, column=0, columnspan=2, sticky="ew")
        row_frame.columnconfigure(0, weight=1)
        ttk.Entry(row_frame, textvariable=self.filename_gain_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(row_frame, text="...", width=3, command=self._browse_gain_file).grid(row=0, column=1, sticky="e", padx=(6, 0))

        # ----- Plot settings section -----
        lf_plot = ttk.LabelFrame(parent, text="PLOT SETTINGS", style="Section.TLabelframe", padding=10)
        lf_plot.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        lf_plot.columnconfigure(1, weight=1)

        ttk.Label(lf_plot, text="Plot update period [ms]:", style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(lf_plot, textvariable=self.plot_refresh_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(lf_plot, text="Y axis max (line):", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(lf_plot, textvariable=self.line_ylim_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(lf_plot, text="Y axis max (2D map):", style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(lf_plot, textvariable=self.map_maxval_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Checkbutton(lf_plot, text="Invert y axis",
                        variable=self.plot_invyaxis_opt).grid(row=3, column=0, columnspan=2, sticky="w", pady=3)

        self.selection_status_var.set("Selected pixels: " + ", ".join(str(p) for p in self.selected_pixels))
        ttk.Label(lf_plot, textvariable=self.selection_status_var, wraplength=430,
                  style="Status.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(lf_plot, text="Ground truth pixels / wells:", style="Muted.TLabel").grid(row=5, column=0, sticky="w", pady=(8, 3))
        ttk.Entry(lf_plot, textvariable=self.ground_truth_var).grid(row=5, column=1, sticky="ew", pady=(8, 3))
        ttk.Label(lf_plot, text="Comma separated pixel numbers.", style="Muted.TLabel").grid(row=6, column=0, columnspan=2, sticky="w")
        self.ground_truth_var.trace_add("write", lambda *_: self.set_ground_truth_pixels(self.ground_truth_var.get()))

        # ----- Connection / actions section -----
        control = ttk.LabelFrame(parent, text="CONNECTION / ACTIONS", style="Section.TLabelframe", padding=10)
        control.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        control.columnconfigure(2, weight=1)
        control.columnconfigure(3, weight=1)

        self.list_btn = tk.Button(control, text="refresh PORTS", fg="white", bg=Palette.ACCENT,
                                  activebackground=Palette.ACCENT_DARK, activeforeground="white",
                                  relief="flat", bd=0, height=2, font=self.base_font, cursor="hand2")
        self.list_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(control, text="Serial port:", style="Muted.TLabel").grid(row=0, column=1, sticky="e", padx=(0, 6))
        self.serial_port_combo = ttk.Combobox(control, textvariable=self.input_serial_port, state="readonly")
        self.serial_port_combo.grid(row=0, column=2, columnspan=2, sticky="ew", padx=(0, 0))

        self.run_btn = tk.Button(control, text="RUN", fg="white", bg=Palette.GREEN,
                                 activebackground=Palette.GREEN_DARK, activeforeground="white",
                                 relief="flat", bd=0, height=2, font=(self.base_font[0], self.base_font[1], "bold"),
                                 cursor="hand2")
        self.run_btn.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(10, 0))

        self.finish_btn = tk.Button(control, text="FINISH", fg="white", bg=Palette.RED,
                                    activebackground="#91322B", activeforeground="white",
                                    relief="flat", bd=0, height=2, font=(self.base_font[0], self.base_font[1], "bold"),
                                    cursor="hand2")
        self.finish_btn.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(10, 0))

        self.plot_btn = tk.Button(control, text="stop PLOT", fg="white", bg=Palette.GOLD,
                                  activebackground="#9A651D", activeforeground="white",
                                  relief="flat", bd=0, height=2, font=self.base_font, cursor="hand2")
        self.plot_btn.grid(row=1, column=2, columnspan=2, sticky="ew", padx=(0, 0), pady=(10, 0))

    def _build_console_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.console_text = tk.Text(parent, height=8, wrap="word", state="disabled",
                                    bg="#FBFCFD", fg=Palette.TEXT, relief="flat",
                                    insertbackground=Palette.TEXT, font=("Consolas", 10) if os.name == "nt" else ("DejaVu Sans Mono", 10),
                                    padx=8, pady=6)
        self.console_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.console_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.console_text.configure(yscrollcommand=scroll.set)

    # --------------------------------------------------------------------- #
    #  Status bar                                                           #
    # --------------------------------------------------------------------- #
    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=Palette.PANEL_HEADER, height=28)
        bar.grid(row=1, column=0, sticky="ew")
        bar.grid_propagate(True)

        def _seg(textvar, color=Palette.TEXT_MUTED):
            lbl = tk.Label(bar, textvariable=textvar, bg=Palette.PANEL_HEADER, fg=color,
                           font=(self.base_font[0], self.base_font[1] - 1))
            return lbl

        self._status_dot = tk.Canvas(bar, width=12, height=12, highlightthickness=0, bg=Palette.PANEL_HEADER, bd=0)
        self._status_dot_id = self._status_dot.create_oval(2, 2, 11, 11, fill=Palette.BADGE_IDLE, outline="")
        self._status_dot.pack(side="left", padx=(12, 6), pady=6)

        _seg(self.status_conn_var, Palette.TEXT).pack(side="left", padx=(0, 16))
        tk.Frame(bar, width=1, bg=Palette.BORDER).pack(side="left", fill="y", pady=6)
        _seg(self.status_state_var).pack(side="left", padx=16)
        tk.Frame(bar, width=1, bg=Palette.BORDER).pack(side="left", fill="y", pady=6)
        _seg(self.status_port_var).pack(side="left", padx=16)

        tk.Label(bar, text=f"{APP_NAME}  •  {APP_VERSION}", bg=Palette.PANEL_HEADER,
                 fg=Palette.TEXT_MUTED, font=(self.base_font[0], self.base_font[1] - 1)).pack(side="right", padx=12)

    def _set_status(self, connected: Optional[bool] = None, state: Optional[str] = None,
                    badge: Optional[str] = None):
        if connected is not None:
            self.status_conn_var.set("Connected" if connected else "Disconnected")
        if state is not None:
            self.status_state_var.set(state)
        if badge is not None:
            try:
                self._status_dot.itemconfigure(self._status_dot_id, fill=badge)
            except tk.TclError:
                pass

    # --------------------------------------------------------------------- #
    #  Event wiring                                                         #
    # --------------------------------------------------------------------- #
    def _wire_events(self):
        self.list_btn.config(command=self.on_list_ports)
        self.run_btn.config(command=self.on_run)
        self.finish_btn.config(command=self.on_finish_cancel, state="disabled")
        self.plot_btn.config(command=self.on_plot)

    def _reset_layout(self):
        self.dock.reset_layout()
        self.log_message("Layout reset to default")

    def _browse_gain_file(self):
        path = filedialog.askopenfilename(
            title="Select gain calibration file",
            defaultextension=".dataframe",
            filetypes=[("Dataframe (pickle) files", "*.dataframe")],
        )
        if path:
            self.filename_gain_var.set(path)

    # --------------------------------------------------------------------- #
    #  Pixel selection helpers                                              #
    # --------------------------------------------------------------------- #
    def normalize_selected_pixels(self, pixel_values):
        unique_pixels = []
        for pixel in pixel_values:
            try:
                pixel_index = int(pixel)
            except (TypeError, ValueError):
                continue
            if 0 <= pixel_index < ARRAY_SIZE and pixel_index not in unique_pixels:
                unique_pixels.append(pixel_index)
        return unique_pixels[:MAX_PLOTTED_PIXELS]

    def get_selected_pixels(self):
        return self.normalize_selected_pixels(self.selected_pixels)

    def refresh_selected_pixels_label(self):
        pixels = self.get_selected_pixels()
        if pixels:
            self.selection_status_var.set("Selected pixels: " + ", ".join(str(p) for p in pixels))
        else:
            self.selection_status_var.set("Selected pixels: none")

    @staticmethod
    def pixel_index_to_xy(pixel_index):
        return divmod(int(pixel_index), GRID_SIDE)

    @staticmethod
    def parse_pixel_list(raw_value):
        if raw_value is None:
            return []
        normalized = str(raw_value).replace(";", ",").replace(" ", ",")
        parsed = []
        for token in normalized.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                pixel_index = int(token)
            except ValueError:
                continue
            if 0 <= pixel_index < ARRAY_SIZE and pixel_index not in parsed:
                parsed.append(pixel_index)
        return parsed

    def set_ground_truth_pixels(self, pixel_values):
        self.ground_truth_pixels = self.parse_pixel_list(pixel_values)
        self.refresh_ground_truth_overlay()

    def set_selected_pixels(self, pixel_values):
        self.selected_pixels = self.normalize_selected_pixels(pixel_values)
        self.refresh_selected_pixels_label()
        self.refresh_selection_overlay()
        self.refresh_line_labels_and_reset()

    # --------------------------------------------------------------------- #
    #  Overlays on the 2D map                                               #
    # --------------------------------------------------------------------- #
    def refresh_ground_truth_overlay(self):
        if self.ax_2D is None or self.canvas_2D is None:
            return
        ground_truth_xy = np.array(
            [self.pixel_index_to_xy(p) for p in self.ground_truth_pixels], dtype=float)
        if self.ground_truth_artist is None:
            # Open ring marker: clearly visible on any cell colour, does not
            # hide the live value underneath it.
            self.ground_truth_artist = self.ax_2D.scatter(
                [], [], s=230, marker="o", facecolors="none",
                edgecolors="#111A16", linewidths=2.2, zorder=6)
        if ground_truth_xy.size == 0:
            self.ground_truth_artist.set_offsets(np.empty((0, 2)))
        else:
            self.ground_truth_artist.set_offsets(ground_truth_xy)
        self.canvas_2D.draw_idle()

    def refresh_selection_overlay(self):
        if self.ax_2D is None or self.canvas_2D is None:
            return
        for patch in self.selected_overlay_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        self.selected_overlay_patches = []
        for pixel_index in self.get_selected_pixels():
            x_value, y_value = self.pixel_index_to_xy(pixel_index)
            # Slight darkening (so the selected pixel reads as "a touch darker"
            # as requested) PLUS a bold accent border that stays visible on top
            # of any live colour.
            shade = Rectangle((x_value - 0.5, y_value - 0.5), 1, 1,
                              facecolor=(0.0, 0.0, 0.0, 0.16), edgecolor="none",
                              zorder=4)
            border = Rectangle((x_value - 0.5, y_value - 0.5), 1, 1, fill=False,
                               edgecolor=Palette.ACCENT, linewidth=2.0, zorder=5)
            self.ax_2D.add_patch(shade)
            self.ax_2D.add_patch(border)
            self.selected_overlay_patches.extend([shade, border])
        self.canvas_2D.draw_idle()

    def refresh_line_labels_and_reset(self):
        if self.fig is None or self.ax is None or self.lines is None or self.canvas_lines is None:
            return
        pixels = self.get_selected_pixels()
        while len(pixels) < MAX_PLOTTED_PIXELS:
            pixels.append(None)
        labels = [f"pix {p}" if p is not None else "_nolegend_" for p in pixels] + ["OF"]
        for line in self.lines:
            line.set_data([], [])
        for line, label in zip(self.lines, labels):
            line.set_label(label)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend(ncols=2)
        self.canvas_lines.draw_idle()

    def on_map_click(self, event):
        if event.inaxes != self.ax_2D or event.xdata is None or event.ydata is None:
            return
        x = int(np.clip(round(event.xdata), 0, GRID_SIDE - 1))
        y = int(np.clip(round(event.ydata), 0, GRID_SIDE - 1))
        pixel_index = GRID_SIDE * x + y

        current = self.get_selected_pixels()
        if pixel_index in current:
            current.remove(pixel_index)
            self.log_message(f"Pixel {pixel_index} removed from line plot selection")
        else:
            if len(current) >= MAX_PLOTTED_PIXELS:
                self.log_message(f"Selection limited to {MAX_PLOTTED_PIXELS} pixels")
                return
            current.append(pixel_index)
            self.log_message(f"Pixel {pixel_index} added to line plot selection")
        self.set_selected_pixels(current)

    def on_map_hover(self, event):
        if self.ax_2D is None or self.canvas_2D is None:
            return
        if self.map_hover_annotation is None or self.map_hover_marker is None:
            return
        if event.inaxes != self.ax_2D or event.xdata is None or event.ydata is None:
            if self.map_hover_annotation.get_visible() or self.map_hover_marker.get_visible():
                self.map_hover_annotation.set_visible(False)
                self.map_hover_marker.set_visible(False)
                self.canvas_2D.draw_idle()
            return

        x = int(np.clip(round(event.xdata), 0, GRID_SIDE - 1))
        y = int(np.clip(round(event.ydata), 0, GRID_SIDE - 1))
        pixel_index = GRID_SIDE * x + y

        value_text = ""
        if self.map_data_current is not None:
            try:
                value = float(self.map_data_current[y][x])
                value_text = f"\nval {value:.1f}"
            except (IndexError, ValueError, TypeError):
                value_text = ""

        self.map_hover_marker.set_xy((x - 0.5, y - 0.5))
        self.map_hover_marker.set_visible(True)
        self.map_hover_annotation.xy = (x, y)
        self.map_hover_annotation.set_text(f"pixel {pixel_index}{value_text}")
        self.map_hover_annotation.set_visible(True)
        self.canvas_2D.draw_idle()

    # --------------------------------------------------------------------- #
    #  Plot construction / updates                                          #
    # --------------------------------------------------------------------- #
    def _initialize_dashboard_preview(self):
        preview_meas = {
            "measurement_time_s": int(self.meas_time_var.get()),
            "apply_gain_calib": bool(self.apply_gain_calib.get()),
        }
        preview_plot = {
            "line_ylim": int(self.line_ylim_var.get()),
            "map2D_maxval": int(self.map_maxval_var.get()),
            "map2D_minval": -int(self.map_maxval_var.get()),
        }
        self.start_plot(preview_meas, preview_plot)

    def start_plot(self, measurement_configuration, plot_configuration):
        print("STARTING PLOT IN MAIN THREAD")
        measurement_time_s = measurement_configuration["measurement_time_s"]
        apply_gain_calib = measurement_configuration["apply_gain_calib"]
        line_ylim = plot_configuration["line_ylim"]
        map2D_maxval = plot_configuration["map2D_maxval"]
        map2D_minval = plot_configuration["map2D_minval"]

        ylabel = "mV" if apply_gain_calib else "Eps"

        selected_pixels_live = self.get_selected_pixels()
        while len(selected_pixels_live) < MAX_PLOTTED_PIXELS:
            selected_pixels_live.append(None)

        self.selected_overlay_patches = []
        self.ground_truth_artist = None

        self.fig, self.ax, self.lines, self.canvas_lines = self._start_dynamic_line_plot(
            self.line_plot_container, measurement_time_s, Nlines=MAX_PLOTTED_PIXELS + 1,
            xlabel="time [s]", ylabel=ylabel, title="SmartLoC2",
            labels=[f"pix{n}" if n is not None else "_nolegend_" for n in selected_pixels_live] + ["OF"])
        if line_ylim is not None:
            self.ax.set_ylim(-line_ylim, line_ylim)

        self.fig_2D, self.image_2D, self.ax_2D, self.canvas_2D = self._start_dynamic_map(
            self.map_plot_container, val_max=map2D_maxval, val_min=map2D_minval,
            annotate=True, title="SmartLoC2", zlabel=ylabel)

        self.refresh_selection_overlay()
        self.refresh_selected_pixels_label()
        self.refresh_ground_truth_overlay()
        self.map_click_cid = self.canvas_2D.mpl_connect("button_press_event", self.on_map_click)
        self.map_hover_cid = self.canvas_2D.mpl_connect("motion_notify_event", self.on_map_hover)

    def _start_dynamic_line_plot(self, container, xmax, Nlines=1, xlabel="", ylabel="",
                                 title="", labels=None):
        for child in container.winfo_children():
            child.destroy()

        fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
        fig.patch.set_facecolor(Palette.PANEL_BG)
        ax.set_facecolor("#FBFCFD")

        canvas_line = FigureCanvasTkAgg(fig, master=container)
        canvas_widget = canvas_line.get_tk_widget()
        toolbar = NavigationToolbar2Tk(canvas_line, container)
        toolbar.update()
        toolbar.pack(side="top", fill="x")
        canvas_widget.pack(side="top", fill="both", expand=True)
        canvas_line.draw()

        lines = [ax.plot([], [])[0] for _ in range(Nlines)]

        ax.set_autoscaley_on(True)
        ax.set_xlim(0, xmax)
        self._configure_time_ticks(ax, xmax)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.title.set_text(title)

        if labels is None:
            labels = []
        if len(lines) == len(labels):
            for line, label in zip(lines, labels):
                line.set_label(label)
            ax.legend(ncols=2)

        return fig, ax, lines, canvas_line

    @staticmethod
    def _configure_time_ticks(ax, xmax):
        if xmax >= 14400:
            ax.set_xticks(np.arange(0, xmax + 3600, 3600))
        elif xmax >= 7200:
            ax.set_xticks(np.arange(0, xmax + 600, 600))
        elif xmax >= 1200:
            ax.set_xticks(np.arange(0, xmax + 300, 300))
        elif xmax >= 600:
            ax.set_xticks(np.arange(0, xmax + 60, 60))
        elif xmax >= 120:
            ax.set_xticks(np.arange(0, xmax + 30, 30))
        elif xmax >= 60:
            ax.set_xticks(np.arange(0, xmax + 5, 5))
        elif xmax != 0:
            ax.set_xticks(np.arange(0, xmax + 2, 2))
        else:
            ax.set_autoscalex_on(True)

    def _start_dynamic_map(self, container, val_max, val_min, annotate=False,
                           xlabel="", ylabel="", title="", zlabel=""):
        for child in container.winfo_children():
            child.destroy()

        # No fixed figsize: let the figure grow to fill the panel. The map keeps
        # a square aspect via set_box_aspect so it never drifts to one side and
        # always uses the available space symmetrically.
        fig, ax = plt.subplots(constrained_layout=True)
        fig.patch.set_facecolor(Palette.PANEL_BG)
        ax.set_facecolor("#FBFCFD")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.title.set_text(title)
        # Tick every other pixel so labels stay readable on a 16x16 grid.
        tick_positions = list(range(0, GRID_SIDE, 2))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(labelsize=9, length=0)
        ax.set_xlim(-0.5, GRID_SIDE - 0.5)
        ax.set_ylim(GRID_SIDE - 0.5, -0.5)
        # "auto" aspect: the heat map stretches to fill the whole panel so no
        # space is wasted. Cells may become slightly rectangular when the panel
        # is not square, which is fine for a 16x16 sensor array.
        ax.set_aspect("auto")
        for spine in ax.spines.values():
            spine.set_edgecolor("#C4D0D6")

        data2d = np.zeros((GRID_SIDE, GRID_SIDE))
        data2d[0][0] = val_max
        data2d[0][1] = val_min
        image = ax.imshow(data2d, cmap="RdBu", interpolation="nearest",
                          vmin=val_min, vmax=val_max, aspect="auto")
        self.map_data_current = data2d

        # Thin grid lines between cells so single pixels stay distinguishable.
        ax.set_xticks(np.arange(-0.5, GRID_SIDE, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, GRID_SIDE, 1), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=0.5, alpha=0.45)
        ax.tick_params(which="minor", length=0)

        # Discreet pixel-index labels: only on the outer frame of the grid so
        # the live colour of each cell stays fully readable. Hover shows the
        # exact index + value for any cell (see _on_map_hover).
        self.map_index_labels = []
        if annotate:
            for x in range(GRID_SIDE):
                for y in range(GRID_SIDE):
                    on_edge = x in (0, GRID_SIDE - 1) or y in (0, GRID_SIDE - 1)
                    if not on_edge:
                        continue
                    label = ax.text(x, y, f"{int(GRID_SIDE * x + y)}",
                                    ha="center", va="center", fontsize=6.5,
                                    color="#6B7780", alpha=0.8, zorder=3)
                    self.map_index_labels.append(label)

        cbar = fig.colorbar(image, ax=ax, label=zlabel, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        # Hover marker + annotation, hidden until the cursor enters a cell.
        self.map_hover_marker = Rectangle((-1, -1), 1, 1, fill=False,
                                          edgecolor="#18211D", linewidth=1.4,
                                          zorder=8, visible=False)
        ax.add_patch(self.map_hover_marker)
        self.map_hover_annotation = ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            fontsize=9, color="#18211D", zorder=9, visible=False,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF",
                      ec=Palette.ACCENT, lw=1.0, alpha=0.95))

        canvas_2d = FigureCanvasTkAgg(fig, master=container)
        canvas_widget = canvas_2d.get_tk_widget()
        toolbar = NavigationToolbar2Tk(canvas_2d, container)
        toolbar.update()
        toolbar.pack(side="top", fill="x")
        canvas_widget.pack(side="top", fill="both", expand=True)
        canvas_2d.draw()

        return fig, image, ax, canvas_2d

    def update_plot(self, xdata, ydata):
        if self.lines is None:
            return
        if len(self.lines) == len(xdata) == len(ydata):
            for line, x, y in zip(self.lines, xdata, ydata):
                line.set_xdata(np.append(line.get_xdata(), x))
                line.set_ydata(np.append(line.get_ydata(), y))
            self.ax.relim()
            self.ax.autoscale_view()
            if self.canvas_lines is not None:
                self.canvas_lines.draw_idle()
        else:
            print("Cannot plot. Array sizes do not match.")

    def update_plot_2D(self, data2d, title):
        if self.image_2D is None:
            return
        self.ax_2D.title.set_text(title)
        self.image_2D.set_data(data2d)
        self.map_data_current = data2d
        if self.canvas_2D is not None:
            self.canvas_2D.draw_idle()

    # --------------------------------------------------------------------- #
    #  Serial port handling                                                 #
    # --------------------------------------------------------------------- #
    def on_list_ports(self):
        self.refresh_serial_ports(manual=True)

    def refresh_serial_ports(self, manual=False):
        ports = list(serial.tools.list_ports.comports())
        port_values = [p.device for p in ports]

        self.serial_port_combo.configure(values=port_values)
        current_value = self.input_serial_port.get().strip()
        if port_values:
            if current_value not in port_values:
                self.input_serial_port.set(port_values[0])
            self.status_port_var.set(f"Port: {self.input_serial_port.get()}")
        else:
            self.input_serial_port.set("")
            self.status_port_var.set("Port: -")

        if manual:
            self.log_message("===================")
            if port_values:
                for port in ports:
                    self.log_message(f"{port.device} : {port.description}")
            else:
                self.log_message("No serial ports found")

        if not manual:
            self.root.after(2000, self.refresh_serial_ports)

    # --------------------------------------------------------------------- #
    #  Measurement control                                                  #
    # --------------------------------------------------------------------- #
    def on_run(self):
        if self.running:
            self.log_message("RUN ignored: already running.")
            return
        self.running = True
        self.cancel_event.clear()
        self.run_btn.config(state="disabled")
        self.finish_btn.config(state="normal")
        self._set_status(state="Running", badge=Palette.BADGE_OK)
        self.panel_config.set_badge(Palette.BADGE_OK)

        measurement_time_s = int(self.meas_time_var.get())
        disFB_time_s = int(self.disfb_time_var.get())
        sampling_period_ms = int(self.sampling_period_var.get())
        AZERO_time_s = 5

        f_sampling = 1 / (sampling_period_ms * 1e-3)
        AZERO_sample = int(AZERO_time_s * f_sampling)
        disFB_sample = int(disFB_time_s * f_sampling)

        measurement_configuration = dict(
            sampling_period_ms=sampling_period_ms,
            measurement_time_s=measurement_time_s,
            f_sampling=f_sampling,
            AZERO_time_s=AZERO_time_s,
            AZERO_sample=AZERO_sample,
            disFB_time_s=disFB_time_s,
            disFB_sample=disFB_sample,
            do_gain_calib=self.do_gain_calib.get(),
            apply_gain_calib=self.apply_gain_calib.get(),
            gain_calib_name=self.filename_gain_var.get(),
            save_measurement=self.save_measurement_var.get(),
        )

        plot_configuration = dict(
            pixels_to_plot=self.get_selected_pixels(),
            plot_refresh_time_ms=int(self.plot_refresh_var.get()),
            line_ylim=int(self.line_ylim_var.get()),
            map2D_maxval=int(self.map_maxval_var.get()),
            map2D_minval=-1 * int(self.map_maxval_var.get()),
            enable_plot=True,
        )

        filepath = "./measurements/"
        os.makedirs(filepath, exist_ok=True)
        file_to_save = filepath + self.filename_var.get()

        serial_port = self.input_serial_port.get().strip()
        if not serial_port:
            self.log_message("No serial port selected.")
            self.running = False
            self.run_btn.config(state="normal")
            self.finish_btn.config(state="disabled")
            self._set_status(state="Idle", badge=Palette.BADGE_IDLE)
            self.panel_config.set_badge(Palette.BADGE_IDLE)
            return

        if serial_port[:3] == "COM":
            ser = serial_connect(port=serial_port, baudrate=DEFAULT_BAUDRATE)
        else:
            ser = serial_connect(port="/dev/" + serial_port, baudrate=DEFAULT_BAUDRATE)

        self._set_status(connected=True)
        do_chip_reset(ser)
        do_array_config(ser)

        t = threading.Thread(target=self.measurement_worker,
                             args=(ser, measurement_configuration, plot_configuration, file_to_save),
                             daemon=True)
        t.start()

    def on_finish_cancel(self):
        """FINISH acts as Cancel: signal the worker to stop."""
        if self.running and not self.cancel_event.is_set():
            self.cancel_event.set()
        else:
            self.log_message("No measurement to stop.")

    def on_plot(self):
        if self.active_plot:
            self.active_plot = False
            self.log_message("set plot non interactive")
            self.plot_btn.config(text="edit PLOT")
            self.panel_line.set_badge(Palette.BADGE_WARN)
            self.panel_map.set_badge(Palette.BADGE_WARN)
        else:
            self.active_plot = True
            self.log_message("set plot interactive")
            self.plot_btn.config(text="stop PLOT")
            self.panel_line.set_badge(Palette.BADGE_OK)
            self.panel_map.set_badge(Palette.BADGE_OK)

    def measurement_worker(self, ser, measurement_configuration, plot_configuration, file_to_save):
        """Runs in a background thread; never touch Tk widgets directly here --
        always marshal UI updates through ``root.after``."""
        df_out = None
        device = None
        amplitudes = None
        initial_delay = None
        pulse_period = None
        temp_measurement_path = None
        extension = ".dataframe"
        try:
            self.log_message("===================")
            self.log_message("Measurement started")

            df_in = create_dataframe()

            if measurement_configuration["do_gain_calib"]:
                self.log_message("Gain calibration measurement")

                amplitudes = np.array([20e-3, 40e-3, 60e-3, 80e-3, -20e-3, -40e-3, -60e-3, -80e-3])
                pulse_period = 10
                sampling_period = 0.5
                initial_delay = measurement_configuration["disFB_time_s"] + 5
                measurement_time_s = initial_delay + pulse_period * len(amplitudes)
                measurement_configuration["measurement_time_s"] = measurement_time_s

                dwf = DwfLibrary()
                device_count = dwf.deviceEnum.enumerateDevices()
                self.log_message(f"Number of Digilent Waveforms devices found: {device_count}")
                for n in range(device_count):
                    self.log_message(f" - {dwf.deviceEnum.deviceName(n)}")

                device = openDwfDevice(dwf, score_func=maximize_analog_out_buffer_size)
                self.log_message("Connected to Analog Discovery 2")

                t_final = pulse_period * len(amplitudes) + initial_delay
                timestamps = np.arange(0.0, float(t_final), float(sampling_period), dtype=float)
                waveform = custom_pulsed_waveform(timestamps, amplitudes, pulse_period,
                                                  initial_delay=initial_delay)
                analog_out_custom_waveform(device.analogOut, waveform, t_final)

            df_out = self.perform_measurement(ser, df_in, measurement_configuration, plot_configuration)

        except Exception as e:  # noqa: BLE001  (surface any hardware error in the console)
            self.log_message(f"ERROR: {e}")
            self.root.after(0, lambda: self._set_status(state="Error", badge=Palette.BADGE_ERR))

        finally:
            save_measurement = measurement_configuration.get("save_measurement", True)

            if df_out is not None:
                if save_measurement or measurement_configuration.get("do_gain_calib"):
                    target_file = file_to_save
                    if not save_measurement and measurement_configuration.get("do_gain_calib"):
                        target_file = file_to_save + "_temp_gain_analysis_" + str(int(time.time()))
                        temp_measurement_path = target_file + extension
                    elif os.path.isfile(target_file + extension):
                        target_file = target_file + "_" + str(int(time.time()))
                        self.log_message(f'File exists! Saving as "{target_file}" ...')

                    df_out.to_pickle(target_file + extension)
                    self.log_message(f'"{target_file + extension}" stored in disk')
                else:
                    self.log_message("Measurement file save disabled by user.")

            if measurement_configuration["do_gain_calib"] and device is not None:
                device.close()
                print("Analog discovery closed")

            ser.close()
            print("Serial port closed")

            if (measurement_configuration["do_gain_calib"] and df_out is not None and amplitudes is not None):
                analysis_source = file_to_save
                if temp_measurement_path is not None:
                    analysis_source = temp_measurement_path[:-len(extension)]
                gain_analysis(analysis_source, amplitudes, initial_delay, pulse_period, plot_configuration)
                if temp_measurement_path is not None and os.path.isfile(temp_measurement_path):
                    os.remove(temp_measurement_path)

            self.root.after(0, self._measurement_cleanup)

    def _measurement_cleanup(self):
        self.running = False
        self.run_btn.config(state="normal")
        self.finish_btn.config(state="disabled")
        self._set_status(state="Idle", connected=False, badge=Palette.BADGE_IDLE)
        self.panel_config.set_badge(Palette.BADGE_IDLE)

    def perform_measurement(self, ser, df_in, measurement_configuration, plot_configuration):
        df_out = df_in

        measurement_time_s = measurement_configuration["measurement_time_s"]
        sampling_period_ms = measurement_configuration["sampling_period_ms"]
        f_sampling = measurement_configuration["f_sampling"]
        AZERO_sample = measurement_configuration["AZERO_sample"]
        disFB_sample = measurement_configuration["disFB_sample"]
        apply_gain_calib = measurement_configuration["apply_gain_calib"]
        gain_calib_name = measurement_configuration["gain_calib_name"]

        enable_plot = plot_configuration["enable_plot"]
        plot_refresh_time_ms = plot_configuration["plot_refresh_time_ms"]

        self.root.after(0, lambda: self.start_plot(measurement_configuration, plot_configuration))

        gain_positive = np.ones(ARRAY_SIZE)
        gain_negative = np.ones(ARRAY_SIZE)

        if apply_gain_calib:
            if os.path.isfile(gain_calib_name):
                self.log_message(f'Applying gain calibration from: "{gain_calib_name}"')
                df_gains = pd.read_pickle(gain_calib_name)
                gain_positive = df_gains.Gains_positive.to_numpy()
                gain_negative = df_gains.Gains_negative.to_numpy()
            else:
                self.log_message(f'File "{gain_calib_name}" does not exist!!')
                measurement_configuration["apply_gain_calib"] = 0
                apply_gain_calib = 0

        plot_Nlines = MAX_PLOTTED_PIXELS + 1
        number_of_samples = int(measurement_time_s / (sampling_period_ms * 1e-3))

        array_events = np.zeros((number_of_samples, ARRAY_SIZE))
        array_eps = np.zeros((number_of_samples, ARRAY_SIZE))
        array_milivolts = np.zeros((number_of_samples, ARRAY_SIZE))
        array_OF = np.zeros((number_of_samples))

        timestamps = np.arange(0, measurement_time_s, sampling_period_ms * 1e-3)
        stop_time_s = measurement_time_s

        # acquisition configuration
        s_L = sampling_period_ms & 0xFF
        s_H = (sampling_period_ms & 0xFF00) >> 8
        do_acq_config(ser)
        ser.write(bytes([s_L]))
        ser.write(bytes([s_H]))

        nsample = 0
        print("=== START ACQUISITION ===")
        do_acq_loop(ser)

        while True:
            array_OF[nsample] = read_overflow(ser)
            events_8b, array_events[nsample] = read_mem(ser, size=ARRAY_SIZE, acq=True, debug=False)

            if nsample == AZERO_sample:
                ser.write(bytes([ord("A")]))  # do AZERO
                print("do AZERO")
            elif nsample == disFB_sample:
                ser.write(bytes([ord("F")]))  # disable FB
                print("disable FB")
            else:
                ser.write(bytes([ord("C")]))  # continue acquisition

            array_eps[nsample] = array_events[nsample] * f_sampling

            if apply_gain_calib:
                array_milivolts[nsample] = np.array([
                    linear_function_inverse(data, gain_pos, gain_neg)
                    for data, gain_pos, gain_neg in zip(array_eps[nsample], gain_positive, gain_negative)
                ])

            if enable_plot and ((timestamps[nsample] * 1e3 % plot_refresh_time_ms) == 0):
                data_to_plot = array_milivolts[nsample] if apply_gain_calib else array_eps[nsample]

                selected_pixels_live = self.get_selected_pixels()
                selected_values = list(data_to_plot[selected_pixels_live]) if selected_pixels_live else []
                while len(selected_values) < MAX_PLOTTED_PIXELS:
                    selected_values.append(np.nan)

                timestamps_to_plot = np.repeat(timestamps[nsample], plot_Nlines)
                self.root.after(0, lambda ts=timestamps_to_plot,
                                values=np.append(selected_values, array_OF[nsample]):
                                self.update_plot(ts, values))

                data_to_plot_2D = reshape_data_smartloc2(data_to_plot)
                self.root.after(0, lambda data2d=data_to_plot_2D, stamp=timestamps[nsample]:
                                self.update_plot_2D(data2d, f"SmartLoC2 (t = {str(stamp)} s)"))

            if self.cancel_event.is_set():
                self.log_message("Measurement canceled by user")
                break

            if nsample == number_of_samples - 1:
                break
            nsample += 1

        ser.write(bytes([ord("S")]))  # stop acquisition
        self.log_message("Measurement finished")

        measurement_configuration["stop_time_s"] = round(stop_time_s, 3)
        if np.sum(array_OF):
            self.log_message("OJO! Overflow detected!")

        df_out.at[0, "Timestamps"] = timestamps
        df_out.at[0, "Measurement_configuration"] = measurement_configuration

        for pix_index in range(ARRAY_SIZE):
            events = np.transpose(array_events)[pix_index]
            eps = np.transpose(array_eps)[pix_index]
            mV = np.transpose(array_milivolts)[pix_index]
            df_out.at[pix_index, "Events"] = events.astype(int)
            df_out.at[pix_index, "Eps"] = eps.astype(int)
            if apply_gain_calib:
                df_out.at[pix_index, "Milivolts"] = mV.astype(float)

        print("Data stored to the dataframe")
        return df_out

    # --------------------------------------------------------------------- #
    #  Lifecycle                                                            #
    # --------------------------------------------------------------------- #
    def on_exit(self):
        """Exit the app; if a measurement is running, cancel it first."""
        self.dock.save_layout()
        if self.running and not self.cancel_event.is_set():
            self.log_message("Exit requested: canceling measurement...")
            self.cancel_event.set()
            self.root.after(150, self.root.destroy)
        else:
            self.root.destroy()

    def run(self):
        self.log_message("Welcome to SmartLoC2 GUI :)")
        self.root.mainloop()


# =========================================================================== #
#  2D plot helper (also used by the gain analysis)                            #
# =========================================================================== #

def plot_2D_plt(data_2d, title="", minval=None, maxval=None, annot=False,
                fmt=None, annot_size=10, barlabel=None):
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    ax.title.set_text(title)

    data2d = np.ones((GRID_SIDE, GRID_SIDE))
    data2d[0][0] = maxval
    data2d[0][1] = minval
    image = ax.imshow(data2d, cmap="coolwarm")
    image.set_data(data_2d)

    if annot:
        for y in range(GRID_SIDE):
            for x in range(GRID_SIDE):
                plt.text(x, y, f"{data_2d[y, x]:{fmt}}", ha="center", va="center", fontsize=annot_size)

    plt.colorbar(image, label=barlabel)
    return fig


# =========================================================================== #
#  Entry point                                                                #
# =========================================================================== #

if __name__ == "__main__":
    app = SmartLoC2App()
    app.run()
