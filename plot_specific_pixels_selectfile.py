#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Attempt date: 22/01/2026 — baseline-subtracted version

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os


def _fill_nonfinite_1d(data):
    """Replace NaN/Inf in 1D data using linear interpolation."""
    arr = np.asarray(data, dtype=float).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not np.any(finite):
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def butter_LPF(data, order, f_cutoff, f_sampling):
    """Zero-phase Butterworth low-pass filter for 1D data."""
    data = _fill_nonfinite_1d(data)
    if data.size < 5:
        return data
    sos = signal.butter(order, f_cutoff, fs=f_sampling, output='sos')
    try:
        return signal.sosfiltfilt(sos, data)
    except ValueError:
        return data


def savgol_denoise(data, window_length, polyorder):
    """Savitzky-Golay smoothing filter for noise removal."""
    data = _fill_nonfinite_1d(data)

    if data.size < 3:
        return data

    win = min(window_length, data.size if data.size % 2 == 1 else data.size - 1)
    if win < 3:
        return data

    poly = min(polyorder, win - 1)

    try:
        return signal.savgol_filter(data, win, poly)
    except np.linalg.LinAlgError:
        return data


# === CONFIG ==================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def choose_dataframe_file(initial_dir=BASE_DIR):
    # Try to use tkinter file dialog; fall back to console input
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        tk = None

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        return sys.argv[1]

    if tk is not None:
        try:
            root = tk.Tk()
            root.withdraw()
            fname = filedialog.askopenfilename(title='Select measurement .dataframe', initialdir=initial_dir,
                                               filetypes=[('DataFrame', '*.dataframe'), ('All files', '*.*')])
            root.destroy()
            if fname:
                return fname
        except Exception:
            pass

    # Fallback: ask in console
    try:
        fname = input(f'Enter path to .dataframe file (default dir: {initial_dir}): ').strip()
    except Exception:
        fname = ''
    if not fname:
        print('No file selected. Exiting.')
        sys.exit(1)
    return fname


# Use interactive selection (or CLI arg) instead of hardcoded filename
DATAFRAME_FILE = choose_dataframe_file()

# === PIXELS TO PLOT ==========================================================
# Default pixel lists (can be overridden interactively or via --pixels)
default_pixels_bead    = [79, 193, 178, 161, 177, 158, 174, 238, 107, 106, 130, 176]
default_pixels_no_bead = [144, 35, 184, 16]


def choose_pixels(default_list):
    """Return a list of pixel IDs from CLI, tkinter dialog, or console input."""
    # Check for a --pixels option (comma-separated)
    pixels_arg = None
    for i, a in enumerate(sys.argv):
        if a == '--pixels' and i + 1 < len(sys.argv):
            pixels_arg = sys.argv[i + 1]
            break
        if a.startswith('--pixels='):
            pixels_arg = a.split('=', 1)[1]
            break

    if pixels_arg:
        try:
            return [int(x) for x in pixels_arg.split(',') if x.strip()]
        except Exception:
            print('Failed to parse --pixels argument; falling back to interactive input')

    # Try tkinter simple dialog
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        s = simpledialog.askstring('Pixels', 'Enter comma-separated bead pixel IDs', initialvalue=','.join(map(str, default_list)))
        root.destroy()
        if s:
            return [int(x) for x in s.split(',') if x.strip()]
    except Exception:
        pass

    # Console fallback
    try:
        s = input(f'Enter comma-separated bead pixel IDs (default: {default_list}): ').strip()
    except Exception:
        s = ''
    if not s:
        return list(default_list)
    try:
        return [int(x) for x in s.split(',') if x.strip()]
    except Exception:
        print('Invalid input; using default pixel list')
        return list(default_list)

# Get pixels to plot (interactive or CLI)
pixels_bead = choose_pixels(default_pixels_bead)
pixels_no_bead = list(default_pixels_no_bead)

do_LPF    = True
f_LPF     = 0.25      # Hz
lpf_order = 2

do_savgol       = True
savgol_window   = 51    # Must be odd; larger = more smoothing
savgol_polyorder = 3    # Polynomial order; lower = more smoothing

# Must match detect_defective_pixels_clustering.py
span_high_threshold_mV = 500.0
diff_std_threshold_mV = 50.0
span_flat_threshold_mV = 1.0

# High-quality export
save_plots = True
save_formats = ['svg', 'pdf']
save_basename = 'grafik'

# Baseline: average the first `baseline_duration` seconds of the recording
baseline_duration = 10  # s

# Colors: high-contrast distinct colors for each curve

# === LOAD ====================================================================
if not os.path.exists(DATAFRAME_FILE):
    raise FileNotFoundError(f"Dataframe not found: {DATAFRAME_FILE}")

print(f"Loaded file: {DATAFRAME_FILE}")

df_read = pd.read_pickle(DATAFRAME_FILE)

timestamps         = np.asarray(df_read.Timestamps[0], dtype=float)
measurement_config = df_read.Measurement_configuration[0]
f_sampling         = measurement_config['f_sampling']

print(f'Sampling frequency: {f_sampling} Hz')


def _safe_span_mV(signal_values):
    finite_values = signal_values[np.isfinite(signal_values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.max(finite_values) - np.min(finite_values))


def _safe_diff_std_mV(signal_values):
    finite_values = signal_values[np.isfinite(signal_values)]
    if finite_values.size < 2:
        return 0.0
    diffs = np.diff(finite_values)
    if diffs.size == 0:
        return 0.0
    return float(np.std(diffs))


# Exclude wells with the same rules used in detect_defective_pixels_clustering.py
excluded_pixels = set()
for p in range(len(df_read)):
    mv_raw = np.asarray(df_read.Milivolts[p], dtype=float)
    has_nonfinite = bool(np.any(~np.isfinite(mv_raw)))
    span_mV = _safe_span_mV(mv_raw)
    diff_std_mV = _safe_diff_std_mV(mv_raw)

    # Exclude only if the signal contains non-finite values (NaN/Inf).
    if has_nonfinite:
        excluded_pixels.add(p)

included_bead_pixels = [p for p in pixels_bead if p not in excluded_pixels]

print(f'Excluded wells by criteria: {len(excluded_pixels)}')
print(f'Bead wells total: {len(pixels_bead)}')
print(f'Bead wells included for plotting: {len(included_bead_pixels)}')
print(f'Included bead well IDs: {included_bead_pixels}')

if len(included_bead_pixels) == 0:
    raise ValueError('No bead wells left after exclusion criteria. Nothing to plot.')

# Pre-compute filtered mV for all pixels
all_pixels  = included_bead_pixels
color_map = plt.get_cmap('tab20', max(len(all_pixels), 1))
all_colors  = [color_map(i) for i in range(len(all_pixels))]
all_styles  = ['-'] * len(all_pixels)
all_labels  = [f'pix. {p} (w/ bead)' for p in all_pixels]

pixel_data = {}
for p in all_pixels:
    mv = np.asarray(df_read.Milivolts[p], dtype=float)
    if do_LPF:
        mv = butter_LPF(mv, order=lpf_order, f_cutoff=f_LPF, f_sampling=f_sampling)
    if do_savgol:
        mv = savgol_denoise(mv, window_length=savgol_window, polyorder=savgol_polyorder)
    pixel_data[p] = mv

# Pre-compute baseline mask for fallback (first `baseline_duration` seconds)
t0 = float(timestamps[0])
bl_mask = (timestamps >= t0) & (timestamps < t0 + baseline_duration)
baseline_fallback = {p: float(np.mean(pixel_data[p][bl_mask])) for p in all_pixels}


# === BASELINE SUBTRACTION & PLOTTING =========================================

# Define desired plot windows (seconds)
# User-requested ranges: 400–450 and 640–690 (will be clipped to recording)
t_min = float(np.min(timestamps))
t_max = float(np.max(timestamps))
requested_windows = [(410.0, 450.0), (640.0, 690.0)]
plot_windows = []
for (a, b) in requested_windows:
    s = max(a, t_min)
    e = min(b, t_max)
    if s < e:
        plot_windows.append((s, e))

if len(plot_windows) == 0:
    # Fallback: if none of the requested windows overlap the recording,
    # fall back to showing the full recording range.
    plot_windows = [(t_min, t_max)]

# Baseline mask: first `baseline_duration` seconds from the start
t0 = float(timestamps[0])
bl_mask = (timestamps >= t0) & (timestamps < t0 + baseline_duration)

# Well opening times corresponding to requested windows (s)
well_openings = [420.0, 660.0]

for i, (t_start, t_end) in enumerate(plot_windows):
    mask = (timestamps >= t_start) & (timestamps <= t_end)
    t_window = timestamps[mask]
    # Determine well opening time for this window (fallback to window start if not provided)
    open_t = well_openings[i] if i < len(well_openings) else t_start

    # Baseline window: 20 s prior to opening time
    base_start = open_t - 20.0
    base_end = open_t
    base_mask = (timestamps >= base_start) & (timestamps < base_end)

    # If baseline-range is empty, fallback to initial baseline_fallback
    if not np.any(base_mask):
        baseline_per_pixel = dict(baseline_fallback)
    else:
        baseline_per_pixel = {p: float(np.mean(pixel_data[p][base_mask])) for p in all_pixels}

    # Compute absolute change (span) per pixel inside this window and rank pixels
    spans = {}
    for p in all_pixels:
        y_plot = pixel_data[p][mask] - baseline_per_pixel[p]
        finite = y_plot[np.isfinite(y_plot)]
        if finite.size == 0:
            spans[p] = 0.0
        else:
            spans[p] = float(np.max(finite) - np.min(finite))

    sorted_spans = sorted(spans.items(), key=lambda x: x[1], reverse=True)
    rank_map = {p: i + 1 for i, (p, _) in enumerate(sorted_spans)}

    print('\nRanking for window: {:.1f}–{:.1f} s (baseline: {:.1f}–{:.1f} s)'.format(t_start, t_end, base_start, base_end))
    for j, (p, s) in enumerate(sorted_spans, start=1):
        print(f'{j:2d}. pixel {p:3d}: span = {s:.3f} mV')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('ΔmV')
    ax.grid(True, alpha=0.4)
    ax.set_title(f'Digital Detection | SmartLoC 2 | {int(t_start)}–{int(t_end)} s')
    ax.set_xlim(t_start, t_end)

    for p, color, ls, label in zip(all_pixels, all_colors, all_styles, all_labels):
        baseline = baseline_per_pixel[p]
        y_plot = pixel_data[p][mask] - baseline
        line = ax.plot(t_window, y_plot,
                       color=color, linestyle=ls, label=label, linewidth=1.4)

        # Add inline label near the right end of each curve for unique identification
        try:
            if t_window.size > 0 and y_plot.size > 0:
                x_pos = t_window[-1]
                # small horizontal offset (fraction of window width)
                dx = 0.01 * (t_end - t_start) if (t_end - t_start) != 0 else 0.1
                y_pos = y_plot[-1]
                # include rank in the label
                r = rank_map.get(p, '')
                ax.text(x_pos + dx, y_pos, f'{p} ({r})', color=color, fontsize=8,
                        verticalalignment='center', horizontalalignment='left', clip_on=False)
        except Exception:
            pass

    ax.autoscale(enable=True, axis='y')
    fig.canvas.draw()

    ax.legend(ncols=2, fontsize=8, loc='upper left')
    plt.tight_layout()

    if save_plots:
        for fmt in save_formats:
            out_name = f'{save_basename}_{int(t_start)}_{int(t_end)}.{fmt}'
            fig.savefig(out_name, format=fmt, bbox_inches='tight')
            print(f'Saved plot: {out_name}')

plt.show()
