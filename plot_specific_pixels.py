#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Attempt date: 22/01/2026 — baseline-subtracted version

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

# Always use this exact dataframe from the Digital_Detection folder.
# Change only this filename if you want to plot another measurement.
DATAFRAME_FILE = os.path.join(BASE_DIR, "filename_1769101165.dataframe")

# === PIXELS TO PLOT ==========================================================
# Edit the lists below to choose which pixels are plotted.
# - `pixels_bead`: pixels shown as bead / greenish traces
# - `pixels_no_bead`: pixels shown as no-bead / grayish traces
# Keep pixel IDs as integers in the 0..255 range.
pixels_bead    = [79, 193, 178, 161, 177, 158, 174, 238, 107, 106, 130, 176]
pixels_no_bead = [144, 35, 184, 16]

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

# Plot windows
plot_windows = [
    (200, 280),
]

# High-quality export
save_plots = True
save_formats = ['svg', 'pdf']
save_basename = 'grafik'

# Well opened times (s)
well_opened_times = [230]

# Baseline: average the 10 s interval immediately before each well-open event
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

    if has_nonfinite:
        excluded_pixels.add(p)
    if np.isfinite(span_mV) and span_mV > span_high_threshold_mV:
        excluded_pixels.add(p)
    if diff_std_mV > diff_std_threshold_mV:
        excluded_pixels.add(p)
    if np.isfinite(span_mV) and span_mV < span_flat_threshold_mV:
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


# === BASELINE SUBTRACTION & PLOTTING =========================================

for (t_start, t_end), t_open in zip(plot_windows, well_opened_times):
    # Baseline mask: 10 s immediately before the well-open event
    bl_mask = (timestamps >= t_open - baseline_duration) & (timestamps < t_open)

    mask = (timestamps >= t_start) & (timestamps <= t_end)
    t_window = timestamps[mask]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('ΔmV')
    ax.grid(True, alpha=0.4)
    ax.set_title(f'Digital Detection | SmartLoC 2 | Att. 22nd of Jan. 2026  |  '
                 f't [s] = [{t_start}, {t_end}]')
    ax.set_xlim(t_start, t_end)

    for p, color, ls, label in zip(all_pixels, all_colors, all_styles, all_labels):
        baseline = np.mean(pixel_data[p][bl_mask])
        y_plot = pixel_data[p][mask] - baseline
        ax.plot(t_window, y_plot,
                color=color, linestyle=ls, label=label, linewidth=1.4)

    ax.autoscale(enable=True, axis='y')
    fig.canvas.draw()

    # Mark well-opened event
    if t_start <= t_open <= t_end:
        ax.axvline(t_open, color='black', linestyle='--', linewidth=1.2, alpha=0.85)
        ymin, ymax = ax.get_ylim()
        ax.text(t_open + (t_end - t_start) * 0.005, ymin + (ymax - ymin) * 0.02,
                'well opened', color='black', fontsize=8, va='bottom', rotation=90)

    ax.legend(ncols=2, fontsize=8, loc='upper left')
    plt.tight_layout()

    if save_plots:
        for fmt in save_formats:
            out_name = f'{save_basename}_{int(t_start)}_{int(t_end)}.{fmt}'
            fig.savefig(out_name, format=fmt, bbox_inches='tight')
            print(f'Saved plot: {out_name}')

plt.show()