#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
import sys
# Ensure repository root is on sys.path so imports work when running from this subfolder
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from smartloc2_functions import reshape_data_smartloc2


# === CONFIG ==================================================================
extension = ".dataframe"
# Prefer Digital_Detection folder for selecting .dataframe files
filename_root = r"C:\Users\MErtl\Desktop\Digital ISFET Detection\SMARTLOC2\PYTHON\Digital_Detection\\"
filename = "filename_1769101165"

# Rule-based exclusion thresholds (mV)
span_high_threshold_mV = 500.0
diff_std_threshold_mV = 500.0
span_flat_threshold_mV = 10.0

# 2D well-map plotting
array_rows = 16
array_cols = 16
plot_title = "Excluded vs Included Wells (Rule-Based)"


# === HELPERS =================================================================
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


# === LOAD ====================================================================
# If a .dataframe exists in filename_root, pick it automatically to avoid GUI blocking
import glob
candidates = glob.glob(os.path.join(filename_root, "*.dataframe"))
file_path = None
if candidates:
    # prefer exact filename if provided
    target = os.path.join(filename_root, filename + extension)
    if os.path.exists(target):
        file_path = target
    else:
        file_path = candidates[0]
else:
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a .dataframe file",
        initialdir=filename_root,
        filetypes=[("Dataframe files", "*.dataframe"), ("All files", "*.*")],
    )
    root.destroy()

if not file_path:
    raise ValueError("No file selected. Exiting.")

df = pd.read_pickle(file_path)
filename_root, file_name = os.path.split(file_path)
filename, _ = os.path.splitext(file_name)

n_pixels = len(df)
pixel_ids = list(range(n_pixels))


# === RULE-BASED DETECTION ====================================================
excluded_by_nonfinite = set()
excluded_by_span_high = set()
excluded_by_diff_std = set()
excluded_by_flat = set()

for p in pixel_ids:
    mv = np.asarray(df.Milivolts.iloc[p], dtype=float)

    has_nonfinite = bool(np.any(~np.isfinite(mv)))
    span_mV = _safe_span_mV(mv)
    diff_std_mV = _safe_diff_std_mV(mv)

    if has_nonfinite:
        excluded_by_nonfinite.add(p)
    if np.isfinite(span_mV) and span_mV > span_high_threshold_mV:
        excluded_by_span_high.add(p)
    if diff_std_mV > diff_std_threshold_mV:
        excluded_by_diff_std.add(p)
    if np.isfinite(span_mV) and span_mV < span_flat_threshold_mV:
        excluded_by_flat.add(p)

excluded_set = (
    excluded_by_nonfinite
    | excluded_by_span_high
    | excluded_by_diff_std
    | excluded_by_flat
)
pred_removed = sorted(excluded_set)


# === REPORT ===================================================================
print("=== Automatic Defective Pixel Detection (Rule-Based) ===")
print(f"File: {file_name}")
print(f"Pixels total: {n_pixels}")
print("Exclusion criteria:")
print("1) Signal contains NaN or Inf")
print(f"2) Voltage span > {span_high_threshold_mV:.1f} mV")
print(f"3) Sample-to-sample fluctuation (diff std) > {diff_std_threshold_mV:.1f} mV")
print(f"4) Flat signal (span < {span_flat_threshold_mV:.1f} mV)")
print("")

print(f"Excluded pixels ({len(pred_removed)}):")
print(pred_removed)
print("")

print("Exclusion counts by criterion:")
print(f"NaN/Inf:            {len(excluded_by_nonfinite)}")
print(f"Span > threshold:   {len(excluded_by_span_high)}")
print(f"Diff std > threshold: {len(excluded_by_diff_std)}")
print(f"Flat span < threshold: {len(excluded_by_flat)}")


# === 2D ARRAY PLOT ===========================================================
if array_rows * array_cols != n_pixels:
    raise ValueError(
        f"Array shape mismatch: rows*cols={array_rows * array_cols} but n_pixels={n_pixels}."
    )

excluded_set = set(pred_removed)
included_set = set(pixel_ids) - excluded_set

well_number_map = reshape_data_smartloc2(np.arange(n_pixels))
well_positions = {}
for row in range(well_number_map.shape[0]):
    for col in range(well_number_map.shape[1]):
        well_positions[int(well_number_map[row, col])] = (row, col)

excluded_coords = [well_positions[p] for p in sorted(excluded_set)]
included_coords = [well_positions[p] for p in sorted(included_set)]

excluded_rows = np.array([rc[0] for rc in excluded_coords], dtype=int)
excluded_cols = np.array([rc[1] for rc in excluded_coords], dtype=int)
included_rows = np.array([rc[0] for rc in included_coords], dtype=int)
included_cols = np.array([rc[1] for rc in included_coords], dtype=int)

fig, ax = plt.subplots(figsize=(8, 8))

if included_rows.size > 0:
    ax.scatter(
        included_cols,
        included_rows,
        s=90,
        c="#4C9F70",
        edgecolors="white",
        linewidths=0.5,
        marker="s",
        label=f"Included ({included_rows.size})",
    )

if excluded_rows.size > 0:
    ax.scatter(
        excluded_cols,
        excluded_rows,
        s=120,
        c="#D1495B",
        edgecolors="black",
        linewidths=0.6,
        marker="X",
        label=f"Excluded ({excluded_rows.size})",
    )

ax.set_title(plot_title)
ax.set_xlabel("Well Layout X")
ax.set_ylabel("Well Layout Y")
ax.set_xticks(np.arange(array_cols))
ax.set_yticks(np.arange(array_rows))
ax.set_xlim(-0.5, array_cols - 0.5)
ax.set_ylim(-0.5, array_rows - 0.5)
ax.invert_yaxis()
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_aspect("equal", adjustable="box")

for well_id, (row, col) in well_positions.items():
    ax.text(
        col,
        row,
        str(well_id),
        ha="center",
        va="center",
        fontsize=6,
        color="black",
        bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=0.2),
    )

ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
