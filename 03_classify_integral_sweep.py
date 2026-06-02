#!/usr/bin/env python3
"""
Script 8: DC-Subtracted Rectified Integral – Multi-Parameter Sweep
====================================================================
Classifies wells by:
  1. Estimating the DC baseline from a window before the opening event
  2. Subtracting the baseline, rectifying (abs), integrating in the
     opening window
  3. Using the max integral across both opening events as the feature

Sweep parameters:
  • Baseline window duration (s)
  • Integration window duration (s)
  • Integration window start offset (s)
  • Classification threshold (mV·s)

Uses exclusion list from 06_pixel_spatial_classification.json.

Input:  data/raw_measurement.dataframe
        output/pixel_spatial_classification.json
Output: output/rectified_integral_sweep_results.json
        output/rectified_integral_best_per_pixel.csv
"""

import numpy as np
import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Configuration ──────────────────────────────────────────────────────────
INPUT_DATA = os.path.join(BASE_DIR, "data", "raw_measurement.dataframe")
INPUT_CLASS = os.path.join(BASE_DIR, "output", "pixel_spatial_classification.json")
OUTPUT_JSON = os.path.join("output", "rectified_integral_sweep_results.json")
OUTPUT_CSV = os.path.join("output", "rectified_integral_best_per_pixel.csv")

F_SAMPLING = 2.0
N_SAMPLES = 1800
DT = 1.0 / F_SAMPLING
TIMESTAMPS = np.arange(N_SAMPLES) * DT

OPEN_TIMES = [400.0, 640.0]

BEAD_WELLS_ALL = {79, 158, 174, 238, 253, 107, 106, 227, 130, 178, 161, 177, 193, 176}

# ── Sweep ranges ────────────────────────────────────────────────────────────
BASELINE_DURATIONS = [15, 20, 25, 30]           # seconds before open
INTEG_START_OFFSETS = np.arange(-5, 10.1, 5)    # -5, 0, 5, 10  (rel. to open)
INTEG_DURATIONS = [15, 20, 25, 30]              # seconds of integration
THR_MIN, THR_MAX, THR_STEP = 5.0, 500.0, 5.0


# ─── Load ───────────────────────────────────────────────────────────────────
# Prefer any .dataframe in the script folder (Digital_Detection)
ALT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_measurement.dataframe")
if os.path.exists(ALT_INPUT):
    INPUT_DATA = ALT_INPUT
else:
    import glob
    candidates = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.dataframe"))
    if candidates:
        INPUT_DATA = candidates[0]

df = pd.read_pickle(INPUT_DATA)
with open(INPUT_CLASS) as f:
    classif = json.load(f)

excluded = set(classif["excluded_pixels"])
included = classif["included_pixels"]
bead_included = set(classif["bead_wells_included"])
n_incl = len(included)

print(f"Included pixels: {n_incl}, bead wells in data: {len(bead_included)}")


# ─── Helpers ────────────────────────────────────────────────────────────────
def compute_metrics(predicted_set, actual_set, total_n):
    tp = len(predicted_set & actual_set)
    fp = len(predicted_set - actual_set)
    fn = len(actual_set - predicted_set)
    tn = total_n - tp - fp - fn
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    acc = (tp + tn) / total_n if total_n > 0 else 0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, accuracy=acc)


# ─── Sweep ──────────────────────────────────────────────────────────────────
best_f1 = -1
best_config = None
thresholds = np.arange(THR_MIN, THR_MAX + THR_STEP / 2, THR_STEP)
n_configs = 0

for bl_dur in BASELINE_DURATIONS:
    for ig_off in INTEG_START_OFFSETS:
        for ig_dur in INTEG_DURATIONS:
            # Compute feature for each included pixel
            features = {}
            for p in included:
                mv = np.asarray(df.iloc[p]["Milivolts"], dtype=float)
                max_integral = 0.0

                for t_open in OPEN_TIMES:
                    # Baseline window: [open - bl_dur, open]
                    bl_start = t_open - bl_dur
                    bl_end = t_open
                    bl_mask = (TIMESTAMPS >= bl_start) & (TIMESTAMPS < bl_end)
                    baseline = np.mean(mv[bl_mask]) if bl_mask.sum() > 0 else 0.0

                    # Integration window
                    ig_start = t_open + ig_off
                    ig_end = ig_start + ig_dur
                    ig_mask = (TIMESTAMPS >= ig_start) & (TIMESTAMPS <= ig_end)
                    if ig_mask.sum() > 0:
                        rectified = np.abs(mv[ig_mask] - baseline)
                        integral = np.trapezoid(rectified, dx=DT)
                        max_integral = max(max_integral, integral)

                features[p] = max_integral

            # Threshold sweep
            for thr in thresholds:
                predicted = {p for p, v in features.items() if v > thr}
                m = compute_metrics(predicted, bead_included, n_incl)
                n_configs += 1
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    best_config = {
                        "baseline_duration_s": float(bl_dur),
                        "integ_start_offset_s": float(ig_off),
                        "integ_duration_s": float(ig_dur),
                        "threshold_mVs": round(float(thr), 2),
                        **m,
                    }
                    best_features = dict(features)

print(f"\nSweep complete: {n_configs} configurations tested")
print(f"Best F1 = {best_f1:.4f}")
print(f"Best config: baseline_dur={best_config['baseline_duration_s']}s, "
      f"integ_offset={best_config['integ_start_offset_s']}s, "
      f"integ_dur={best_config['integ_duration_s']}s, "
      f"threshold={best_config['threshold_mVs']} mV·s")
print(f"  TP={best_config['tp']} FP={best_config['fp']} "
      f"FN={best_config['fn']} TN={best_config['tn']}")
print(f"  Sensitivity={best_config['sensitivity']:.4f} "
      f"Specificity={best_config['specificity']:.4f} "
      f"Precision={best_config['precision']:.4f}")


# ─── Per-pixel results at best config ──────────────────────────────────────
thr_best = best_config["threshold_mVs"]
per_pixel = []
for p in included:
    feat_val = best_features[p]
    predicted_pos = feat_val > thr_best
    actual_pos = p in bead_included
    cat = classif["classification"][str(p)]
    per_pixel.append({
        "pixel_id": p,
        "category": cat["category"],
        "chebyshev_dist": cat["chebyshev_dist_to_nearest_bead"],
        "rectified_integral_mVs": round(feat_val, 4),
        "predicted_positive": predicted_pos,
        "actual_positive": actual_pos,
        "result": ("TP" if predicted_pos and actual_pos else
                   "FP" if predicted_pos and not actual_pos else
                   "FN" if not predicted_pos and actual_pos else "TN"),
    })

pp_df = pd.DataFrame(per_pixel)

# ─── Save ───────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

out = {
    "method": "rectified_integral",
    "best_config": best_config,
    "best_f1": best_f1,
    "sweep_params": {
        "baseline_durations_s": BASELINE_DURATIONS,
        "integ_start_offsets_s": [float(x) for x in INTEG_START_OFFSETS],
        "integ_durations_s": INTEG_DURATIONS,
        "threshold_range": [float(THR_MIN), float(THR_MAX), float(THR_STEP)],
    },
    "per_pixel": per_pixel,
    "metrics": {
        "sensitivity": best_config["sensitivity"],
        "specificity": best_config["specificity"],
        "precision": best_config["precision"],
        "f1": best_config["f1"],
        "accuracy": best_config["accuracy"],
    },
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else None)

pp_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_JSON}")
print(f"Saved {OUTPUT_CSV}")


# ─── 2D Prediction Map ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_MAP = os.path.join("output", "map_prediction_rectified_integral.png")

ARRAY_ROWS, ARRAY_COLS = 16, 16

def reshape_data_smartloc2(data_in):
    """Map 1-D pixel indices to the physical 16×16 grid layout."""
    d = np.copy(data_in)
    d2 = d.reshape(-1, 16)
    out = np.transpose(d2)
    d2[0::1, :] = d2[0::1, ::-1]
    return out

well_map = reshape_data_smartloc2(np.arange(256))
pixel_to_pos = {}
for r in range(ARRAY_ROWS):
    for c in range(ARRAY_COLS):
        pixel_to_pos[int(well_map[r, c])] = (r, c)

result_lookup = {px["pixel_id"]: px["result"] for px in per_pixel}

RESULT_COLORS = {
    "TP": "#63BE7B",
    "TN": "#BCE2E7",
    "FP": "#DD6974",
    "FN": "#FFC553",
}
EXCLUDED_COLOR = "#D4D1CA"

fig, ax = plt.subplots(figsize=(10, 10))

for r in range(ARRAY_ROWS):
    for c in range(ARRAY_COLS):
        pid = int(well_map[r, c])
        if pid in excluded:
            color = EXCLUDED_COLOR
            txt_color = "#7A7974"
        else:
            res = result_lookup.get(pid, "TN")
            color = RESULT_COLORS.get(res, "#F7F6F2")
            txt_color = "#FFFFFF" if res in ("TP", "FP") else "#28251D"

        rect = mpatches.FancyBboxPatch(
            (c - 0.45, r - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.02", facecolor=color,
            edgecolor="#7A7974", linewidth=0.5)
        ax.add_patch(rect)

        fontweight = "bold" if result_lookup.get(pid) in ("TP", "FP", "FN") else "normal"
        fontsize = 6.5 if pid not in excluded else 5.5
        ax.text(c, r, str(pid), ha="center", va="center",
                fontsize=fontsize, color=txt_color, fontweight=fontweight)

ax.set_xlim(-0.6, ARRAY_COLS - 0.4)
ax.set_ylim(-0.6, ARRAY_ROWS - 0.4)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_xticks(range(ARRAY_COLS))
ax.set_yticks(range(ARRAY_ROWS))
ax.set_xlabel("Grid Column", fontsize=10)
ax.set_ylabel("Grid Row", fontsize=10)

n_tp = sum(1 for px in per_pixel if px["result"] == "TP")
n_tn = sum(1 for px in per_pixel if px["result"] == "TN")
n_fp = sum(1 for px in per_pixel if px["result"] == "FP")
n_fn = sum(1 for px in per_pixel if px["result"] == "FN")

ax.set_title("Prediction Map \u2014 Rectified Integral (Best Config)",
             fontsize=13, fontweight="bold", pad=12)

legend_patches = [
    mpatches.Patch(facecolor=RESULT_COLORS["TP"], edgecolor="#7A7974", label=f"TP ({n_tp})"),
    mpatches.Patch(facecolor=RESULT_COLORS["TN"], edgecolor="#7A7974", label=f"TN ({n_tn})"),
    mpatches.Patch(facecolor=RESULT_COLORS["FP"], edgecolor="#7A7974", label=f"FP ({n_fp})"),
    mpatches.Patch(facecolor=RESULT_COLORS["FN"], edgecolor="#7A7974", label=f"FN ({n_fn})"),
    mpatches.Patch(facecolor=EXCLUDED_COLOR, edgecolor="#7A7974", label=f"Excluded ({len(excluded)})"),
]
ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.01, 1),
          fontsize=9, frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig(OUTPUT_MAP, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {OUTPUT_MAP}")
