#!/usr/bin/env python3
"""
Script 6: Pixel / Well Spatial Classification
===============================================
Classifies every non-excluded pixel into one of three categories:

  * Occupied       – well contains a microbead (ground truth)
  * Adjacent       – at least one 8-connected neighbour is Occupied
  * Distal         – neither Occupied nor Adjacent;
                     Chebyshev distance to nearest Occupied well is reported

Uses the same rule-based exclusion as PYTHON/detect_defective_pixels_clustering.py.

Input:  data/raw_measurement.dataframe
Output: output/pixel_spatial_classification.csv
        output/pixel_spatial_classification.json   (for downstream scripts)
        output/map_spatial_classification.png
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw_measurement.dataframe")
# Prefer any .dataframe located directly in the script folder (Digital_Detection)
ALT_INPUT = os.path.join(BASE_DIR, "raw_measurement.dataframe")
if os.path.exists(ALT_INPUT):
    INPUT_FILE = ALT_INPUT
else:
    import glob
    candidates = glob.glob(os.path.join(BASE_DIR, "*.dataframe"))
    if candidates:
        INPUT_FILE = candidates[0]
OUTPUT_JSON = os.path.join(BASE_DIR, "output", "pixel_spatial_classification.json")

ARRAY_ROWS = 16
ARRAY_COLS = 16

BEAD_WELLS = {79, 158, 174, 238, 253, 107, 106, 227, 130, 178, 161, 177, 193, 176}

# Rule-based exclusion thresholds (same as detect_defective_pixels_clustering.py)
SPAN_HIGH_MV = 500.0
DIFF_STD_MV = 50.0
SPAN_FLAT_MV = 1.0


# ─── Helpers ────────────────────────────────────────────────────────────────
def reshape_data_smartloc2(data_in):
    """Map 1-D pixel indices to the physical 16×16 grid layout."""
    d = np.copy(data_in)
    d2 = d.reshape(-1, 16)
    out = np.transpose(d2)
    d2[0::1, :] = d2[0::1, ::-1]
    return out


def chebyshev_distance(pos_a, pos_b):
    return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))


# ─── Build grid position lookup ────────────────────────────────────────────
well_map = reshape_data_smartloc2(np.arange(256))
pixel_to_pos = {}
for r in range(ARRAY_ROWS):
    for c in range(ARRAY_COLS):
        pixel_to_pos[int(well_map[r, c])] = (r, c)


# ─── Exclusion (same logic as detect_defective_pixels_clustering.py) ───────
df = pd.read_pickle(INPUT_FILE)
excluded = set()
for p in range(256):
    mv = np.asarray(df.iloc[p]["Milivolts"], dtype=float)
    if np.any(~np.isfinite(mv)):
        excluded.add(p)
        continue
    span = mv.max() - mv.min()
    diff_std = np.std(np.diff(mv))
    if span > SPAN_HIGH_MV or diff_std > DIFF_STD_MV or span < SPAN_FLAT_MV:
        excluded.add(p)

included = sorted(set(range(256)) - excluded)
bead_wells_in_data = sorted(BEAD_WELLS - excluded)

print(f"Total pixels:       256")
print(f"Excluded pixels:    {len(excluded)}")
print(f"Included pixels:    {len(included)}")
print(f"Bead wells total:   {len(BEAD_WELLS)}")
print(f"Bead wells incl.:   {len(bead_wells_in_data)}  {bead_wells_in_data}")


# ─── Spatial classification ─────────────────────────────────────────────────
# Positions of ALL bead wells (including excluded ones) for neighbour calc
bead_positions = {pixel_to_pos[w] for w in BEAD_WELLS}

rows = []
for p in included:
    pos = pixel_to_pos[p]
    is_bead = p in BEAD_WELLS

    # Check 8-connected neighbours for any bead well
    is_adjacent = False
    if not is_bead:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = pos[0] + dr, pos[1] + dc
                if 0 <= nr < ARRAY_ROWS and 0 <= nc < ARRAY_COLS:
                    if (nr, nc) in bead_positions:
                        is_adjacent = True

    # Chebyshev distance to nearest bead
    min_dist = min(chebyshev_distance(pos, bp) for bp in bead_positions)

    if is_bead:
        category = "Occupied"
    elif is_adjacent:
        category = "Adjacent"
    else:
        category = "Distal"

    rows.append({
        "pixel_id": p,
        "grid_row": pos[0],
        "grid_col": pos[1],
        "is_occupied": is_bead,
        "is_adjacent": is_adjacent,
        "category": category,
        "chebyshev_dist_to_nearest_bead": min_dist,
    })

result_df = pd.DataFrame(rows)

# ─── Summary ────────────────────────────────────────────────────────────────
print(f"\nSpatial classification summary:")
for cat in ["Occupied", "Adjacent", "Distal"]:
    n = (result_df["category"] == cat).sum()
    print(f"  {cat:12s}: {n}")

print(f"\nChebyshev distance distribution for Distal wells:")
distal = result_df[result_df["category"] == "Distal"]["chebyshev_dist_to_nearest_bead"]
if len(distal) > 0:
    print(f"  min={distal.min()}, max={distal.max()}, mean={distal.mean():.1f}")

# ─── Save JSON for downstream scripts (e.g., Script 8) ─────────────────────
os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
output_data = {
    "excluded_pixels": sorted(excluded),
    "included_pixels": included,
    "bead_wells_all": sorted(BEAD_WELLS),
    "bead_wells_included": bead_wells_in_data,
    "classification": {str(r["pixel_id"]): {k: v for k, v in r.items()} for _, r in result_df.iterrows()},
}

def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

with open(OUTPUT_JSON, "w") as f:
    json.dump(output_data, f, indent=2, default=convert)

print(f"\nSaved {OUTPUT_JSON}")


# ─── 2D Map Plot ────────────────────────────────────────────────────────────
# Build classification map for ALL 256 pixels
cat_map = np.full((ARRAY_ROWS, ARRAY_COLS), -1, dtype=int)  # -1 = excluded
label_map = np.full((ARRAY_ROWS, ARRAY_COLS), -1, dtype=int)

# Categories: 0=Excluded, 1=Distal, 2=Adjacent, 3=Occupied, 4=Occupied+Excluded (striped)
for p in range(256):
    r, c = pixel_to_pos[p]
    if p in excluded and p in BEAD_WELLS:
        cat_map[r, c] = 4  # occupied + excluded (striped)
    elif p in excluded:
        cat_map[r, c] = 0  # excluded
    elif p in BEAD_WELLS:
        cat_map[r, c] = 3  # occupied
    else:
        # check from result_df
        match = result_df[result_df["pixel_id"] == p]
        if len(match) > 0:
            cat_str = match.iloc[0]["category"]
            if cat_str == "Adjacent":
                cat_map[r, c] = 2
            else:
                cat_map[r, c] = 1  # distal
    label_map[r, c] = p

# Colors
COLORS = {
    0: "#D4D1CA",   # Excluded — grey
    1: "#F7F6F2",   # Distal — off-white
    2: "#FFE699",   # Adjacent — amber
    3: "#20808D",   # Occupied — teal
}

fig, ax = plt.subplots(figsize=(10, 10))

for r in range(ARRAY_ROWS):
    for c in range(ARRAY_COLS):
        cat = cat_map[r, c]
        color = COLORS[0] if cat == 4 else COLORS[cat]
        rect = mpatches.FancyBboxPatch(
            (c - 0.45, r - 0.45), 0.9, 0.9,
            boxstyle="round,pad=0.02", facecolor=color,
            edgecolor="#7A7974", linewidth=0.5)
        ax.add_patch(rect)

        if cat == 4:
            striped = mpatches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02", facecolor="none",
                edgecolor=COLORS[3], linewidth=0.0, hatch="///")
            ax.add_patch(striped)

        pid = label_map[r, c]
        txt_color = "#FFFFFF" if cat == 3 else "#28251D"
        fontweight = "bold" if cat in (3, 4) else "normal"
        fontsize = 6.5 if cat in (3, 4) else 5.5
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
ax.set_title("Well Spatial Classification — 16×16 ISFET Array", fontsize=13, fontweight="bold", pad=12)

# Legend
legend_patches = [
    mpatches.Patch(facecolor=COLORS[3], edgecolor="#7A7974", label=f"Occupied ({(cat_map == 3).sum()})"),
    mpatches.Patch(facecolor=COLORS[0], edgecolor=COLORS[3], hatch="///",
                   label=f"Occupied + Excluded ({(cat_map == 4).sum()})"),
    mpatches.Patch(facecolor=COLORS[2], edgecolor="#7A7974", label=f"Adjacent ({(cat_map == 2).sum()})"),
    mpatches.Patch(facecolor=COLORS[1], edgecolor="#7A7974", label=f"Distal ({(cat_map == 1).sum()})"),
    mpatches.Patch(facecolor=COLORS[0], edgecolor="#7A7974", label=f"Excluded ({(cat_map == 0).sum()})"),
]
ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.01, 1),
          fontsize=9, frameon=True, fancybox=True)

plt.tight_layout()
plt.show()
