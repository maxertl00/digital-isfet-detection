#!/usr/bin/env python3
"""
Script 16B: Advanced Detection Methods — Both Events (maps + best params)
=========================================================================
Runs the advanced detection pipeline separately for Event 1 and Event 2,
selects the best method/configuration per event by F1, prints the optimum
parameters, and shows side-by-side 2D TP/TN/FP/FN maps.
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA = os.path.join(BASE_DIR, "data", "raw_measurement.dataframe")
# prefer any .dataframe in the script folder
ALT_INPUT = os.path.join(BASE_DIR, "raw_measurement.dataframe")
if os.path.exists(ALT_INPUT):
    INPUT_DATA = ALT_INPUT
else:
    import glob
    candidates = glob.glob(os.path.join(BASE_DIR, "*.dataframe"))
    if candidates:
        INPUT_DATA = candidates[0]
INPUT_CLASS = os.path.join(BASE_DIR, "output", "pixel_spatial_classification.json")

F_SAMPLING = 2.0
N_SAMPLES = 1800
DT = 1.0 / F_SAMPLING
TIMESTAMPS = np.arange(N_SAMPLES) * DT
OPEN_TIMES = [400.0, 640.0]

TP_TEMPLATE_WELLS = {130, 177, 178, 193}

BASELINE_DURATIONS = [10, 15, 20, 25, 30, 40, 50]
POST_DURATIONS = [10, 15, 20, 25, 30, 40, 50]
POST_OFFSETS = [0, 2, 5]

ARRAY_ROWS, ARRAY_COLS = 16, 16

RESULT_COLORS = {
    "TP": "#1F8A3B",  # pronounced green
    "TN": "#D8F0D2",  # light green
    "FP": "#C62828",  # pronounced red
    "FN": "#F6C1C1",  # light red
}
EXCLUDED_COLOR = "#D4D1CA"


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
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, sensitivity=sens,
                specificity=spec, precision=prec, f1=f1, accuracy=acc)


def sweep_threshold(features, actual_set, total_n, thr_values):
    best_f1 = -1.0
    best = None
    for thr in thr_values:
        predicted = {p for p, v in features.items() if v > thr}
        m = compute_metrics(predicted, actual_set, total_n)
        if m["f1"] > best_f1 or (m["f1"] == best_f1 and best is not None and m["sensitivity"] > best["sensitivity"]):
            best_f1 = m["f1"]
            best = {"threshold": float(thr), **m}
    return best_f1, best


def sweep_threshold_lower(features, actual_set, total_n, thr_values):
    best_f1 = -1.0
    best = None
    for thr in thr_values:
        predicted = {p for p, v in features.items() if v < thr}
        m = compute_metrics(predicted, actual_set, total_n)
        if m["f1"] > best_f1 or (m["f1"] == best_f1 and best is not None and m["sensitivity"] > best["sensitivity"]):
            best_f1 = m["f1"]
            best = {"threshold": float(thr), **m}
    return best_f1, best


def run_baseline_integral_comparison(t_open, included, bead_included, signals):
    """Compare baseline integral with rectification vs signed (no rectification)."""
    baseline_durations = [15, 20, 25, 30]
    integ_offsets = np.arange(-5, 10.1, 5)
    integ_durations = [15, 20, 25, 30]

    def sweep(rectify):
        thresholds = np.arange(5.0, 500.0 + 2.5, 5.0) if rectify else np.arange(-500.0, 500.0 + 2.5, 5.0)
        best = {"f1": -1.0}
        n_incl = len(included)

        for bl_dur in baseline_durations:
            for ig_off in integ_offsets:
                for ig_dur in integ_durations:
                    features = {}
                    for p in included:
                        mv = signals[p]
                        bl_mask = (TIMESTAMPS >= t_open - bl_dur) & (TIMESTAMPS < t_open)
                        baseline = float(np.mean(mv[bl_mask])) if bl_mask.sum() > 0 else 0.0

                        ig_start = t_open + ig_off
                        ig_end = ig_start + ig_dur
                        ig_mask = (TIMESTAMPS >= ig_start) & (TIMESTAMPS <= ig_end)
                        if ig_mask.sum() > 0:
                            centered = mv[ig_mask] - baseline
                            val = np.trapezoid(np.abs(centered), dx=DT) if rectify else np.trapezoid(centered, dx=DT)
                        else:
                            val = 0.0
                        features[p] = float(val)

                    for thr in thresholds:
                        pred_gt = {p for p, v in features.items() if v > thr}
                        m_gt = compute_metrics(pred_gt, bead_included, n_incl)
                        if (m_gt["f1"] > best["f1"] or
                            (m_gt["f1"] == best["f1"] and m_gt["sensitivity"] > best.get("sensitivity", 0))):
                            best = {
                                "baseline_duration_s": float(bl_dur),
                                "integ_start_offset_s": float(ig_off),
                                "integ_duration_s": float(ig_dur),
                                "threshold_mVs": round(float(thr), 2),
                                "decision_direction": ">",
                                **m_gt,
                            }

                        if not rectify:
                            pred_lt = {p for p, v in features.items() if v < thr}
                            m_lt = compute_metrics(pred_lt, bead_included, n_incl)
                            if (m_lt["f1"] > best["f1"] or
                                (m_lt["f1"] == best["f1"] and m_lt["sensitivity"] > best.get("sensitivity", 0))):
                                best = {
                                    "baseline_duration_s": float(bl_dur),
                                    "integ_start_offset_s": float(ig_off),
                                    "integ_duration_s": float(ig_dur),
                                    "threshold_mVs": round(float(thr), 2),
                                    "decision_direction": "<",
                                    **m_lt,
                                }
        return best

    rectified = sweep(rectify=True)
    signed = sweep(rectify=False)
    return rectified, signed, signed["f1"] - rectified["f1"]


def percentile_ranks(features_dict, pixel_list):
    vals = np.array([features_dict.get(p, 0.0) for p in pixel_list], dtype=float)
    ranks = sp_stats.rankdata(vals, method="average") / len(vals)
    return {p: r for p, r in zip(pixel_list, ranks)}


def reshape_data_smartloc2(data_in):
    d = np.copy(data_in)
    d2 = d.reshape(-1, 16)
    out = np.transpose(d2)
    d2[0::1, :] = d2[0::1, ::-1]
    return out


def run_advanced_for_event(t_open, included, bead_included, signals, classif):
    n_incl = len(included)

    best_a = {"f1": -1.0}
    thresholds_a = np.arange(-200, 200.1, 1.0)
    features_a = {p: 0.0 for p in included}

    for bl_dur in BASELINE_DURATIONS:
        for post_off in POST_OFFSETS:
            for post_dur in POST_DURATIONS:
                features = {}
                for p in included:
                    mv = signals[p]
                    bl_mask = (TIMESTAMPS >= t_open - bl_dur) & (TIMESTAMPS < t_open)
                    baseline = float(np.mean(mv[bl_mask])) if bl_mask.sum() > 0 else 0.0
                    post_start = t_open + post_off
                    post_mask = (TIMESTAMPS >= post_start) & (TIMESTAMPS <= post_start + post_dur)
                    if post_mask.sum() > 0:
                        signed = mv[post_mask] - baseline
                        features[p] = float(np.trapezoid(signed, dx=DT))
                    else:
                        features[p] = 0.0

                f1, cfg = sweep_threshold(features, bead_included, n_incl, thresholds_a)
                if cfg is not None and (f1 > best_a["f1"] or (f1 == best_a["f1"] and cfg["sensitivity"] > best_a.get("sensitivity", 0))):
                    best_a = {
                        "method": "signed_integral",
                        "baseline_dur_s": bl_dur,
                        "post_offset_s": post_off,
                        "post_dur_s": post_dur,
                        "direction": "positive",
                        **cfg,
                        "f1": f1,
                    }
                    features_a = dict(features)

                f1n, cfgn = sweep_threshold_lower(features, bead_included, n_incl, thresholds_a)
                if cfgn is not None and (f1n > best_a["f1"] or (f1n == best_a["f1"] and cfgn["sensitivity"] > best_a.get("sensitivity", 0))):
                    best_a = {
                        "method": "signed_integral",
                        "baseline_dur_s": bl_dur,
                        "post_offset_s": post_off,
                        "post_dur_s": post_dur,
                        "direction": "negative",
                        **cfgn,
                        "f1": f1n,
                    }
                    features_a = {p: -v for p, v in features.items()}

    best_b = {"f1": -1.0}
    thresholds_b = np.arange(-10, 10.01, 0.05)
    features_b = {p: 0.0 for p in included}

    for bl_dur in BASELINE_DURATIONS:
        for post_off in POST_OFFSETS:
            for post_dur in POST_DURATIONS:
                features = {}
                for p in included:
                    mv = signals[p]
                    bl_mask = (TIMESTAMPS >= t_open - bl_dur) & (TIMESTAMPS < t_open)
                    post_start = t_open + post_off
                    post_mask = (TIMESTAMPS >= post_start) & (TIMESTAMPS <= post_start + post_dur)

                    pre_vals = mv[bl_mask]
                    post_vals = mv[post_mask]
                    mean_pre = float(np.mean(pre_vals)) if len(pre_vals) > 0 else 0.0
                    std_pre = float(np.std(pre_vals, ddof=1)) if len(pre_vals) > 1 else 0.0
                    mean_post = float(np.mean(post_vals)) if len(post_vals) > 0 else mean_pre
                    features[p] = (mean_post - mean_pre) / std_pre if std_pre > 1e-9 else 0.0

                f1, cfg = sweep_threshold(features, bead_included, n_incl, thresholds_b)
                if cfg is not None and (f1 > best_b["f1"] or (f1 == best_b["f1"] and cfg["sensitivity"] > best_b.get("sensitivity", 0))):
                    best_b = {
                        "method": "norm_mean_shift",
                        "baseline_dur_s": bl_dur,
                        "post_offset_s": post_off,
                        "post_dur_s": post_dur,
                        "direction": "positive",
                        **cfg,
                        "f1": f1,
                    }
                    features_b = dict(features)

                f1n, cfgn = sweep_threshold_lower(features, bead_included, n_incl, thresholds_b)
                if cfgn is not None and (f1n > best_b["f1"] or (f1n == best_b["f1"] and cfgn["sensitivity"] > best_b.get("sensitivity", 0))):
                    best_b = {
                        "method": "norm_mean_shift",
                        "baseline_dur_s": bl_dur,
                        "post_offset_s": post_off,
                        "post_dur_s": post_dur,
                        "direction": "negative",
                        **cfgn,
                        "f1": f1n,
                    }
                    features_b = {p: -v for p, v in features.items()}

    best_c = {"f1": -1.0}
    thresholds_t = np.arange(0, 15.01, 0.1)
    thresholds_p = np.arange(0, 15.01, 0.1)
    features_c = {p: 0.0 for p in included}

    for bl_dur in BASELINE_DURATIONS:
        for post_off in POST_OFFSETS:
            for post_dur in POST_DURATIONS:
                feat_t = {}
                feat_p = {}
                for p in included:
                    mv = signals[p]
                    bl_mask = (TIMESTAMPS >= t_open - bl_dur) & (TIMESTAMPS < t_open)
                    post_start = t_open + post_off
                    post_mask = (TIMESTAMPS >= post_start) & (TIMESTAMPS <= post_start + post_dur)

                    pre_vals = mv[bl_mask]
                    post_vals = mv[post_mask]
                    if len(pre_vals) > 1 and len(post_vals) > 1:
                        t_stat, p_val = sp_stats.ttest_ind(post_vals, pre_vals, equal_var=False)
                        feat_t[p] = float(abs(t_stat))
                        feat_p[p] = float(-np.log10(max(float(p_val), 1e-300)))
                    else:
                        feat_t[p] = 0.0
                        feat_p[p] = 0.0

                f1, cfg = sweep_threshold(feat_t, bead_included, n_incl, thresholds_t)
                if cfg is not None and (f1 > best_c["f1"] or (f1 == best_c["f1"] and cfg["sensitivity"] > best_c.get("sensitivity", 0))):
                    best_c = {
                        "method": "welch_t_test",
                        "baseline_dur_s": bl_dur,
                        "post_offset_s": post_off,
                        "post_dur_s": post_dur,
                        "score_type": "abs_t_stat",
                        **cfg,
                        "f1": f1,
                    }
                    features_c = dict(feat_t)

                f1p, cfgp = sweep_threshold(feat_p, bead_included, n_incl, thresholds_p)
                if cfgp is not None and (f1p > best_c["f1"] or (f1p == best_c["f1"] and cfgp["sensitivity"] > best_c.get("sensitivity", 0))):
                    best_c = {
                        "method": "welch_t_test",
                        "baseline_dur_s": bl_dur,
                        "post_offset_s": post_off,
                        "post_dur_s": post_dur,
                        "score_type": "neg_log10_p",
                        **cfgp,
                        "f1": f1p,
                    }
                    features_c = dict(feat_p)

    best_d = {"f1": -1.0}
    thresholds_d = np.arange(-0.5, 1.001, 0.01)
    features_d = {p: 0.0 for p in included}

    template_wells = sorted((TP_TEMPLATE_WELLS & set(included)) or (bead_included & set(included)))

    for bl_dur in BASELINE_DURATIONS:
        for post_off in POST_OFFSETS:
            for post_dur in POST_DURATIONS:
                bl_mask = (TIMESTAMPS >= t_open - bl_dur) & (TIMESTAMPS < t_open)
                post_start = t_open + post_off
                post_mask = (TIMESTAMPS >= post_start) & (TIMESTAMPS <= post_start + post_dur)
                analysis_mask = bl_mask | post_mask

                if analysis_mask.sum() < 4:
                    continue

                template_signals = []
                for tp_well in template_wells:
                    mv = signals[tp_well]
                    seg = mv[analysis_mask]
                    seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-12)
                    template_signals.append(seg_norm)

                if not template_signals:
                    continue

                template = np.mean(template_signals, axis=0)
                template = (template - np.mean(template)) / (np.std(template) + 1e-12)

                features = {}
                for p in included:
                    mv = signals[p]
                    seg = mv[analysis_mask]
                    seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-12)
                    corr = float(np.corrcoef(seg_norm, template)[0, 1])
                    features[p] = corr if not np.isnan(corr) else 0.0

                f1, cfg = sweep_threshold(features, bead_included, n_incl, thresholds_d)
                if cfg is not None and (f1 > best_d["f1"] or (f1 == best_d["f1"] and cfg["sensitivity"] > best_d.get("sensitivity", 0))):
                    best_d = {
                        "method": "template_correlation",
                        "baseline_dur_s": bl_dur,
                        "post_offset_s": post_off,
                        "post_dur_s": post_dur,
                        **cfg,
                        "f1": f1,
                    }
                    features_d = dict(features)

    all_pixels = sorted(included)
    ranks_a = percentile_ranks(features_a, all_pixels)
    ranks_b = percentile_ranks(features_b, all_pixels)
    ranks_c = percentile_ranks(features_c, all_pixels)
    ranks_d = percentile_ranks(features_d, all_pixels)

    features_e_avg = {p: (ranks_a[p] + ranks_b[p] + ranks_c[p] + ranks_d[p]) / 4.0 for p in all_pixels}
    method_f1s = {
        "a": best_a["f1"],
        "b": best_b["f1"],
        "c": best_c["f1"],
        "d": best_d["f1"],
    }
    total_f1 = sum(method_f1s.values()) + 1e-12
    weights = {k: v / total_f1 for k, v in method_f1s.items()}
    features_e_weighted = {
        p: (weights["a"] * ranks_a[p] + weights["b"] * ranks_b[p] + weights["c"] * ranks_c[p] + weights["d"] * ranks_d[p])
        for p in all_pixels
    }
    features_e_max = {p: max(ranks_a[p], ranks_b[p], ranks_c[p], ranks_d[p]) for p in all_pixels}
    features_e_min = {p: min(ranks_a[p], ranks_b[p], ranks_c[p], ranks_d[p]) for p in all_pixels}

    thresholds_e = np.arange(0, 1.001, 0.005)
    best_e = {"f1": -1.0}
    features_e = dict(features_e_avg)

    f1_avg, cfg_avg = sweep_threshold(features_e_avg, bead_included, n_incl, thresholds_e)
    if cfg_avg is not None:
        best_e = {"method": "combined_avg_rank", **cfg_avg, "f1": f1_avg}
        features_e = dict(features_e_avg)

    f1_w, cfg_w = sweep_threshold(features_e_weighted, bead_included, n_incl, thresholds_e)
    if cfg_w is not None and (f1_w > best_e["f1"] or (f1_w == best_e["f1"] and cfg_w["sensitivity"] > best_e.get("sensitivity", 0))):
        best_e = {"method": "combined_weighted_rank", **cfg_w, "f1": f1_w}
        features_e = dict(features_e_weighted)

    f1_max, cfg_max = sweep_threshold(features_e_max, bead_included, n_incl, thresholds_e)
    if cfg_max is not None and (f1_max > best_e["f1"] or (f1_max == best_e["f1"] and cfg_max["sensitivity"] > best_e.get("sensitivity", 0))):
        best_e = {"method": "combined_max_rank", **cfg_max, "f1": f1_max}
        features_e = dict(features_e_max)

    f1_min, cfg_min = sweep_threshold(features_e_min, bead_included, n_incl, thresholds_e)
    if cfg_min is not None and (f1_min > best_e["f1"] or (f1_min == best_e["f1"] and cfg_min["sensitivity"] > best_e.get("sensitivity", 0))):
        best_e = {"method": "combined_min_rank", **cfg_min, "f1": f1_min}
        features_e = dict(features_e_min)

    all_methods = [
        ("A: Signed Integral", best_a, features_a),
        ("B: Normalized Mean Shift", best_b, features_b),
        ("C: Welch's t-Test", best_c, features_c),
        ("D: Template Correlation", best_d, features_d),
        ("E: Combined Rank Score", best_e, features_e),
    ]

    overall_best_name, overall_best, overall_features = max(
        all_methods,
        key=lambda x: (x[1]["f1"], x[1].get("sensitivity", 0)),
    )

    per_pixel = []
    thr_best = float(overall_best["threshold"])
    for p in included:
        feat_val = float(overall_features[p])
        predicted_pos = feat_val > thr_best
        actual_pos = p in bead_included
        cat = classif["classification"][str(p)]
        per_pixel.append(
            {
                "pixel_id": p,
                "category": cat["category"],
                "score": feat_val,
                "predicted_positive": bool(predicted_pos),
                "actual_positive": bool(actual_pos),
                "result": (
                    "TP"
                    if predicted_pos and actual_pos
                    else "FP"
                    if predicted_pos and not actual_pos
                    else "FN"
                    if not predicted_pos and actual_pos
                    else "TN"
                ),
            }
        )

    methods_detail = {
        "A_signed_integral": best_a,
        "B_norm_mean_shift": best_b,
        "C_welch_t_test": best_c,
        "D_template_correlation": best_d,
        "E_combined_rank": best_e,
    }

    return {
        "event": t_open,
        "overall_best_name": overall_best_name,
        "overall_best": overall_best,
        "methods_detail": methods_detail,
        "per_pixel": per_pixel,
    }


def plot_event_map(ax, well_map, excluded, per_pixel, title):
    result_lookup = {px["pixel_id"]: px["result"] for px in per_pixel}

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
                (c - 0.45, r - 0.45),
                0.9,
                0.9,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="#7A7974",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            if pid in excluded:
                hatch_patch = mpatches.FancyBboxPatch(
                    (c - 0.45, r - 0.45),
                    0.9,
                    0.9,
                    boxstyle="round,pad=0.02",
                    facecolor="none",
                    edgecolor="#7A7974",
                    linewidth=0.0,
                    hatch="///",
                )
                ax.add_patch(hatch_patch)

            fw = "bold" if result_lookup.get(pid) in ("TP", "FP", "FN") else "normal"
            fs = 6.3 if pid not in excluded else 5.3
            ax.text(c, r, str(pid), ha="center", va="center", fontsize=fs, color=txt_color, fontweight=fw)

    n_tp = sum(1 for px in per_pixel if px["result"] == "TP")
    n_tn = sum(1 for px in per_pixel if px["result"] == "TN")
    n_fp = sum(1 for px in per_pixel if px["result"] == "FP")
    n_fn = sum(1 for px in per_pixel if px["result"] == "FN")

    ax.set_xlim(-0.6, ARRAY_COLS - 0.4)
    ax.set_ylim(-0.6, ARRAY_ROWS - 0.4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(range(ARRAY_COLS))
    ax.set_yticks(range(ARRAY_ROWS))
    ax.set_xlabel("Grid Column", fontsize=10)
    ax.set_ylabel("Grid Row", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    legend_patches = [
        mpatches.Patch(facecolor=RESULT_COLORS["TP"], edgecolor="#7A7974", label=f"TP ({n_tp})"),
        mpatches.Patch(facecolor=RESULT_COLORS["TN"], edgecolor="#7A7974", label=f"TN ({n_tn})"),
        mpatches.Patch(facecolor=RESULT_COLORS["FP"], edgecolor="#7A7974", label=f"FP ({n_fp})"),
        mpatches.Patch(facecolor=RESULT_COLORS["FN"], edgecolor="#7A7974", label=f"FN ({n_fn})"),
        mpatches.Patch(facecolor=EXCLUDED_COLOR, edgecolor="#7A7974", hatch="///", label=f"Excluded ({len(excluded)})"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, frameon=True)


def main():
    df = pd.read_pickle(INPUT_DATA)
    with open(INPUT_CLASS) as f:
        classif = json.load(f)

    excluded = set(classif["excluded_pixels"])
    included = classif["included_pixels"]
    bead_included = set(classif["bead_wells_included"])

    signals = {p: np.asarray(df.iloc[p]["Milivolts"], dtype=float) for p in included}

    print(f"Included pixels: {len(included)}, bead wells in data: {len(bead_included)}")

    event_results = []
    for t_open in OPEN_TIMES:
        print(f"\nRunning advanced sweep for Event @ {t_open:.0f}s ...")
        res = run_advanced_for_event(t_open, included, bead_included, signals, classif)
        event_results.append(res)

        baseline_rect, baseline_signed, delta_f1 = run_baseline_integral_comparison(
            t_open, included, bead_included, signals
        )

        best = res["overall_best"]
        print(f"Best method: {res['overall_best_name']}")
        print(
            "Best metrics: "
            f"F1={best['f1']:.4f}, Sens={best['sensitivity']:.4f}, Prec={best['precision']:.4f}, "
            f"TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}"
        )
        print(
            "Baseline integral comparison: "
            f"rectified F1={baseline_rect['f1']:.4f} vs signed(no rect.) F1={baseline_signed['f1']:.4f} "
            f"(delta={delta_f1:+.4f})"
        )
        print(
            f"  Signed decision: feature {baseline_signed['decision_direction']} "
            f"{baseline_signed['threshold_mVs']} mV·s"
        )
        print("Best parameters:")
        for k, v in res["methods_detail"].items():
            pass
        for key in ["baseline_dur_s", "post_offset_s", "post_dur_s", "direction", "score_type", "threshold"]:
            if key in best:
                print(f"  - {key}: {best[key]}")

    well_map = reshape_data_smartloc2(np.arange(256))

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for idx, res in enumerate(event_results):
        best = res["overall_best"]
        title = (
            f"Event {idx + 1} @ {res['event']:.0f}s\n"
            f"{res['overall_best_name']} | F1={best['f1']:.3f}"
        )
        plot_event_map(axes[idx], well_map, excluded, res["per_pixel"], title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
