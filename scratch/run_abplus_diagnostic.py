import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import hashlib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    CLASS_NAMES,
    PROJECT_ROOT,
    DATA_DIR
)
from src.predictor import BloodGroupPredictor
from src.preprocessing import preprocess_for_ensemble

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "mfs100_abplus_diagnostic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Define Cohort of real MFS100 AB+ subjects and captures
AB_COHORT = [
    {
        "subject_id": "S005",
        "split": "train",
        "true_label": "AB+",
        "captures": [
            "capture_1788759809513.bmp",
            "capture_1788759811570.bmp",
            "capture_1788759813142.bmp"
        ]
    },
    {
        "subject_id": "S006",
        "split": "val",
        "true_label": "AB+",
        "captures": [
            "capture_1788760098032.bmp",
            "capture_1788760099328.bmp",
            "capture_1788760100615.bmp"
        ]
    },
    {
        "subject_id": "S026",
        "split": "test",
        "true_label": "AB+",
        "captures": [
            "capture_1788798700652.bmp",
            "capture_1788798702179.bmp",
            "capture_1788798703500.bmp"
        ]
    },
    {
        "subject_id": "BG-000615",
        "split": "live_test",
        "true_label": "AB+",
        "captures": [
            "capture_1789115056057.bmp",
            "capture_1789115057360.bmp",
            "capture_1789115058674.bmp"
        ]
    },
    {
        "subject_id": "BG-000614",
        "split": "live_test",
        "true_label": "AB+",
        "captures": [
            "capture_1789115014276.bmp",
            "capture_1789115016007.bmp",
            "capture_1789115017307.bmp"
        ]
    }
]

# S001 audit (to address prompt: "Include S001 if applicable")
S001_AUDIT = {
    "subject_id": "S001",
    "split": "train",
    "true_label": "A+", # ground truth
    "captures": [
        "capture_1788759751645.bmp",
        "capture_1788759752958.bmp",
        "capture_1788759754260.bmp"
    ]
}

def extract_features(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    h, w = gray.shape
    pixels = gray.flatten().astype(np.float32)
    mean_val = float(np.mean(pixels))
    std_val = float(np.std(pixels))
    rms_contrast = float(np.sqrt(np.mean((pixels - mean_val) ** 2)))
    p5, p95 = np.percentile(pixels, 5), np.percentile(pixels, 95)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    fg_pixels = int(np.count_nonzero(mask))
    total_pixels = int(h * w)
    fg_percentage = float(fg_pixels / total_pixels) * 100.0
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_large = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        if c_large:
            all_pts = np.vstack(c_large)
            bx, by, bw, bh = cv2.boundingRect(all_pts)
        else:
            bx, by, bw, bh = cv2.boundingRect(contours[0])
    else:
        bx, by, bw, bh = 0, 0, w, h
        
    norm_cx = round((bx + bw / 2.0) / w, 4)
    norm_cy = round((by + bh / 2.0) / h, 4)
    norm_bw = round(bw / w, 4)
    norm_bh = round(bh / h, 4)
    
    # FFT Frequency Spectrum
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    total_energy = float(np.sum(magnitude_spectrum ** 2))
    
    y_coords, x_coords = np.ogrid[:h, :w]
    center_y, center_x = h / 2.0, w / 2.0
    r = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_r = np.sqrt(center_x ** 2 + center_y ** 2)
    
    low_mask = r <= (0.15 * max_r)
    mid_mask = (r > (0.15 * max_r)) & (r <= (0.50 * max_r))
    high_mask = r > (0.50 * max_r)
    
    low_energy_ratio = float(np.sum(magnitude_spectrum[low_mask] ** 2) / (total_energy + 1e-12))
    mid_energy_ratio = float(np.sum(magnitude_spectrum[mid_mask] ** 2) / (total_energy + 1e-12))
    high_energy_ratio = float(np.sum(magnitude_spectrum[high_mask] ** 2) / (total_energy + 1e-12))
    
    # 2D FFT patch ridge wavelength estimator
    patch_size = 64
    step = 32
    wavelengths = []
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            p = gray[y:y+patch_size, x:x+patch_size]
            if np.std(p) < 10:
                continue
            fp = np.fft.fftshift(np.fft.fft2(p.astype(np.float32) - np.mean(p)))
            mag = np.abs(fp)
            py_c, px_c = patch_size // 2, patch_size // 2
            yy, xx = np.ogrid[:patch_size, :patch_size]
            rr = np.sqrt((xx - px_c)**2 + (yy - py_c)**2)
            # radial profile
            r_int = np.round(rr).astype(int)
            tbin = np.bincount(r_int.ravel(), mag.ravel())
            nr = np.bincount(r_int.ravel())
            radial = tbin / np.maximum(nr, 1)
            # frequency band 0.047 to 0.328 cycles/px -> radius 3 to 21
            band = radial[3:22]
            if len(band) > 0 and np.max(band) > 1.2 * np.mean(band):
                peak_r = 3 + np.argmax(band)
                if peak_r > 0:
                    wl = patch_size / float(peak_r)
                    wavelengths.append(wl)
    
    median_wl = float(np.median(wavelengths)) if len(wavelengths) > 0 else np.nan
    median_freq = float(1.0 / median_wl) if not np.isnan(median_wl) and median_wl > 0 else np.nan

    return {
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "rms_contrast": round(rms_contrast, 2),
        "p5": round(float(p5), 2),
        "p95": round(float(p95), 2),
        "laplacian_sharpness": round(sharpness, 2),
        "foreground_area_pct": round(fg_percentage, 2),
        "bbox_cx": norm_cx,
        "bbox_cy": norm_cy,
        "bbox_w": norm_bw,
        "bbox_h": norm_bh,
        "spectral_low_freq": round(low_energy_ratio, 4),
        "spectral_mid_freq": round(mid_energy_ratio, 4),
        "spectral_high_freq": round(high_energy_ratio, 4),
        "ridge_wavelength": round(median_wl, 2) if not np.isnan(median_wl) else None,
        "ridge_frequency": round(median_freq, 4) if not np.isnan(median_freq) else None
    }

def main():
    print("[*] Loading Champion Ensemble V2 predictor...")
    predictor = BloodGroupPredictor(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)
    class_names = predictor.class_names
    
    weights = predictor.weights # {'resnet50': 0.58, 'vgg16': 0.38, 'efficientnetb0': 0.04}
    
    # 1. Run predictions on all captures
    all_captures_data = []
    all_models_data = []
    all_probs_data = []
    consensus_data = []
    
    # Include AB_COHORT plus S001
    full_cohort = AB_COHORT + [S001_AUDIT]
    
    for subject_info in full_cohort:
        sub_id = subject_info["subject_id"]
        true_lbl = subject_info["true_label"]
        split_name = subject_info["split"]
        c_files = subject_info["captures"]
        
        full_paths = [os.path.join(DATA_DIR, "captured", f) for f in c_files]
        
        # Individual predictions
        sub_ind_res = []
        for f, fp in zip(c_files, full_paths):
            if not os.path.exists(fp):
                print(f"[-] File missing: {fp}")
                continue
            
            # Predict ensemble
            res = predictor.predict(fp, source="mfs100")
            sub_ind_res.append(res)
            
            # Predict individual models
            prep = preprocess_for_ensemble(fp, source="mfs100")
            p_res = predictor.resnet_model.predict(prep["resnet50"], verbose=0)[0]
            p_vgg = predictor.vgg_model.predict(prep["vgg16"], verbose=0)[0]
            p_eff = predictor.eff_model.predict(prep["efficientnetb0"], verbose=0)[0]
            
            res_pred_idx = np.argmax(p_res)
            vgg_pred_idx = np.argmax(p_vgg)
            eff_pred_idx = np.argmax(p_eff)
            
            res_pred = class_names[res_pred_idx]
            vgg_pred = class_names[vgg_pred_idx]
            eff_pred = class_names[eff_pred_idx]
            
            res_conf = float(p_res[res_pred_idx])
            vgg_conf = float(p_vgg[vgg_pred_idx])
            eff_conf = float(p_eff[eff_pred_idx])
            
            # Derived ABO for each model
            res_abo_dict = {abo: p_res[class_names.index(f"{abo}+")] + p_res[class_names.index(f"{abo}-")] for abo in ["A", "B", "AB", "O"]}
            vgg_abo_dict = {abo: p_vgg[class_names.index(f"{abo}+")] + p_vgg[class_names.index(f"{abo}-")] for abo in ["A", "B", "AB", "O"]}
            eff_abo_dict = {abo: p_eff[class_names.index(f"{abo}+")] + p_eff[class_names.index(f"{abo}-")] for abo in ["A", "B", "AB", "O"]}
            
            res_top_abo = max(res_abo_dict.items(), key=lambda x: x[1])[0]
            vgg_top_abo = max(vgg_abo_dict.items(), key=lambda x: x[1])[0]
            eff_top_abo = max(eff_abo_dict.items(), key=lambda x: x[1])[0]
            
            abo_analysis = res.get("abo_analysis", {})
            pred_abo = abo_analysis.get("abo_group", "Unknown")
            pred_abo_conf = abo_analysis.get("abo_confidence", 0.0)
            pred_rh = abo_analysis.get("rh_factor", "Unknown")
            
            p_dict = res["probabilities"]
            p_A = p_dict.get("A+", 0.0) + p_dict.get("A-", 0.0)
            p_AB = p_dict.get("AB+", 0.0) + p_dict.get("AB-", 0.0)
            p_B = p_dict.get("B+", 0.0) + p_dict.get("B-", 0.0)
            p_O = p_dict.get("O+", 0.0) + p_dict.get("O-", 0.0)
            
            models_agree = (res_pred == vgg_pred == eff_pred)
            
            # Which model dominated the error
            # Check if ResNet50 has the highest weight (0.58) and drove the prediction
            dominating = "ResNet50" if res["blood_group"] == res_pred else ("VGG16" if res["blood_group"] == vgg_pred else "EfficientNetB0")
            
            # Capture record
            all_captures_data.append({
                "subject_id": sub_id,
                "split": split_name,
                "capture_filename": f,
                "true_label": true_lbl,
                "prediction": res["blood_group"],
                "confidence": res["confidence"],
                "ABO_prediction": pred_abo,
                "ABO_confidence": round(pred_abo_conf, 4),
                "Rh_prediction": pred_rh,
                "Rh_uncertainty": abo_analysis.get("rh_uncertainty", False),
                "correct_8class": bool(res["blood_group"] == true_lbl),
                "correct_ABO": bool(pred_abo == true_lbl[:-1]),
                "advisory_status": res.get("advisory_level"),
                "advisory_reason": res.get("advisory_reason"),
                "p_A": round(p_A, 4),
                "p_AB": round(p_AB, 4),
                "p_B": round(p_B, 4),
                "p_O": round(p_O, 4),
                "p_AB_plus": p_dict.get("AB+", 0.0),
                "p_O_pos": p_dict.get("O+", 0.0),
                "p_O_neg": p_dict.get("O-", 0.0),
            })
            
            # Model comparison record
            all_models_data.append({
                "subject_id": sub_id,
                "capture_filename": f,
                "true_label": true_lbl,
                "resnet50_pred": res_pred,
                "resnet50_conf": round(res_conf, 4),
                "resnet50_abo": res_top_abo,
                "vgg16_pred": vgg_pred,
                "vgg16_conf": round(vgg_conf, 4),
                "vgg16_abo": vgg_top_abo,
                "effnet_pred": eff_pred,
                "effnet_conf": round(eff_conf, 4),
                "effnet_abo": eff_top_abo,
                "ensemble_pred": res["blood_group"],
                "ensemble_conf": round(res["confidence"], 4),
                "ensemble_abo": pred_abo,
                "models_agree": models_agree,
                "dominating_model": dominating
            })
            
            # Probability comparison record
            all_probs_data.append({
                "subject_id": sub_id,
                "capture_filename": f,
                "true_label": true_lbl,
                "p_A+": p_dict.get("A+", 0.0),
                "p_A-": p_dict.get("A-", 0.0),
                "p_AB+": p_dict.get("AB+", 0.0),
                "p_AB-": p_dict.get("AB-", 0.0),
                "p_B+": p_dict.get("B+", 0.0),
                "p_B-": p_dict.get("B-", 0.0),
                "p_O+": p_dict.get("O+", 0.0),
                "p_O-": p_dict.get("O-", 0.0),
                "p_ABO_A": round(p_A, 4),
                "p_ABO_AB": round(p_AB, 4),
                "p_ABO_B": round(p_B, 4),
                "p_ABO_O": round(p_O, 4),
                "p_O_minus_p_AB": round(p_O - p_AB, 4),
                "p_O_div_p_AB": round(p_O / (p_AB + 1e-6), 2)
            })
            
        # Consensus prediction across available impressions
        if full_paths:
            c_res = predictor.predict_consensus(full_paths, source="mfs100")
            c_abo = c_res.get("abo_analysis", {})
            consensus_data.append({
                "subject_id": sub_id,
                "split": split_name,
                "true_label": true_lbl,
                "consensus_prediction": c_res["blood_group"],
                "consensus_confidence": round(c_res["confidence"], 4),
                "consensus_ABO": c_abo.get("abo_group", "Unknown"),
                "consensus_ABO_conf": round(c_abo.get("abo_confidence", 0.0), 4),
                "consensus_Rh": c_abo.get("rh_factor", "Unknown"),
                "consensus_correct_8class": bool(c_res["blood_group"] == true_lbl),
                "consensus_correct_ABO": bool(c_abo.get("abo_group") == true_lbl[:-1]),
                "agreement": f"{c_res['consensus_metadata']['agreement_count']}/{c_res['consensus_metadata']['num_scans']}",
                "advisory_status": c_res.get("advisory_level"),
                "advisory_reason": c_res.get("advisory_reason")
            })

    # Convert to DataFrames
    df_captures = pd.DataFrame(all_captures_data)
    df_models = pd.DataFrame(all_models_data)
    df_probs = pd.DataFrame(all_probs_data)
    df_consensus = pd.DataFrame(consensus_data)
    
    # Save CSVs
    df_captures.to_csv(os.path.join(OUTPUT_DIR, "abplus_capture_predictions.csv"), index=False)
    df_models.to_csv(os.path.join(OUTPUT_DIR, "abplus_model_comparison.csv"), index=False)
    df_probs.to_csv(os.path.join(OUTPUT_DIR, "abplus_probability_comparison.csv"), index=False)
    df_consensus.to_csv(os.path.join(OUTPUT_DIR, "abplus_consensus_summary.csv"), index=False)
    
    print("[+] Saved capture predictions, model comparison, probability comparison, and consensus summary CSVs.")
    
    # 2. Domain Features Extraction (Benchmark AB+ TRAIN vs MFS100 AB+ Non-test)
    print("[*] Extracting domain features for Benchmark AB+ TRAIN images...")
    manifest_path = os.path.join(PROJECT_ROOT, "outputs", "dataset_manifest.csv")
    df_manifest = pd.read_csv(manifest_path)
    benchmark_ab_train = df_manifest[(df_manifest["class_name"] == "AB+") & (df_manifest["split"] == "train")]
    
    domain_features_records = []
    
    # Extract for benchmark AB+ TRAIN images (up to 100 images for robust non-overfitting distribution)
    sample_bench = benchmark_ab_train.sample(min(100, len(benchmark_ab_train)), random_state=42)
    for idx, row in sample_bench.iterrows():
        img_p = row["image_path"]
        if not os.path.isabs(img_p):
            img_p = os.path.join(PROJECT_ROOT, img_p)
        if not os.path.exists(img_p):
            continue
        feats = extract_features(img_p)
        if feats:
            feats["domain"] = "Benchmark_AB+_Train"
            feats["subject_id"] = f"bench_{idx}"
            feats["filename"] = os.path.basename(img_p)
            feats["true_label"] = "AB+"
            domain_features_records.append(feats)
            
    print(f"[+] Processed {len(domain_features_records)} Benchmark AB+ TRAIN images.")
    
    # Extract for MFS100 AB+ non-test captures (S005, S006, BG-000615, BG-000614)
    # Exclude test S026 from domain feature comparison as required
    non_test_ab_cohort = [s for s in AB_COHORT if s["split"] != "test"]
    mfs_count = 0
    for s in non_test_ab_cohort:
        for f in s["captures"]:
            fp = os.path.join(DATA_DIR, "captured", f)
            if not os.path.exists(fp):
                continue
            feats = extract_features(fp)
            if feats:
                feats["domain"] = "MFS100_AB+_NonTest"
                feats["subject_id"] = s["subject_id"]
                feats["filename"] = f
                feats["true_label"] = "AB+"
                domain_features_records.append(feats)
                mfs_count += 1
                
    df_features = pd.DataFrame(domain_features_records)
    df_features.to_csv(os.path.join(OUTPUT_DIR, "abplus_domain_features.csv"), index=False)
    print(f"[+] Processed {mfs_count} MFS100 AB+ non-test captures. Saved abplus_domain_features.csv.")
    
    # 3. Generate Visual Plots
    print("[*] Generating visual diagnostic plots...")
    
    # Plot 1: Probability Comparison (AB+ vs O vs A vs B across AB+ cases)
    ab_only_probs = df_probs[df_probs["true_label"] == "AB+"]
    plt.figure(figsize=(12, 6))
    x = np.arange(len(ab_only_probs))
    width = 0.2
    
    plt.bar(x - 1.5*width, ab_only_probs["p_ABO_AB"], width, label="P(AB) Total", color="#3b82f6", alpha=0.9)
    plt.bar(x - 0.5*width, ab_only_probs["p_ABO_O"], width, label="P(O) Total", color="#ef4444", alpha=0.9)
    plt.bar(x + 0.5*width, ab_only_probs["p_ABO_A"], width, label="P(A) Total", color="#f59e0b", alpha=0.9)
    plt.bar(x + 1.5*width, ab_only_probs["p_ABO_B"], width, label="P(B) Total", color="#10b981", alpha=0.9)
    
    labels = [f"{r.subject_id}\n{r.capture_filename[-8:]}" for _, r in ab_only_probs.iterrows()]
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Probability Mass", fontsize=11)
    plt.title("ABO Group Probability Mass Distribution for MFS100 AB+ Captures\n(Showing Strong Bias toward Group O and Rejection of Group AB)", fontsize=12, fontweight="bold")
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "abplus_probability_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 2: Model-by-Model Agreement
    ab_only_models = df_models[df_models["true_label"] == "AB+"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for ax, (m_name, m_col) in zip(axes, [("ResNet50 (58%)", "resnet50_abo"), ("VGG16 (38%)", "vgg16_abo"), ("EfficientNetB0 (4%)", "effnet_abo")]):
        counts = ab_only_models[m_col].value_counts().reindex(["O", "A", "B", "AB"], fill_value=0)
        colors = ["#ef4444" if k == "O" else ("#3b82f6" if k == "AB" else "#6b7280") for k in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=1.2)
        ax.set_title(m_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Derived ABO Group", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Capture Count", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h}", (bar.get_x() + bar.get_width() / 2, h + 0.1), ha="center", va="bottom", fontsize=10)
            
    fig.suptitle("Pre-Ensemble Architectural Agreement on Ground-Truth AB+ MFS100 Captures", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "abplus_model_agreement.png"), dpi=300)
    plt.close()
    
    # Plot 3: Domain Comparison (Benchmark AB+ Train vs MFS100 AB+ Non-test)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ("rms_contrast", "RMS Contrast", axes[0, 0]),
        ("laplacian_sharpness", "Laplacian Sharpness", axes[0, 1]),
        ("foreground_area_pct", "Foreground Contact Area (%)", axes[1, 0]),
        ("spectral_mid_freq", "Mid-Frequency Spectral Energy Ratio", axes[1, 1])
    ]
    
    for m_col, m_title, ax in metrics:
        bench_vals = df_features[df_features["domain"] == "Benchmark_AB+_Train"][m_col].dropna()
        mfs_vals = df_features[df_features["domain"] == "MFS100_AB+_NonTest"][m_col].dropna()
        
        data_to_plot = [bench_vals, mfs_vals]
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=["Benchmark AB+\n(Train)", "MFS100 AB+\n(Non-Test)"])
        bp['boxes'][0].set_facecolor("#93c5fd")
        bp['boxes'][1].set_facecolor("#fca5a5")
        ax.set_title(m_title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        
    fig.suptitle("Biophysical & Optical Feature Shift: Benchmark AB+ Train vs MFS100 AB+ Captures", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "abplus_domain_comparison.png"), dpi=300)
    plt.close()
    
    print("[+] Saved all 3 visualization plots.")
    
    # Print high-level metrics for report compilation
    print("\n" + "="*70)
    print("MFS100 AB+ DIAGNOSTIC SUMMARY")
    print("="*70)
    
    ab_captures_only = df_captures[df_captures["true_label"] == "AB+"]
    n_ab_impressions = len(ab_captures_only)
    n_ab_corr_8class = int(ab_captures_only["correct_8class"].sum())
    n_ab_corr_abo = int(ab_captures_only["correct_ABO"].sum())
    
    err_counts = ab_captures_only["ABO_prediction"].value_counts().to_dict()
    
    print(f"Total AB+ impressions evaluated: {n_ab_impressions}")
    print(f"AB+ Impression 8-class accuracy: {n_ab_corr_8class}/{n_ab_impressions} ({n_ab_corr_8class/n_ab_impressions*100:.2f}%)")
    print(f"AB+ Impression ABO accuracy:     {n_ab_corr_abo}/{n_ab_impressions} ({n_ab_corr_abo/n_ab_impressions*100:.2f}%)")
    print(f"AB+ -> O errors: {err_counts.get('O', 0)}")
    print(f"AB+ -> A errors: {err_counts.get('A', 0)}")
    print(f"AB+ -> B errors: {err_counts.get('B', 0)}")
    print(f"AB+ -> AB (correct): {err_counts.get('AB', 0)}")
    
    ab_consensus_only = df_consensus[df_consensus["true_label"] == "AB+"]
    n_cons = len(ab_consensus_only)
    n_cons_corr = int(ab_consensus_only["consensus_correct_8class"].sum())
    print(f"\nSubject-level consensus accuracy (AB+): {n_cons_corr}/{n_cons} ({n_cons_corr/n_cons*100:.2f}%)")
    for _, r in ab_consensus_only.iterrows():
        print(f"  {r.subject_id} ({r.split}): Consensus -> {r.consensus_prediction} ({r.consensus_confidence*100:.2f}%, ABO={r.consensus_ABO}, Agreement={r.agreement})")
        
    print("\nIndividual Model ABO Predictions on AB+ Captures:")
    for m_col in ["resnet50_abo", "vgg16_abo", "effnet_abo", "ensemble_abo"]:
        print(f"  {m_col}: {dict(ab_only_models[m_col].value_counts())}")
        
    # Check S001 result
    s001_res = df_captures[df_captures["subject_id"] == "S001"]
    print("\nS001 Audit (True Label: A+):")
    for _, r in s001_res.iterrows():
        print(f"  {r.capture_filename}: Predicted {r.prediction} (Conf {r.confidence*100:.2f}%, ABO={r.ABO_prediction})")
    print("="*70)

if __name__ == "__main__":
    main()
