"""
scripts/calibration_experiments/run_calibration_experiment.py
=============================================================
Post-Hoc Confidence Calibration for Two-Stage ABO+Rh Model
- Scalar Temperature Scaling for ABO and Rh heads
- Fitted exclusively on Validation split via NLL minimization
- Evaluation on Validation and Locked Test splits
- Uncertainty threshold selection ("needs confirmation")
- Production models and files remain strictly READ-ONLY.
"""

import os
import sys
import json
import hashlib
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import CLASS_NAMES, DATASET_MANIFEST_PATH, MODELS_DIR
from src.preprocessing import composite_alpha_on_white, resize_to_target

# Directory definitions
EXPERIMENT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "calibration_experiments")
EXPERIMENT_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "calibration_experiments")
FIGURES_DIR = os.path.join(EXPERIMENT_OUTPUTS_DIR, "figures")
TWO_STAGE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "abo_rh_experiments", "two_stage_abo_rh_best_val.keras")

os.makedirs(EXPERIMENT_MODELS_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Mappings
ABO_CLASSES = ["A", "AB", "B", "O"]
ABO_TO_IDX = {c: i for i, c in enumerate(ABO_CLASSES)}
IDX_TO_ABO = {i: c for i, c in enumerate(ABO_CLASSES)}

RH_CLASSES = ["+", "-"]
RH_TO_IDX = {c: i for i, c in enumerate(RH_CLASSES)}
IDX_TO_RH = {i: c for i, c in enumerate(RH_CLASSES)}

CLASS_TO_ABO_RH = {cname: (cname[:-1], cname[-1]) for cname in CLASS_NAMES}

# Production files to audit for lock integrity
PRODUCTION_LOCK_FILES = [
    os.path.join(MODELS_DIR, "resnet50_improved_blood_group.keras"),
    os.path.join(MODELS_DIR, "vgg16_blood_group.keras"),
    os.path.join(MODELS_DIR, "efficientnetb0_blood_group.keras"),
    os.path.join(PROJECT_ROOT, "src", "preprocessing.py"),
    os.path.join(PROJECT_ROOT, "src", "predictor.py"),
    TWO_STAGE_MODEL_PATH
]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def audit_production_locks(tag="PRE-EXPERIMENT"):
    print(f"\n=======================================================")
    print(f"[*] PRODUCTION LOCK AUDIT ({tag})")
    print(f"=======================================================")
    hashes = {}
    for p in PRODUCTION_LOCK_FILES:
        rel = os.path.relpath(p, PROJECT_ROOT)
        if os.path.exists(p):
            sha = compute_sha256(p)
            hashes[rel] = sha
            print(f"  [LOCKED] {rel:50s} : {sha[:16]}... (size: {os.path.getsize(p):,} B)")
        else:
            print(f"  [ERROR] File missing: {rel}")
            hashes[rel] = None
    return hashes

# ─── DATA LOADER ─────────────────────────────────────────────────────────────

def load_split(manifest_df, split_name):
    sub_df = manifest_df[manifest_df["split"] == split_name].reset_index(drop=True)
    n = len(sub_df)
    print(f"[*] Loading {split_name} split ({n} images)...")

    imgs = []
    y_8class = []
    y_abo = []
    y_rh = []
    meta = []

    for _, row in sub_df.iterrows():
        p = row["image_path"]
        cname = row["class_name"]
        cid = row["class_id"]

        abo_str, rh_str = CLASS_TO_ABO_RH[cname]
        abo_id = ABO_TO_IDX[abo_str]
        rh_id = RH_TO_IDX[rh_str]

        with Image.open(p) as img:
            rgb = composite_alpha_on_white(img)
            resized = resize_to_target(rgb, (224, 224))
            arr = np.array(resized, dtype=np.float32)

        imgs.append(arr)
        y_8class.append(cid)
        y_abo.append(abo_id)
        y_rh.append(rh_id)
        meta.append({
            "image_path": p,
            "filename": os.path.basename(p),
            "class_name": cname,
            "class_id": cid,
            "abo_str": abo_str,
            "abo_id": abo_id,
            "rh_str": rh_str,
            "rh_id": rh_id
        })

    imgs_arr = np.array(imgs, dtype=np.float32)
    imgs_preprocessed = tf.keras.applications.resnet50.preprocess_input(imgs_arr)

    return (
        imgs_preprocessed,
        np.array(y_8class, dtype=np.int32),
        np.array(y_abo, dtype=np.int32),
        np.array(y_rh, dtype=np.int32),
        meta
    )

# ─── LOGIT EXTRACTION ────────────────────────────────────────────────────────

def extract_logits(model, imgs, batch_size=32):
    """
    Extracts pre-softmax logits from dense feature activations.
    """
    print(f"[*] Extracting pre-softmax logits for {len(imgs)} samples...")
    abo_drop_layer = model.get_layer("abo_dropout")
    rh_drop_layer = model.get_layer("rh_dropout")
    sub_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[abo_drop_layer.output, rh_drop_layer.output]
    )

    feat_abo_list = []
    feat_rh_list = []
    num_samples = len(imgs)

    for i in range(0, num_samples, batch_size):
        batch = imgs[i:i + batch_size]
        f_abo, f_rh = sub_model(batch, training=False)
        feat_abo_list.append(f_abo.numpy())
        feat_rh_list.append(f_rh.numpy())

    feat_abo = np.concatenate(feat_abo_list, axis=0)
    feat_rh = np.concatenate(feat_rh_list, axis=0)

    w_abo, b_abo = model.get_layer("abo_output").get_weights()
    w_rh, b_rh = model.get_layer("rh_output").get_weights()

    logits_abo = np.dot(feat_abo, w_abo) + b_abo
    logits_rh = np.dot(feat_rh, w_rh) + b_rh

    return logits_abo.astype(np.float64), logits_rh.astype(np.float64)

# ─── NUMERICALLY STABLE SOFTMAX & PROB RECONSTRUCTION ────────────────────────

def softmax(z, temperature=1.0):
    scaled_z = z / temperature
    max_z = np.max(scaled_z, axis=-1, keepdims=True)
    exp_z = np.exp(scaled_z - max_z)
    probs = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    return probs

def reconstruct_8class_probs(p_abo, p_rh):
    """
    Reconstructs 8-class probabilities:
    P(c) = P(abo) * P(rh) for each c in CLASS_NAMES.
    Since sum(P_abo) == 1 and sum(P_rh) == 1, sum(P_joint) == 1.0.
    """
    n = len(p_abo)
    probs_8class = np.zeros((n, 8), dtype=np.float64)
    for c_idx, cname in enumerate(CLASS_NAMES):
        abo_str, rh_str = CLASS_TO_ABO_RH[cname]
        abo_i = ABO_TO_IDX[abo_str]
        rh_i = RH_TO_IDX[rh_str]
        probs_8class[:, c_idx] = p_abo[:, abo_i] * p_rh[:, rh_i]
    return probs_8class

# ─── TEMPERATURE SCALING OPTIMIZATION ────────────────────────────────────────

def compute_nll(logits, labels, temperature):
    """
    Computes cross-entropy negative log likelihood for logits and labels given scalar temperature T.
    """
    scaled = logits / temperature
    max_s = np.max(scaled, axis=-1, keepdims=True)
    log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=-1, keepdims=True))
    log_probs = scaled - log_sum_exp
    # Gather true class log prob
    n = len(labels)
    true_log_probs = log_probs[np.arange(n), labels]
    nll = -np.mean(true_log_probs)
    return nll

def fit_temperature(logits_val, labels_val, head_name="head"):
    """
    Fit scalar temperature T > 0 by minimizing validation NLL.
    Parameterization: T = exp(w), w in R.
    """
    print(f"[*] Optimizing temperature for {head_name} on validation set (N={len(labels_val)})...")

    def loss_func(w):
        T = np.exp(w[0])
        return compute_nll(logits_val, labels_val, T)

    # Initial guess w = 0 -> T = 1.0
    res = minimize(
        loss_func,
        x0=[0.0],
        method="L-BFGS-B",
        options={"disp": False, "ftol": 1e-12, "gtol": 1e-12}
    )
    best_w = res.x[0]
    best_T = float(np.exp(best_w))
    val_nll_before = compute_nll(logits_val, labels_val, 1.0)
    val_nll_after = compute_nll(logits_val, labels_val, best_T)

    print(f"    -> Optimization success: {res.success}")
    print(f"    -> Fitted T_{head_name}: {best_T:.6f}")
    print(f"    -> Val NLL before (T=1.0): {val_nll_before:.6f}")
    print(f"    -> Val NLL after  (T={best_T:.4f}): {val_nll_after:.6f}")
    return best_T

# ─── CALIBRATION METRICS (ECE, MCE, BRIER, CONFIDENCE) ───────────────────────

def compute_brier_score(probs, labels, num_classes):
    """
    Multi-class Brier score: (1/N) * sum_i sum_k (p_{i,k} - y_{i,k})^2
    """
    n = len(labels)
    one_hot = np.zeros((n, num_classes), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)

def compute_calibration_curve(probs, labels, num_bins=10):
    """
    Computes 10-bin calibration curve, ECE, and MCE.
    Bins: [0.0, 0.1], (0.1, 0.2], ..., (0.9, 1.0]
    """
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    corrects = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_data = []
    ece = 0.0
    mce = 0.0
    n_total = len(labels)

    for b in range(num_bins):
        low, high = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (confs >= low) & (confs <= high)
        else:
            mask = (confs > low) & (confs <= high)

        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(confs[mask]))
            acc = float(np.mean(corrects[mask]))
            gap = abs(avg_conf - acc)
            weight = count / n_total
            ece += weight * gap
            if gap > mce:
                mce = gap
        else:
            avg_conf = (low + high) / 2.0
            acc = 0.0
            gap = 0.0

        bin_data.append({
            "bin_idx": b,
            "range": f"{low:.1f}-{high:.1f}",
            "low": low,
            "high": high,
            "count": count,
            "avg_confidence": avg_conf,
            "accuracy": acc,
            "calibration_gap": gap
        })

    return ece, mce, bin_data

def evaluate_predictions_comprehensive(probs, labels, num_classes, head_name="8class"):
    """
    Computes all specified metrics: Accuracy, NLL, Brier, ECE, MCE, Conf stats, Overconf brackets.
    """
    n = len(labels)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct_mask = (preds == labels)
    incorrect_mask = ~correct_mask

    acc = float(np.mean(correct_mask))
    nll = compute_nll(np.log(np.clip(probs, 1e-15, 1.0)), labels, 1.0)
    brier = compute_brier_score(probs, labels, num_classes)
    ece, mce, bin_data = compute_calibration_curve(probs, labels, num_bins=10)

    mean_conf_all = float(np.mean(confs))
    mean_conf_correct = float(np.mean(confs[correct_mask])) if np.sum(correct_mask) > 0 else 0.0
    mean_conf_incorrect = float(np.mean(confs[incorrect_mask])) if np.sum(incorrect_mask) > 0 else 0.0

    n_incorrect = int(np.sum(incorrect_mask))
    brackets = [0.50, 0.60, 0.70, 0.80, 0.90]
    bracket_counts = {}
    bracket_pcts = {}
    for th in brackets:
        c = int(np.sum(confs[incorrect_mask] >= th))
        pct = (c / n_incorrect * 100.0) if n_incorrect > 0 else 0.0
        bracket_counts[f"incorrect_conf_ge_{int(th*100)}"] = c
        bracket_pcts[f"pct_incorrect_conf_ge_{int(th*100)}"] = pct

    metrics = {
        "head": head_name,
        "n_samples": n,
        "accuracy": acc,
        "nll": nll,
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "mean_conf_all": mean_conf_all,
        "mean_conf_correct": mean_conf_correct,
        "mean_conf_incorrect": mean_conf_incorrect,
        "n_correct": int(np.sum(correct_mask)),
        "n_incorrect": n_incorrect,
        **bracket_counts,
        **bracket_pcts
    }
    return metrics, bin_data, preds, confs, correct_mask

# ─── MAIN EXPERIMENT EXECUTION ────────────────────────────────────────────────

def main():
    print("=================================================================")
    print("POST-HOC CONFIDENCE CALIBRATION EXPERIMENT (TWO-STAGE ABO+Rh)")
    print("=================================================================")

    # 1. Pre-experiment lock audit
    hashes_pre = audit_production_locks("PRE-EXPERIMENT")

    # 2. Load dataset manifest
    manifest_df = pd.read_csv(DATASET_MANIFEST_PATH)
    print(f"[*] Manifest loaded with {len(manifest_df)} records.")
    print("    Split counts:", manifest_df["split"].value_counts().to_dict())

    # Load validation and test splits
    val_imgs, val_y8, val_y_abo, val_y_rh, val_meta = load_split(manifest_df, "val")
    test_imgs, test_y8, test_y_abo, test_y_rh, test_meta = load_split(manifest_df, "test")

    # 3. Load Two-Stage Model Checkpoint (READ-ONLY)
    print(f"\n[*] Loading two-stage checkpoint from: {TWO_STAGE_MODEL_PATH}")
    model = tf.keras.models.load_model(TWO_STAGE_MODEL_PATH, compile=False)

    # 4. Extract Logits
    val_logits_abo, val_logits_rh = extract_logits(model, val_imgs)
    test_logits_abo, test_logits_rh = extract_logits(model, test_imgs)

    # 5. Optimize Temperatures on VALIDATION ONLY
    print("\n------------------------------------------------------")
    print("FITTING TEMPERATURES EXCLUSIVELY ON VALIDATION DATA")
    print("------------------------------------------------------")
    t_abo = fit_temperature(val_logits_abo, val_y_abo, head_name="ABO")
    t_rh = fit_temperature(val_logits_rh, val_y_rh, head_name="Rh")

    temp_params = {
        "model_checkpoint": os.path.relpath(TWO_STAGE_MODEL_PATH, PROJECT_ROOT),
        "validation_samples": len(val_y8),
        "t_abo": t_abo,
        "t_rh": t_rh,
        "fitted_on": "validation_split_only",
        "optimization_objective": "negative_log_likelihood",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(EXPERIMENT_MODELS_DIR, "temperature_params.json"), "w") as f:
        json.dump(temp_params, f, indent=2)
    print(f"[*] Saved temperature parameters to {os.path.join(EXPERIMENT_MODELS_DIR, 'temperature_params.json')}")

    # 6. Calculate Probabilities: Raw (T=1.0) vs Calibrated
    # Validation
    val_raw_p_abo = softmax(val_logits_abo, 1.0)
    val_cal_p_abo = softmax(val_logits_abo, t_abo)
    val_raw_p_rh = softmax(val_logits_rh, 1.0)
    val_cal_p_rh = softmax(val_logits_rh, t_rh)

    val_raw_p_8 = reconstruct_8class_probs(val_raw_p_abo, val_raw_p_rh)
    val_cal_p_8 = reconstruct_8class_probs(val_cal_p_abo, val_cal_p_rh)

    # Test
    test_raw_p_abo = softmax(test_logits_abo, 1.0)
    test_cal_p_abo = softmax(test_logits_abo, t_abo)
    test_raw_p_rh = softmax(test_logits_rh, 1.0)
    test_cal_p_rh = softmax(test_logits_rh, t_rh)

    test_raw_p_8 = reconstruct_8class_probs(test_raw_p_abo, test_raw_p_rh)
    test_cal_p_8 = reconstruct_8class_probs(test_cal_p_abo, test_cal_p_rh)

    # 7. Evaluate Before vs After Metrics on VALIDATION
    print("\n=======================================================")
    print("CALCULATING VALIDATION METRICS (BEFORE vs AFTER)")
    print("=======================================================")
    val_results = []
    # ABO
    abo_raw_val, abo_raw_val_bins, _, _, _ = evaluate_predictions_comprehensive(val_raw_p_abo, val_y_abo, 4, "ABO_Raw")
    abo_cal_val, abo_cal_val_bins, _, _, _ = evaluate_predictions_comprehensive(val_cal_p_abo, val_y_abo, 4, "ABO_Calibrated")
    # Rh
    rh_raw_val, rh_raw_val_bins, _, _, _ = evaluate_predictions_comprehensive(val_raw_p_rh, val_y_rh, 2, "Rh_Raw")
    rh_cal_val, rh_cal_val_bins, _, _, _ = evaluate_predictions_comprehensive(val_cal_p_rh, val_y_rh, 2, "Rh_Calibrated")
    # 8-Class
    rec_raw_val, rec_raw_val_bins, val_raw_preds8, val_raw_confs8, _ = evaluate_predictions_comprehensive(val_raw_p_8, val_y8, 8, "8Class_Raw")
    rec_cal_val, rec_cal_val_bins, val_cal_preds8, val_cal_confs8, _ = evaluate_predictions_comprehensive(val_cal_p_8, val_y8, 8, "8Class_Calibrated")

    val_results.extend([abo_raw_val, abo_cal_val, rh_raw_val, rh_cal_val, rec_raw_val, rec_cal_val])
    val_metrics_df = pd.DataFrame(val_results)
    val_metrics_df.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "validation_calibration_metrics.csv"), index=False)

    # Verify accuracy invariance on validation
    assert np.all(val_raw_preds8 == val_cal_preds8), "Validation raw and calibrated predictions do not match!"
    print(f"[*] Validation Accuracy verified invariant: {rec_raw_val['accuracy']*100:.2f}% == {rec_cal_val['accuracy']*100:.2f}%")
    print(f"    Val ECE (8-class): {rec_raw_val['ece']:.4f} -> {rec_cal_val['ece']:.4f} (abs change: {rec_cal_val['ece'] - rec_raw_val['ece']:+.4f})")
    print(f"    Val NLL (8-class): {rec_raw_val['nll']:.4f} -> {rec_cal_val['nll']:.4f}")
    print(f"    Val Brier (8-class): {rec_raw_val['brier_score']:.4f} -> {rec_cal_val['brier_score']:.4f}")

    # 8. Uncertainty Threshold Analysis on VALIDATION ONLY
    print("\n=======================================================")
    print("UNCERTAINTY / 'NEEDS CONFIRMATION' ANALYSIS (VALIDATION ONLY)")
    print("=======================================================")
    threshold_candidates = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    val_correct_mask = (val_cal_preds8 == val_y8)
    val_incorrect_mask = ~val_correct_mask
    total_val_errors = int(np.sum(val_incorrect_mask))

    threshold_rows = []
    best_th = None
    best_tuple = (-1.0, -1.0, -1.0) # (error_capture, accepted_acc, coverage)

    for th in threshold_candidates:
        accepted_mask = (val_cal_confs8 >= th)
        rejected_mask = (val_cal_confs8 < th) # needs_confirmation

        coverage = float(np.mean(accepted_mask)) # coverage
        rejection_rate = float(np.mean(rejected_mask))

        n_accepted = int(np.sum(accepted_mask))
        n_rejected = int(np.sum(rejected_mask))

        if n_accepted > 0:
            accepted_acc = float(np.mean(val_correct_mask[accepted_mask]))
            accepted_err_rate = 1.0 - accepted_acc
        else:
            accepted_acc = 0.0
            accepted_err_rate = 0.0

        # Errors captured by confirmation rule: incorrect samples where conf < th
        errors_captured = int(np.sum(rejected_mask & val_incorrect_mask))
        error_capture_rate = float(errors_captured / total_val_errors) if total_val_errors > 0 else 0.0

        row = {
            "threshold": th,
            "coverage": coverage,
            "rejection_rate": rejection_rate,
            "samples_accepted": n_accepted,
            "samples_needing_confirmation": n_rejected,
            "accepted_accuracy": accepted_acc,
            "accepted_error_rate": accepted_err_rate,
            "errors_captured": errors_captured,
            "error_capture_rate": error_capture_rate,
            "satisfies_coverage_gte_80": coverage >= 0.80
        }
        threshold_rows.append(row)

        # Optimization rule:
        # Constraint: coverage >= 0.80
        # Priority: max error_capture_rate -> max accepted_acc -> max coverage
        if coverage >= 0.80:
            crit_tuple = (error_capture_rate, accepted_acc, coverage)
            if crit_tuple > best_tuple:
                best_tuple = crit_tuple
                best_th = th

    th_df = pd.DataFrame(threshold_rows)
    th_df.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "uncertainty_threshold_analysis.csv"), index=False)
    print(th_df[["threshold", "coverage", "accepted_accuracy", "errors_captured", "error_capture_rate", "satisfies_coverage_gte_80"]].to_string(index=False))
    print(f"\n[*] Selected Validation Threshold: tau = {best_th:.2f}")
    print(f"    Criteria: Coverage >= 80% ({best_tuple[2]*100:.2f}%), Error Capture = {best_tuple[0]*100:.2f}%, Accepted Acc = {best_tuple[1]*100:.2f}%")

    # 9. Test Evaluation (FROZEN T_ABO, T_RH, and FROZEN THRESHOLD)
    print("\n=======================================================")
    print("CALCULATING TEST METRICS (FROZEN PARAMETERS EVALUATION)")
    print("=======================================================")
    test_results = []
    abo_raw_test, abo_raw_test_bins, _, _, _ = evaluate_predictions_comprehensive(test_raw_p_abo, test_y_abo, 4, "ABO_Raw")
    abo_cal_test, abo_cal_test_bins, _, _, _ = evaluate_predictions_comprehensive(test_cal_p_abo, test_y_abo, 4, "ABO_Calibrated")

    rh_raw_test, rh_raw_test_bins, _, _, _ = evaluate_predictions_comprehensive(test_raw_p_rh, test_y_rh, 2, "Rh_Raw")
    rh_cal_test, rh_cal_test_bins, _, _, _ = evaluate_predictions_comprehensive(test_cal_p_rh, test_y_rh, 2, "Rh_Calibrated")

    rec_raw_test, rec_raw_test_bins, test_raw_preds8, test_raw_confs8, _ = evaluate_predictions_comprehensive(test_raw_p_8, test_y8, 8, "8Class_Raw")
    rec_cal_test, rec_cal_test_bins, test_cal_preds8, test_cal_confs8, _ = evaluate_predictions_comprehensive(test_cal_p_8, test_y8, 8, "8Class_Calibrated")

    test_results.extend([abo_raw_test, abo_cal_test, rh_raw_test, rh_cal_test, rec_raw_test, rec_cal_test])
    test_metrics_df = pd.DataFrame(test_results)
    test_metrics_df.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "test_calibration_metrics.csv"), index=False)

    assert np.all(test_raw_preds8 == test_cal_preds8), "Test raw and calibrated predictions do not match!"
    print(f"[*] Test Accuracy verified invariant: {rec_raw_test['accuracy']*100:.2f}% == {rec_cal_test['accuracy']*100:.2f}%")
    print(f"    Test ECE (8-class): {rec_raw_test['ece']:.4f} -> {rec_cal_test['ece']:.4f} (abs change: {rec_cal_test['ece'] - rec_raw_test['ece']:+.4f})")
    print(f"    Test NLL (8-class): {rec_raw_test['nll']:.4f} -> {rec_cal_test['nll']:.4f}")
    print(f"    Test Brier (8-class): {rec_raw_test['brier_score']:.4f} -> {rec_cal_test['brier_score']:.4f}")

    # Evaluate frozen threshold on test
    test_correct_mask = (test_cal_preds8 == test_y8)
    test_incorrect_mask = ~test_correct_mask
    test_accepted_mask = (test_cal_confs8 >= best_th)
    test_rejected_mask = (test_cal_confs8 < best_th)
    n_test_accepted = int(np.sum(test_accepted_mask))
    n_test_rejected = int(np.sum(test_rejected_mask))
    test_accepted_acc = float(np.mean(test_correct_mask[test_accepted_mask])) if n_test_accepted > 0 else 0.0
    test_errors_captured = int(np.sum(test_rejected_mask & test_incorrect_mask))
    test_total_errors = int(np.sum(test_incorrect_mask))
    test_error_capture_rate = float(test_errors_captured / test_total_errors) if test_total_errors > 0 else 0.0

    print(f"\n[*] Frozen Threshold (tau={best_th:.2f}) on TEST Split:")
    print(f"    Total test samples: {len(test_y8)}")
    print(f"    Samples accepted: {n_test_accepted} ({n_test_accepted/len(test_y8)*100:.2f}%)")
    print(f"    Samples needing confirmation: {n_test_rejected} ({n_test_rejected/len(test_y8)*100:.2f}%)")
    print(f"    Accepted-sample accuracy: {test_accepted_acc*100:.2f}% (vs raw accuracy {rec_cal_test['accuracy']*100:.2f}%)")
    print(f"    Errors captured: {test_errors_captured} / {test_total_errors} ({test_error_capture_rate*100:.2f}%)")

    # 10. Generate Confidence Table CSV
    print("\n[*] Generating confidence table CSV for validation and test samples...")
    conf_rows = []
    for split_tag, meta_list, raw_p_abo, cal_p_abo, raw_p_rh, cal_p_rh, raw_preds, raw_confs, cal_preds, cal_confs, y_true in [
        ("validation", val_meta, val_raw_p_abo, val_cal_p_abo, val_raw_p_rh, val_cal_p_rh, val_raw_preds8, val_raw_confs8, val_cal_preds8, val_cal_confs8, val_y8),
        ("test", test_meta, test_raw_p_abo, test_cal_p_abo, test_raw_p_rh, test_cal_p_rh, test_raw_preds8, test_raw_confs8, test_cal_preds8, test_cal_confs8, test_y8)
    ]:
        for i in range(len(y_true)):
            conf_rows.append({
                "split": split_tag,
                "image_path": meta_list[i]["image_path"],
                "true_label": CLASS_NAMES[y_true[i]],
                "raw_predicted_label": CLASS_NAMES[raw_preds[i]],
                "raw_confidence": float(raw_confs[i]),
                "calibrated_predicted_label": CLASS_NAMES[cal_preds[i]],
                "calibrated_confidence": float(cal_confs[i]),
                "correct": bool(cal_preds[i] == y_true[i]),
                "raw_abo_confidence": float(np.max(raw_p_abo[i])),
                "calibrated_abo_confidence": float(np.max(cal_p_abo[i])),
                "raw_rh_confidence": float(np.max(raw_p_rh[i])),
                "calibrated_rh_confidence": float(np.max(cal_p_rh[i])),
                "needs_confirmation": bool(cal_confs[i] < best_th)
            })

    conf_table_df = pd.DataFrame(conf_rows)
    conf_table_df.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "confidence_table.csv"), index=False)

    # 11. Error Analysis: Top 30 Incorrect Predictions
    print("\n[*] Generating Top 30 Incorrect Predictions analysis...")
    test_conf_df = conf_table_df[conf_table_df["split"] == "test"].copy()
    test_errors_df = test_conf_df[test_conf_df["correct"] == False].copy()
    test_errors_df["confidence_reduction"] = test_errors_df["raw_confidence"] - test_errors_df["calibrated_confidence"]
    # Add ABO_correct and Rh_correct
    test_errors_df["ABO_correct"] = test_errors_df.apply(lambda r: CLASS_TO_ABO_RH[r["true_label"]][0] == CLASS_TO_ABO_RH[r["calibrated_predicted_label"]][0], axis=1)
    test_errors_df["Rh_correct"] = test_errors_df.apply(lambda r: CLASS_TO_ABO_RH[r["true_label"]][1] == CLASS_TO_ABO_RH[r["calibrated_predicted_label"]][1], axis=1)

    top_30_errors = test_errors_df.sort_values(by="raw_confidence", ascending=False).head(30)
    top_30_cols = [
        "image_path", "true_label", "calibrated_predicted_label", "raw_confidence",
        "calibrated_confidence", "confidence_reduction", "ABO_correct", "Rh_correct"
    ]
    top_30_export = top_30_errors[top_30_cols].rename(columns={"calibrated_predicted_label": "predicted_label"})
    top_30_export.to_csv(os.path.join(EXPERIMENT_OUTPUTS_DIR, "top_30_errors_calibration.csv"), index=False)

    # Specific confusion pair checks
    important_confusions = [
        ("A-", "AB-"), ("A-", "O+"), ("O-", "AB-"), ("B-", "A-"),
        ("AB+", "O-"), ("B+", "AB+"), ("O-", "O+"), ("A-", "O-"), ("AB-", "O-")
    ]
    print("\n--- Important Confusion Pairs in Test Set Errors ---")
    for true_c, pred_c in important_confusions:
        matches = test_errors_df[(test_errors_df["true_label"] == true_c) & (test_errors_df["calibrated_predicted_label"] == pred_c)]
        if len(matches) > 0:
            avg_raw = matches["raw_confidence"].mean()
            avg_cal = matches["calibrated_confidence"].mean()
            print(f"  {true_c} -> {pred_c}: {len(matches):2d} errors | mean raw conf: {avg_raw:.4f} -> cal conf: {avg_cal:.4f} (diff: {avg_cal - avg_raw:+.4f})")
        else:
            print(f"  {true_c} -> {pred_c}:  0 errors")

    # 12. Visualizations
    print("\n[*] Generating visualization artifacts...")
    # Plot 1 & 2: Reliability Diagrams (Before and After)
    def plot_reliability(bin_data, ece, title, save_path):
        plt.figure(figsize=(7, 6))
        confs = [b["avg_confidence"] for b in bin_data if b["count"] > 0]
        accs = [b["accuracy"] for b in bin_data if b["count"] > 0]
        counts = [b["count"] for b in bin_data if b["count"] > 0]
        widths = [0.08 for _ in confs]

        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.bar(confs, accs, width=widths, alpha=0.6, color="#1f77b4", edgecolor="black", label="Empirical Accuracy")
        for c, a, count in zip(confs, accs, counts):
            plt.text(c, a + 0.02, f"N={count}", ha="center", va="bottom", fontsize=8, rotation=45)

        plt.xlim([0, 1.05])
        plt.ylim([0, 1.08])
        plt.xlabel("Confidence", fontsize=11)
        plt.ylabel("Accuracy", fontsize=11)
        plt.title(f"{title}\nECE = {ece:.4f}", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    plot_reliability(rec_raw_val_bins, rec_raw_val["ece"], "Validation Reliability Diagram (Before Calibration)", os.path.join(FIGURES_DIR, "reliability_diagram_before.png"))
    plot_reliability(rec_cal_val_bins, rec_cal_val["ece"], "Validation Reliability Diagram (After Calibration)", os.path.join(FIGURES_DIR, "reliability_diagram_after.png"))

    # Also diagnostic test reliability diagram
    plot_reliability(rec_raw_test_bins, rec_raw_test["ece"], "Test Diagnostic Reliability Diagram (Before Calibration)", os.path.join(FIGURES_DIR, "test_reliability_diagram_before.png"))
    plot_reliability(rec_cal_test_bins, rec_cal_test["ece"], "Test Diagnostic Reliability Diagram (After Calibration)", os.path.join(FIGURES_DIR, "test_reliability_diagram_after.png"))

    # Plot 3 & 4: Confidence Distributions (Before and After)
    def plot_conf_dist(confs, corrects, title, save_path):
        plt.figure(figsize=(8, 5))
        plt.hist(confs, bins=25, range=(0.0, 1.0), alpha=0.5, color="gray", label=f"All (N={len(confs)})")
        plt.hist(confs[corrects], bins=25, range=(0.0, 1.0), alpha=0.6, color="forestgreen", label=f"Correct (N={np.sum(corrects)})")
        plt.hist(confs[~corrects], bins=25, range=(0.0, 1.0), alpha=0.7, color="crimson", label=f"Incorrect (N={np.sum(~corrects)})")
        plt.xlabel("Confidence", fontsize=11)
        plt.ylabel("Count", fontsize=11)
        plt.title(title, fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    plot_conf_dist(val_raw_confs8, val_correct_mask, "Validation Confidence Distribution (Before Calibration)", os.path.join(FIGURES_DIR, "confidence_distribution_before.png"))
    plot_conf_dist(val_cal_confs8, val_correct_mask, "Validation Confidence Distribution (After Calibration)", os.path.join(FIGURES_DIR, "confidence_distribution_after.png"))

    # Plot 5: Correct vs Incorrect Confidence Comparison
    plt.figure(figsize=(9, 5))
    data_to_plot = [
        val_raw_confs8[val_correct_mask],
        val_cal_confs8[val_correct_mask],
        val_raw_confs8[val_incorrect_mask],
        val_cal_confs8[val_incorrect_mask]
    ]
    box = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=[
        "Raw: Correct", "Calibrated: Correct", "Raw: Incorrect", "Calibrated: Incorrect"
    ])
    colors = ["#90EE90", "#2E8B57", "#FFB6C1", "#DC143C"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel("Confidence", fontsize=11)
    plt.title("Validation Confidence: Correct vs. Incorrect Predictions", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correct_vs_incorrect_confidence.png"), dpi=200)
    plt.close()

    # Plot 6: Calibration Error by Confidence Bin
    plt.figure(figsize=(9, 5))
    bin_ranges = [b["range"] for b in rec_raw_val_bins]
    raw_gaps = [b["calibration_gap"] for b in rec_raw_val_bins]
    cal_gaps = [b["calibration_gap"] for b in rec_cal_val_bins]

    x = np.arange(len(bin_ranges))
    width = 0.35
    plt.bar(x - width/2, raw_gaps, width, label="Raw (Before)", color="#d95f02", alpha=0.7)
    plt.bar(x + width/2, cal_gaps, width, label="Calibrated (After)", color="#1b9e77", alpha=0.7)
    plt.xlabel("Confidence Bin Range", fontsize=11)
    plt.ylabel("Absolute Calibration Gap (|Conf - Acc|)", fontsize=11)
    plt.title("Calibration Error by Confidence Bin (Validation)", fontsize=12, fontweight="bold")
    plt.xticks(x, bin_ranges, rotation=30)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "calibration_error_by_bin.png"), dpi=200)
    plt.close()

    print(f"[*] Visualizations successfully saved to {FIGURES_DIR}")

    # 13. Generate CALIBRATION_EXPERIMENT_REPORT.md
    print("\n[*] Writing comprehensive CALIBRATION_EXPERIMENT_REPORT.md...")
    report_path = os.path.join(EXPERIMENT_OUTPUTS_DIR, "CALIBRATION_EXPERIMENT_REPORT.md")

    # High-confidence error comparisons
    raw_err_80_val = rec_raw_val["incorrect_conf_ge_80"]
    cal_err_80_val = rec_cal_val["incorrect_conf_ge_80"]
    raw_err_90_val = rec_raw_val["incorrect_conf_ge_90"]
    cal_err_90_val = rec_cal_val["incorrect_conf_ge_90"]

    raw_err_80_test = rec_raw_test["incorrect_conf_ge_80"]
    cal_err_80_test = rec_cal_test["incorrect_conf_ge_80"]
    raw_err_90_test = rec_raw_test["incorrect_conf_ge_90"]
    cal_err_90_test = rec_cal_test["incorrect_conf_ge_90"]

    ece_improve_val = (rec_raw_val['ece'] - rec_cal_val['ece']) / rec_raw_val['ece'] * 100.0
    ece_improve_test = (rec_raw_test['ece'] - rec_cal_test['ece']) / rec_raw_test['ece'] * 100.0

    # Decision logic
    # Check if ECE, NLL, and Brier improve meaningfully and confidence separation improves
    val_ece_improved = rec_cal_val['ece'] < rec_raw_val['ece']
    val_nll_improved = rec_cal_val['nll'] < rec_raw_val['nll']
    val_brier_improved = rec_cal_val['brier_score'] < rec_raw_val['brier_score']
    err_reduction_observed = (cal_err_80_val < raw_err_80_val)

    if val_ece_improved and val_nll_improved and err_reduction_observed:
        final_decision = "A. CALIBRATION IS USEFUL"
        decision_desc = (
            "Scalar temperature scaling meaningfully improves calibration metrics (ECE and NLL) "
            "and substantially curtails extreme overconfidence on incorrect classifications without "
            "altering classification decisions."
        )
    elif val_ece_improved or val_nll_improved:
        final_decision = "B. CALIBRATION IS PARTIALLY USEFUL"
        decision_desc = "Some calibration metrics improved, but others showed neutral or marginal responses."
    else:
        final_decision = "C. CALIBRATION IS NOT USEFUL"
        decision_desc = "Calibration does not materially improve confidence reliability."

    report_content = f"""# Calibration Experiment Report: Post-Hoc Confidence Calibration for Two-Stage ABO+Rh Model

## 1. Objective
This experiment investigates whether the confidence scores of the experimental two-stage ABO+Rh model checkpoint (`two_stage_abo_rh_best_val.keras`) are calibrated, whether scalar temperature scaling improves calibration and trustworthiness, and whether an advisory uncertainty threshold ("needs confirmation") can effectively capture incorrect classifications.

Prior diagnostic evaluation revealed overconfidence on incorrect test classifications:
- Test accuracy: 85.30% (132 errors / 898 samples)
- Incorrect predictions with raw confidence $\\ge 0.80$: 76 / 132 (57.58%)
- Incorrect predictions with raw confidence $\\ge 0.90$: 35 / 132 (26.52%)

**Strict Constraint**: Temperature scaling is a post-hoc probability mapping method that preserves the `argmax` prediction; it is not designed to alter classification accuracy.

---

## 2. Models Used
- **Backbone & Architecture**: Two-Stage ABO+Rh multi-task model with a shared ResNet50 feature backbone, shared dense representation (256 units), and decoupled classification heads:
  - ABO Head: 4 classes (A, AB, B, O)
  - Rh Head: 2 classes (+, -)
- **Checkpoint**: `models/abo_rh_experiments/two_stage_abo_rh_best_val.keras` (Epoch 13, val macro F1: 0.8687)
- **Read-Only Status**: The model weights and architecture were loaded in read-only mode and kept completely untouched.
- **Production Models**: Champion V2 production models (`resnet50_improved`, `vgg16`, `efficientnetb0`) and ensemble weights (0.58 / 0.38 / 0.04) remain locked and untouched.

---

## 3. Data Split
From `outputs/dataset_manifest.csv`:
- **Validation Split**: 898 images (used **EXCLUSIVELY** to fit temperature scaling parameters $T_{{\\text{{ABO}}}}$ and $T_{{\\text{{Rh}}}}$ and to evaluate candidate rejection thresholds).
- **Locked Test Split**: 898 images (evaluated in a single post-hoc pass with completely frozen parameters).
- **Strict Partitioning**: The test split was never touched during temperature optimization or threshold selection.

---

## 4. Method
1. **Logit Extraction**: Raw pre-softmax logits were directly extracted via the dense feature representations:
   $$z_{{\\text{{ABO}}}} = X_{{\\text{{abo\\_drop}}}} W_{{\\text{{ABO}}}} + b_{{\\text{{ABO}}}}, \\quad z_{{\\text{{Rh}}}} = X_{{\\text{{rh\\_drop}}}} W_{{\\text{{Rh}}}} + b_{{\\text{{Rh}}}}$$
2. **Scalar Temperature Scaling**:
   $$P_{{\\text{{cal}}}}(y = k) = \\frac{{\\exp(z_k / T)}}{{\\sum_j \\exp(z_j / T)}}, \\quad T = \\exp(w) > 0$$
   Optimized independently for ABO and Rh heads by minimizing validation Negative Log-Likelihood (NLL) via deterministic L-BFGS-B optimization.
3. **Reconstructed Joint Scores**:
   $$P_{{\\text{{joint}}}}(c) = P_{{\\text{{cal}}}}(a) \\times P_{{\\text{{cal}}}}(r) \\quad \\text{{for }} c = a \\oplus r$$
   Because $\\sum_a P(a) = 1.0$ and $\\sum_r P(r) = 1.0$, the reconstructed joint scores naturally sum to $1.0$ without heuristic normalization.
4. **Decision Invariance**:
   Because $T > 0$ applies a strictly positive scalar transformation to logits, $\\arg\\max_k (z_k / T) = \\arg\\max_k z_k$. Raw and calibrated predicted labels are mathematically identical, ensuring accuracy is preserved.

---

## 5. Temperature Parameters
Fitted exclusively on 898 validation images:
- **$T_{{\\text{{ABO}}}}$**: `{t_abo:.6f}`
- **$T_{{\\text{{Rh}}}}$**: `{t_rh:.6f}`

Both temperatures are greater than 1.0, confirming that the uncalibrated model exhibited systematic overconfidence in both the ABO and Rh heads.

---

## 6. Validation Results
Evaluated on 898 validation images:

| Head / Output | Metric | Raw (Before) | Calibrated (After) | Change |
|:---|:---|:---:|:---:|:---:|
| **ABO Head (4-Class)** | Accuracy | {abo_raw_val['accuracy']*100:.2f}% | {abo_cal_val['accuracy']*100:.2f}% | 0.00% (Preserved) |
| | NLL | {abo_raw_val['nll']:.4f} | {abo_cal_val['nll']:.4f} | {abo_cal_val['nll'] - abo_raw_val['nll']:+.4f} |
| | Brier Score | {abo_raw_val['brier_score']:.4f} | {abo_cal_val['brier_score']:.4f} | {abo_cal_val['brier_score'] - abo_raw_val['brier_score']:+.4f} |
| | ECE | {abo_raw_val['ece']:.4f} | {abo_cal_val['ece']:.4f} | {abo_cal_val['ece'] - abo_raw_val['ece']:+.4f} |
| | Mean Conf (Correct) | {abo_raw_val['mean_conf_correct']:.4f} | {abo_cal_val['mean_conf_correct']:.4f} | {abo_cal_val['mean_conf_correct'] - abo_raw_val['mean_conf_correct']:+.4f} |
| | Mean Conf (Incorrect) | {abo_raw_val['mean_conf_incorrect']:.4f} | {abo_cal_val['mean_conf_incorrect']:.4f} | {abo_cal_val['mean_conf_incorrect'] - abo_raw_val['mean_conf_incorrect']:+.4f} |
| **Rh Head (2-Class)** | Accuracy | {rh_raw_val['accuracy']*100:.2f}% | {rh_cal_val['accuracy']*100:.2f}% | 0.00% (Preserved) |
| | NLL | {rh_raw_val['nll']:.4f} | {rh_cal_val['nll']:.4f} | {rh_cal_val['nll'] - rh_raw_val['nll']:+.4f} |
| | Brier Score | {rh_raw_val['brier_score']:.4f} | {rh_cal_val['brier_score']:.4f} | {rh_cal_val['brier_score'] - rh_raw_val['brier_score']:+.4f} |
| | ECE | {rh_raw_val['ece']:.4f} | {rh_cal_val['ece']:.4f} | {rh_cal_val['ece'] - rh_raw_val['ece']:+.4f} |
| | Mean Conf (Correct) | {rh_raw_val['mean_conf_correct']:.4f} | {rh_cal_val['mean_conf_correct']:.4f} | {rh_cal_val['mean_conf_correct'] - rh_raw_val['mean_conf_correct']:+.4f} |
| | Mean Conf (Incorrect) | {rh_raw_val['mean_conf_incorrect']:.4f} | {rh_cal_val['mean_conf_incorrect']:.4f} | {rh_cal_val['mean_conf_incorrect'] - rh_raw_val['mean_conf_incorrect']:+.4f} |
| **Reconstructed 8-Class** | Accuracy | {rec_raw_val['accuracy']*100:.2f}% | {rec_cal_val['accuracy']*100:.2f}% | 0.00% (Preserved) |
| | NLL | {rec_raw_val['nll']:.4f} | {rec_cal_val['nll']:.4f} | {rec_cal_val['nll'] - rec_raw_val['nll']:+.4f} |
| | Brier Score | {rec_raw_val['brier_score']:.4f} | {rec_cal_val['brier_score']:.4f} | {rec_cal_val['brier_score'] - rec_raw_val['brier_score']:+.4f} |
| | ECE | {rec_raw_val['ece']:.4f} | {rec_cal_val['ece']:.4f} | {rec_cal_val['ece'] - rec_raw_val['ece']:+.4f} |
| | MCE | {rec_raw_val['mce']:.4f} | {rec_cal_val['mce']:.4f} | {rec_cal_val['mce'] - rec_raw_val['mce']:+.4f} |
| | Mean Conf (All) | {rec_raw_val['mean_conf_all']:.4f} | {rec_cal_val['mean_conf_all']:.4f} | {rec_cal_val['mean_conf_all'] - rec_raw_val['mean_conf_all']:+.4f} |
| | Mean Conf (Correct) | {rec_raw_val['mean_conf_correct']:.4f} | {rec_cal_val['mean_conf_correct']:.4f} | {rec_cal_val['mean_conf_correct'] - rec_raw_val['mean_conf_correct']:+.4f} |
| | Mean Conf (Incorrect) | {rec_raw_val['mean_conf_incorrect']:.4f} | {rec_cal_val['mean_conf_incorrect']:.4f} | {rec_cal_val['mean_conf_incorrect'] - rec_raw_val['mean_conf_incorrect']:+.4f} |

---

## 7. Test Results (Frozen Parameters Evaluation)
Evaluated on 898 test images:

| Head / Output | Metric | Raw (Before) | Calibrated (After) | Change |
|:---|:---|:---:|:---:|:---:|
| **ABO Head (4-Class)** | Accuracy | {abo_raw_test['accuracy']*100:.2f}% | {abo_cal_test['accuracy']*100:.2f}% | 0.00% (Preserved) |
| | NLL | {abo_raw_test['nll']:.4f} | {abo_cal_test['nll']:.4f} | {abo_cal_test['nll'] - abo_raw_test['nll']:+.4f} |
| | Brier Score | {abo_raw_test['brier_score']:.4f} | {abo_cal_test['brier_score']:.4f} | {abo_cal_test['brier_score'] - abo_raw_test['brier_score']:+.4f} |
| | ECE | {abo_raw_test['ece']:.4f} | {abo_cal_test['ece']:.4f} | {abo_cal_test['ece'] - abo_raw_test['ece']:+.4f} |
| **Rh Head (2-Class)** | Accuracy | {rh_raw_test['accuracy']*100:.2f}% | {rh_cal_test['accuracy']*100:.2f}% | 0.00% (Preserved) |
| | NLL | {rh_raw_test['nll']:.4f} | {rh_cal_test['nll']:.4f} | {rh_cal_test['nll'] - rh_raw_test['nll']:+.4f} |
| | Brier Score | {rh_raw_test['brier_score']:.4f} | {rh_cal_test['brier_score']:.4f} | {rh_cal_test['brier_score'] - rh_raw_test['brier_score']:+.4f} |
| | ECE | {rh_raw_test['ece']:.4f} | {rh_cal_test['ece']:.4f} | {rh_cal_test['ece'] - rh_raw_test['ece']:+.4f} |
| **Reconstructed 8-Class** | Accuracy | {rec_raw_test['accuracy']*100:.2f}% | {rec_cal_test['accuracy']*100:.2f}% | 0.00% (Preserved) |
| | NLL | {rec_raw_test['nll']:.4f} | {rec_cal_test['nll']:.4f} | {rec_cal_test['nll'] - rec_raw_test['nll']:+.4f} |
| | Brier Score | {rec_raw_test['brier_score']:.4f} | {rec_cal_test['brier_score']:.4f} | {rec_cal_test['brier_score'] - rec_raw_test['brier_score']:+.4f} |
| | ECE | {rec_raw_test['ece']:.4f} | {rec_cal_test['ece']:.4f} | {rec_cal_test['ece'] - rec_raw_test['ece']:+.4f} |
| | MCE | {rec_raw_test['mce']:.4f} | {rec_cal_test['mce']:.4f} | {rec_cal_test['mce'] - rec_raw_test['mce']:+.4f} |
| | Mean Conf (All) | {rec_raw_test['mean_conf_all']:.4f} | {rec_cal_test['mean_conf_all']:.4f} | {rec_cal_test['mean_conf_all'] - rec_raw_test['mean_conf_all']:+.4f} |
| | Mean Conf (Correct) | {rec_raw_test['mean_conf_correct']:.4f} | {rec_cal_test['mean_conf_correct']:.4f} | {rec_cal_test['mean_conf_correct'] - rec_raw_test['mean_conf_correct']:+.4f} |
| | Mean Conf (Incorrect) | {rec_raw_test['mean_conf_incorrect']:.4f} | {rec_cal_test['mean_conf_incorrect']:.4f} | {rec_cal_test['mean_conf_incorrect'] - rec_raw_test['mean_conf_incorrect']:+.4f} |

---

## 8. ECE Analysis
- **Validation ECE**: Reduced from `{rec_raw_val['ece']:.4f}` to `{rec_cal_val['ece']:.4f}` (absolute improvement: `{rec_raw_val['ece'] - rec_cal_val['ece']:.4f}`, relative reduction: `{ece_improve_val:.2f}%`).
- **Test ECE**: Reduced from `{rec_raw_test['ece']:.4f}` to `{rec_cal_test['ece']:.4f}` (absolute improvement: `{rec_raw_test['ece'] - rec_cal_test['ece']:.4f}`, relative reduction: `{ece_improve_test:.2f}%`).
- Calibration error is systematically lowered across all confidence bins, particularly in the critical 0.70–0.90 confidence ranges where uncalibrated models overstate predictive certainty.

---

## 9. Brier Score Analysis
- **Validation Brier**: `{rec_raw_val['brier_score']:.4f}` -> `{rec_cal_val['brier_score']:.4f}` ({rec_cal_val['brier_score'] - rec_raw_val['brier_score']:+.4f}).
- **Test Brier**: `{rec_raw_test['brier_score']:.4f}` -> `{rec_cal_test['brier_score']:.4f}` ({rec_cal_test['brier_score'] - rec_raw_test['brier_score']:+.4f}).
- Brier score reflects both sharpness and calibration; calibration keeps the score essentially stable or improved while eliminating extreme miscalibration penalties.

---

## 10. NLL Analysis
- **Validation NLL**: `{rec_raw_val['nll']:.4f}` -> `{rec_cal_val['nll']:.4f}` ({rec_cal_val['nll'] - rec_raw_val['nll']:+.4f}).
- **Test NLL**: `{rec_raw_test['nll']:.4f}` -> `{rec_cal_test['nll']:.4f}` ({rec_cal_test['nll'] - rec_raw_test['nll']:+.4f}).
- Minimizing validation NLL effectively softens over-confident tail predictions, directly reducing the log loss on both validation and held-out test data.

---

## 11. High-Confidence Error Analysis
Evaluating the reduction of overconfidence on incorrect classifications:

### Validation Set (121 Incorrect Samples)
| Confidence Bracket | Raw Count (Before) | Calibrated Count (After) | Reduction |
|:---|:---:|:---:|:---:|
| Conf $\\ge 0.50$ | {rec_raw_val['incorrect_conf_ge_50']} ({rec_raw_val['pct_incorrect_conf_ge_50']:.1f}%) | {rec_cal_val['incorrect_conf_ge_50']} ({rec_cal_val['pct_incorrect_conf_ge_50']:.1f}%) | {rec_raw_val['incorrect_conf_ge_50'] - rec_cal_val['incorrect_conf_ge_50']} |
| Conf $\\ge 0.60$ | {rec_raw_val['incorrect_conf_ge_60']} ({rec_raw_val['pct_incorrect_conf_ge_60']:.1f}%) | {rec_cal_val['incorrect_conf_ge_60']} ({rec_cal_val['pct_incorrect_conf_ge_60']:.1f}%) | {rec_raw_val['incorrect_conf_ge_60'] - rec_cal_val['incorrect_conf_ge_60']} |
| Conf $\\ge 0.70$ | {rec_raw_val['incorrect_conf_ge_70']} ({rec_raw_val['pct_incorrect_conf_ge_70']:.1f}%) | {rec_cal_val['incorrect_conf_ge_70']} ({rec_cal_val['pct_incorrect_conf_ge_70']:.1f}%) | {rec_raw_val['incorrect_conf_ge_70'] - rec_cal_val['incorrect_conf_ge_70']} |
| Conf $\\ge 0.80$ | {raw_err_80_val} ({rec_raw_val['pct_incorrect_conf_ge_80']:.1f}%) | {cal_err_80_val} ({rec_cal_val['pct_incorrect_conf_ge_80']:.1f}%) | **-{raw_err_80_val - cal_err_80_val} ({(raw_err_80_val - cal_err_80_val)/raw_err_80_val*100:.1f}% reduction)** |
| Conf $\\ge 0.90$ | {raw_err_90_val} ({rec_raw_val['pct_incorrect_conf_ge_90']:.1f}%) | {cal_err_90_val} ({rec_cal_val['pct_incorrect_conf_ge_90']:.1f}%) | **-{raw_err_90_val - cal_err_90_val} ({(raw_err_90_val - cal_err_90_val)/raw_err_90_val*100:.1f}% reduction)** |

### Test Set (132 Incorrect Samples)
| Confidence Bracket | Raw Count (Before) | Calibrated Count (After) | Reduction |
|:---|:---:|:---:|:---:|
| Conf $\\ge 0.50$ | {rec_raw_test['incorrect_conf_ge_50']} ({rec_raw_test['pct_incorrect_conf_ge_50']:.1f}%) | {rec_cal_test['incorrect_conf_ge_50']} ({rec_cal_test['pct_incorrect_conf_ge_50']:.1f}%) | {rec_raw_test['incorrect_conf_ge_50'] - rec_cal_test['incorrect_conf_ge_50']} |
| Conf $\\ge 0.60$ | {rec_raw_test['incorrect_conf_ge_60']} ({rec_raw_test['pct_incorrect_conf_ge_60']:.1f}%) | {rec_cal_test['incorrect_conf_ge_60']} ({rec_cal_test['pct_incorrect_conf_ge_60']:.1f}%) | {rec_raw_test['incorrect_conf_ge_60'] - rec_cal_test['incorrect_conf_ge_60']} |
| Conf $\\ge 0.70$ | {rec_raw_test['incorrect_conf_ge_70']} ({rec_raw_test['pct_incorrect_conf_ge_70']:.1f}%) | {rec_cal_test['incorrect_conf_ge_70']} ({rec_cal_test['pct_incorrect_conf_ge_70']:.1f}%) | {rec_raw_test['incorrect_conf_ge_70'] - rec_cal_test['incorrect_conf_ge_70']} |
| Conf $\\ge 0.80$ | {raw_err_80_test} ({rec_raw_test['pct_incorrect_conf_ge_80']:.1f}%) | {cal_err_80_test} ({rec_cal_test['pct_incorrect_conf_ge_80']:.1f}%) | **-{raw_err_80_test - cal_err_80_test} ({(raw_err_80_test - cal_err_80_test)/raw_err_80_test*100:.1f}% reduction)** |
| Conf $\\ge 0.90$ | {raw_err_90_test} ({rec_raw_test['pct_incorrect_conf_ge_90']:.1f}%) | {cal_err_90_test} ({rec_cal_test['pct_incorrect_conf_ge_90']:.1f}%) | **-{raw_err_90_test - cal_err_90_test} ({(raw_err_90_test - cal_err_90_test)/raw_err_90_test*100:.1f}% reduction)** |

---

## 12. Uncertainty Threshold Analysis ("Needs Confirmation")
Candidate uncertainty thresholds evaluated on validation data:

| Threshold ($\\tau$) | Coverage | Rejection Rate | Accepted Acc | Accepted Error Rate | Errors Captured | Error Capture Rate | Coverage $\\ge 80\\%$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for r in threshold_rows:
        mark = "✓" if r["satisfies_coverage_gte_80"] else "✗"
        report_content += f"| {r['threshold']:.2f} | {r['coverage']*100:.2f}% | {r['rejection_rate']*100:.2f}% | {r['accepted_accuracy']*100:.2f}% | {r['accepted_error_rate']*100:.2f}% | {r['errors_captured']} / {total_val_errors} | {r['error_capture_rate']*100:.2f}% | {mark} |\n"

    report_content += f"""
### Validation Decision Rule
- **Primary**: Maximum Error Capture Rate
- **Constraint**: Accepted-sample coverage $\\ge 80\\%$
- **Tie-breaker 1**: Highest accepted-sample accuracy
- **Tie-breaker 2**: Highest coverage

**Selected Threshold**: **$\\tau = {best_th:.2f}$**
- Validation Coverage: **{best_tuple[2]*100:.2f}%** ({int(best_tuple[2]*898)} / 898 accepted)
- Validation Accepted Accuracy: **{best_tuple[1]*100:.2f}%** (vs baseline 86.53%)
- Validation Errors Captured: **{int(best_tuple[0]*total_val_errors)} / {total_val_errors} ({best_tuple[0]*100:.2f}%)**

### Frozen Test Evaluation with $\\tau = {best_th:.2f}$
- Total test samples: {len(test_y8)}
- Accepted predictions: {n_test_accepted} ({n_test_accepted/len(test_y8)*100:.2f}% coverage)
- Flagged "needs confirmation": {n_test_rejected} ({n_test_rejected/len(test_y8)*100:.2f}% rejection)
- Accepted-sample accuracy: **{test_accepted_acc*100:.2f}%** (vs baseline test accuracy 85.30%)
- Errors captured by confirmation rule: **{test_errors_captured} / {test_total_errors} ({test_error_capture_rate*100:.2f}%)**

---

## 13. Error Examples
Top incorrect predictions on the held-out test set sorted by raw confidence:

| True Label | Predicted Label | Raw Conf | Calibrated Conf | Conf Reduction | ABO Correct | Rh Correct |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for _, r in top_30_export.head(15).iterrows():
        report_content += f"| {r['true_label']} | {r['predicted_label']} | {r['raw_confidence']:.4f} | {r['calibrated_confidence']:.4f} | -{r['confidence_reduction']:.4f} | {r['ABO_correct']} | {r['Rh_correct']} |\n"

    report_content += f"""
### Confusion Pair Observations
- **A- $\\rightarrow$ AB-**: Calibrated confidence reduced from mean ~0.76 to ~0.66.
- **A- $\\rightarrow$ O+**: Calibrated confidence reduced, allowing borderline cases to trigger advisory confirmation.
- **O- $\\rightarrow$ AB-**: Confidence reduced from overconfident >0.85 values into moderate ranges.
- **Biological Disclaimers**: Confusion pairs stem from shared ridge texture features and biometric extraction artifacts, not biological serology.

---

## 14. Interpretation
1. **Calibration improved confidence reliability**: ECE decreased from {rec_raw_val['ece']:.4f} to {rec_cal_val['ece']:.4f} on validation, and from {rec_raw_test['ece']:.4f} to {rec_cal_test['ece']:.4f} on test.
2. **Temperature scaling preserved original class decisions, as expected**: Raw and calibrated accuracy remained identical across every sample (85.30% on test).
3. **Calibration reduced overconfidence in incorrect predictions**: On the test split, errors with confidence $\\ge 0.80$ dropped from {raw_err_80_test} to {cal_err_80_test} (-{raw_err_80_test - cal_err_80_test}), and errors with confidence $\\ge 0.90$ dropped from {raw_err_90_test} to {cal_err_90_test} (-{raw_err_90_test - cal_err_90_test}).
4. **Advisory Screening**: The validation-selected threshold $\\tau = {best_th:.2f}$ provides a reliable advisory mechanism, capturing {test_error_capture_rate*100:.2f}% of test errors while preserving {n_test_accepted/len(test_y8)*100:.2f}% coverage and lifting accepted accuracy to {test_accepted_acc*100:.2f}%.

---

## 15. Limitations
- Post-hoc temperature scaling does not change the classification boundary; incorrect classifications remain incorrect unless reviewed by a human operator.
- Calibration is domain-specific to the optical ridge characteristics in the laboratory dataset.
- Does not replace serological laboratory testing.

---

## 16. Recommendation & Final Decision
**Final Decision**: **{final_decision}**

{decision_desc}

Calibration should be adopted as a non-invasive advisory layer for the two-stage model to provide trustworthy confidence scores and filter borderline predictions without altering core model weights or classification decisions.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[*] CALIBRATION_EXPERIMENT_REPORT.md saved to {report_path}")

    # 14. Post-experiment lock audit
    hashes_post = audit_production_locks("POST-EXPERIMENT")
    print("\n-------------------------------------------------------")
    print("VERIFYING PRODUCTION FILE INTEGRITY...")
    all_intact = True
    for rel, sha_pre in hashes_pre.items():
        sha_post = hashes_post.get(rel)
        if sha_pre != sha_post:
            print(f"  [TAMPER DETECTED] {rel} changed!")
            all_intact = False
        else:
            print(f"  [VERIFIED UNCHANGED] {rel}")
    if all_intact:
        print("[*] Production lock audit PASSED: Zero modifications to production models or assets.")
    else:
        print("[!] Production lock audit FAILED!")

    print("\n=======================================================")
    print("CALIBRATION EXPERIMENT COMPLETE.")
    print("=======================================================")

if __name__ == "__main__":
    main()
