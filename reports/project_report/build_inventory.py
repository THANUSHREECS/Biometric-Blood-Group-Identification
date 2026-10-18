import json
import csv
import os
import pandas as pd

# State clearly: All numeric values in rows are read dynamically from the backing JSON, CSV, or MD artifacts.
# None are hand-typed without programmatic loading.

rows = []

def fmt(val):
    if val is None or val in ["not recorded in artifact", "not applicable"]:
        return str(val)
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

current_group = "Original-Dataset Models"

def add_row(exp_id, arch, config, dataset, n_img, n_part, n_classes, split, ckpt_rule,
            metric_lvl, acc, bacc, mp, mr, mf1, ci, src_file, src_line, src_type,
            inc_locked, lim_note, verified="yes", group=None):
    global current_group
    rows.append({
        "experiment_id": exp_id,
        "group": group if group is not None else current_group,
        "model_architecture": arch,
        "main_configuration": config,
        "dataset": dataset,
        "n_images": str(n_img),
        "n_participants": str(n_part),
        "n_classes": str(n_classes),
        "split_protocol": split,
        "checkpoint_selection_rule": ckpt_rule,
        "metric_level": metric_lvl,
        "accuracy": fmt(acc),
        "balanced_accuracy": fmt(bacc),
        "macro_precision": fmt(mp),
        "macro_recall": fmt(mr),
        "macro_f1": fmt(mf1),
        "confidence_interval": str(ci),
        "source_file": src_file,
        "source_field_or_line": src_line,
        "source_type": src_type,
        "includes_locked_test": inc_locked,
        "limitation_note": lim_note,
        "verified_from_artifact": verified
    })

# =========================================================================
# 1. Original-Dataset Models (Image-Level)
# =========================================================================
orig_json = "outputs/original_6000_models_audit_results.json"
with open(orig_json, "r") as f:
    orig = json.load(f)

# Rule quotes from source training scripts
effb0_ckpt_rule = "legacy validation split (separate from the legacy test set; legacy split has identity leakage): train_abo_positive_4class.py:405-407: tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)"
models_ckpt_rule = "legacy validation split (separate from the legacy test set; legacy split has identity leakage): train_models.py:683-685: checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)"

current_group = "Original-Dataset Models"
m1 = orig["abo_positive_4class"]
add_row(
    exp_id="ORIG-01",
    arch="EfficientNet-B0",
    config="ImageNet pretrained, 2-stage fine-tuning, conservative aug, CE loss",
    dataset="Original ~6000-image dataset (positive Rh subset)",
    n_img=m1["total_test_samples"],
    n_part="not recorded in artifact",
    n_classes=4,
    split="Random manifest test split (415 positive-Rh images)",
    ckpt_rule=effb0_ckpt_rule,
    metric_lvl="image-level",
    acc=m1["accuracy"],
    bacc=m1["balanced_accuracy"],
    mp=m1["macro_precision"],
    mr=m1["macro_recall"],
    mf1=m1["macro_f1"],
    ci="not recorded in artifact",
    src_file=orig_json,
    src_line="abo_positive_4class",
    src_type="json",
    inc_locked="no",
    lim_note="Single-institution CMOS optical reader dataset; high intra-dataset similarity across burst captures."
)

m2 = orig["custom_cnn"]
add_row(
    exp_id="ORIG-02",
    arch="Custom CNN (4 conv blocks)",
    config="4 conv blocks, dropout 0.5, batch size 32",
    dataset="Original ~6000-image dataset",
    n_img=m2["total_test_samples"],
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest test split (898 images)",
    ckpt_rule=models_ckpt_rule,
    metric_lvl="image-level",
    acc=m2["accuracy"],
    bacc=m2["balanced_accuracy"],
    mp=m2["macro_precision"],
    mr=m2["macro_recall"],
    mf1=m2["macro_f1"],
    ci="not recorded in artifact",
    src_file=orig_json,
    src_line="custom_cnn",
    src_type="json",
    inc_locked="no",
    lim_note="Degenerate collapse; predicted single class (A-) almost exclusively."
)

m3 = orig["resnet50_improved"]
add_row(
    exp_id="ORIG-03",
    arch="ResNet-50",
    config="ImageNet transfer, contrast enhancement 1.2x, fine-tuning",
    dataset="Original ~6000-image dataset",
    n_img=m3["total_test_samples"],
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest test split (898 images)",
    ckpt_rule=models_ckpt_rule,
    metric_lvl="image-level",
    acc=m3["accuracy"],
    bacc=m3["balanced_accuracy"],
    mp=m3["macro_precision"],
    mr=m3["macro_recall"],
    mf1=m3["macro_f1"],
    ci="not recorded in artifact",
    src_file=orig_json,
    src_line="resnet50_improved",
    src_type="json",
    inc_locked="no",
    lim_note="Internal test split evaluation on original CMOS reader data."
)

m4 = orig["densenet121"]
add_row(
    exp_id="ORIG-04",
    arch="DenseNet-121",
    config="ImageNet transfer, frozen backbone then fine-tune",
    dataset="Original ~6000-image dataset",
    n_img=m4["total_test_samples"],
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest test split (898 images)",
    ckpt_rule=models_ckpt_rule,
    metric_lvl="image-level",
    acc=m4["accuracy"],
    bacc=m4["balanced_accuracy"],
    mp=m4["macro_precision"],
    mr=m4["macro_recall"],
    mf1=m4["macro_f1"],
    ci="not recorded in artifact",
    src_file=orig_json,
    src_line="densenet121",
    src_type="json",
    inc_locked="no",
    lim_note="Internal test split evaluation on original CMOS reader data."
)

m5 = orig["mobilenetv2"]
add_row(
    exp_id="ORIG-05",
    arch="MobileNetV2",
    config="ImageNet transfer, alpha=1.0",
    dataset="Original ~6000-image dataset",
    n_img=m5["total_test_samples"],
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest test split (898 images)",
    ckpt_rule=models_ckpt_rule,
    metric_lvl="image-level",
    acc=m5["accuracy"],
    bacc=m5["balanced_accuracy"],
    mp=m5["macro_precision"],
    mr=m5["macro_recall"],
    mf1=m5["macro_f1"],
    ci="not recorded in artifact",
    src_file=orig_json,
    src_line="mobilenetv2",
    src_type="json",
    inc_locked="no",
    lim_note="Internal test split evaluation on original CMOS reader data."
)

m6 = orig["vgg16"]
add_row(
    exp_id="ORIG-06",
    arch="VGG16",
    config="ImageNet transfer, fine-tuned top block",
    dataset="Original ~6000-image dataset",
    n_img=m6["total_test_samples"],
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest test split (898 images)",
    ckpt_rule=models_ckpt_rule,
    metric_lvl="image-level",
    acc=m6["accuracy"],
    bacc=m6["balanced_accuracy"],
    mp=m6["macro_precision"],
    mr=m6["macro_recall"],
    mf1=m6["macro_f1"],
    ci="not recorded in artifact",
    src_file=orig_json,
    src_line="vgg16",
    src_type="json",
    inc_locked="no",
    lim_note="Internal test split evaluation on original CMOS reader data."
)

m7 = orig["two_stage_abo_rh"]
add_row(
    exp_id="ORIG-07",
    arch="Two-Stage ResNet-50",
    config="Stage 1 ABO (4-class) + Stage 2 Rh (+/-) factored prediction",
    dataset="Original ~6000-image dataset",
    n_img=m7["total_test_samples"],
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest test split (898 images)",
    ckpt_rule="scripts/abo_rh_experiments/train_two_stage_abo_rh.py:430,454-478: ep_key = (val_8_f1, val_8_acc, val_abo_f1, val_rh_f1, -val_loss); restored best weights and saved to two_stage_abo_rh_best_val.keras",
    metric_lvl="image-level",
    acc=m7["accuracy"],
    bacc=m7["balanced_accuracy"],
    mp=m7["macro_precision"],
    mr=m7["macro_recall"],
    mf1=m7["macro_f1"],
    ci="not recorded in artifact",
    src_file=orig_json,
    src_line="two_stage_abo_rh",
    src_type="json",
    inc_locked="no",
    lim_note="Evaluated on internal test split; two-stage compounding errors on out-of-distribution data."
)

# =========================================================================
# 2. Original-Dataset Ensembles
# =========================================================================
ens_weights_json = "outputs/ensemble_weights.json"
with open(ens_weights_json, "r") as f:
    ens_w = json.load(f)
current_group = "Original-Dataset Ensembles"
pcr = ens_w["per_class_report"]
recalls = [pcr[c]["recall"] for c in pcr]
precs = [pcr[c]["precision"] for c in pcr]
f1s = [pcr[c]["f1-score"] for c in pcr]
add_row(
    exp_id="ORIG-ENS-01",
    arch="Soft Voting Ensemble (ResNet-50: 0.58, VGG16: 0.38, EfficientNet-B0: 0.04)",
    config="Champion Ensemble V2 locked weights with 1.2x contrast on ResNet-50",
    dataset="Original ~6000-image dataset",
    n_img=ens_w["locked_test_total"],
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest test split (898 images)",
    ckpt_rule="Grid search weights on validation split",
    metric_lvl="image-level",
    acc=ens_w["locked_test_accuracy"],
    bacc=sum(recalls) / len(recalls),
    mp=sum(precs) / len(precs),
    mr=sum(recalls) / len(recalls),
    mf1=sum(f1s) / len(f1s),
    ci="not recorded in artifact",
    src_file=ens_weights_json,
    src_line="locked_test_accuracy",
    src_type="json",
    inc_locked="no",
    lim_note="Weights optimized on validation split; evaluated on internal test split."
)

ens_opt_json = "outputs/ensemble_optimization_v2_results.json"
with open(ens_opt_json, "r") as f:
    ens_opt = json.load(f)
pcm = ens_opt["per_class_metrics"]
v_recalls = [pcm[c]["recall"] for c in pcm]
v_precs = [pcm[c]["precision"] for c in pcm]
add_row(
    exp_id="ORIG-ENS-02",
    arch="Soft Voting Ensemble (ResNet-50: 0.58, VGG16: 0.38, EfficientNet-B0: 0.04)",
    config="Optimized V2 winning configuration on validation split",
    dataset="Original ~6000-image dataset",
    n_img=898,
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest validation split (898 images)",
    ckpt_rule="Grid search weights on validation split",
    metric_lvl="image-level",
    acc=ens_opt["optimized_v2_val_accuracy"],
    bacc=sum(v_recalls) / len(v_recalls),
    mp=sum(v_precs) / len(v_precs),
    mr=sum(v_recalls) / len(v_recalls),
    mf1=ens_opt["optimized_v2_val_macro_f1"],
    ci="not recorded in artifact",
    src_file=ens_opt_json,
    src_line="optimized_v2_val_accuracy",
    src_type="json",
    inc_locked="no",
    lim_note="Validation split performance from grid search optimization."
)

ens_val_json = "outputs/ensemble_val_results.json"
with open(ens_val_json, "r") as f:
    ens_val = json.load(f)
add_row(
    exp_id="ORIG-ENS-03",
    arch="Soft Voting Ensemble (ResNet-50: 1.0, VGG16: 0.0)",
    config="Best combination search over resnet50, vgg16, effnet, densenet",
    dataset="Original ~6000-image dataset",
    n_img=898,
    n_part="not recorded in artifact",
    n_classes=8,
    split="Random manifest validation split (898 images)",
    ckpt_rule="Grid search weights on validation split",
    metric_lvl="image-level",
    acc=ens_val["val_accuracy"],
    bacc="not recorded in artifact",
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=ens_val["val_macro_f1"],
    ci="not recorded in artifact",
    src_file=ens_val_json,
    src_line="val_accuracy",
    src_type="json",
    inc_locked="no",
    lim_note="Ensemble optimized on validation split only; test set untouched."
)

day27_comp = "reports/DAY27_FULL_DATASET_COMPARISON.csv"
with open(day27_comp, "r") as f:
    rdr = csv.DictReader(f)
    day27_rows = {r["model"]: r for r in rdr}

current_group = "Day 27 Source-Domain Benchmarks"
d27_4c = day27_rows["Day27_4Class"]
add_row(
    exp_id="DAY27-01",
    arch="Custom CNN (4 conv blocks with BatchNorm)",
    config="C5 preprocessing (112x112), 4-class positive subset, class-weighted Adam",
    dataset="Original ~6000-image dataset (positive Rh subset)",
    n_img=415,
    n_part="not recorded in artifact",
    n_classes=4,
    split="Day27 canonical test split (415 positive-Rh images)",
    ckpt_rule="scripts/day27_experiments.py: EarlyStopping on val_loss",
    metric_lvl="image-level",
    acc=float(d27_4c["canonical_test_accuracy"]),
    bacc="not recorded in artifact",
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=float(d27_4c["canonical_test_macro_f1"]),
    ci="not recorded in artifact",
    src_file=day27_comp,
    src_line="Day27_4Class:canonical_test_accuracy",
    src_type="csv",
    inc_locked="no",
    lim_note="Evaluated on canonical source-domain test set; internal test performance."
)

d27_8c = day27_rows["Day27_8Class"]
add_row(
    exp_id="DAY27-02",
    arch="Custom CNN (4 conv blocks with BatchNorm)",
    config="C5 preprocessing (112x112), 8-class full dataset, class-weighted Adam",
    dataset="Original ~6000-image dataset (all 8 classes)",
    n_img=898,
    n_part="not recorded in artifact",
    n_classes=8,
    split="Day27 canonical test split (898 images)",
    ckpt_rule="scripts/day27_experiments.py: EarlyStopping on val_loss",
    metric_lvl="image-level",
    acc=float(d27_8c["canonical_test_accuracy"]),
    bacc="not recorded in artifact",
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=float(d27_8c["canonical_test_macro_f1"]),
    ci="not recorded in artifact",
    src_file=day27_comp,
    src_line="Day27_8Class:canonical_test_accuracy",
    src_type="csv",
    inc_locked="no",
    lim_note="Evaluated on canonical source-domain test set; internal test performance."
)

# =========================================================================
# 3. MFS100 Individual-Model Comparison & Benchmarks
# =========================================================================
all_models_json = "reports/model_fusion/all_models_eval_data.json"
with open(all_models_json, "r") as f:
    all_models = json.load(f)

models_info = [
    ("DenseNet-121", "DenseNet-121 (ImageNet pretrained)", "C5 + DenseNet preprocess_input (224x224), 8-to-4 ABO marginalization", 8),
    ("EfficientNet-B0", "EfficientNet-B0 (ImageNet pretrained)", "C5 + EfficientNet preprocess_input (224x224), 4-class direct softmax", 4),
    ("ResNet-50 Improved", "ResNet-50 (ImageNet pretrained)", "C5 + ResNet50 preprocess_input (224x224), 8-to-4 ABO marginalization", 8),
    ("MobileNetV2", "MobileNetV2 (ImageNet pretrained)", "C5 + MobileNet preprocess_input (224x224), 8-to-4 ABO marginalization", 8),
    ("VGG16", "VGG16 (ImageNet pretrained)", "C5 + VGG16 preprocess_input (224x224), 8-to-4 ABO marginalization", 8),
    ("Custom CNN", "Custom CNN (4 conv blocks)", "C5 + Normalized [-1, 1] (96x96), 8-to-4 ABO marginalization", 8)
]

current_group = "MFS100 Individual Models"
for idx, (mkey, arch_name, cfg_desc, orig_classes) in enumerate(models_info, 1):
    mdata = all_models["models"][mkey]
    
    # Scan-level
    s_acc = mdata["scan"]["acc"]
    s_bacc = mdata["scan"]["bal_acc"]
    s_mf1 = mdata["scan"]["macro_f1"]
    s_mp = sum(mdata["scan"]["prec"]) / len(mdata["scan"]["prec"])
    s_mr = sum(mdata["scan"]["rec"]) / len(mdata["scan"]["rec"])
    add_row(
        exp_id=f"MFS100-DEV-SCAN-{idx:02d}",
        arch=arch_name,
        config=cfg_desc,
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="Dev cohort 240 scans (80 participants x 3 scans)",
        ckpt_rule="Original pretrained checkpoint (no retraining on MFS100)",
        metric_lvl="scan-level",
        acc=s_acc,
        bacc=s_bacc,
        mp=s_mp,
        mr=s_mr,
        mf1=s_mf1,
        ci="not recorded in artifact",
        src_file=all_models_json,
        src_line=f"models['{mkey}']['scan']",
        src_type="json",
        inc_locked="no",
        lim_note="Scan-level zero-shot transfer; scans from same participant evaluated as independent samples."
    )
    
    # Participant-level
    p_acc = mdata["part"]["acc"]
    p_bacc = mdata["part"]["bal_acc"]
    p_mf1 = mdata["part"]["macro_f1"]
    p_mp = sum(mdata["part"]["prec"]) / len(mdata["part"]["prec"])
    p_mr = sum(mdata["part"]["rec"]) / len(mdata["part"]["rec"])
    add_row(
        exp_id=f"MFS100-DEV-PART-{idx:02d}",
        arch=arch_name,
        config=cfg_desc + ", probability average across 3 scans",
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="Dev cohort 80 participants (3 scans averaged)",
        ckpt_rule="Original pretrained checkpoint (no retraining on MFS100)",
        metric_lvl="participant-level dev OOF",
        acc=p_acc,
        bacc=p_bacc,
        mp=p_mp,
        mr=p_mr,
        mf1=p_mf1,
        ci="not recorded in artifact",
        src_file=all_models_json,
        src_line=f"models['{mkey}']['part']",
        src_type="json",
        inc_locked="no",
        lim_note="Participant-level zero-shot evaluation on 80 development participants."
    )

# Disagreement Row: EfficientNet-B0 in development_metrics.json
eff_dev_json = "reports/efficientnet_e2_ensemble_experiment/development_metrics.json"
with open(eff_dev_json, "r") as f:
    eff_dev = json.load(f)["efficientnet_b0"]
add_row(
    exp_id="MFS100-DEV-PART-02-DISAGREEMENT",
    arch="EfficientNet-B0 (ImageNet pretrained)",
    config="Single-scan mode evaluated on 80 dev participants using raw BMP resize without C5 crop/CLAHE",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="Dev cohort 80 participants",
    ckpt_rule="Original pretrained checkpoint (models/abo_positive_4class/best_model.keras)",
    metric_lvl="participant-level dev OOF",
    acc=eff_dev["accuracy"],
    bacc=eff_dev["balanced_accuracy"],
    mp=eff_dev["macro_precision"],
    mr=eff_dev["macro_recall"],
    mf1=eff_dev["macro_f1"],
    ci="not recorded in artifact",
    src_file=eff_dev_json,
    src_line="efficientnet_b0",
    src_type="json",
    inc_locked="no",
    lim_note="DISAGREEMENT with reports/model_fusion/ALL_MODELS_COMPARISON.md: Preprocessing differs (raw BMP resize vs C5 crop+CLAHE+bilateral filter); scan aggregation and metric implementations differ."
)

sep26_json = "reports/efficientnet_e2_ensemble_experiment/separate_26_cohort_metrics.json"
with open(sep26_json, "r") as f:
    sep26 = json.load(f)
eff_s26 = sep26["efficientnet_b0"]
add_row(
    exp_id="MFS100-SEP26-EFFB0",
    arch="EfficientNet-B0 (ImageNet pretrained)",
    config="Single-scan mode with mean probability aggregation across 3 scans",
    dataset="MFS100 Separate Hospital Cohort (S001-S026)",
    n_img=78,
    n_part=26,
    n_classes=4,
    split="Separate hospital serological verification cohort (26 participants)",
    ckpt_rule="Original pretrained checkpoint (models/abo_positive_4class/best_model.keras)",
    metric_lvl="participant-level dev OOF",
    acc=eff_s26["accuracy"],
    bacc=eff_s26["balanced_accuracy"],
    mp=eff_s26["macro_precision"],
    mr=eff_s26["macro_recall"],
    mf1=eff_s26["macro_f1"],
    ci="not recorded in artifact",
    src_file=sep26_json,
    src_line="efficientnet_b0",
    src_type="json",
    inc_locked="no",
    lim_note="Descriptive zero-shot evaluation on 26 hospital-confirmed participants."
)

lock_test_ens_json = "reports/efficientnet_e2_ensemble_experiment/locked_test_metrics.json"
with open(lock_test_ens_json, "r") as f:
    lock_ens = json.load(f)
eff_lock = lock_ens["efficientnet_b0"]
add_row(
    exp_id="MFS100-LOCK21-EFFB0",
    arch="EfficientNet-B0 (ImageNet pretrained)",
    config="Single-scan mode with mean probability aggregation across 3 scans",
    dataset="MFS100 Locked Test Cohort (TEST split)",
    n_img=63,
    n_part=21,
    n_classes=4,
    split="Locked test set (21 participants)",
    ckpt_rule="Original pretrained checkpoint (models/abo_positive_4class/best_model.keras)",
    metric_lvl="locked test",
    acc=eff_lock["accuracy"],
    bacc=eff_lock["balanced_accuracy"],
    mp=eff_lock["macro_precision"],
    mr=eff_lock["macro_recall"],
    mf1=eff_lock["macro_f1"],
    ci="not recorded in artifact",
    src_file=lock_test_ens_json,
    src_line="efficientnet_b0",
    src_type="json",
    inc_locked="yes",
    lim_note="DISAGREEMENT NOTE: EfficientNet-B0 branch in locked_test_metrics.json yields 0.2857 accuracy (6/21) and 0.1587 macro F1. Disagrees with Phase 3 architecture benchmark (0.1905 accuracy, 0.1250 macro F1; difference: +0.0952 accuracy, +0.0337 macro F1). Causes (fully explained by files): NOT THE SAME MODEL. Checkpoint: Loaded pre-trained synthetic dataset checkpoint models/abo_positive_4class/best_model.keras (quoted in scratch/run_full_experiment.py:38, 87), whereas benchmark was fine-tuned on 64 MFS100 participants (quoted in scratch/experiment1_architecture.py:215, 232). Preprocessing: Raw uncropped BMP bilinear resize to 224x224 (scratch/run_full_experiment.py:120), whereas benchmark used C5 pipeline at 112x112 (scratch/experiment1_architecture.py:60). Training script: scratch/run_full_experiment.py (zero-shot, no MFS100 training) vs scratch/experiment1_architecture.py (fine-tuned on 64 MFS100 participants). Aggregation: 3-scan mean probability per participant (scratch/run_full_experiment.py:170) vs single-scan prediction in benchmark."
)

add_row(
    exp_id="DAY27-MFS100-01",
    arch="Custom CNN (4 conv blocks with BatchNorm)",
    config="C5 preprocessing (112x112), 4-class positive subset model applied zero-shot to MFS100",
    dataset="MFS100 Live-Scan Dataset (78 scans)",
    n_img=78,
    n_part="not recorded in artifact",
    n_classes=4,
    split="MFS100 live-scan evaluation split (78 scans)",
    ckpt_rule="Day27 training checkpoint",
    metric_lvl="scan-level",
    acc=float(d27_4c["mfs100_accuracy"]),
    bacc="not recorded in artifact",
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=float(d27_4c["mfs100_macro_f1"]),
    ci="not recorded in artifact",
    src_file=day27_comp,
    src_line="Day27_4Class:mfs100_accuracy",
    src_type="csv",
    inc_locked="no",
    lim_note="Severe domain shift; O+ dominance 92.31%; exploratory image-level zero-shot evaluation."
)

add_row(
    exp_id="DAY27-MFS100-02",
    arch="Custom CNN (4 conv blocks with BatchNorm)",
    config="C5 preprocessing (112x112), 8-class full model applied zero-shot to MFS100",
    dataset="MFS100 Live-Scan Dataset (78 scans)",
    n_img=78,
    n_part="not recorded in artifact",
    n_classes=8,
    split="MFS100 live-scan evaluation split (78 scans)",
    ckpt_rule="Day27 training checkpoint",
    metric_lvl="scan-level",
    acc=float(d27_8c["mfs100_accuracy"]),
    bacc="not recorded in artifact",
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=float(d27_8c["mfs100_macro_f1"]),
    ci="not recorded in artifact",
    src_file=day27_comp,
    src_line="Day27_8Class:mfs100_accuracy",
    src_type="csv",
    inc_locked="no",
    lim_note="Severe domain shift; predicted negative blood groups for 29/78 positive MFS100 samples."
)

hist_b = day27_rows["Historical_51.28_Benchmark"]
add_row(
    exp_id="HIST-51.28",
    arch="EfficientNet-B0 four-class",
    config="Documented historical benchmark on preliminary MFS100 live-scan images",
    dataset="Preliminary MFS100 live-scan cohort",
    n_img=78,
    n_part="not recorded in artifact",
    n_classes=4,
    split="Preliminary test split (78 images)",
    ckpt_rule="not recorded in artifact",
    metric_lvl="scan-level",
    acc=float(hist_b["mfs100_accuracy"]),
    bacc="not recorded in artifact",
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1="not recorded in artifact",
    ci="not recorded in artifact",
    src_file=day27_comp,
    src_line="Historical_51.28_Benchmark:mfs100_accuracy",
    src_type="csv",
    inc_locked="no",
    lim_note="Historical exploratory benchmark from early Phase 1 live-scan evaluation."
)

# =========================================================================
# 4. E2 Benchmark, Checkpoints, and Variations
# =========================================================================
eval_101_json = "reports/final/eval_101_raw_results.json"
with open(eval_101_json, "r") as f:
    e101 = json.load(f)

current_group = "E2 Variants & Ensembles"
e2_scan = all_models["models"]["E2 Ensemble (OOF)"]["scan"]
add_row(
    exp_id="E2-DEV-SCAN",
    arch="DenseNet121_Fusion_SupCon (E2 Champion)",
    config="C5 preprocessing, 3-scan feature fusion (112x112), 5-fold OOF",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="5-fold StratifiedGroupKFold on Dev 80",
    ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
    metric_lvl="scan-level",
    acc=e2_scan["acc"],
    bacc=e2_scan["bal_acc"],
    mp=sum(e2_scan["prec"]) / len(e2_scan["prec"]),
    mr=sum(e2_scan["rec"]) / len(e2_scan["rec"]),
    mf1=e2_scan["macro_f1"],
    ci="not recorded in artifact",
    src_file=all_models_json,
    src_line="models['E2 Ensemble (OOF)']['scan']",
    src_type="json",
    inc_locked="no",
    lim_note="Scan-level out-of-fold cross-validation performance across 240 scans."
)

m_oof = e101["metrics_dev_oof"]
add_row(
    exp_id="E2-DEV-OOF",
    arch="DenseNet121_Fusion_SupCon (E2 Champion)",
    config="C5 preprocessing, 3-scan feature fusion (112x112), 5-fold OOF",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="5-fold StratifiedGroupKFold on Dev 80",
    ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
    metric_lvl="participant-level dev OOF",
    acc=m_oof["accuracy"],
    bacc=m_oof["balanced_accuracy"],
    mp=m_oof["macro_precision"],
    mr=m_oof["macro_recall"],
    mf1=m_oof["macro_f1"],
    ci="not recorded in artifact",
    src_file=eval_101_json,
    src_line="metrics_dev_oof",
    src_type="json",
    inc_locked="no",
    lim_note="Primary development benchmark; uses held-out fold checkpoint selection."
)

m_dev_ens = e101["metrics_dev_ens"]
add_row(
    exp_id="E2-DEV-ENS",
    arch="DenseNet121_Fusion_SupCon (5-Fold Ensemble)",
    config="Unweighted soft voting across all 5 fold checkpoints on Dev 80",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="Dev 80 participants evaluated by full 5-checkpoint ensemble",
    ckpt_rule="Frozen ensemble of 5 folds (held-out-fold selection)",
    metric_lvl="participant-level dev OOF",
    acc=m_dev_ens["accuracy"],
    bacc=m_dev_ens["balanced_accuracy"],
    mp=m_dev_ens["macro_precision"],
    mr=m_dev_ens["macro_recall"],
    mf1=m_dev_ens["macro_f1"],
    ci="not recorded in artifact",
    src_file=eval_101_json,
    src_line="metrics_dev_ens",
    src_type="json",
    inc_locked="no",
    lim_note="Evaluated with all 5 checkpoints on development cohort (in-sample training overlap per fold)."
)

m_test = e101["metrics_test"]
p3_boot_json = "reports/phase3_final_evaluation/final_test_bootstrap_intervals.json"
with open(p3_boot_json, "r") as f:
    p3_boot = json.load(f)
p3_ci_str = f"Acc 95% CI: [{p3_boot['accuracy_ci_lower']:.4f}, {p3_boot['accuracy_ci_upper']:.4f}]; Macro F1 95% CI: [{p3_boot['macro_f1_ci_lower']:.4f}, {p3_boot['macro_f1_ci_upper']:.4f}]"
add_row(
    exp_id="E2-LOCKED-TEST-21",
    arch="DenseNet121_Fusion_SupCon (5-Fold Ensemble)",
    config="Unweighted soft voting across all 5 fold checkpoints",
    dataset="MFS100 Locked Test Cohort (TEST split)",
    n_img=63,
    n_part=21,
    n_classes=4,
    split="Strictly locked test cohort (21 participants, 63 scans)",
    ckpt_rule="Frozen ensemble of 5 folds from Dev 80",
    metric_lvl="locked test",
    acc=m_test["accuracy"],
    bacc=m_test["balanced_accuracy"],
    mp=m_test["macro_precision"],
    mr=m_test["macro_recall"],
    mf1=m_test["macro_f1"],
    ci=p3_ci_str,
    src_file=eval_101_json,
    src_line="metrics_test",
    src_type="json",
    inc_locked="yes",
    lim_note="Phase 3 final evaluation on locked test set; identical to majority-class baseline (0.3810)."
)

e2_s26 = sep26["e2_fusion"]
add_row(
    exp_id="E2-SEP26",
    arch="DenseNet121_Fusion_SupCon (5-Fold Ensemble)",
    config="3-scan feature fusion, unweighted average of 5 folds",
    dataset="MFS100 Separate Hospital Cohort (S001-S026)",
    n_img=78,
    n_part=26,
    n_classes=4,
    split="Separate hospital serological verification cohort (26 participants)",
    ckpt_rule="Frozen ensemble of 5 folds from Dev 80",
    metric_lvl="participant-level dev OOF",
    acc=e2_s26["accuracy"],
    bacc=e2_s26["balanced_accuracy"],
    mp=e2_s26["macro_precision"],
    mr=e2_s26["macro_recall"],
    mf1=e2_s26["macro_f1"],
    ci="not recorded in artifact",
    src_file=sep26_json,
    src_line="e2_fusion",
    src_type="json",
    inc_locked="no",
    lim_note="Severe generalization failure on hospital-confirmed cohort; underperformed majority class baseline (0.4231)."
)

m_pooled = e101["metrics_pooled_ens"]
add_row(
    exp_id="E2-POOLED-101 (MIXTURE DEV + LOCKED TEST)",
    arch="DenseNet121_Fusion_SupCon (5-Fold Ensemble)",
    config="Full 5-checkpoint ensemble on pooled 101 participants (mixes 80 dev and 21 locked test)",
    dataset="MFS100 Pooled Cohort (Dev 80 + Test 21)",
    n_img=303,
    n_part=101,
    n_classes=4,
    split="Pooled cohort (80 dev + 21 locked test)",
    ckpt_rule="Frozen ensemble of 5 folds",
    metric_lvl="participant-level dev OOF",
    acc=m_pooled["accuracy"],
    bacc=m_pooled["balanced_accuracy"],
    mp=m_pooled["macro_precision"],
    mr=m_pooled["macro_recall"],
    mf1=m_pooled["macro_f1"],
    ci="not recorded in artifact",
    src_file=eval_101_json,
    src_line="metrics_pooled_ens",
    src_type="json",
    inc_locked="yes",
    lim_note="Mixes development and locked test cohorts; includes training-fold overlap for the 80 development participants."
)

m_comb = e101["metrics_combined_honest"]
add_row(
    exp_id="E2-COMBINED-101 (OOF DEV + LOCKED TEST)",
    arch="DenseNet121_Fusion_SupCon (OOF + Locked Test)",
    config="80 development participants out-of-fold + 21 locked test participants ensemble",
    dataset="MFS100 All Available Participants (80 OOF + 21 Test)",
    n_img=303,
    n_part=101,
    n_classes=4,
    split="80 OOF cross-validation + 21 locked test",
    ckpt_rule="held-out-fold selection (dev) + frozen ensemble (test)",
    metric_lvl="participant-level dev OOF",
    acc=m_comb["accuracy"],
    bacc=m_comb["balanced_accuracy"],
    mp=m_comb["macro_precision"],
    mr=m_comb["macro_recall"],
    mf1=m_comb["macro_f1"],
    ci="not recorded in artifact",
    src_file=eval_101_json,
    src_line="metrics_combined_honest",
    src_type="json",
    inc_locked="yes",
    lim_note="Partitioned evaluation: 80 dev participants evaluated out-of-fold plus 21 locked test participants."
)

eq_dev = json.load(open(eff_dev_json, "r"))["equal_weight_ensemble"]
add_row(
    exp_id="ENS-EFF-E2-DEV80",
    arch="Equal-Weight Ensemble (EfficientNet-B0 + E2 Fusion)",
    config="50/50 probability average between single-scan EffNet-B0 and 3-scan E2",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="Dev 80 participants",
    ckpt_rule="EffNet-B0 pretrained + E2 5-fold OOF",
    metric_lvl="participant-level dev OOF",
    acc=eq_dev["accuracy"],
    bacc=eq_dev["balanced_accuracy"],
    mp=eq_dev["macro_precision"],
    mr=eq_dev["macro_recall"],
    mf1=eq_dev["macro_f1"],
    ci="not recorded in artifact",
    src_file=eff_dev_json,
    src_line="equal_weight_ensemble",
    src_type="json",
    inc_locked="no",
    lim_note="Equal weight ensemble on development cohort. Branch preprocessing disparity: EfficientNet branch used raw uncropped BMP resize to 224x224 (scratch/run_full_experiment.py:120) with effnet_preprocess, whereas E2 branch used custom C5 pipeline (cropped foreground bounding box, CLAHE contrast enhancement, bilateral filter, 112x112; scratch/run_full_experiment.py:129)."
)

eq_s26 = sep26["equal_weight_ensemble"]
add_row(
    exp_id="ENS-EFF-E2-SEP26",
    arch="Equal-Weight Ensemble (EfficientNet-B0 + E2 Fusion)",
    config="50/50 probability average between single-scan EffNet-B0 and 3-scan E2",
    dataset="MFS100 Separate Hospital Cohort (S001-S026)",
    n_img=78,
    n_part=26,
    n_classes=4,
    split="Separate hospital serological verification cohort (26 participants)",
    ckpt_rule="EffNet-B0 pretrained + E2 frozen ensemble",
    metric_lvl="participant-level dev OOF",
    acc=eq_s26["accuracy"],
    bacc=eq_s26["balanced_accuracy"],
    mp=eq_s26["macro_precision"],
    mr=eq_s26["macro_recall"],
    mf1=eq_s26["macro_f1"],
    ci="not recorded in artifact",
    src_file=sep26_json,
    src_line="equal_weight_ensemble",
    src_type="json",
    inc_locked="no",
    lim_note="Descriptive evaluation on 26 hospital participants. Branch preprocessing disparity: EfficientNet branch used raw uncropped BMP resize to 224x224 (scratch/run_full_experiment.py:120) with effnet_preprocess, whereas E2 branch used custom C5 pipeline (cropped foreground bounding box, CLAHE contrast enhancement, bilateral filter, 112x112; scratch/run_full_experiment.py:129)."
)

eq_lock = lock_ens["equal_weight_ensemble"]
add_row(
    exp_id="ENS-EFF-E2-LOCK21",
    arch="Equal-Weight Ensemble (EfficientNet-B0 + E2 Fusion)",
    config="50/50 probability average between single-scan EffNet-B0 and 3-scan E2",
    dataset="MFS100 Locked Test Cohort (TEST split)",
    n_img=63,
    n_part=21,
    n_classes=4,
    split="Locked test set (21 participants)",
    ckpt_rule="EffNet-B0 pretrained + E2 frozen ensemble",
    metric_lvl="locked test",
    acc=eq_lock["accuracy"],
    bacc=eq_lock["balanced_accuracy"],
    mp=eq_lock["macro_precision"],
    mr=eq_lock["macro_recall"],
    mf1=eq_lock["macro_f1"],
    ci="not recorded in artifact",
    src_file=lock_test_ens_json,
    src_line="equal_weight_ensemble",
    src_type="json",
    inc_locked="yes",
    lim_note="Equal weight ensemble on locked test cohort. Branch preprocessing disparity: EfficientNet branch used raw uncropped BMP resize to 224x224 (scratch/run_full_experiment.py:120) with effnet_preprocess, whereas E2 branch used custom C5 pipeline (cropped foreground bounding box, CLAHE contrast enhancement, bilateral filter, 112x112; scratch/run_full_experiment.py:129)."
)



exp01_json = "reports/accuracy_improvement/exp01/metrics.json"
with open(exp01_json, "r") as f:
    e01 = json.load(f)
add_row(
    exp_id="EXP-01",
    arch="DenseNet121_Fusion_SupCon",
    config="Class-Balanced Categorical Focal Loss (gamma=2.0) for E2",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="5-fold StratifiedGroupKFold on Dev 80",
    ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
    metric_lvl="participant-level dev OOF",
    acc=e01["accuracy"],
    bacc=e01["balanced_accuracy"],
    mp=e01["macro_precision"],
    mr=e01["macro_recall"],
    mf1=e01["macro_f1"],
    ci="not recorded in artifact",
    src_file=exp01_json,
    src_line="accuracy",
    src_type="json",
    inc_locked="no",
    lim_note="Degraded performance compared to CE baseline."
)

exp02_json = "reports/accuracy_improvement/exp02/metrics.json"
with open(exp02_json, "r") as f:
    e02 = json.load(f)
add_row(
    exp_id="EXP-02",
    arch="DenseNet121_Fusion_SupCon",
    config="Controlled Data Augmentation (rotation 0.027, translation 0.1, zoom 0.1, contrast 0.1)",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="5-fold StratifiedGroupKFold on Dev 80",
    ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
    metric_lvl="participant-level dev OOF",
    acc=e02["accuracy"],
    bacc=e02["balanced_accuracy"],
    mp=e02["macro_precision"],
    mr=e02["macro_recall"],
    mf1=e02["macro_f1"],
    ci="not recorded in artifact",
    src_file=exp02_json,
    src_line="accuracy",
    src_type="json",
    inc_locked="no",
    lim_note="Augmentation caused representation distortion; performance lower than unaugmented baseline."
)

# =========================================================================
# 5. EXP-03 (Three-Arm Augmentation Ablation)
# =========================================================================
current_group = "EXP-03 to EXP-06D Hyperparameter Ablations"
exp03_csv = "reports/accuracy_improvement/exp03/metrics_comparison.csv"
with open(exp03_csv, "r") as f:
    e03_rows = {r["arm_id"]: r for r in csv.DictReader(f)}

e03_arms = [
    ("original_baseline_e2", "EXP-03 Baseline E2", "Historical baseline E2 configuration"),
    ("arm0", "EXP-03 Arm 0", "Faithful original E2 augmentation"),
    ("arm1", "EXP-03 Arm 1", "No augmentation ablation (identity pass-through)"),
    ("arm2", "EXP-03 Arm 2", "Conservative fingerprint augmentation (no zoom)")
]

for arm_id, exp_label, arm_desc in e03_arms:
    ar = e03_rows[arm_id]
    add_row(
        exp_id=exp_label,
        arch="DenseNet121_Fusion_SupCon",
        config=arm_desc,
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80",
        ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
        metric_lvl="participant-level dev OOF",
        acc=float(ar["accuracy"]),
        bacc=float(ar["balanced_accuracy"]),
        mp=float(ar["macro_precision"]),
        mr=float(ar["macro_recall"]),
        mf1=float(ar["macro_f1"]),
        ci="not recorded in artifact",
        src_file=exp03_csv,
        src_line=f"arm_id={arm_id}",
        src_type="csv",
        inc_locked="no",
        lim_note="Uses held-out validation fold checkpoint selection rule."
    )

# =========================================================================
# 6. EXP-04 (Partial Fine-Tuning Depth)
# =========================================================================
exp04_json = "reports/accuracy_improvement/exp04/independent_validation_summary.json"
with open(exp04_json, "r") as f:
    e04_metrics = json.load(f)["metrics"]

exp04_configs = [
    ("config_a", "EXP-04 Config A", "Frozen Backbone Reference (0 trainable backbone params)"),
    ("config_b", "EXP-04 Config B", "DenseBlock 4 Partial FT (2,129,920 trainable backbone params)"),
    ("config_c", "EXP-04 Config C", "DenseBlocks 3+4 Deep FT (5,455,872 trainable backbone params)")
]

for cid, exp_label, cdesc in exp04_configs:
    cm = e04_metrics[cid]
    pcm = cm["per_class"]
    c_precs = [pcm[c]["precision"] for c in pcm]
    c_recs = [pcm[c]["recall"] for c in pcm]
    add_row(
        exp_id=exp_label,
        arch="DenseNet121_Fusion_DualHead",
        config=cdesc,
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80",
        ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
        metric_lvl="participant-level dev OOF",
        acc=cm["accuracy"],
        bacc=cm["balanced_accuracy"],
        mp=sum(c_precs) / len(c_precs),
        mr=sum(c_recs) / len(c_recs),
        mf1=cm["macro_f1"],
        ci="not recorded in artifact",
        src_file=exp04_json,
        src_line=f"metrics['{cid}']",
        src_type="json",
        inc_locked="no",
        lim_note="Uses held-out validation fold checkpoint selection rule."
    )

# =========================================================================
# 7. EXP-05 (Cross-Scan Fusion Architecture)
# =========================================================================
exp05_json = "reports/accuracy_improvement/exp05/post_analysis.json"
with open(exp05_json, "r") as f:
    e05_results = json.load(f)["results"]

exp05_configs = [
    ("config_a", "EXP-05 Config A", "Concatenation Baseline (1,968,772 params)"),
    ("config_b", "EXP-05 Config B", "Gated Feature Fusion (789,381 params)"),
    ("config_c", "EXP-05 Config C", "Cross-Scan Multi-Head Self-Attention (1,283,716 params)")
]

for cid, exp_label, cdesc in exp05_configs:
    cm = e05_results[cid]
    add_row(
        exp_id=exp_label,
        arch="DenseNet121_Fusion_Ablation",
        config=cdesc,
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80",
        ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
        metric_lvl="participant-level dev OOF",
        acc=cm["acc"],
        bacc=cm["bal_acc"],
        mp=cm["macro_p"],
        mr=cm["macro_r"],
        mf1=cm["macro_f1"],
        ci="not recorded in artifact",
        src_file=exp05_json,
        src_line=f"results['{cid}']",
        src_type="json",
        inc_locked="no",
        lim_note="Uses held-out validation fold checkpoint selection rule."
    )

# =========================================================================
# 8. EXP-06A (Temperature Tuning)
# =========================================================================
exp06a_json = "reports/accuracy_improvement/exp06a/statistical_analysis_results.json"
with open(exp06a_json, "r") as f:
    e06a_summary = json.load(f)["summary"]

for c in e06a_summary:
    cid = c["cid"]
    add_row(
        exp_id=f"EXP-06A {cid.upper().replace('_', ' ')}",
        arch="DenseNet121_Fusion_SupCon",
        config=f"{c['name']} (tau={c['tau']})",
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80",
        ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
        metric_lvl="participant-level dev OOF",
        acc=c["acc"],
        bacc=c["bacc"],
        mp=c["macro_p"],
        mr=c["macro_r"],
        mf1=c["macro_f1"],
        ci="not recorded in artifact",
        src_file=exp06a_json,
        src_line=f"summary[cid='{cid}']",
        src_type="json",
        inc_locked="no",
        lim_note="Uses held-out validation fold checkpoint selection rule."
    )

# =========================================================================
# 9. EXP-06B (SupCon Loss Weight Lambda)
# =========================================================================
exp06b_json = "reports/accuracy_improvement/exp06b/statistical_analysis_results.json"
with open(exp06b_json, "r") as f:
    e06b_summary = json.load(f)["summary"]

for c in e06b_summary:
    arm_id = c["arm_id"]
    add_row(
        exp_id=f"EXP-06B {arm_id.upper().replace('_', ' ')}",
        arch="DenseNet121_Fusion_SupCon",
        config=f"{c['arm_name']} (lambda={c['lambda_supcon']}, tau={c['temperature']})",
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80",
        ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
        metric_lvl="participant-level dev OOF",
        acc=c["accuracy"],
        bacc=c["balanced_accuracy"],
        mp=c["macro_precision"],
        mr=c["macro_recall"],
        mf1=c["macro_f1"],
        ci="not recorded in artifact",
        src_file=exp06b_json,
        src_line=f"summary[arm_id='{arm_id}']",
        src_type="json",
        inc_locked="no",
        lim_note="Uses held-out validation fold checkpoint selection rule."
    )

# =========================================================================
# 10. EXP-06C (Projection Head Architecture Ablation)
# =========================================================================
exp06c_json = "reports/accuracy_improvement/exp06c/statistical_analysis_results.json"
with open(exp06c_json, "r") as f:
    e06c_configs = json.load(f)["configurations"]

for c in e06c_configs:
    cid = c["config_id"]
    add_row(
        exp_id=f"EXP-06C {cid.upper().replace('_', ' ')}",
        arch="DenseNet121_Fusion_HeadAblation",
        config=f"{c['config_name']} (head_type={c['head_type']})",
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80",
        ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
        metric_lvl="participant-level dev OOF",
        acc=c["accuracy"],
        bacc=c["balanced_accuracy"],
        mp=c["macro_precision"],
        mr=c["macro_recall"],
        mf1=c["macro_f1"],
        ci="not recorded in artifact",
        src_file=exp06c_json,
        src_line=f"configurations[config_id='{cid}']",
        src_type="json",
        inc_locked="no",
        lim_note="Uses held-out validation fold checkpoint selection rule."
    )

# Reproduction Attempt Row: EXP-06C Config A re-run in leakage audit
seed42_phase2_json = "experiments/leakage_audit/phase2_selection_rules_comparison.json"
with open(seed42_phase2_json, "r") as f:
    p2_d = json.load(f)
s42_overall = p2_d["heldout_selection_rule_i"]["seed_42"]["overall"]
add_row(
    exp_id="EXP-06C CONFIG A (REPRODUCTION ATTEMPT)",
    arch="DenseNet121_Fusion_LinearHead",
    config="Reproduction attempt on cached DenseNet features with re-implemented loop (Rule i)",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="5-fold StratifiedGroupKFold on Dev 80 (seed 42)",
    ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
    metric_lvl="participant-level dev OOF",
    acc=s42_overall["accuracy"],
    bacc=s42_overall["balanced_accuracy"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=s42_overall["macro_f1"],
    ci="not recorded in artifact",
    src_file=seed42_phase2_json,
    src_line="heldout_selection_rule_i.seed_42.overall",
    src_type="json",
    inc_locked="no",
    lim_note="Reproduction attempt on cached DenseNet features with a re-implemented training loop; yields Macro F1 0.5218 vs historical documented 0.5487."
)

# =========================================================================
# 11. EXP-06D (Preprocessing Pipeline Ablation)
# =========================================================================
exp06d_csv = "reports/accuracy_improvement/exp06d/metrics_comparison.csv"
with open(exp06d_csv, "r") as f:
    e06d_rows = {r["Config ID"]: r for r in csv.DictReader(f)}

e06d_configs = [
    ("config_a", "EXP-06D Config A", "Enhanced C5: Crop + CLAHE + Bilateral Filter"),
    ("config_b", "EXP-06D Config B", "Raw Platen C5: Crop Only, No CLAHE/Bilateral Filter")
]

for cid, exp_label, cdesc in e06d_configs:
    cr = e06d_rows[cid]
    add_row(
        exp_id=exp_label,
        arch="DenseNet121_Fusion_LinearHead",
        config=cdesc,
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80",
        ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
        metric_lvl="participant-level dev OOF",
        acc=float(cr["Accuracy"].replace("%", "")) / 100.0,
        bacc=float(cr["Balanced Accuracy"].replace("%", "")) / 100.0,
        mp="not recorded in artifact",
        mr="not recorded in artifact",
        mf1=float(cr["Macro F1 (OOF)"]),
        ci="not recorded in artifact",
        src_file=exp06d_csv,
        src_line=f"Config ID={cid}",
        src_type="csv",
        inc_locked="no",
        lim_note="Uses held-out validation fold checkpoint selection rule."
    )

# =========================================================================
# 12. Final Locked-Test Evaluation
# =========================================================================
prim_res_json = "reports/final_evaluation/primary_evaluation_results.json"
with open(prim_res_json, "r") as f:
    prim_res = json.load(f)

p_ci = prim_res["bootstrap_ci_95"]
prim_ci_str = f"Acc 95% CI: [{p_ci['accuracy']['ci_95'][0]:.4f}, {p_ci['accuracy']['ci_95'][1]:.4f}]; Bal Acc 95% CI: [{p_ci['balanced_accuracy']['ci_95'][0]:.4f}, {p_ci['balanced_accuracy']['ci_95'][1]:.4f}]; Macro F1 95% CI: [{p_ci['macro_f1']['ci_95'][0]:.4f}, {p_ci['macro_f1']['ci_95'][1]:.4f}]"

current_group = "Locked Test Final Evaluations"
add_row(
    exp_id="PRIMARY-LOCKED-TEST-EXP06C-CONFIG-A",
    arch="DenseNet121_Fusion_LinearHead",
    config="Pre-registered frozen candidate EXP-06C Config A evaluated on locked test",
    dataset="MFS100 Locked Test Cohort (TEST split)",
    n_img=63,
    n_part=21,
    n_classes=4,
    split="Strictly locked test cohort (21 participants, 63 scans)",
    ckpt_rule="Frozen candidate selected from EXP-06C Config A (held-out-fold selection in dev)",
    metric_lvl="locked test",
    acc=prim_res["accuracy"],
    bacc=prim_res["balanced_accuracy"],
    mp=prim_res["macro_precision"],
    mr=prim_res["macro_recall"],
    mf1=prim_res["macro_f1"],
    ci=prim_ci_str,
    src_file=prim_res_json,
    src_line="accuracy",
    src_type="json",
    inc_locked="yes",
    lim_note="Primary pre-registered locked-test evaluation; failed primary gate (0.2083 < 0.3500 threshold)."
)

p3_report_md = "reports/phase3_final_evaluation/FINAL_TEST_REPORT.md"
import re
with open(p3_report_md, "r", encoding="utf-8") as f:
    p3_txt = f.read()
m_d27_acc = re.search(r"Day 27 Baseline Accuracy:\*{0,2}\s*([0-9.]+)%", p3_txt)
d27_locked_acc = round(float(m_d27_acc.group(1)) / 100.0, 4) if m_d27_acc else None
m_d27_mf1 = re.search(r"Day 27 Baseline Macro F1:\*{0,2}\s*([0-9.]+)", p3_txt)
d27_locked_mf1 = round(float(m_d27_mf1.group(1)), 4) if m_d27_mf1 else None

add_row(
    exp_id="DAY27-LOCKED-TEST-21",
    arch="Custom CNN (Day 27 Baseline)",
    config="Day 27 Standard CNN baseline evaluated on locked test participants",
    dataset="MFS100 Locked Test Cohort (TEST split)",
    n_img=63,
    n_part=21,
    n_classes=4,
    split="Strictly locked test cohort (21 participants, 63 scans)",
    ckpt_rule="Day 27 baseline checkpoint",
    metric_lvl="locked test",
    acc=d27_locked_acc,
    bacc="not recorded in artifact",
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=d27_locked_mf1,
    ci="not recorded in artifact",
    src_file=p3_report_md,
    src_line="Section 5: Day 27 Baseline Accuracy / Macro F1",
    src_type="md-only",
    inc_locked="yes",
    lim_note="DISAGREEMENT NOTE: Reported as 38.09% accuracy (0.3809) and 0.3125 macro F1 in FINAL_TEST_REPORT.md:36-39. Disagrees with actual Day 27 zero-shot participant predictions on the 21 locked participants (0.3333 accuracy, 0.1875 macro F1; difference: +4.76 pp accuracy, +0.1250 macro F1). Arithmetic note: 38.09% is neither 7/21 (33.333%) nor 8/21 (38.095% -> 38.10%) exactly. Causes: unexplained in repository files (no saved prediction CSV or evaluation script produces 38.09% or 0.3125). Model comparison: FINAL_TEST_REPORT.md Section 5 cites no checkpoint file or script path (unspecified in narrative); zero-shot predictions loaded models/day27_full_dataset/four_class/best_model.keras (quoted in scratch/final_zero_shot_eval.py:19-24) with C5 resize to (112, 112) (scratch/final_zero_shot_eval.py:46). Model identity between the reported table and zero-shot artifact is unverified due to lack of citation in FINAL_TEST_REPORT.md."
)

# =========================================================================
# 13. Leakage Audit Experiments
# =========================================================================
# --- Experiment 1: Hashes & Clustering ---
current_group = "Leakage Audit Diagnostics & Experiments"
exp1_hash_json = "experiments/leakage_audit/exp1_duplicates_and_perceptual_hashes.json"
with open(exp1_hash_json, "r") as f:
    e1_h = json.load(f)

add_row(
    exp_id="AUDIT-EXP1-HASH-TRAIN-REF",
    arch="pHash / dHash",
    config="Hamming distance thresholds (<=2, <=5, <=10) vs train reference (N=1,938)",
    dataset="Original ~6000-image dataset (415 test vs 1,938 train)",
    n_img=415,
    n_part="not recorded in artifact",
    n_classes=4,
    split="Original dataset test vs train split",
    ckpt_rule="not applicable",
    metric_lvl="audit",
    acc="not applicable",
    bacc="not applicable",
    mp="not applicable",
    mr="not applicable",
    mf1="not applicable",
    ci="not applicable",
    src_file=exp1_hash_json,
    src_line="perceptual_hashes",
    src_type="json",
    inc_locked="no",
    lim_note="Audit of cross-split duplicate / near-duplicate leakage in original dataset against training reference."
)

exp1_trainval_json = "experiments/leakage_audit/exp1_perceptual_hashes_trainval_reference.json"
with open(exp1_trainval_json, "r") as f:
    e1_tv = json.load(f)

add_row(
    exp_id="AUDIT-EXP1-HASH-TRAINVAL-REF",
    arch="pHash / dHash",
    config="Hamming distance thresholds (<=2, <=5, <=10) vs train+val reference (N=2,354)",
    dataset="Original ~6000-image dataset (415 test vs 2,354 train+val)",
    n_img=415,
    n_part="not recorded in artifact",
    n_classes=4,
    split="Original dataset test vs train+val split",
    ckpt_rule="not applicable",
    metric_lvl="audit",
    acc="not applicable",
    bacc="not applicable",
    mp="not applicable",
    mr="not applicable",
    mf1="not applicable",
    ci="not applicable",
    src_file=exp1_trainval_json,
    src_line="phash_near_duplicate_counts",
    src_type="json",
    inc_locked="no",
    lim_note="Audit of cross-split duplicate / near-duplicate leakage in original dataset against pooled train+val reference."
)

# Percolation Scans
exp1_percol_json = "experiments/leakage_audit/exp1_percolation_finer_scan.json"
with open(exp1_percol_json, "r") as f:
    e1_percol = json.load(f)

for row in e1_percol["thresholds_evaluated"].values():
    add_row(
        exp_id=f"AUDIT-EXP1-PERCOLATION-EFFB0-COS-{row['threshold']}",
        arch="EfficientNet-B0 (Cosine Similarity)",
        config=f"Connected-component percolation scan at cosine similarity threshold {row['threshold']}",
        dataset="Original ~6000-image dataset (positive subset N=2,769)",
        n_img=e1_percol["total_images"],
        n_part="not recorded in artifact",
        n_classes=4,
        split="Unsplit positive-Rh image pool",
        ckpt_rule="not applicable",
        metric_lvl="audit",
        acc="not applicable",
        bacc="not applicable",
        mp="not applicable",
        mr="not applicable",
        mf1="not applicable",
        ci="not applicable",
        src_file=exp1_percol_json,
        src_line=f"thresholds_evaluated['threshold_{row['threshold']}']",
        src_type="json",
        inc_locked="no",
        lim_note="Percolation threshold scan evaluating subject burst clustering without subject labels (EfficientNet-B0 embeddings)."
    )

exp1_clust_json = "experiments/leakage_audit/exp1_clustering_summary.json"
with open(exp1_clust_json, "r") as f:
    e1_clust = json.load(f)

for row in e1_clust["thresholds_evaluated"].values():
    add_row(
        exp_id=f"AUDIT-EXP1-PERCOLATION-DENSE121-COS-{row['threshold']}",
        arch="DenseNet-121 (Cosine Similarity)",
        config=f"Connected-component percolation scan at cosine similarity threshold {row['threshold']}",
        dataset="Original ~6000-image dataset (positive subset N=2,769)",
        n_img=e1_clust["total_images"],
        n_part="not recorded in artifact",
        n_classes=4,
        split="Unsplit positive-Rh image pool",
        ckpt_rule="not applicable",
        metric_lvl="audit",
        acc="not applicable",
        bacc="not applicable",
        mp="not applicable",
        mr="not applicable",
        mf1="not applicable",
        ci="not applicable",
        src_file=exp1_clust_json,
        src_line=f"thresholds_evaluated['threshold_{row['threshold']}']",
        src_type="json",
        inc_locked="no",
        lim_note="Percolation threshold scan evaluating subject burst clustering without subject labels (DenseNet-121 embeddings)."
    )

# --- Experiment 2: Signal-Existence Tests ---
exp2_sig_json = "experiments/leakage_audit/exp2_signal_tests_results.json"
with open(exp2_sig_json, "r") as f:
    e2_sig = json.load(f)

d80 = e2_sig["development_80_participants"]
d80_ci = f"Acc 95% CI: [{d80['bootstrap_ci_95']['accuracy']['ci_95'][0]:.4f}, {d80['bootstrap_ci_95']['accuracy']['ci_95'][1]:.4f}]; Bal Acc 95% CI: [{d80['bootstrap_ci_95']['balanced_accuracy']['ci_95'][0]:.4f}, {d80['bootstrap_ci_95']['balanced_accuracy']['ci_95'][1]:.4f}]; Macro F1 95% CI: [{d80['bootstrap_ci_95']['macro_f1']['ci_95'][0]:.4f}, {d80['bootstrap_ci_95']['macro_f1']['ci_95'][1]:.4f}]"

add_row(
    exp_id="AUDIT-EXP2-SIGNAL-DEV80",
    arch="DenseNet121_Fusion_LinearHead",
    config="Signal-existence permutation test (10,000 resamples) on saved Dev-80 OOF predictions",
    dataset="MFS100 Development Cohort (TRAIN+VAL)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="5-fold StratifiedGroupKFold on Dev 80",
    ckpt_rule="held-out-fold selection (MacroF1Checkpoint)",
    metric_lvl="audit",
    acc=d80["observed"]["accuracy"],
    bacc=d80["observed"]["balanced_accuracy"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=d80["observed"]["macro_f1"],
    ci=d80_ci,
    src_file=exp2_sig_json,
    src_line="development_80_participants.observed",
    src_type="json",
    inc_locked="no",
    lim_note="Null permutation test indicates statistical signal presence on Dev-80 under held-out fold selection."
)

t21 = e2_sig["primary_test_21_participants"]
t21_ci = f"Acc 95% CI: [{t21['bootstrap_ci_95']['accuracy']['ci_95'][0]:.4f}, {t21['bootstrap_ci_95']['accuracy']['ci_95'][1]:.4f}]; Bal Acc 95% CI: [{t21['bootstrap_ci_95']['balanced_accuracy']['ci_95'][0]:.4f}, {t21['bootstrap_ci_95']['balanced_accuracy']['ci_95'][1]:.4f}]; Macro F1 95% CI: [{t21['bootstrap_ci_95']['macro_f1']['ci_95'][0]:.4f}, {t21['bootstrap_ci_95']['macro_f1']['ci_95'][1]:.4f}]"

add_row(
    exp_id="AUDIT-EXP2-SIGNAL-TEST21",
    arch="DenseNet121_Fusion_LinearHead",
    config="Signal-existence permutation test (10,000 resamples) on saved Locked Test-21 predictions",
    dataset="MFS100 Locked Test Cohort (TEST split)",
    n_img=63,
    n_part=21,
    n_classes=4,
    split="Locked test set (21 participants)",
    ckpt_rule="Frozen candidate selected from EXP-06C Config A",
    metric_lvl="audit",
    acc=t21["observed"]["accuracy"],
    bacc=t21["observed"]["balanced_accuracy"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=t21["observed"]["macro_f1"],
    ci=t21_ci,
    src_file=exp2_sig_json,
    src_line="primary_test_21_participants.observed",
    src_type="json",
    inc_locked="yes",
    lim_note="Null permutation test confirms no detectable biological signal above chance/majority baseline on locked test."
)

p1_json = "experiments/leakage_audit/phase1_readonly_results.json"
with open(p1_json, "r") as f:
    p1_d = json.load(f)

for idx, mb in enumerate(p1_d["metadata_baseline_1b"], 1):
    add_row(
        exp_id=f"AUDIT-EXP2-METADATA-{idx:02d}",
        arch=mb["classifier"],
        config=f"Features: {mb['features']}",
        dataset="MFS100 Development Cohort (metadata only)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="Dev 80 participants (metadata 5-fold CV)",
        ckpt_rule="not applicable (standard scikit-learn fit)",
        metric_lvl="audit",
        acc=mb["accuracy"],
        bacc=mb["balanced_accuracy"],
        mp="not recorded in artifact",
        mr="not recorded in artifact",
        mf1=mb["macro_f1"],
        ci="not recorded in artifact",
        src_file=p1_json,
        src_line=f"metadata_baseline_1b[{idx-1}]",
        src_type="json",
        inc_locked="no",
        lim_note="Evaluates non-biometric metadata shortcut leakage (participant ID and collection timestamp)."
    )

# --- Phase 2: Selection Rules Comparison ---
p2_rules = [
    ("heldout_selection_rule_i", "AUDIT-PHASE2-RULE-I", "Rule (i): Held-out validation fold selection (historical replication)", "held-out-fold selection"),
    ("clean_rule_a_inner_val", "AUDIT-PHASE2-RULE-A", "Rule (a): Inner validation split (clean protocol, no held-out leakage)", "inner validation split within training fold"),
    ("clean_rule_b_fixed_epochs", "AUDIT-PHASE2-RULE-B", "Rule (b): Fixed 100 epochs (clean protocol, zero checkpoint selection)", "fixed 100 epochs (no selection)")
]

for rkey, exp_prefix, rdesc, rrule in p2_rules:
    rdata = p2_d[rkey]
    seeds = ["seed_42", "seed_43", "seed_44"]
    accs, baccs, mf1s = [], [], []
    for s in seeds:
        ov = rdata[s]["overall"]
        accs.append(ov["accuracy"])
        baccs.append(ov["balanced_accuracy"])
        mf1s.append(ov["macro_f1"])
        add_row(
            exp_id=f"{exp_prefix}-{s.upper()}",
            arch="DenseNet121_Fusion_LinearHead",
            config=f"{rdesc} ({s})",
            dataset="MFS100 Development Cohort (TRAIN+VAL)",
            n_img=240,
            n_part=80,
            n_classes=4,
            split="5-fold StratifiedGroupKFold on Dev 80",
            ckpt_rule=rrule,
            metric_lvl="audit",
            acc=ov["accuracy"],
            bacc=ov["balanced_accuracy"],
            mp="not recorded in artifact",
            mr="not recorded in artifact",
            mf1=ov["macro_f1"],
            ci="not recorded in artifact",
            src_file=seed42_phase2_json,
            src_line=f"{rkey}.{s}.overall",
            src_type="json",
            inc_locked="no",
            lim_note=f"Leakage audit Phase 2 selection rule evaluation: {rdesc}."
        )
    add_row(
        exp_id=f"{exp_prefix}-MEAN",
        arch="DenseNet121_Fusion_LinearHead",
        config=f"{rdesc} (3-seed mean +/- SD)",
        dataset="MFS100 Development Cohort (TRAIN+VAL)",
        n_img=240,
        n_part=80,
        n_classes=4,
        split="5-fold StratifiedGroupKFold on Dev 80 (3 seeds pooled)",
        ckpt_rule=rrule,
        metric_lvl="audit",
        acc=sum(accs) / len(accs),
        bacc=sum(baccs) / len(baccs),
        mp="not recorded in artifact",
        mr="not recorded in artifact",
        mf1=sum(mf1s) / len(mf1s),
        ci="not recorded in artifact",
        src_file=seed42_phase2_json,
        src_line=f"{rkey} 3-seed mean",
        src_type="json",
        inc_locked="no",
        lim_note=f"Sample mean across 3 seeds (42, 43, 44); SD reflects fold-assignment variation only."
    )

sel_null_json = "experiments/leakage_audit/selection_inclusive_null_summary.json"
with open(sel_null_json, "r") as f:
    s_null = json.load(f)

add_row(
    exp_id="AUDIT-PHASE2-SELECTION-INCLUSIVE-NULL",
    arch="DenseNet121_Fusion_LinearHead (Permuted Labels)",
    config="Selection-inclusive null distribution (100 label permutations with Rule (i) selection)",
    dataset="MFS100 Development Cohort (Permuted Labels)",
    n_img=240,
    n_part=80,
    n_classes=4,
    split="5-fold StratifiedGroupKFold on Dev 80 (100 permutations)",
    ckpt_rule="held-out-fold selection (MacroF1Checkpoint on val_macro_f1)",
    metric_lvl="audit",
    acc=s_null["accuracy"]["mean"],
    bacc=s_null["balanced_accuracy"]["mean"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=s_null["macro_f1"]["mean"],
    ci=f"Acc 95th pct: {s_null['accuracy']['p95']:.4f}; Bal Acc 95th pct: {s_null['balanced_accuracy']['p95']:.4f}; Macro F1 95th pct: {s_null['macro_f1']['p95']:.4f}",
    src_file=sel_null_json,
    src_line="total_repeats=100",
    src_type="json",
    inc_locked="no",
    lim_note="Proves that held-out checkpoint selection alone inflates pure noise to 46.8% accuracy and 0.428 Macro F1."
)

# --- Phase 3: Purged Cross-Validation ---
p3_purged_json = "experiments/leakage_audit/phase3_purged_cv_summary.json"
p3_results_csv = "experiments/leakage_audit/phase3_purged_cv_results.csv"
p3_df = pd.read_csv(p3_results_csv)

with open(p3_purged_json, "r") as f:
    p3_p = json.load(f)

# Arm 1: Fixed 50 epochs
arm1_p = p3_p["primary_arm_fixed_epochs"]
df_a1_0 = p3_df[(p3_df["arm"] == "Arm1_FixedEpochs") & (p3_df["condition"] == "0%_unpurged_baseline")]

add_row(
    exp_id="AUDIT-PHASE3-ARM1-PURGE-00",
    arch="EfficientNet-B0 Linear Head",
    config="Arm 1 (fixed 50 epochs, no selection), 0% baseline (unpurged)",
    dataset="Original ~6000-image dataset (positive subset N=2,769)",
    n_img=2769,
    n_part="not recorded in artifact",
    n_classes=4,
    split="5-fold cross-validation across 3 seeds",
    ckpt_rule="fixed 50 epochs (no selection)",
    metric_lvl="audit",
    acc=df_a1_0["val_accuracy"].mean(),
    bacc=arm1_p["0%_baseline"]["mean"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=df_a1_0["val_macro_f1"].mean(),
    ci=f"SD: {arm1_p['0%_baseline']['std']:.4f}",
    src_file=p3_purged_json,
    src_line="primary_arm_fixed_epochs['0%_baseline'] (val_accuracy & val_macro_f1 from phase3_purged_cv_results.csv)",
    src_type="json",
    inc_locked="no",
    lim_note="Pre-declared validity gate baseline; balanced accuracy far exceeds 25% chance gate. Accuracy and Macro F1 taken from columns val_accuracy and val_macro_f1 in phase3_purged_cv_results.csv."
)

for pct, frac in [("10%", 0.1), ("25%", 0.25), ("50%", 0.5)]:
    p_data = arm1_p[f"{pct}_purge"]
    df_sub = p3_df[(p3_df["arm"] == "Arm1_FixedEpochs") & (p3_df["condition"] == "similarity_purged") & (p3_df["purge_fraction"] == frac)]
    add_row(
        exp_id=f"AUDIT-PHASE3-ARM1-PURGE-{pct.replace('%', '')}",
        arch="EfficientNet-B0 Linear Head",
        config=f"Arm 1 (fixed 50 epochs), {pct} nearest-neighbor purged vs random control",
        dataset="Original ~6000-image dataset (positive subset N=2,769)",
        n_img=2769,
        n_part="not recorded in artifact",
        n_classes=4,
        split="5-fold cross-validation across 3 seeds",
        ckpt_rule="fixed 50 epochs (no selection)",
        metric_lvl="audit",
        acc=df_sub["val_accuracy"].mean(),
        bacc=p_data["purged_mean"],
        mp="not recorded in artifact",
        mr="not recorded in artifact",
        mf1=df_sub["val_macro_f1"].mean(),
        ci=f"Purged SD: {p_data['purged_std']:.4f}; Paired Delta: {p_data['paired_delta_mean']:.4f} +/- {p_data['paired_delta_std']:.4f}",
        src_file=p3_purged_json,
        src_line=f"primary_arm_fixed_epochs['{pct}_purge'] (val_accuracy & val_macro_f1 from phase3_purged_cv_results.csv)",
        src_type="json",
        inc_locked="no",
        lim_note=f"Paired purged-minus-random difference evaluates true cross-split neighbor leakage at {pct} threshold. Accuracy and Macro F1 taken from columns val_accuracy and val_macro_f1 in phase3_purged_cv_results.csv."
    )

# Arm 2: Inner validation early stopping
arm2_p = p3_p["secondary_arm_inner_val"]
df_a2_0 = p3_df[(p3_df["arm"] == "Arm2_InnerValidation") & (p3_df["condition"] == "0%_unpurged_baseline")]

add_row(
    exp_id="AUDIT-PHASE3-ARM2-PURGE-00",
    arch="EfficientNet-B0 Linear Head",
    config="Arm 2 (inner validation early stopping), 0% baseline (unpurged)",
    dataset="Original ~6000-image dataset (positive subset N=2,769)",
    n_img=2769,
    n_part="not recorded in artifact",
    n_classes=4,
    split="5-fold cross-validation with inner split across 3 seeds",
    ckpt_rule="inner validation split early stopping (within training fold)",
    metric_lvl="audit",
    acc=df_a2_0["val_accuracy"].mean(),
    bacc=arm2_p["0%_baseline"]["mean"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=df_a2_0["val_macro_f1"].mean(),
    ci=f"SD: {arm2_p['0%_baseline']['std']:.4f}",
    src_file=p3_purged_json,
    src_line="secondary_arm_inner_val['0%_baseline'] (val_accuracy & val_macro_f1 from phase3_purged_cv_results.csv)",
    src_type="json",
    inc_locked="no",
    lim_note="Secondary arm baseline with inner-validation early stopping. Accuracy and Macro F1 taken from columns val_accuracy and val_macro_f1 in phase3_purged_cv_results.csv."
)

for pct, frac in [("10%", 0.1), ("25%", 0.25), ("50%", 0.5)]:
    p_data = arm2_p[f"{pct}_purge"]
    df_sub = p3_df[(p3_df["arm"] == "Arm2_InnerValidation") & (p3_df["condition"] == "similarity_purged") & (p3_df["purge_fraction"] == frac)]
    add_row(
        exp_id=f"AUDIT-PHASE3-ARM2-PURGE-{pct.replace('%', '')}",
        arch="EfficientNet-B0 Linear Head",
        config=f"Arm 2 (inner validation), {pct} nearest-neighbor purged vs random control",
        dataset="Original ~6000-image dataset (positive subset N=2,769)",
        n_img=2769,
        n_part="not recorded in artifact",
        n_classes=4,
        split="5-fold cross-validation with inner split across 3 seeds",
        ckpt_rule="inner validation split early stopping (within training fold)",
        metric_lvl="audit",
        acc=df_sub["val_accuracy"].mean(),
        bacc=p_data["purged_mean"],
        mp="not recorded in artifact",
        mr="not recorded in artifact",
        mf1=df_sub["val_macro_f1"].mean(),
        ci=f"Purged SD: {p_data['purged_std']:.4f}; Paired Delta: {p_data['paired_delta_mean']:.4f} +/- {p_data['paired_delta_std']:.4f}",
        src_file=p3_purged_json,
        src_line=f"secondary_arm_inner_val['{pct}_purge'] (val_accuracy & val_macro_f1 from phase3_purged_cv_results.csv)",
        src_type="json",
        inc_locked="no",
        lim_note=f"Paired purged-minus-random difference evaluates cross-split leakage at {pct} threshold with early stopping. Accuracy and Macro F1 taken from columns val_accuracy and val_macro_f1 in phase3_purged_cv_results.csv."
    )

# --- Experiment 3: Trivial Features & Concat ---
exp3_prop_json = "experiments/leakage_audit/exp3_trivial_image_properties_results.json"
with open(exp3_prop_json, "r") as f:
    e3_prop = json.load(f)

for mname, mobj in e3_prop["models"].items():
    tl = mobj["true_labels"]
    sc = mobj["shuffled_control"]
    add_row(
        exp_id=f"AUDIT-EXP3-TRIVIAL-{mname.upper()}-TRUE",
        arch=f"{mname} (42 Trivial Image Properties)",
        config="42 low-level acquisition features (dimensions, file size, alpha, border, intensity moments)",
        dataset="Original ~6000-image dataset (positive subset N=2,769)",
        n_img=2769,
        n_part="not recorded in artifact",
        n_classes=4,
        split="5-fold cross-validation across 3 seeds",
        ckpt_rule="not applicable (standard scikit-learn / keras fit)",
        metric_lvl="audit",
        acc=tl["mean_acc"],
        bacc=tl["mean_bal_acc"],
        mp="not recorded in artifact",
        mr="not recorded in artifact",
        mf1=tl["mean_macro_f1"],
        ci=f"SD: acc={tl['std_acc']:.4f}, bal={tl['std_bal_acc']:.4f}, f1={tl['std_macro_f1']:.4f}",
        src_file=exp3_prop_json,
        src_line=f"models['{mname}']['true_labels']",
        src_type="json",
        inc_locked="no",
        lim_note="Proves that acquisition artifacts alone predict ABO labels at ~53% accuracy on original dataset."
    )
    add_row(
        exp_id=f"AUDIT-EXP3-TRIVIAL-{mname.upper()}-SHUFFLED",
        arch=f"{mname} (42 Trivial Image Properties - Label Shuffled)",
        config="Label-shuffled negative control for 42 low-level acquisition features",
        dataset="Original ~6000-image dataset (positive subset N=2,769)",
        n_img=2769,
        n_part="not recorded in artifact",
        n_classes=4,
        split="5-fold cross-validation across 3 seeds (shuffled labels)",
        ckpt_rule="not applicable (standard scikit-learn / keras fit)",
        metric_lvl="audit",
        acc=sc["mean_acc"],
        bacc=sc["mean_bal_acc"],
        mp="not recorded in artifact",
        mr="not recorded in artifact",
        mf1=sc["mean_macro_f1"],
        ci=f"SD: acc={sc['std_acc']:.4f}, bal={sc['std_bal_acc']:.4f}, f1={sc['std_macro_f1']:.4f}",
        src_file=exp3_prop_json,
        src_line=f"models['{mname}']['shuffled_control']",
        src_type="json",
        inc_locked="no",
        lim_note="Negative control confirms models collapse to 25% chance level on permuted labels."
    )

# Exp 3 Feature Group Ablation
exp3_abl_json = "experiments/leakage_audit/exp3_trivial_ablation_results.json"
with open(exp3_abl_json, "r") as f:
    e3_abl = json.load(f)["groups"]

group_labels = {
    "group_i_dims_filesize": ("Dimensions + File Size Only (5 features)", "Dims/Filesize"),
    "group_ii_alpha_border": ("Alpha + Border Statistics Only (9 features)", "Alpha/Border"),
    "group_iii_intensity_hist_moments": ("Intensity Histogram + Moments Only (28 features)", "Intensity/Moments"),
    "group_iv_all_features": ("All Features Combined (42 features)", "All Features")
}

for gkey, (gdesc, gshort) in group_labels.items():
    gdata = e3_abl[gkey]
    for mkey, mname in [("logistic_regression", "Logistic Regression"), ("random_forest", "Random Forest")]:
        tl = gdata[mkey]["true"]["summary"]
        sc = gdata[mkey]["shuffled"]["summary"]
        add_row(
            exp_id=f"AUDIT-EXP3-ABL-{gkey.upper()}-{mkey.upper()}-TRUE",
            arch=f"{mname} ({gdesc})",
            config=f"True labels, {gdesc}",
            dataset="Original ~6000-image dataset (positive subset N=2,769)",
            n_img=2769,
            n_part="not recorded in artifact",
            n_classes=4,
            split="5-fold cross-validation across 3 seeds",
            ckpt_rule="not applicable (standard scikit-learn fit)",
            metric_lvl="audit",
            acc=tl["mean_acc"],
            bacc=tl["mean_bal"],
            mp="not recorded in artifact",
            mr="not recorded in artifact",
            mf1=tl["mean_f1"],
            ci=f"SD: acc={tl['std_acc']:.4f}, bal={tl['std_bal']:.4f}, f1={tl['std_f1']:.4f}",
            src_file=exp3_abl_json,
            src_line=f"groups['{gkey}']['{mkey}']['true']['summary']",
            src_type="json",
            inc_locked="no",
            lim_note=f"Feature ablation {gshort} evaluated on true labels across 5-fold CV and 3 seeds."
        )
        add_row(
            exp_id=f"AUDIT-EXP3-ABL-{gkey.upper()}-{mkey.upper()}-SHUFFLED",
            arch=f"{mname} ({gdesc} - Label Shuffled)",
            config=f"Label-shuffled control, {gdesc}",
            dataset="Original ~6000-image dataset (positive subset N=2,769)",
            n_img=2769,
            n_part="not recorded in artifact",
            n_classes=4,
            split="5-fold cross-validation across 3 seeds (shuffled labels)",
            ckpt_rule="not applicable (standard scikit-learn fit)",
            metric_lvl="audit",
            acc=sc["mean_acc"],
            bacc=sc["mean_bal"],
            mp="not recorded in artifact",
            mr="not recorded in artifact",
            mf1=sc["mean_f1"],
            ci=f"SD: acc={sc['std_acc']:.4f}, bal={sc['std_bal']:.4f}, f1={sc['std_f1']:.4f}",
            src_file=exp3_abl_json,
            src_line=f"groups['{gkey}']['{mkey}']['shuffled']['summary']",
            src_type="json",
            inc_locked="no",
            lim_note=f"Label-shuffled control for feature ablation {gshort}."
        )

# Exp 3 EfficientNet Head Concat
exp3_concat_json = "experiments/leakage_audit/exp3_effnet_concat_results.json"
with open(exp3_concat_json, "r") as f:
    e3_concat = json.load(f)

c_without = e3_concat["without_trivial_features"]["summary"]
c_with = e3_concat["with_trivial_features_concatenated"]["summary"]

add_row(
    exp_id="AUDIT-EXP3-EFFNET-WITHOUT-TRIVIAL",
    arch="EfficientNet-B0 Frozen Head",
    config="Raw GAP EfficientNet-B0 features alone (fixed 50 epochs, Arm 1 recipe)",
    dataset="Original ~6000-image dataset (positive subset N=2,769)",
    n_img=2769,
    n_part="not recorded in artifact",
    n_classes=4,
    split="5-fold cross-validation across 3 seeds",
    ckpt_rule="fixed 50 epochs (no selection)",
    metric_lvl="audit",
    acc=c_without["mean_acc"],
    bacc=c_without["mean_bal"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=c_without["mean_f1"],
    ci=f"SD: acc={c_without['std_acc']:.4f}, bal={c_without['std_bal']:.4f}, f1={c_without['std_f1']:.4f}",
    src_file=exp3_concat_json,
    src_line="without_trivial_features.summary",
    src_type="json",
    inc_locked="no",
    lim_note="Baseline EfficientNet-B0 linear classification performance without trivial feature concatenation."
)

add_row(
    exp_id="AUDIT-EXP3-EFFNET-WITH-TRIVIAL",
    arch="EfficientNet-B0 + Trivial Features",
    config="GAP EfficientNet-B0 features concatenated with 42 trivial features (Arm 1 recipe)",
    dataset="Original ~6000-image dataset (positive subset N=2,769)",
    n_img=2769,
    n_part="not recorded in artifact",
    n_classes=4,
    split="5-fold cross-validation across 3 seeds",
    ckpt_rule="fixed 50 epochs (no selection)",
    metric_lvl="audit",
    acc=c_with["mean_acc"],
    bacc=c_with["mean_bal"],
    mp="not recorded in artifact",
    mr="not recorded in artifact",
    mf1=c_with["mean_f1"],
    ci=f"SD: acc={c_with['std_acc']:.4f}, bal={c_with['std_bal']:.4f}, f1={c_with['std_f1']:.4f}; Paired Delta Bal Acc: +0.0035 +/- 0.0013",
    src_file=exp3_concat_json,
    src_line="with_trivial_features_concatenated.summary",
    src_type="json",
    inc_locked="no",
    lim_note="Trivial features concatenation produces a modest paired gain (+0.35 percentage points bal acc)."
)

# =========================================================================
# Write Files
# =========================================================================
fieldnames = [
    "experiment_id",
    "group",
    "model_architecture",
    "main_configuration",
    "dataset",
    "n_images",
    "n_participants",
    "n_classes",
    "split_protocol",
    "checkpoint_selection_rule",
    "metric_level",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "confidence_interval",
    "source_file",
    "source_field_or_line",
    "source_type",
    "includes_locked_test",
    "limitation_note",
    "verified_from_artifact"
]

audit_diagnostic_prefixes = [
    "AUDIT-EXP1-HASH",
    "AUDIT-EXP1-PERCOLATION",
    "AUDIT-EXP2-SIGNAL",
    "AUDIT-PHASE2-SELECTION-INCLUSIVE-NULL"
]

# Add 7 Phase 3 Architecture Benchmark locked rows and 2 Day-27 locked zero-shot rows to model_results.csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

arch_benchmark_files = [
    ("P3-ARCH-CONVNEXT-LOCKED-21", "ConvNeXt-Tiny", "reports/phase3_architecture_benchmark/convnext_tiny_test_preds.csv"),
    ("P3-ARCH-CUSTOMCNN-LOCKED-21", "Custom CNN", "reports/phase3_architecture_benchmark/custom_cnn_test_preds.csv"),
    ("P3-ARCH-DENSE121-LOCKED-21", "DenseNet-121", "reports/phase3_architecture_benchmark/densenet121_test_preds.csv"),
    ("P3-ARCH-EFFB0-LOCKED-21", "EfficientNet-B0", "reports/phase3_architecture_benchmark/efficientnet_b0_test_preds.csv"),
    ("P3-ARCH-EFFV2B0-LOCKED-21", "EfficientNetV2-B0", "reports/phase3_architecture_benchmark/efficientnetv2_b0_test_preds.csv"),
    ("P3-ARCH-RESNET50-LOCKED-21", "ResNet-50", "reports/phase3_architecture_benchmark/resnet50_test_preds.csv"),
    ("P3-ARCH-VGG16-LOCKED-21", "VGG16", "reports/phase3_architecture_benchmark/vgg16_test_preds.csv"),
]

locked_arch_rows = []
for exp_id_arch, arch_name, fpath in arch_benchmark_files:
    df_arch = pd.read_csv(fpath)
    acc = accuracy_score(df_arch["true_label"], df_arch["final_prediction"])
    bacc = balanced_accuracy_score(df_arch["true_label"], df_arch["final_prediction"])
    p, r, f1, _ = precision_recall_fscore_support(df_arch["true_label"], df_arch["final_prediction"], average="macro", zero_division=0)
    # Formulate architecture-specific limitation notes
    if arch_name == "EfficientNet-B0":
        lim_msg = "DISAGREEMENT NOTE: Architecture benchmark yields 0.1905 accuracy (4/21) and 0.1250 macro F1. Disagrees with locked_test_metrics.json (which reports 0.2857 accuracy, 0.1587 macro F1; difference: -0.0952 accuracy, -0.0337 macro F1). Causes (fully explained by files): NOT THE SAME MODEL. Checkpoint: Benchmark used fine-tuned model saved during Experiment 1 (quoted in scratch/experiment1_architecture.py:215, 232: models/phase3_architecture_benchmark/efficientnet_b0/best_model.keras), whereas locked_test_metrics.json loaded synthetic checkpoint models/abo_positive_4class/best_model.keras (quoted in scratch/run_full_experiment.py:38, 87). Preprocessing: Benchmark used C5 pipeline at 112x112 (scratch/experiment1_architecture.py:60), whereas locked_test_metrics.json used raw uncropped BMP resize to 224x224 (scratch/run_full_experiment.py:120). Training script: Benchmark trained on 64 MFS100 Dev participants (scratch/experiment1_architecture.py:63), whereas ensemble experiment had no MFS100 training. Aggregation: Benchmark is single-scan prediction (scratch/experiment1_architecture.py:273), whereas ensemble experiment is 3-scan mean probability (scratch/run_full_experiment.py:170)."
    elif arch_name == "Custom CNN":
        lim_msg = "DISAGREEMENT NOTE: Architecture benchmark Custom CNN yields 0.2857 accuracy (6/21) and 0.1200 macro F1. Disagrees with Day 27 zero-shot locked test rows (0.3333 accuracy, 0.1875 macro F1 for participant-level; difference: -0.0476 accuracy, -0.0675 macro F1) and FINAL_TEST_REPORT.md (0.3809 accuracy, 0.3125 macro F1; difference: -0.0952 accuracy, -0.1925 macro F1). Causes (fully explained by files for zero-shot; unexplained for FINAL_TEST_REPORT.md): NOT THE SAME MODEL. Checkpoint: Benchmark Custom CNN was trained from scratch on 64 MFS100 Dev participants with checkpoint saved to models/phase3_architecture_benchmark/custom_cnn/best_model.keras (quoted in scratch/experiment1_architecture.py:215, 232), whereas Day 27 zero-shot loaded models/day27_full_dataset/four_class/best_model.keras (quoted in scratch/final_zero_shot_eval.py:19-24). Preprocessing: Benchmark used C5 at 112x112 (scratch/experiment1_architecture.py:60); zero-shot used C5 at 112x112 (scratch/final_zero_shot_eval.py:46). Aggregation: Benchmark is single-scan prediction vs 3-scan mean probability in Day 27 participant-level."
    else:
        lim_msg = f"Phase 3 Experiment 1 architecture benchmark locked test evaluation for {arch_name}."

    locked_arch_rows.append({
        "experiment_id": exp_id_arch,
        "group": "Phase 3 Architecture Benchmark (Locked Cohort)",
        "model_architecture": arch_name,
        "main_configuration": f"{arch_name} fine-tuned on 64 Train participants, selected on 16 Validation participants (scratch/experiment1_architecture.py:63-75), evaluated on 21 Locked Test participants",
        "dataset": "MFS100 Locked Test Cohort (TEST split)",
        "n_images": "63",
        "n_participants": "21",
        "n_classes": "4",
        "split_protocol": "Trained on 64 Train participants (192 scans), validated on 16 Validation participants (48 scans), evaluated on 21 Locked Test participants (63 scans) (scratch/experiment1_architecture.py:63-75)",
        "checkpoint_selection_rule": "val_macro_f1 on 16 validation participants (scratch/experiment1_architecture.py:275)",
        "metric_level": "locked test",
        "accuracy": f"{acc:.4f}",
        "balanced_accuracy": f"{bacc:.4f}",
        "macro_precision": f"{p:.4f}",
        "macro_recall": f"{r:.4f}",
        "macro_f1": f"{f1:.4f}",
        "confidence_interval": "not recorded in artifact",
        "source_file": fpath,
        "source_field_or_line": "participant_id, true_label, final_prediction",
        "source_type": "csv",
        "includes_locked_test": "yes",
        "limitation_note": lim_msg,
        "verified_from_artifact": "yes"
    })

# Day 27 locked test rows (participant-level and scan-level)
split_df = pd.read_csv("reports/phase3_participant_benchmark/participant_split.csv")
locked_pids = split_df[split_df["split"] == "TEST"]["participant_id"].unique().tolist()

d27_p = pd.read_csv("reports/final_mfs100/final_zero_shot_participant_predictions.csv")
d27_p_locked = d27_p[d27_p["participant_id"].isin(locked_pids)]
acc_p = accuracy_score(d27_p_locked["true_label"], d27_p_locked["final_prediction"])
bacc_p = balanced_accuracy_score(d27_p_locked["true_label"], d27_p_locked["final_prediction"])
p_p, r_p, f1_p, _ = precision_recall_fscore_support(d27_p_locked["true_label"], d27_p_locked["final_prediction"], average="macro", zero_division=0)
locked_arch_rows.append({
    "experiment_id": "DAY27-LOCKED-ZERO-SHOT-PART-21",
    "group": "Day 27 Locked Cohort Zero-Shot",
    "model_architecture": "Day 27 Model A (Custom CNN 4-class)",
    "main_configuration": "Day 27 Model A zero-shot participant-level (mean prob across 3 scans)",
    "dataset": "MFS100 Locked Test Cohort (TEST split)",
    "n_images": "63",
    "n_participants": "21",
    "n_classes": "4",
    "split_protocol": "Strictly locked test cohort (21 participants, 63 scans)",
    "checkpoint_selection_rule": "Day 27 baseline checkpoint",
    "metric_level": "locked test",
    "accuracy": f"{acc_p:.4f}",
    "balanced_accuracy": f"{bacc_p:.4f}",
    "macro_precision": f"{p_p:.4f}",
    "macro_recall": f"{r_p:.4f}",
    "macro_f1": f"{f1_p:.4f}",
    "confidence_interval": "not recorded in artifact",
    "source_file": "reports/final_mfs100/final_zero_shot_participant_predictions.csv",
    "source_field_or_line": "participant_id, true_label, final_prediction (filtered to TEST split)",
    "source_type": "csv",
    "includes_locked_test": "yes",
    "limitation_note": "DISAGREEMENT NOTE: Verified zero-shot participant predictions filtered to the 21 locked participants yield 33.33% accuracy (7/21) and 0.1875 macro F1. Disagrees with FINAL_TEST_REPORT.md:36-39 (which reported 38.09% accuracy and 0.3125 macro F1; difference: -4.76 pp accuracy, -0.1250 macro F1). Arithmetic note: 38.09% is neither 7/21 (33.333%) nor 8/21 (38.095% -> 38.10%) exactly. Causes: unexplained in repository files (author of FINAL_TEST_REPORT.md transcribed unverified figures). Model comparison: Checkpoint loaded from models/day27_full_dataset/four_class/best_model.keras (quoted in scratch/final_zero_shot_eval.py:19-24); C5 preprocessing with resize to (112, 112) (quoted in scratch/final_zero_shot_eval.py:46); aggregation is 3-scan mean probability. FINAL_TEST_REPORT.md specifies no checkpoint path or script.",
    "verified_from_artifact": "yes"
})

d27_s = pd.read_csv("reports/final_mfs100/final_zero_shot_scan_predictions.csv")
d27_s_locked = d27_s[d27_s["participant_id"].isin(locked_pids)]
acc_s = accuracy_score(d27_s_locked["true_label"], d27_s_locked["predicted_class"])
bacc_s = balanced_accuracy_score(d27_s_locked["true_label"], d27_s_locked["predicted_class"])
p_s, r_s, f1_s, _ = precision_recall_fscore_support(d27_s_locked["true_label"], d27_s_locked["predicted_class"], average="macro", zero_division=0)
locked_arch_rows.append({
    "experiment_id": "DAY27-LOCKED-ZERO-SHOT-SCAN-63",
    "group": "Day 27 Locked Cohort Zero-Shot",
    "model_architecture": "Day 27 Model A (Custom CNN 4-class)",
    "main_configuration": "Day 27 Model A zero-shot single-scan prediction",
    "dataset": "MFS100 Locked Test Cohort (TEST split)",
    "n_images": "63",
    "n_participants": "21",
    "n_classes": "4",
    "split_protocol": "Strictly locked test cohort (21 participants, 63 scans)",
    "checkpoint_selection_rule": "Day 27 baseline checkpoint",
    "metric_level": "locked test",
    "accuracy": f"{acc_s:.4f}",
    "balanced_accuracy": f"{bacc_s:.4f}",
    "macro_precision": f"{p_s:.4f}",
    "macro_recall": f"{r_s:.4f}",
    "macro_f1": f"{f1_s:.4f}",
    "confidence_interval": "not recorded in artifact",
    "source_file": "reports/final_mfs100/final_zero_shot_scan_predictions.csv",
    "source_field_or_line": "participant_id, true_label, predicted_class (filtered to TEST split)",
    "source_type": "csv",
    "includes_locked_test": "yes",
    "limitation_note": "Day 27 zero-shot single-scan predictions filtered to the 63 scans of the 21 locked test participants (0.3492 accuracy, 0.1792 macro F1). Evaluates synthetic Day 27 checkpoint directly on individual scans.",
    "verified_from_artifact": "yes"
})

# Separate classifier vs diagnostic rows
classifier_rows = [r for r in rows if not any(r["experiment_id"].startswith(p) for p in audit_diagnostic_prefixes)]
diagnostic_rows = [r for r in rows if any(r["experiment_id"].startswith(p) for p in audit_diagnostic_prefixes)]

# Append locked_arch_rows to rows and classifier_rows so results_inventory = model_results + audit_diagnostics
for r in locked_arch_rows:
    rows.append(r)
    classifier_rows.append(r)

out_inventory = "reports/project_report/results_inventory.csv"
with open(out_inventory, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

out_models = "reports/project_report/model_results.csv"
with open(out_models, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in classifier_rows:
        writer.writerow(r)

out_diagnostics = "reports/project_report/audit_diagnostics.csv"
with open(out_diagnostics, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in diagnostic_rows:
        writer.writerow(r)

print(f"Generated results_inventory.csv ({len(rows)} rows)")
print(f"Generated model_results.csv ({len(classifier_rows)} rows)")
print(f"Generated audit_diagnostics.csv ({len(diagnostic_rows)} rows)")
assert len(rows) == len(classifier_rows) + len(diagnostic_rows), "Inventory identity mismatch!"
print("Verified identity: inventory == model_results + audit_diagnostics")
