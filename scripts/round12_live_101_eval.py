import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from src.preprocessing import transform_c5

def wilson_ci(k, n, confidence=0.95):
    """Computes Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return center - half_width, center + half_width

def composite_alpha_on_white(img_bgra):
    if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
        alpha = img_bgra[:, :, 3].astype(np.float32) / 255.0
        bgr = img_bgra[:, :, :3].astype(np.float32)
        white = np.ones_like(bgr) * 255.0
        out = bgr * alpha[:, :, np.newaxis] + white * (1.0 - alpha[:, :, np.newaxis])
        return np.clip(out, 0, 255).astype(np.uint8)
    return img_bgra

def preprocess_effnet(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    comp = composite_alpha_on_white(img)
    rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB) if comp.ndim == 3 else cv2.cvtColor(comp, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return resized

def print_evaluation_block(title, y_true, y_pred, classes):
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    k = np.sum(y_true == y_pred)
    ci_low, ci_high = wilson_ci(k, n)
    
    # Class counts
    true_counts = pd.Series(y_true).value_counts().to_dict()
    pred_counts = pd.Series(y_pred).value_counts().to_dict()
    for c in classes:
        true_counts.setdefault(c, 0)
        pred_counts.setdefault(c, 0)
        
    # Always-O+ baseline
    o_count = true_counts.get("O+", 0)
    always_o_baseline = o_count / n
    
    # Prior-matched chance
    prior_matched = sum((true_counts[c] / n) * (pred_counts[c] / n) for c in classes)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    print("\n" + "="*65)
    print(f" {title} ")
    print("="*65)
    print(f"Total Participants: {n}")
    print(f"Accuracy: {acc*100:.2f}% ({k}/{n})")
    print(f"Wilson 95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")
    print(f"Always-O+ Majority Baseline: {always_o_baseline*100:.2f}% ({o_count}/{n})")
    print(f"Prior-Matched Chance: {prior_matched*100:.2f}%")
    print("\nPredicted-Class Counts:")
    for c in classes:
        print(f"  {c}: {pred_counts[c]} ({pred_counts[c]/n*100:.1f}%)")
    print("\nTrue-Class Counts:")
    for c in classes:
        print(f"  {c}: {true_counts[c]} ({true_counts[c]/n*100:.1f}%)")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    header = f"{'True/Pred':<10}" + "".join([f"{c:>8}" for c in classes])
    print(header)
    for i, true_c in enumerate(classes):
        row_str = f"{true_c:<10}" + "".join([f"{cm[i, j]:>8}" for j in range(len(classes))])
        print(row_str)

def main():
    part_df = pd.read_csv("data/mfs100_final/metadata/participants.csv")
    log_df = pd.read_csv("data/mfs100_final/metadata/collection_log.csv")
    merged = pd.merge(log_df, part_df, on="participant_id", how="left")
    
    # Filter 4-class participants
    valid_classes = ["A+", "B+", "AB+", "O+"]
    merged = merged[merged["blood_group"].isin(valid_classes)]
    
    groups = merged.groupby("participant_id")
    participant_data = []
    for pid, grp in groups:
        paths = grp["image_path"].tolist()
        if len(paths) == 3:
            participant_data.append({
                "pid": pid,
                "label": grp["blood_group"].iloc[0],
                "paths": paths
            })
            
    n_part = len(participant_data)
    print(f"Loaded {n_part} participants with exactly 3 scans.")
    
    # -------------------------------------------------------------
    # 1 & 2: E2 Ensemble (Deployed vs B+/AB+ Swapped)
    # -------------------------------------------------------------
    from src.predictor import BloodGroupPredictor
    predictor = BloodGroupPredictor()
    model = predictor.model
    cps = predictor.checkpoint_paths
    
    # Build batch for all participants
    print("Preprocessing live participant scans using C5...")
    X_live = []
    y_true = []
    for item in participant_data:
        scans = [transform_c5(p)[0] for p in item["paths"]]
        X_live.append(scans)
        y_true.append(item["label"])
    X_live = np.array(X_live, dtype=np.float32) # (101, 3, 112, 112, 3)
    y_true = np.array(y_true)
    
    print("Running E2 5-fold ensemble inference...")
    fold_probs = []
    for cp in cps:
        model.load_weights(cp)
        preds = model.predict(X_live, batch_size=16, verbose=0)[0]
        fold_probs.append(preds)
    ensemble_probs = np.mean(fold_probs, axis=0) # (101, 4)
    best_indices = np.argmax(ensemble_probs, axis=1)
    
    # Deployed Class Order: ['A+', 'B+', 'AB+', 'O+']
    classes_deployed = ["A+", "B+", "AB+", "O+"]
    y_pred_deployed = np.array([classes_deployed[i] for i in best_indices])
    print_evaluation_block("E2 ENSEMBLE - DEPLOYED CLASS ORDER ['A+', 'B+', 'AB+', 'O+']", y_true, y_pred_deployed, classes_deployed)
    
    # Swapped Class Order: ['A+', 'AB+', 'B+', 'O+']
    classes_swapped = ["A+", "AB+", "B+", "O+"]
    y_pred_swapped = np.array([classes_swapped[i] for i in best_indices])
    print_evaluation_block("E2 ENSEMBLE - SWAPPED CLASS ORDER ['A+', 'AB+', 'B+', 'O+']", y_true, y_pred_swapped, classes_swapped)
    
    # -------------------------------------------------------------
    # 3: Exploratory EfficientNetB0 on 101 Participants
    # -------------------------------------------------------------
    effnet_path = "models/abo_positive_4class/best_model.keras"
    if os.path.exists(effnet_path):
        effnet_model = tf.keras.models.load_model(effnet_path, compile=False)
        classes_effnet = ["A+", "AB+", "B+", "O+"]
        
        y_pred_effnet = []
        for item in participant_data:
            scan_probs = []
            for p in item["paths"]:
                arr = preprocess_effnet(p)
                inp = np.expand_dims(arr, axis=0)
                pred = effnet_model(inp, training=False).numpy()[0]
                scan_probs.append(pred)
            avg_prob = np.mean(scan_probs, axis=0)
            pred_class = classes_effnet[np.argmax(avg_prob)]
            y_pred_effnet.append(pred_class)
            
        y_pred_effnet = np.array(y_pred_effnet)
        print_evaluation_block("EXPLORATORY: EfficientNetB0 (Avg of 3 Scans, 224x224, No Crop/CLAHE)", y_true, y_pred_effnet, classes_effnet)

if __name__ == "__main__":
    main()
