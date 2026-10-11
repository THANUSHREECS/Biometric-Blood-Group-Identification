import itertools
import os
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from src.predictor import BloodGroupPredictor
from src.preprocessing import transform_c5

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def transform_training_style(img_path):
    """Training-style preprocessing: composite alpha, resize to 112x112, DenseNet preprocess. No crop, no CLAHE."""
    img_bgra = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img_bgra is None:
        raise ValueError(f"Could not read image: {img_path}")
    if img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
        alpha = img_bgra[:, :, 3].astype(np.float32) / 255.0
        bgr = img_bgra[:, :, :3].astype(np.float32)
        white = np.ones_like(bgr) * 255.0
        out = bgr * alpha[:, :, np.newaxis] + white * (1.0 - alpha[:, :, np.newaxis])
        img = np.clip(out, 0, 255).astype(np.uint8)
    else:
        img = img_bgra
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA)
    arr = tf.keras.applications.densenet.preprocess_input(resized.astype(np.float32))
    return arr

def transform_c5_wrapper(img_path):
    arr, _, _ = transform_c5(img_path)
    return arr

def run_ensemble_batch(model, checkpoint_paths, X_batch):
    """Runs the 5-fold ensemble efficiently across X_batch (N, 3, 112, 112, 3)."""
    fold_probs = []
    for cp in checkpoint_paths:
        model.load_weights(cp)
        preds = model.predict(X_batch, batch_size=32, verbose=0)[0] # clf output
        fold_probs.append(preds)
    return np.mean(fold_probs, axis=0) # (N, 4)

def evaluate_permutations(raw_probs, true_labels, classes=("A+", "B+", "AB+", "O+")):
    """
    Evaluates all 24 permutations of class order mapping.
    raw_probs: (N, 4)
    true_labels: list of true class names length N
    """
    perms = list(itertools.permutations(classes))
    results = []
    best_idx = np.argmax(raw_probs, axis=1)
    
    for perm in perms:
        preds = [perm[i] for i in best_idx]
        acc = np.mean([p == t for p, t in zip(preds, true_labels)])
        results.append((perm, acc))
        
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def main():
    print("="*70)
    print("ROUND 12: ITEM 2 & 3 - CLASS ORDER AND TRIPLET PROBE")
    print("probe: E2's training split is unknown")
    print("="*70)
    
    predictor = BloodGroupPredictor()
    model = predictor.model
    cps = predictor.checkpoint_paths
    
    # Load 415 test images
    test_csv = "outputs/abo_positive_4class/test_predictions.csv"
    df_test = pd.read_csv(test_csv)
    test_paths = df_test["image_path"].tolist()
    label_col = "ground_truth_class" if "ground_truth_class" in df_test.columns else "true_blood_group"
    test_labels = df_test[label_col].tolist()
    n_test = len(test_paths)
    print(f"Loaded {n_test} test images.")
    
    # Sample 400 random training images (100 per class from data/processed)
    classes = ["A+", "B+", "AB+", "O+"]
    train_paths, train_labels = [], []
    for c in classes:
        folder = os.path.join("data", "processed", c)
        all_imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.upper().endswith(".BMP")]
        sampled = random.sample(all_imgs, 100)
        train_paths.extend(sampled)
        train_labels.extend([c] * 100)
    print(f"Sampled {len(train_paths)} training images (100 per class).")
    
    # -------------------------------------------------------------
    # ITEM 2: Run E2 on 415 test and 400 train, C5 vs Training-style
    # -------------------------------------------------------------
    print("\n--- Preprocessing 415 Test Images (triplet = same x3) ---")
    X_test_c5 = np.array([[transform_c5_wrapper(p)]*3 for p in test_paths], dtype=np.float32)
    X_test_tr = np.array([[transform_training_style(p)]*3 for p in test_paths], dtype=np.float32)
    
    print("--- Running Inference on 415 Test Images ---")
    probs_test_c5 = run_ensemble_batch(model, cps, X_test_c5)
    probs_test_tr = run_ensemble_batch(model, cps, X_test_tr)
    
    print("\n--- Preprocessing 400 Training Images (triplet = same x3) ---")
    X_train_c5 = np.array([[transform_c5_wrapper(p)]*3 for p in train_paths], dtype=np.float32)
    X_train_tr = np.array([[transform_training_style(p)]*3 for p in train_paths], dtype=np.float32)
    
    print("--- Running Inference on 400 Training Images ---")
    probs_train_c5 = run_ensemble_batch(model, cps, X_train_c5)
    probs_train_tr = run_ensemble_batch(model, cps, X_train_tr)
    
    print("\n" + "="*70)
    print("ITEM 2 RESULTS: 24 CLASS-ORDER PERMUTATIONS")
    print("probe: E2's training split is unknown")
    print("="*70)
    
    res_test_c5 = evaluate_permutations(probs_test_c5, test_labels, classes)
    res_test_tr = evaluate_permutations(probs_test_tr, test_labels, classes)
    res_train_c5 = evaluate_permutations(probs_train_c5, train_labels, classes)
    res_train_tr = evaluate_permutations(probs_train_tr, train_labels, classes)
    
    print("\n[415 Test Images - C5 Preprocessing]")
    for perm, acc in res_test_c5:
        print(f"  {perm} -> {acc*100:.2f}%")
        
    print("\n[415 Test Images - Training-style Preprocessing]")
    for perm, acc in res_test_tr:
        print(f"  {perm} -> {acc*100:.2f}%")
        
    print("\n[400 Train Images - C5 Preprocessing]")
    for perm, acc in res_train_c5:
        print(f"  {perm} -> {acc*100:.2f}%")
        
    print("\n[400 Train Images - Training-style Preprocessing]")
    for perm, acc in res_train_tr:
        print(f"  {perm} -> {acc*100:.2f}%")
        
    # Check standing out
    deployed_perm = ("A+", "B+", "AB+", "O+")
    swapped_perm = ("A+", "AB+", "B+", "O+")
    print("\nSummary Comparison for Deployed vs Swapped vs Best:")
    print(f"Deployed ('A+', 'B+', 'AB+', 'O+'):")
    print(f"  Test C5: {dict(res_test_c5)[deployed_perm]*100:.2f}% | Test Tr: {dict(res_test_tr)[deployed_perm]*100:.2f}%")
    print(f"  Train C5: {dict(res_train_c5)[deployed_perm]*100:.2f}% | Train Tr: {dict(res_train_tr)[deployed_perm]*100:.2f}%")
    print(f"Swapped ('A+', 'AB+', 'B+', 'O+'):")
    print(f"  Test C5: {dict(res_test_c5)[swapped_perm]*100:.2f}% | Test Tr: {dict(res_test_tr)[swapped_perm]*100:.2f}%")
    print(f"  Train C5: {dict(res_train_c5)[swapped_perm]*100:.2f}% | Train Tr: {dict(res_train_tr)[swapped_perm]*100:.2f}%")
    print(f"Best Test C5: {res_test_c5[0][0]} -> {res_test_c5[0][1]*100:.2f}%")
    print(f"Best Train C5: {res_train_c5[0][0]} -> {res_train_c5[0][1]*100:.2f}%")
    print(f"Best Train Tr: {res_train_tr[0][0]} -> {res_train_tr[0][1]*100:.2f}%")

    # -------------------------------------------------------------
    # ITEM 3: TRIPLET PROBE on 415 test images (Deployed Permutation & C5)
    # (a) same image x3
    # (b) three random images of same class
    # (c) three random images of any class
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("ITEM 3: TRIPLET PROBE ON 415 TEST IMAGES")
    print("="*70)
    
    # (a) already computed as probs_test_c5
    acc_a = dict(res_test_c5)[deployed_perm]
    
    # (b) three random images of same class
    by_class = {c: df_test[df_test[label_col] == c]["image_path"].tolist() for c in classes}
    triplets_b = []
    for p, label in zip(test_paths, test_labels):
        pool = by_class[label]
        other_two = random.sample(pool, 2)
        triplets_b.append([transform_c5_wrapper(p), transform_c5_wrapper(other_two[0]), transform_c5_wrapper(other_two[1])])
    X_test_b = np.array(triplets_b, dtype=np.float32)
    probs_b = run_ensemble_batch(model, cps, X_test_b)
    preds_b = [deployed_perm[i] for i in np.argmax(probs_b, axis=1)]
    acc_b = np.mean([p == t for p, t in zip(preds_b, test_labels)])
    
    # (c) three random images of any class
    all_pool = test_paths
    triplets_c = []
    for p in test_paths:
        other_two = random.sample(all_pool, 2)
        triplets_c.append([transform_c5_wrapper(p), transform_c5_wrapper(other_two[0]), transform_c5_wrapper(other_two[1])])
    X_test_c = np.array(triplets_c, dtype=np.float32)
    probs_c = run_ensemble_batch(model, cps, X_test_c)
    preds_c = [deployed_perm[i] for i in np.argmax(probs_c, axis=1)]
    acc_c = np.mean([p == t for p, t in zip(preds_c, test_labels)])
    
    print(f"(a) Same image x3:                           {acc_a*100:.2f}% ({int(round(acc_a*n_test))}/{n_test})")
    print(f"(b) Three random images of SAME class:       {acc_b*100:.2f}% ({int(round(acc_b*n_test))}/{n_test})")
    print(f"(c) Three random images of ANY class:        {acc_c*100:.2f}% ({int(round(acc_c*n_test))}/{n_test})")

if __name__ == "__main__":
    main()
