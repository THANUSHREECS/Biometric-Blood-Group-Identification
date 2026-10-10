import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_image_to_pil, composite_alpha_on_white, crop_mfs100_foreground, enhance_mfs100_ridge_contrast, resize_to_target
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

def transform_c5(image_input, target_size=(112, 112)):
    """
    C5 = C1 background norm + C2 scale norm + C3 spatial framing + C4 intensity norm
    Reusing existing implementations from src.preprocessing
    """
    # C1 Background
    pil_img = load_image_to_pil(image_input)
    rgb_img = composite_alpha_on_white(pil_img)
    
    # C3 Spatial Framing
    cropped, _ = crop_mfs100_foreground(rgb_img)
    
    # C4 Intensity Normalization
    enhanced = enhance_mfs100_ridge_contrast(cropped)
    
    # C2 Scale Normalization
    resized = resize_to_target(enhanced, target_size=target_size)
    
    # Convert to array [0, 255] float32
    return np.array(resized, dtype=np.float32)

def build_custom_cnn(input_shape=(112, 112, 3), num_classes=4):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def load_data(df, target_classes):
    df_filtered = df[df["true_label"].isin(target_classes)].copy()
    class_to_id = {c: i for i, c in enumerate(target_classes)}
    paths = df_filtered["image_path"].tolist()
    labels = df_filtered["true_label"].map(class_to_id).values
    
    X = []
    for p in paths:
        X.append(transform_c5(p))
    return np.array(X), labels, paths

def augment_data(X, y):
    """Canonical augmentation policy"""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1, fill_mode='nearest'),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'),
        tf.keras.layers.RandomZoom(0.1, fill_mode='nearest'),
    ])
    
    # We apply this via tf.data dataset in the training pipeline
    return data_augmentation

def train_model(model_name, target_classes, X_train, y_train, X_val, y_val):
    model = build_custom_cnn(num_classes=len(target_classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Compute class weights ONLY from train
    classes_arr = np.array(y_train)
    unique_classes = np.unique(classes_arr)
    raw_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=classes_arr)
    class_weight = {c: w for c, w in zip(unique_classes, raw_weights)}
    
    # Augmentation
    aug = augment_data(None, None)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train)).batch(32).map(lambda x, y: (aug(x, training=True), y))
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
    
    out_dir = f"models/day27_full_dataset/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{out_dir}/best_model.keras",
        monitor="val_loss",  # Using val_loss because keras might complain if val_macro_f1 isn't built in.
        save_best_only=True
    )
    
    # Early stopping only AFTER epoch 100
    class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
        def on_epoch_end(self, epoch, logs=None):
            if epoch < 100:
                self.wait = 0
            else:
                super().on_epoch_end(epoch, logs)
                
    cb_early = CustomEarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=150, class_weight=class_weight, callbacks=[cb_checkpoint, cb_early], verbose=2)
    pd.DataFrame(history.history).to_csv(f"reports/day27_{model_name}_history.csv", index=False)
    
    return model

def bootstrap_ci(y_true, y_pred, n_iterations=10000):
    n_size = len(y_true)
    accuracies = []
    f1s = []
    
    for _ in range(n_iterations):
        indices = np.random.randint(0, n_size, n_size)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        accuracies.append(accuracy_score(y_true_boot, y_pred_boot))
        _, _, f1, _ = precision_recall_fscore_support(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        f1s.append(f1)
        
    acc_lower, acc_upper = np.percentile(accuracies, [2.5, 97.5])
    f1_lower, f1_upper = np.percentile(f1s, [2.5, 97.5])
    
    return (np.mean(accuracies), acc_lower, acc_upper), (np.mean(f1s), f1_lower, f1_upper)

def evaluate_and_report(model, X_test, y_test, target_classes, model_name, suffix):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    
    rep = classification_report(y_test, y_pred, target_names=target_classes, output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(f"reports/day27_{model_name}_{suffix}_classification_report.csv")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_classes, yticklabels=target_classes)
    plt.title(f"{model_name} {suffix} Confusion Matrix")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"reports/day27_{model_name}_{suffix}_confusion_matrix.png")
    plt.close()
    
    metrics = {
        "accuracy": acc,
        "macro_precision": p_mac,
        "macro_recall": r_mac,
        "macro_f1": f1_mac
    }
    
    # Per class metrics
    for i, cls in enumerate(target_classes):
        metrics[f"{cls}_precision"] = rep[cls]['precision']
        metrics[f"{cls}_recall"] = rep[cls]['recall']
        metrics[f"{cls}_f1"] = rep[cls]['f1-score']
        
    pd.DataFrame([metrics]).to_csv(f"reports/day27_{model_name}_{suffix}_canonical_results.csv", index=False)
    return metrics, y_pred_probs, y_pred

def main():
    print("Loading data splits...")
    df_train = pd.read_csv("reports/day27_full_dataset_split_train.csv")
    df_val = pd.read_csv("reports/day27_full_dataset_split_val.csv")
    df_test = pd.read_csv("reports/day27_full_dataset_split_test.csv")
    
    # MFS100 
    mfs100_df = pd.read_csv("data/mfs100_ground_truth.csv")
    mfs100_df["image_path"] = "data/captured/" + mfs100_df["capture_filename"]
    # Only evaluate on MFS100 classes that exist
    mfs100_df = mfs100_df[mfs100_df["confirmed_blood_group"].isin(["A+", "B+", "AB+", "O+"])]
    
    four_classes = ["A+", "AB+", "B+", "O+"]
    eight_classes = ["A-", "A+", "AB-", "AB+", "B-", "B+", "O-", "O+"]
    
    print("Processing Four Class Data...")
    X_train_4, y_train_4, _ = load_data(df_train, four_classes)
    X_val_4, y_val_4, _ = load_data(df_val, four_classes)
    X_test_4, y_test_4, _ = load_data(df_test, four_classes)
    
    print("Training Model A...")
    model_a = train_model("four_class", four_classes, X_train_4, y_train_4, X_val_4, y_val_4)
    
    print("Evaluating Model A Canonical...")
    metrics_a, _, _ = evaluate_and_report(model_a, X_test_4, y_test_4, four_classes, "four_class", "canonical")
    
    print("Processing MFS100 Data for Model A...")
    X_mfs_4, y_mfs_4, mfs_paths_4 = load_data(mfs100_df.rename(columns={"confirmed_blood_group": "true_label"}), four_classes)
    metrics_a_mfs, probs_a_mfs, preds_a_mfs = evaluate_and_report(model_a, X_mfs_4, y_mfs_4, four_classes, "four_class", "mfs100")
    
    print("Processing Eight Class Data...")
    X_train_8, y_train_8, _ = load_data(df_train, eight_classes)
    X_val_8, y_val_8, _ = load_data(df_val, eight_classes)
    X_test_8, y_test_8, _ = load_data(df_test, eight_classes)
    
    print("Training Model B...")
    model_b = train_model("eight_class", eight_classes, X_train_8, y_train_8, X_val_8, y_val_8)
    
    print("Evaluating Model B Canonical...")
    metrics_b, _, _ = evaluate_and_report(model_b, X_test_8, y_test_8, eight_classes, "eight_class", "canonical")
    
    print("Evaluating Model B MFS100...")
    X_mfs_8, y_mfs_8, mfs_paths_8 = load_data(mfs100_df.rename(columns={"confirmed_blood_group": "true_label"}), eight_classes)
    metrics_b_mfs, probs_b_mfs, preds_b_mfs = evaluate_and_report(model_b, X_mfs_8, y_mfs_8, eight_classes, "eight_class", "mfs100")
    
    print("Creating prediction CSV...")
    # mfs_paths_4 and mfs_paths_8 should be identical
    pred_data = []
    for i in range(len(mfs_paths_4)):
        pred_data.append({
            "image_path": mfs_paths_4[i],
            "true_label": mfs100_df.iloc[i]["confirmed_blood_group"],
            "model": "Model A",
            "predicted_label": four_classes[preds_a_mfs[i]],
            "confidence": float(np.max(probs_a_mfs[i])),
            "prob_A_minus": None,
            "prob_A_plus": probs_a_mfs[i][four_classes.index("A+")],
            "prob_AB_minus": None,
            "prob_AB_plus": probs_a_mfs[i][four_classes.index("AB+")],
            "prob_B_minus": None,
            "prob_B_plus": probs_a_mfs[i][four_classes.index("B+")],
            "prob_O_minus": None,
            "prob_O_plus": probs_a_mfs[i][four_classes.index("O+")]
        })
        pred_data.append({
            "image_path": mfs_paths_8[i],
            "true_label": mfs100_df.iloc[i]["confirmed_blood_group"],
            "model": "Model B",
            "predicted_label": eight_classes[preds_b_mfs[i]],
            "confidence": float(np.max(probs_b_mfs[i])),
            "prob_A_minus": probs_b_mfs[i][eight_classes.index("A-")],
            "prob_A_plus": probs_b_mfs[i][eight_classes.index("A+")],
            "prob_AB_minus": probs_b_mfs[i][eight_classes.index("AB-")],
            "prob_AB_plus": probs_b_mfs[i][eight_classes.index("AB+")],
            "prob_B_minus": probs_b_mfs[i][eight_classes.index("B-")],
            "prob_B_plus": probs_b_mfs[i][eight_classes.index("B+")],
            "prob_O_minus": probs_b_mfs[i][eight_classes.index("O-")],
            "prob_O_plus": probs_b_mfs[i][eight_classes.index("O+")]
        })
    pd.DataFrame(pred_data).to_csv("reports/day27_mfs100_predictions.csv", index=False)
    
    # Bootstrapping
    print("Bootstrapping MFS100...")
    acc_ci_a, f1_ci_a = bootstrap_ci(y_mfs_4, preds_a_mfs)
    acc_ci_b, f1_ci_b = bootstrap_ci(y_mfs_8, preds_b_mfs)
    
    print("Primary Positive-Class Evaluation for Model B...")
    # Calculate Model B positive-class subset metrics for MFS100
    pos_preds_b = []
    pos_true_b = []
    neg_predictions_count = {"A-": 0, "B-": 0, "AB-": 0, "O-": 0}
    
    for i in range(len(preds_b_mfs)):
        pred_lbl = eight_classes[preds_b_mfs[i]]
        true_lbl = mfs100_df.iloc[i]["confirmed_blood_group"]
        if pred_lbl in four_classes:
            pos_preds_b.append(pred_lbl)
        else:
            neg_predictions_count[pred_lbl] += 1
            pos_preds_b.append(pred_lbl) # Still keep it as misclassified for positive class recall?
            # Actually, if we are evaluating positive-class accuracy, a negative prediction is just a wrong prediction.
            # So we map to strings to do classification_report with four_classes.
            
        pos_true_b.append(true_lbl)
    
    # We report negative counts
    print(f"Model B MFS100 predicted as negative groups: {neg_predictions_count}")
    
    # Primary 4 positive metrics (using four_classes subset)
    rep_b_pos = classification_report(pos_true_b, pos_preds_b, labels=four_classes, output_dict=True, zero_division=0)
    b_pos_acc = accuracy_score([t for t, p in zip(pos_true_b, pos_preds_b) if p in four_classes], 
                               [p for t, p in zip(pos_true_b, pos_preds_b) if p in four_classes]) if len([p for p in pos_preds_b if p in four_classes]) > 0 else 0.0

    print("Generating Comparison Table...")
    comp_df = pd.DataFrame([{
        "model": "Historical_51.28_Benchmark",
        "training_classes": 4,
        "training_images": 1938,
        "canonical_test_accuracy": None,
        "canonical_test_macro_f1": None,
        "mfs100_accuracy": 0.5128,
        "mfs100_macro_f1": None,
        "Aplus_recall": None,
        "Bplus_recall": None,
        "ABplus_recall": None,
        "Oplus_recall": None,
        "Oplus_prediction_share": None
    }, {
        "model": "Day27_4Class",
        "training_classes": 4,
        "training_images": len(X_train_4),
        "canonical_test_accuracy": metrics_a["accuracy"],
        "canonical_test_macro_f1": metrics_a["macro_f1"],
        "mfs100_accuracy": metrics_a_mfs["accuracy"],
        "mfs100_macro_f1": metrics_a_mfs["macro_f1"],
        "Aplus_recall": metrics_a_mfs["A+_recall"],
        "Bplus_recall": metrics_a_mfs["B+_recall"],
        "ABplus_recall": metrics_a_mfs["AB+_recall"],
        "Oplus_recall": metrics_a_mfs["O+_recall"],
        "Oplus_prediction_share": sum([1 for p in preds_a_mfs if four_classes[p] == "O+"]) / len(preds_a_mfs)
    }, {
        "model": "Day27_8Class",
        "training_classes": 8,
        "training_images": len(X_train_8),
        "canonical_test_accuracy": metrics_b["accuracy"],
        "canonical_test_macro_f1": metrics_b["macro_f1"],
        "mfs100_accuracy": metrics_b_mfs["accuracy"],
        "mfs100_macro_f1": metrics_b_mfs["macro_f1"],
        "Aplus_recall": metrics_b_mfs["A+_recall"],
        "Bplus_recall": metrics_b_mfs["B+_recall"],
        "ABplus_recall": metrics_b_mfs["AB+_recall"],
        "Oplus_recall": metrics_b_mfs["O+_recall"],
        "Oplus_prediction_share": sum([1 for p in preds_b_mfs if eight_classes[p] == "O+"]) / len(preds_b_mfs)
    }])
    comp_df.to_csv("reports/DAY27_FULL_DATASET_COMPARISON.csv", index=False)
    
    print("\nDataset: 6000 Images")
    print("Model A: 4 positive classes")
    print("Model B: 8 classes")
    print(f"\nCanonical results:\nModel A: {metrics_a['accuracy']:.4f}\nModel B: {metrics_b['accuracy']:.4f}")
    print(f"\nMFS100 results:\nHistorical: 0.5128\nModel A: {metrics_a_mfs['accuracy']:.4f}\nModel B: {metrics_b_mfs['accuracy']:.4f}")
    print(f"\nA+ recall: Model A {metrics_a_mfs['A+_recall']:.4f}, Model B {metrics_b_mfs['A+_recall']:.4f}")
    print(f"B+ recall: Model A {metrics_a_mfs['B+_recall']:.4f}, Model B {metrics_b_mfs['B+_recall']:.4f}")
    print(f"AB+ recall: Model A {metrics_a_mfs['AB+_recall']:.4f}, Model B {metrics_b_mfs['AB+_recall']:.4f}")
    print(f"O+ recall: Model A {metrics_a_mfs['O+_recall']:.4f}, Model B {metrics_b_mfs['O+_recall']:.4f}")
    print(f"\nO+ prediction share: Model A {sum([1 for p in preds_a_mfs if four_classes[p] == 'O+']) / len(preds_a_mfs):.4f}")
    
if __name__ == "__main__":
    main()
