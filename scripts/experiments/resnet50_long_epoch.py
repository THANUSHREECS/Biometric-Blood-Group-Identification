#!/usr/bin/env python3
"""
Long‑epoch training experiment for ResNet50 (Champion Ensemble component).

- Uses existing data‑loading pipeline (train/val splits only).
- Two‑stage training: head‑only (5 epochs) then fine‑tune last conv block.
- Early stopping on `val_loss`.
- Generates CSV training log `outputs/long_epoch_training_log.csv`.
- Saves best checkpoint to `models/experiments/resnet50_long_epoch_best.keras`.
- Produces JSON and CSV validation reports.
"""
import os, argparse, time, json, csv
import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

# Add project root to sys.path for local imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import CLASS_NAMES, NUM_CLASSES, DATASET_MANIFEST_PATH, OUTPUTS_DIR, MODELS_DIR
from src.preprocessing import composite_alpha_on_white, resize_to_target
from scripts.train_models import (
    load_split,
    make_preprocess_fn,
    build_augmentation_layer,
    create_tf_dataset,
    compute_mild_class_weights,
    apply_finetune_strategy,
    save_training_curves,
    save_confusion_matrix,
    log_experiment,
    compute_val_metrics,
    MODEL_BUILDERS,
    INPUT_SIZES,
)

def get_args():
    parser = argparse.ArgumentParser(description="Long‑epoch ResNet50 experiment")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum total epochs (stage1+stage2)")
    parser.add_argument("--stage1_epochs", type=int, default=5, help="Head‑only training epochs")
    parser.add_argument("--stage2_epochs", type=int, default=25, help="Fine‑tune epochs (capped by max_epochs)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage1_lr", type=float, default=1e-3)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help="Early‑stopping patience")
    parser.add_argument("--early_stop_metric", type=str, default="val_loss", choices=["val_loss", "val_macro_f1"], help="Metric for early stopping")
    return parser.parse_args()

def main():
    args = get_args()
    arch_name = "resnet50"
    target_size = INPUT_SIZES[arch_name]
    input_shape = (*target_size, 3)
    save_path = os.path.join(MODELS_DIR, "experiments", "resnet50_long_epoch_best.keras")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # -------------------- Data loading --------------------
    train_paths, train_labels = load_split(DATASET_MANIFEST_PATH, "train")
    val_paths, val_labels = load_split(DATASET_MANIFEST_PATH, "val")
    mild_class_weights = compute_mild_class_weights(train_labels)
    augment_layer = build_augmentation_layer(arch_name)
    train_ds = create_tf_dataset(train_paths, train_labels, arch_name, target_size, args.batch_size, is_training=True, augment_layer=augment_layer)
    val_ds = create_tf_dataset(val_paths, val_labels, arch_name, target_size, args.batch_size, is_training=False)

    # -------------------- Model building --------------------
    build_fn = MODEL_BUILDERS[arch_name]
    model, base_model = build_fn(input_shape=input_shape, num_classes=NUM_CLASSES)
    param_count = model.count_params()
    print(f"[+] Total parameters: {param_count:,}")

    # -------------------- Callbacks --------------------
    csv_log_path = os.path.join(OUTPUTS_DIR, "long_epoch_training_log.csv")
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
    csv_file = open(csv_log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"])
    def make_logger(stage_name):
        def on_epoch_end(epoch, logs=None):
            logs = logs or {}
            csv_writer.writerow([
                epoch + 1,
                stage_name,
                logs.get("loss"),
                logs.get("accuracy"),
                logs.get("val_loss"),
                logs.get("val_accuracy"),
                logs.get("val_macro_f1", ""),
                logs.get("lr")
            ])
            csv_file.flush()
        return on_epoch_end
    csv_logger_stage1 = tf.keras.callbacks.LambdaCallback(on_epoch_end=make_logger("stage1"))
    csv_logger_stage2 = tf.keras.callbacks.LambdaCallback(on_epoch_end=make_logger("stage2"))

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=args.early_stop_metric, patience=args.patience, restore_best_weights=True, verbose=1)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # -------------------- Stage 1: head training --------------------
    if base_model is not None:
        base_model.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.stage1_lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.stage1_epochs,
        class_weight=mild_class_weights,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb, csv_logger_stage1],
        verbose=1,
    )

    # -------------------- Stage 2: fine‑tuning --------------------
    finetune_desc = "N/A"
    hist2 = None
    if base_model is not None and args.stage2_epochs > 0:
        finetune_desc = apply_finetune_strategy(arch_name, base_model)
        print(f"[+] Fine-tune strategy: {finetune_desc}")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.stage2_lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.stage2_epochs,
            class_weight=mild_class_weights,
            callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb, csv_logger_stage2],
            verbose=1,
        )
    csv_file.close()

    # -------------------- Evaluation --------------------
    if os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path)
    else:
        print("[!] Checkpoint not found, using last model state.")
    val_pred_labels, val_true_labels, val_macro_f1, val_acc = compute_val_metrics(model, val_ds, val_labels)
    save_confusion_matrix(val_true_labels, val_pred_labels, arch_name, split_name="val")
    # JSON summary
    summary = {
        "arch": arch_name,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "confusion_matrix": os.path.join(OUTPUTS_DIR, f"{arch_name}_confusion_matrix.png"),
        "training_log": csv_log_path,
        "checkpoint": save_path,
        "fine_tune_description": finetune_desc,
    }
    json_path = os.path.join(OUTPUTS_DIR, "long_epoch_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    print(f"[+] Summary written to {json_path}")

    # -------------------- Central experiment log --------------------
    total_epochs = len(hist1.history.get("accuracy", [])) + (len(hist2.history.get("accuracy", [])) if hist2 else 0)
    exp_id = f"resnet50_long_{int(time.time())}"
    log_experiment(
        experiment_id=exp_id,
        arch_name=arch_name,
        best_val_acc=val_acc,
        best_val_loss=min(hist1.history["val_loss"] + (hist2.history["val_loss"] if hist2 else [])),
        val_macro_f1=val_macro_f1,
        epochs_trained=total_epochs,
        training_time_min=(time.time() - start_time) / 60.0,
        param_count=param_count,
        is_pilot=False,
        finetune_desc=finetune_desc,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        notes="Long‑epoch ResNet50 experiment",
    )

if __name__ == "__main__":
    start_time = time.time()
    main()
