"""
src/predictor.py
Centralized prediction engine for Biometric Blood Group Identification.
Supports:
- Experiment 6 E2 Champion Model (4 Classes: A+, B+, AB+, O+)
- 3-scan feature fusion
- 5-fold ensemble aggregation

Speed-up (speed-up-inference branch):
- 5 fold model objects are built and weights loaded ONCE at startup.
- A warm-up prediction is run at startup so the first real request has no cold-start overhead.
- Per-request timing is printed to the console via time.perf_counter.
- No change to preprocessing, class order, thresholds, or prediction values.
"""

import os
import time
import json
import threading
import numpy as np
import tensorflow as tf

from src.config import (
    CLASS_NAMES,
    ID_TO_CLASS,
    ADVISORY_CONFIRMATION_THRESHOLD,
    ADVISORY_CONFIRMATION_THRESHOLD
)

from src.preprocessing import transform_c5

class BloodGroupPredictor:
    """
    Biometric Blood Group Inference Engine.
    Loads the E2 Champion Model checkpoints.
    """

    def __init__(
        self,
        model_path=None,
        class_names_path=None,
        ADVISORY_CONFIRMATION_THRESHOLD=ADVISORY_CONFIRMATION_THRESHOLD,
        advisory_threshold=ADVISORY_CONFIRMATION_THRESHOLD,
        use_ensemble=True
    ):
        self.ADVISORY_CONFIRMATION_THRESHOLD = ADVISORY_CONFIRMATION_THRESHOLD
        self.advisory_threshold = advisory_threshold
        self.class_names = CLASS_NAMES
        self.id_to_class = ID_TO_CLASS

        # Setup E2 model
        self.is_ensemble_ready = False
        self.model = None          # kept for compat; not used for inference after startup
        self.fold_models = []      # one tf.keras.Model per fold, weights pre-loaded
        self.checkpoint_paths = []
        
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Inference lock: prevents concurrent requests from interleaving
        # weight states across the 5 shared fold model objects.
        self._inference_lock = threading.Lock()
        
        self._load_models()

    def _build_E2(self):
        inputs = tf.keras.Input(shape=(3, 112, 112, 3))
        scan1, scan2, scan3 = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        
        base = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet")
        base.trainable = False
        pool = tf.keras.layers.GlobalAveragePooling2D()
        feat1, feat2, feat3 = pool(base(scan1)), pool(base(scan2)), pool(base(scan3))
        concat = tf.keras.layers.Concatenate()([feat1, feat2, feat3])
        
        d = tf.keras.layers.Dense(512, activation="relu")(concat)
        drop = tf.keras.layers.Dropout(0.5)(d)
        clf = tf.keras.layers.Dense(4, activation="softmax", name="clf")(drop)
        emb = tf.keras.layers.Dense(128, activation=None, name="emb")(concat)
        return tf.keras.Model(inputs, [clf, emb])

    def _load_models(self):
        """
        Loads the Champion E2 5-fold ensemble.
        Builds one model object per fold and loads weights at startup so that
        no weight I/O happens during inference requests.
        """
        try:
            t0 = time.perf_counter()
            config_path = os.path.join(self.project_root, "reports", "phase3_final_evaluation", "frozen_model_config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            
            self.checkpoint_paths = [os.path.join(self.project_root, cp.replace("/", os.sep)) for cp in config["checkpoint_paths"]]

            # Build a separate model object for each fold and pre-load weights
            self.fold_models = []
            for i, cp in enumerate(self.checkpoint_paths):
                t_fold = time.perf_counter()
                m = self._build_E2()
                m.load_weights(cp)
                elapsed = time.perf_counter() - t_fold
                print(f"[+] Fold {i+1}/{len(self.checkpoint_paths)} loaded in {elapsed:.2f}s  ({cp})")
                self.fold_models.append(m)

            # Keep self.model pointing at fold 0 for compat (not used in inference)
            self.model = self.fold_models[0] if self.fold_models else None
            self.is_ensemble_ready = True
            elapsed_total = time.perf_counter() - t0
            print(f"[+] E2 Champion 5-Fold Ensemble initialized in {elapsed_total:.2f}s (weights pre-loaded).")

            # Warm-up: run one dummy forward pass through each fold to JIT-compile graph
            self._warmup()

        except Exception as e:
            print(f"[-] Warning: Failed to load E2 models: {e}.")
            self.is_ensemble_ready = False

    def _warmup(self):
        """Run one zero-tensor forward pass per fold to eliminate cold-start latency."""
        try:
            dummy = np.zeros((1, 3, 112, 112, 3), dtype=np.float32)
            t0 = time.perf_counter()
            for i, m in enumerate(self.fold_models):
                m(dummy, training=False)
            elapsed = time.perf_counter() - t0
            print(f"[+] Warm-up complete ({len(self.fold_models)} folds) in {elapsed:.3f}s.")
        except Exception as e:
            print(f"[-] Warm-up failed (non-fatal): {e}")

    def is_model_ready(self):
        return self.is_ensemble_ready

    def get_model_info(self):
        if self.is_ensemble_ready:
            return {
                "name": "E2 Research Ensemble",
                "tag": "E2 Active",
                "accuracy": "8/21 on 21 held-out participants",
                "type": "5-Fold Feature Fusion",
                "classes": self.class_names,
                "ADVISORY_CONFIRMATION_THRESHOLD": self.ADVISORY_CONFIRMATION_THRESHOLD,
                "advisory_layer": {
                    "enabled": True,
                    "threshold": self.advisory_threshold,
                    "status": "healthy"
                }
            }
        else:
            return {"name": "No Model Loaded", "classes": self.class_names}

    def _evaluate_advisory(self, confidence):
        is_low_conf = bool(confidence < self.advisory_threshold)
        if is_low_conf:
            needs_confirmation = True
            advisory_level = "CONFIRMATION_RECOMMENDED"
            advisory_reason = "LOW_CONFIDENCE"
            advisory_message = (
                "Prediction confidence is below the validation-selected advisory threshold. "
                "On 21 held-out participants this research model matched always guessing O+ (8/21). It is not a blood-typing tool. "
                "Confirm blood group using standard serological testing."
            )
        else:
            needs_confirmation = False
            advisory_level = "CONFIDENCE_OK"
            advisory_reason = "NONE"
            advisory_message = (
                "Advisory threshold passed. This does not constitute medical verification. "
                "On 21 held-out participants this research model matched always guessing O+ (8/21). It is not a blood-typing tool."
            )

        return {
            "needs_confirmation": needs_confirmation,
            "advisory_threshold": self.advisory_threshold,
            "advisory_level": advisory_level,
            "advisory_reason": advisory_reason,
            "advisory_message": advisory_message
        }

    def predict_consensus(self, image_inputs, source="mfs100"):
        """
        E2 Model Inference. Requires exactly 3 scans per participant.
        Timing is printed to the console via time.perf_counter.
        """
        t_total = time.perf_counter()

        if not self.is_model_ready():
            raise RuntimeError("No model is loaded for inference.")
            
        if len(image_inputs) != 3:
            raise ValueError(f"E2 Model requires exactly 3 scans. Provided {len(image_inputs)}.")

        # --- Stage: Preprocessing (load_image, composite, crop, CLAHE/bilateral, resize) ---
        t_pre = time.perf_counter()
        scans = []
        crop_metas = []
        display_img = None
        for i, inp in enumerate(image_inputs):
            arr, pil_img, meta = transform_c5(inp)
            scans.append(arr)
            crop_metas.append(meta)
            if i == 0:
                display_img = pil_img
        t_pre_done = time.perf_counter()
        print(f"[TIMING] Preprocessing (3 scans): {(t_pre_done - t_pre)*1000:.1f} ms")

        # --- Stage: Build input tensor ---
        t_tensor = time.perf_counter()
        X_input = np.expand_dims(np.array(scans), axis=0)  # (1, 3, 112, 112, 3)
        t_tensor_done = time.perf_counter()
        print(f"[TIMING] Tensor assembly:          {(t_tensor_done - t_tensor)*1000:.1f} ms")

        # --- Stage: Inference per fold (weights pre-loaded at startup) ---
        # Use direct __call__ (same code path as warm-up) to avoid per-call tf.function retracing.
        # m.predict() traces a *different* tf.function than m(x, training=False) and would add
        # ~10s retracing overhead on every request after the first.
        # threading.Lock: fold models are shared objects; concurrent requests must not interleave.
        all_probs = []
        with self._inference_lock:
            for i, m in enumerate(self.fold_models):
                t_fold = time.perf_counter()
                preds = m(X_input, training=False)   # [clf_tensor, emb_tensor]
                all_probs.append(preds[0].numpy()[0])
                t_fold_done = time.perf_counter()
                print(f"[TIMING] Fold {i+1} inference:          {(t_fold_done - t_fold)*1000:.1f} ms")

        # --- Stage: Ensembling ---
        t_ens = time.perf_counter()
        all_probs = np.array(all_probs)
        ensemble_probs = np.mean(all_probs, axis=0)
        t_ens_done = time.perf_counter()
        print(f"[TIMING] Ensembling:               {(t_ens_done - t_ens)*1000:.1f} ms")
        
        best_idx = int(np.argmax(ensemble_probs))
        predicted_group = self.class_names[best_idx]
        confidence = float(ensemble_probs[best_idx])

        prob_dict = {
            self.class_names[i]: round(float(ensemble_probs[i]), 4)
            for i in range(len(self.class_names))
        }

        is_low = confidence < self.ADVISORY_CONFIRMATION_THRESHOLD
        status = "LOW_CONFIDENCE" if is_low else "NORMAL"
        msg = "Prediction completed successfully via E2 Research Ensemble."

        adv = self._evaluate_advisory(confidence)

        # --- Stage: Display image encoding ---
        t_enc = time.perf_counter()
        # display_img is a PIL Image — encoding happens in app.py if needed; nothing extra here
        t_enc_done = time.perf_counter()
        print(f"[TIMING] Display image encoding:   {(t_enc_done - t_enc)*1000:.1f} ms")

        t_total_done = time.perf_counter()
        print(f"[TIMING] predict_consensus TOTAL:  {(t_total_done - t_total)*1000:.1f} ms")

        return {
            "blood_group": predicted_group,
            "confidence": round(confidence, 4),
            "status": status,
            "message": msg,
            "probabilities": prob_dict,
            "is_low_confidence": is_low,
            "model_name": "E2 Research Ensemble",
            "display_image": display_img,
            "crop_metadata": crop_metas[0],
            "needs_confirmation": adv["needs_confirmation"],
            "advisory_threshold": adv["advisory_threshold"],
            "advisory_level": adv["advisory_level"],
            "advisory_reason": adv["advisory_reason"],
            "advisory_message": adv["advisory_message"]
        }
        
    def predict(self, image_input, source="upload"):
        """
        Fallback for single-scan uploads. Replicates the single scan 3 times to satisfy E2 input shape.
        This is an experimental mode and has NOT been validated as equivalent to three independent scans.
        """
        res = self.predict_consensus([image_input, image_input, image_input], source=source)
        res["message"] = "EXPERIMENTAL: Single image replicated 3 times. Not scientifically validated as equivalent to 3 independent scans."
        res["is_replicated_scan"] = True
        return res
