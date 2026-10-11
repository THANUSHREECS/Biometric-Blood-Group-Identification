"""
src/predictor.py
Centralized prediction engine for Biometric Blood Group Identification.
Supports:
- Experiment 6 E2 Champion Model (4 Classes: A+, B+, AB+, O+)
- 3-scan feature fusion
- 5-fold ensemble aggregation
"""

import os
import json
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
        self.model = None
        self.checkpoint_paths = []
        
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
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
        """Loads the Champion E2 5-fold ensemble."""
        try:
            config_path = os.path.join(self.project_root, "reports", "phase3_final_evaluation", "frozen_model_config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            
            self.checkpoint_paths = [os.path.join(self.project_root, cp.replace("/", os.sep)) for cp in config["checkpoint_paths"]]
            self.model = self._build_E2()
            self.is_ensemble_ready = True
            print("[+] E2 Champion 5-Fold Ensemble initialized successfully.")
        except Exception as e:
            print(f"[-] Warning: Failed to load E2 models: {e}.")
            self.is_ensemble_ready = False

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
        """
        if not self.is_model_ready():
            raise RuntimeError("No model is loaded for inference.")
            
        if len(image_inputs) != 3:
            raise ValueError(f"E2 Model requires exactly 3 scans. Provided {len(image_inputs)}.")
            
        scans = []
        crop_metas = []
        display_img = None
        for i, inp in enumerate(image_inputs):
            arr, pil_img, meta = transform_c5(inp)
            scans.append(arr)
            crop_metas.append(meta)
            if i == 0:
                display_img = pil_img

        X_input = np.expand_dims(np.array(scans), axis=0) # (1, 3, 112, 112, 3)

        all_probs = []
        for cp in self.checkpoint_paths:
            self.model.load_weights(cp)
            preds = self.model.predict(X_input, verbose=0)[0]
            all_probs.append(preds[0])

        all_probs = np.array(all_probs)
        ensemble_probs = np.mean(all_probs, axis=0)
        
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
