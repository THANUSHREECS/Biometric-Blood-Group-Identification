import json
from pathlib import Path

diff = {
    audit_metadata: {
        task: EXP-03 Preparation: Original E2 vs EXP-02 Configuration Audit,
        timestamp: 2026-10-14T13:05:00,
        original_e2_source: scripts/experiment6_fusion_supcon.py,
        exp02_source: experiments/exp02_e2_augmentation.py,
        baseline_oof_source: reports/model_fusion/all_models_participant_probs.npz and reports/efficientnet_e2_ensemble_experiment/e2_predictions.csv
    },
    comparison_dimensions: {
        1_architecture_and_layers: {
            original_e2: 3-scan fusion: DenseNet121 frozen backbone, GlobalAveragePooling2D, Concatenate(3072), Dense(512, relu), Dropout(0.5), Dense(4, softmax, name=clf), Dense(128, activation=None, name=emb),
            exp02: Identical layer graph: DenseNet121 frozen backbone, GlobalAveragePooling2D, Concatenate(3072, name=fused_features), Dense(512, relu, name=dense_512), Dropout(0.5, name=dropout_0.5), Dense(4, softmax, name=clf), Dense(128, activation=None, name=emb),
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Layer graphs are architecturally identical; EXP-02 simply added explicit layer names.
        },
        2_initial_weights_and_backbone: {
            original_e2: DenseNet121(weights='imagenet', include_top=False), trainable=False. Dense layers default Glorot Uniform.,
            exp02: DenseNet121(weights='imagenet', include_top=False), trainable=False. Dense layers default Glorot Uniform.,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Both freeze ImageNet weights for the feature extractor.
        },
        3_classification_loss: {
            original_e2: sparse_categorical_crossentropy,
            exp02: sparse_categorical_crossentropy,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: EXP-02 correctly avoided focal loss and used standard sparse categorical cross-entropy.
        },
        4_supervised_contrastive_loss: {
            original_e2: Custom Supervised Contrastive Loss with temperature=0.07, L2 normalization, masked dot-product logits, numerical stability subtraction.,
            exp02: Custom Supervised Contrastive Loss with temperature=0.07, L2 normalization, masked dot-product logits, numerical stability subtraction.,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Mathematical implementations are functionally and numerically identical.
        },
        5_loss_weights: {
            original_e2: [1.0, 0.1] (clf: 1.0, emb: 0.1, LAMBDA_SUPCON=0.1),
            exp02: [1.0, 0.1] (clf: 1.0, emb: 0.1, LAMBDA_SUPCON=0.1),
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Identical loss balancing.
        },
        6_class_weighting_and_sample_weight: {
            original_e2: Inverse class-frequency weights calculated per training fold: cw = {i: total_train / (4 * train_counts[i])}. Passed to model.fit() via sample_weight=[sw, np.ones_like(sw)] where sw = [cw[y] for y in y_train].,
            exp02: Inverse class-frequency weights calculated per training fold: alpha_weights = [n_train / (4 * train_counts[i])]. Passed to model.fit() via sample_weight=[sw, np.ones_like(sw)] where sw = [alpha_weights[y] for y in y_train].,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: CRITICAL VERIFICATION: In EXP-02, the variable was named alpha_weights (syntactically inherited from EXP-01), but its mathematical value is N / (4 * N_c), which exactly matches cw in original E2. Both scripts applied identical per-fold inverse-frequency sample weights.
        },
        7_optimizer_and_weight_decay: {
            original_e2: AdamW(learning_rate=1e-3, weight_decay=1e-4) if available, else Adam(1e-3),
            exp02: AdamW(learning_rate=1e-3, weight_decay=1e-4) if available, else Adam(1e-3),
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Identical optimizer configuration.
        },
        8_learning_rate_and_schedule: {
            original_e2: Fixed 1e-3 (no learning rate scheduler),
            exp02: Fixed 1e-3 (no learning rate scheduler),
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Identical learning rate.
        },
        9_batch_size_and_epoch_limits: {
            original_e2: BATCH_SIZE = 16, EPOCHS = 150, MIN_EPOCHS = 100,
            exp02: BATCH_SIZE = 16, EPOCHS = 60, MIN_EPOCHS = 15,
            is_different: True,
            intentional: False,
            could_affect_results: True,
            notes: CRITICAL CONFOUNDING FACTOR: In original E2, training ran for at least 100 epochs (up to 150). In EXP-02, MIN_EPOCHS was reduced to 15 and max EPOCHS to 60. As a consequence, folds terminated after only 15 to 27 epochs, severely under-training compared to the baseline.
        },
        10_early_stopping_and_checkpoint_selection: {
            original_e2: PATIENCE = 15, MIN_EPOCHS = 100. Callback F1Checkpoint tracked validation Macro F1 to save fold_{fold}_best.weights.h5. Early stopping could not trigger until epoch >= 100.,
            exp02: PATIENCE = 12, MIN_EPOCHS = 15. Callback tracked validation Macro F1 to save fold_{fold}_best.weights.h5. Early stopping triggered after 12 epochs without improvement once epoch >= 15.,
            is_different: True,
            intentional: False,
            could_affect_results: True,
            notes: CRITICAL CONFOUNDING FACTOR: In original E2, the model was forced to train through initial instability up to at least epoch 100 before early stopping was allowed. In EXP-02, early stopping triggered prematurely (e.g. Fold 5 stopped at epoch 24 with best epoch 12).
        },
        11_input_dimensions_and_preprocessing: {
            original_e2: transform_c5: composite_alpha_on_white -> crop_mfs100_foreground -> enhance_mfs100_ridge_contrast -> resize_to_target((112, 112)). Output float32 [0, 255]. Input shape (3, 112, 112, 3).,
            exp02: transform_c5: composite_alpha_on_white -> crop_mfs100_foreground -> enhance_mfs100_ridge_contrast -> resize_to_target((112, 112)). Output float32 [0, 255]. Input shape (3, 112, 112, 3).,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Preprocessing pipeline and image dimensions are 100% identical.
        },
        12_data_augmentation: {
            original_e2: RandomRotation(0.027) [~9.7 deg], RandomTranslation(0.1, 0.1) [~11.2 px], RandomZoom(0.1) [10% scaling], RandomContrast(0.1) [10% contrast]. fill_mode='nearest'.,
            exp02: RandomRotation((-0.02, 0.02)) [~7.2 deg], RandomTranslation((-0.04, 0.04), (-0.04, 0.04)) [~4.5 px], RandomContrast(0.05) [5% contrast], RandomBrightness(0.05) [5% brightness]. RandomZoom EXCLUDED (0.0). fill_mode='nearest'.,
            is_different: True,
            intentional: True,
            could_affect_results: True,
            notes: CRITICAL INSIGHT: Original E2 was NOT unaugmented; it already contained aggressive data augmentation including 10% zoom, 10% translation, and 10% rotation. EXP-02 tested conservative augmentation without zoom, but did so under truncated training epochs.
        },
        13_random_seeds: {
            original_e2: SEED = 42 for tf.random.set_seed and np.random.seed. No seed passed into augmentation layers.,
            exp02: SEED = 42 for tf.random.set_seed and np.random.seed. Explicit seed=42 passed into each augmentation layer.,
            is_different: True,
            intentional: True,
            could_affect_results: False,
            notes: Passing explicit seeds in Keras augmentation layers increases reproducibility but does not fundamentally alter the learning paradigm.
        },
        14_fold_assignment_and_grouping: {
            original_e2: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on 80 development participants with 3 scans grouped per participant.,
            exp02: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on 80 development participants with 3 scans grouped per participant.,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Fold splits and participant groupings are exactly identical.
        },
        15_prediction_aggregation: {
            original_e2: For development OOF: Each fold checkpoint predicts on its 16 held-out participants. Exactly 80 participant predictions (concatenated into all_models_participant_probs.npz).,
            exp02: For development OOF: Each fold checkpoint predicts on its 16 held-out participants. Exactly 80 participant predictions.,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Prediction aggregation is identical and fair.
        },
        16_class_order_mapping: {
            original_e2: Native output order: [0: A+, 1: B+, 2: AB+, 3: O+]. Remapped to Canonical order [0: A+, 1: AB+, 2: B+, 3: O+] via [p[0], p[2], p[1], p[3]].,
            exp02: Native output order: [0: A+, 1: B+, 2: AB+, 3: O+]. Remapped to Canonical order [0: A+, 1: AB+, 2: B+, 3: O+] via [p[0], p[2], p[1], p[3]].,
            is_different: False,
            intentional: True,
            could_affect_results: False,
            notes: Class order remapping is 100% identical.
        }
    },
    summary_verdict: {
        is_confounded: True,
        confounding_factors: [
            Training duration drastically reduced (EPOCHS=60 vs 150, MIN_EPOCHS=15 vs 100),
            Early stopping patience reduced (PATIENCE=12 vs 15) and triggered after ~15-27 epochs instead of running >= 100 epochs,
            Baseline E2 was not unaugmented (it had aggressive augmentation with 10% zoom and translation, whereas EXP-02 tested conservative augmentation without zoom)
        ],
        conclusion_trustworthiness: UNTRUSTWORTHY: The performance drop in EXP-02 cannot be attributed solely to conservative augmentation because the model was trained for less than one-fourth the number of epochs of the baseline E2 model.
    }
}

out_p = Path('reports/accuracy_improvement/exp03_preparation/configuration_diff.json')
with open(out_p, 'w', encoding='utf-8') as f:
    json.dump(diff, f, indent=2)

print('Saved configuration_diff.json successfully.')
