"""
Round 11 Q3: Which pipeline reproduces test_predictions.csv?
Wrote by: scripts/evaluate_abo_positive_4class.py
Model used: models/abo_positive_4class/best_model.keras (EfficientNetB0, single-scan, 224x224)
Class order in that script: A+=0, AB+=1, B+=2, O+=3  (DIFFERS from src/config.py)
Preprocessing: composite_alpha_on_white -> resize_to_target(224,224) -> NO preprocess_input applied
"""
import sys; sys.path.insert(0,'.')
import numpy as np
import pandas as pd
from PIL import Image
from src.preprocessing import composite_alpha_on_white, resize_to_target, crop_mfs100_foreground, apply_fingerprint_enhancement
import tensorflow as tf

# Load the model that wrote the CSV
model_224 = tf.keras.models.load_model('models/abo_positive_4class/best_model.keras', compile=False)
print('best_model.keras input shape:', model_224.input_shape)

df = pd.read_csv('outputs/abo_positive_4class/test_predictions.csv')
print(f'CSV rows: {len(df)}')

def fix_path(p):
    import os
    if os.path.exists(p): return p
    p2 = p.replace('C:/Users/thanu/OneDrive/Desktop/major project',
                   r'C:\Users\thanu\OneDrive - Don Bosco Institute of Technology\major project')
    if os.path.exists(p2): return p2
    return None

# Variants to test (each returns a 224x224 or 112x112 preprocessed array)
def var_composite_224_raw(p):
    """evaluate_abo_positive_4class.py style: composite -> resize 224 -> raw float32, no preprocess_input"""
    with Image.open(p) as img:
        rgb = composite_alpha_on_white(img)
        resized = resize_to_target(rgb, (224, 224))
    return np.array(resized, dtype=np.float32)

def var_composite_224_effnet(p):
    """composite -> resize 224 -> efficientnet preprocess_input"""
    with Image.open(p) as img:
        rgb = composite_alpha_on_white(img)
        resized = resize_to_target(rgb, (224, 224))
    arr = np.array(resized, dtype=np.float32)
    return tf.keras.applications.efficientnet.preprocess_input(arr)

def var_composite_112_densenet(p):
    """C5 style: composite -> resize 112 -> densenet preprocess_input (no crop, no CLAHE)"""
    with Image.open(p) as img:
        rgb = composite_alpha_on_white(img)
        resized = resize_to_target(rgb, (112, 112))
    arr = np.array(resized, dtype=np.float32)
    return tf.keras.applications.densenet.preprocess_input(arr)

# Load E2 model for 112x112 variants
from src.predictor import BloodGroupPredictor
e2 = BloodGroupPredictor()

variants = {
    "V1: composite+resize224+raw [best_model.keras]": (var_composite_224_raw, model_224, False),
    "V2: composite+resize224+effnet_preprocess [best_model.keras]": (var_composite_224_effnet, model_224, False),
}

results = {}
for vname, (prep_fn, model, is_e2) in variants.items():
    correct = 0
    max_diff = 0.0
    diffs = []
    n = 0
    for _, row in df.iterrows():
        p = fix_path(row['image_path'])
        if p is None:
            continue
        try:
            arr = prep_fn(p)
        except Exception as e:
            continue
        X = np.expand_dims(arr, 0)
        probs = model.predict(X, verbose=0)[0]
        # CSV prob order: prob_A+[0], prob_AB+[1], prob_B+[2], prob_O+[3]
        saved = [row['prob_A+'], row['prob_AB+'], row['prob_B+'], row['prob_O+']]
        for j in range(4):
            d = abs(probs[j] - saved[j])
            diffs.append(d)
            if d > max_diff: max_diff = d
        pred = int(np.argmax(probs))
        if pred == row['predicted_id']:
            correct += 1
        n += 1
        if n % 50 == 0:
            print(f'{vname}: n={n}', flush=True)

    results[vname] = {
        'n': n,
        'max_abs_diff': max_diff,
        'mean_abs_diff': float(np.mean(diffs)) if diffs else None,
        'identical_top1': correct,
        'pct_identical': (correct/n*100) if n > 0 else 0
    }
    print(f'\n=== {vname} ===')
    print(f'n={n}, max_diff={max_diff:.6f}, mean_diff={np.mean(diffs):.6f}, identical_top1={correct}/{n} ({correct/n*100:.1f}%)')

import json
with open('outputs/round11_reproduce_test.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to outputs/round11_reproduce_test.json')
