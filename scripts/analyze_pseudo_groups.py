import pandas as pd
import numpy as np
import networkx as nx
import cv2
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
import json

# Load dataset manifest
df = pd.read_csv("outputs/dataset_manifest.csv")
test_df = df[df["split"] == "test"].copy()

print(f"Total test images: {len(test_df)}")

# Generate groups using ORB
orb = cv2.ORB_create(nfeatures=500)
def extract_orb(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

features = {}
for idx, row in test_df.iterrows():
    features[row["image_path"]] = extract_orb(row["image_path"])

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
G = nx.Graph()
for path in test_df["image_path"]:
    G.add_node(path)

paths = list(test_df["image_path"])
for i in range(len(paths)):
    for j in range(i+1, len(paths)):
        _, des1 = features[paths[i]]
        _, des2 = features[paths[j]]
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            if len(matches) > 50:
                G.add_edge(paths[i], paths[j])

components = list(nx.connected_components(G))
group_map = {}
for g_id, comp in enumerate(components):
    for node in comp:
        group_map[node] = g_id

test_df["pseudo_group"] = test_df["image_path"].map(group_map)
group_counts = test_df["pseudo_group"].value_counts()
print("\n--- PSEUDO-GROUP STATS ---")
print(f"Number of groups: {len(components)}")
print(f"Min group size: {group_counts.min()}")
print(f"Median group size: {group_counts.median()}")
print(f"Max group size: {group_counts.max()}")
print(f"Singleton groups: {(group_counts == 1).sum()}")

# Baseline
majority_class = test_df["class_name"].mode()[0]
baseline_acc = (test_df["class_name"] == majority_class).mean()
print(f"\nMajority baseline: {baseline_acc*100:.2f}%")

# Evaluating links
# For ground truth duplicates, let's assume images that have SSIM > 0.95 or identical hashes are duplicates
# We'll just look at "false links": how often does ORB link different classes?
cross_class_edges = 0
total_edges = G.number_of_edges()
for u, v in G.edges():
    cls_u = test_df[test_df["image_path"] == u]["class_name"].values[0]
    cls_v = test_df[test_df["image_path"] == v]["class_name"].values[0]
    if cls_u != cls_v:
        cross_class_edges += 1
print("\n--- LINKING STATS ---")
print(f"Total ORB links (edges): {total_edges}")
print(f"False links (different blood groups): {cross_class_edges}")
print(f"False link rate: {cross_class_edges/max(1, total_edges)*100:.2f}%")

# Batch Hypothesis: Metadata and Background pixels
print("\n--- BATCH HYPOTHESIS ---")
import struct

def extract_metadata(path):
    size = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [size, mtime, 0, 0, 0]
    
    # Background stats: sample the 4 corners (10x10)
    h, w = img.shape
    c1 = img[0:10, 0:10]
    c2 = img[0:10, w-10:w]
    c3 = img[h-10:h, 0:10]
    c4 = img[h-10:h, w-10:w]
    corners = np.concatenate([c1.flatten(), c2.flatten(), c3.flatten(), c4.flatten()])
    mean_bg = corners.mean()
    std_bg = corners.std()
    
    return [size, mtime, mean_bg, std_bg, img.mean()]

X_meta = []
for p in test_df["image_path"]:
    X_meta.append(extract_metadata(p))
X_meta = np.array(X_meta)
y = test_df["class_name"].values
groups = test_df["pseudo_group"].values

def eval_model(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    model = LogisticRegression(max_iter=1000)
    accs = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], preds))
    return np.mean(accs)

meta_acc = eval_model(X_meta, y, groups)
print(f"Metadata+Background Model Accuracy: {meta_acc*100:.2f}%")

# Bootstrap CI for 72.53%
np.random.seed(42)
group_ids = np.unique(groups)
boot_accs = []
# Assuming the model outputs 72.5% we just want to bootstrap the CI on the test set for a 72.5% accuracy
# But we need actual predictions. We will simulate them or just run the GroupKFold once to get predictions
preds_all = np.zeros(len(y), dtype=y.dtype)
gkf = GroupKFold(n_splits=5)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

model = load_model("models/densenet121_blood_group.keras", compile=False)
from tensorflow.keras.models import Model
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("global_avg_pool").output)

X_densenet = []
for p in test_df["image_path"]:
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    X_densenet.append(img)
X_densenet = np.array(X_densenet)
features_dn = feat_extractor.predict(X_densenet, batch_size=32, verbose=0)

scaler = StandardScaler()
for train_idx, val_idx in gkf.split(features_dn, y, groups):
    X_tr = scaler.fit_transform(features_dn[train_idx])
    X_val = scaler.transform(features_dn[val_idx])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y[train_idx])
    preds_all[val_idx] = clf.predict(X_val)

actual_acc = accuracy_score(y, preds_all)
print(f"DenseNet Backbone Accuracy: {actual_acc*100:.2f}%")

for _ in range(1000):
    sample_g = np.random.choice(group_ids, size=len(group_ids), replace=True)
    sample_idx = np.concatenate([np.where(groups == g)[0] for g in sample_g])
    boot_acc = accuracy_score(y[sample_idx], preds_all[sample_idx])
    boot_accs.append(boot_acc)

ci_lower = np.percentile(boot_accs, 2.5)
ci_upper = np.percentile(boot_accs, 97.5)
print(f"Group Bootstrap 95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")

