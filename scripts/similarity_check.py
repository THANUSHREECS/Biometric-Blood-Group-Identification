import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
import matplotlib.pyplot as plt

print("Task 3: Similarity Check")
os.makedirs("results", exist_ok=True)

manifest = pd.read_csv("outputs/dataset_manifest.csv")
manifest = manifest[manifest["class_name"].isin(["A+", "B+", "AB+", "O+"])].copy()
manifest["cluster_id"] = manifest["image_path"].str.extract(r'cluster_(\d+)_').astype(float)

train_df = manifest[manifest["split"] == "train"].sample(200, random_state=42).reset_index(drop=True)
test_df = manifest[manifest["split"] == "test"].sample(50, random_state=42).reset_index(drop=True)

def extract_features(df):
    base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", pooling="avg")
    def load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_bmp(img, channels=3)
        img = tf.image.resize(img, (112, 112))
        return preprocess_input(img)
    dataset = tf.data.Dataset.from_tensor_slices(df["image_path"].values).map(load_img).batch(32)
    return base_model.predict(dataset, verbose=0)

print("Extracting test embeddings...")
test_emb = extract_features(test_df)
print("Extracting train embeddings...")
train_emb = extract_features(train_df)

from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(test_emb, train_emb)

same_cluster_sims = []
diff_cluster_sims = []
nearest_is_same = 0

for i in range(len(test_df)):
    test_cluster = test_df.iloc[i]["cluster_id"]
    sims = sim_matrix[i]
    
    nearest_idx = np.argmax(sims)
    if train_df.iloc[nearest_idx]["cluster_id"] == test_cluster:
        nearest_is_same += 1
        
    for j in range(len(train_df)):
        if train_df.iloc[j]["cluster_id"] == test_cluster:
            same_cluster_sims.append(sims[j])
        else:
            diff_cluster_sims.append(sims[j])

plt.hist(diff_cluster_sims, bins=50, alpha=0.5, label='Different Cluster', density=True)
plt.hist(same_cluster_sims, bins=50, alpha=0.5, label='Same Cluster', density=True)
plt.legend()
plt.title('Embedding Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.savefig('results/similarity_distributions.png')
plt.close()

print(f"Nearest train neighbour is in the same cluster: {nearest_is_same}/{len(test_df)} ({nearest_is_same/len(test_df)*100:.2f}%)")
