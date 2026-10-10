import pandas as pd
import os

manifest_path = "outputs/dataset_manifest.csv"
if os.path.exists(manifest_path):
    df = pd.read_csv(manifest_path)
    df = df[df["class_name"].isin(["A+", "B+", "AB+", "O+"])]
    print(f"Total 4-class images: {len(df)}")
    
    print("\n--- Split Counts ---")
    print(df["split"].value_counts())
    
    print("\n--- Class Counts ---")
    print(df.groupby(["split", "class_name"]).size())

    df["cluster_id"] = df["image_path"].str.extract(r'cluster_(\d+)_').astype(float)
    
    print("\n--- Cluster Stats ---")
    print(f"Number of clusters: {df['cluster_id'].nunique()}")
    imgs_per_cluster = df.groupby('cluster_id').size()
    print(f"Images per cluster: min={imgs_per_cluster.min()}, median={imgs_per_cluster.median()}, max={imgs_per_cluster.max()}")
    
    clusters_per_bg = df.groupby('class_name')['cluster_id'].nunique()
    print(f"Clusters per blood group:\n{clusters_per_bg}")
    
    splits_per_cluster = df.groupby('cluster_id')['split'].nunique()
    leaked = splits_per_cluster[splits_per_cluster > 1]
    print(f"Clusters appearing in >1 split: {len(leaked)}")
else:
    print("Manifest not found")
