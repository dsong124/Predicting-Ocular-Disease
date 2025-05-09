import os
import ast
import pandas as pd
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('cleaned_df.csv')
df['vec'] = df['target'].apply(ast.literal_eval)
df['class_folder'] = df['vec'].apply(lambda v: f"Class_{v.index(1) + 1}")
df['image_path'] = df.apply(
    lambda r: os.path.join('Organized_Images', r['class_folder'], r['filename']),
    axis=1
)

sample_df = df if len(df) <= 500 else df.sample(500, random_state=42).reset_index(drop=True)

def extract_color_histogram(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

features = []
valid_idx = []
for i, path in enumerate(sample_df['image_path']):
    img = cv2.imread(path)
    if img is None:
        continue
    img = cv2.resize(img, (224, 224))
    features.append(extract_color_histogram(img))
    valid_idx.append(i)

features = np.array(features)
sample_df = sample_df.loc[valid_idx].reset_index(drop=True)

# KMeans clustering
n_clusters = sample_df['class_folder'].nunique()
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
km_labels = kmeans.fit_predict(features)

# PCA reduction
coords = PCA(n_components=2, random_state=42).fit_transform(features)

# Plot PCA of KMeans clusters
plt.figure(figsize=(7,6))
plt.scatter(coords[:, 0], coords[:, 1], c=km_labels, cmap='tab10', s=30)
plt.title('PCA of Sampled Images: KMeans Clusters')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()
