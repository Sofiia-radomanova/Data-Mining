import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
import scipy.cluster.hierarchy as sch


data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

metrics = ['euclidean']
linkage_methods = ['ward', 'complete', 'average', 'single']

plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics):
    for j, method in enumerate(linkage_methods):
        plt.subplot(len(metrics), len(linkage_methods), i * len(linkage_methods) + j + 1)
        linkage_matrix = linkage(df, method=method, metric=metric if method!= 'ward' else 'euclidean')
        dendrogram(linkage_matrix, no_labels=False)
        plt.title(f"{method.capitalize()} - {metric.capitalize()}")

plt.tight_layout()
plt.show()