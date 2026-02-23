import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("cat_breeds_clean.csv", sep=';')
pd.set_option('display.max_columns', None)



encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(df)
df['Cluster_Kmeans'] = y_kmeans
print(df.head(10))
#plt.figure(figsize=(12, 8))
#sns.scatterplot(x=df['Body_length'], y=df['Fur_pattern'], hue=df['Cluster_Kmeans'], palette='viridis')
#plt.title('KMeans Clustering')

#plt.xlabel('Body length')
#plt.ylabel('Fur pattern')
#plt.legend(title='Cluster')
#plt.show()

"""
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df.drop(columns=['Cluster_Kmeans']))
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
"""

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(df.drop(columns=[
'Cluster_Kmeans']))
silhouette_avg = silhouette_score(df.drop(columns=[
'Cluster_Kmeans']), labels)
print(f"Average Silhouette Score: {silhouette_avg:.2f}")
silhouette_vals = silhouette_samples(df.drop(columns=[
'Cluster_Kmeans']), labels)
plt.figure(figsize=(8, 5))
y_lower = 10
for i in range(3):
    ith_silhouette_vals = silhouette_vals[labels == i]
    ith_silhouette_vals.sort()
    y_upper = y_lower + len(ith_silhouette_vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_silhouette_vals)
    y_lower = y_upper + 10
plt.axvline(x=silhouette_avg, linestyle='--', color='red')
plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster')
plt.title('Silhouette Analysis')
plt.show()