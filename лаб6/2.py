import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder

# Завантаження датасету
df = pd.read_csv('Spotify-2000.csv')
pd.set_option('display.max_columns', None)

# Кодуємо жанр у числа
encoder = LabelEncoder()
df['Top Genre'] = encoder.fit_transform(df['Top Genre'])

# Перетворюємо тривалість у число (видаляємо коми)
df['Length (Duration)'] = df['Length (Duration)'].str.replace(',', '').astype(int)

#Вибираємо тільки числові колонки
numeric_df = df.select_dtypes(include=[np.number])


metrics = ['euclidean']
linkage_methods = ['ward', 'complete', 'average', 'single']

# Побудова дендрограм
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics):
    for j, method in enumerate(linkage_methods):
        plt.subplot(len(metrics), len(linkage_methods), i * len(linkage_methods) + j + 1)
        linkage_matrix = linkage(numeric_df, method=method, metric=metric if method != 'ward' else 'euclidean')
        dendrogram(linkage_matrix, no_labels=True)
        plt.title(f"{method.capitalize()} - {metric.capitalize()}")

plt.tight_layout()
plt.show()
