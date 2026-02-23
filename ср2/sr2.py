import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs


def run_dbscan_analysis():
    df_raw = pd.read_csv("Mall_Customers.csv")
    df = df_raw[['Annual Income (k$)', 'Spending Score (1-100)']]


    print("Дані підготовлено. Перші 5 рядків:")
    print(df.head())

    # ЕТАП 2: Масштабування даних (Критично важливо для DBSCAN!)
    # Алгоритм базується на відстані, тому дані мають бути в одному масштабі.
    X_scaled = StandardScaler().fit_transform(df)

    # ЕТАП 3: Кластеризація DBSCAN
    # eps=0.3 -> радіус пошуку сусідів
    # min_samples=5 -> мінімум 5 точок, щоб утворити ядро кластера
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)

    # Додаємо результати в таблицю
    df['Cluster'] = clusters

    # ЕТАП 4: Аналіз та Візуалізація
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f"\n--- Результати ---")
    print(f"Кількість знайдених кластерів: {n_clusters}")
    print(f"Кількість точок шуму (outliers): {n_noise}")

    # Візуалізація
    plt.figure(figsize=(10, 6))

    # Малюємо точки, що належать до кластерів
    unique_clusters = set(clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_clusters))]

    for k, col in zip(unique_clusters, colors):
        if k == -1:
            # Чорний колір використовується для шуму
            col = [0, 0, 0, 1]
            label = "Шум / Викиди"
            marker = 'x'  # Хрестики для шуму
        else:
            label = f"Кластер {k}"
            marker = 'o'  # Кружечки для кластерів

        class_member_mask = (clusters == k)
        xy = df[class_member_mask]

        plt.scatter(xy.iloc[:, 0], xy.iloc[:, 1],
                    c=[col], label=label, marker=marker, edgecolor='k', s=50)

    plt.title(f'DBSCAN Кластеризація (Знайдено: {n_clusters} кластери)')
    plt.xlabel('Річний дохід (нормалізований)')
    plt.ylabel('Оцінка витрат (нормалізована)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_dbscan_analysis()