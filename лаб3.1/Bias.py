import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder

# Завантаження даних
data = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv/Video_Games_Sales_as_at_22_Dec_2016.csv")

print(data.head())
data = data.dropna()
data = data.drop_duplicates()

# Перетворюємо текстові колонки у числа
df = data.copy()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Формування ознак і цільової змінної
X = df.drop(columns=['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'])
threshold = df['Global_Sales'].quantile(0.75)
y = (df['Global_Sales'] >= threshold).astype(int)

# Вибір двох найкращих ознак
selector = SelectKBest(f_classif, k=2)
X_new = selector.fit_transform(X, y)
selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
print(f"Selected features: {selected_features}")

# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Наївний Байєсівський класифікатор
nb = GaussianNB()
nb.fit(X_train, y_train)

# Візуалізація області прийняття рішень
plt.figure(figsize=(8,6))
plot_decision_regions(X_test, y_test.to_numpy(), nb, legend=2)
plt.title("Naive Bayes Classifier (GaussianNB)")
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.show()
