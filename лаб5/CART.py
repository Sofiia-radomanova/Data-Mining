import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Завантаження датасету
data = pd.read_csv('Student_Performance.csv')
pd.set_option('display.max_columns', None)

# Перевірка даних
print(data.head())
print(data.info())

# Заповнення пропусків лише для числових колонок
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Для текстових колонок можна заповнити найчастішим значенням
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Перевірка після заповнення
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

# Кодування категоріальних змінних (якщо є)
if 'gender' in data.columns:
    data['gender'] = data['gender'].map({'M': 0, 'F': 1})

# Вибір ознак та цільової змінної
X = data.drop(columns=['Performance Index'])  # заміни на назву колонки з цільовою змінною
y = data['Performance Index']

# Розподіл на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та навчання дерева регресії
regressor = DecisionTreeRegressor(max_depth=4, random_state=42)
regressor.fit(X_train, y_train)

# Прогнозування
y_pred = regressor.predict(X_test)

# Оцінка моделі
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Візуалізація важливості ознак
plt.figure(figsize=(8, 5))
plt.barh(X.columns, regressor.feature_importances_)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Decision Tree Regressor")
plt.show()

# Візуалізація структури дерева
fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(regressor, filled=True, feature_names=X.columns, rounded=True, fontsize=10, ax=ax)
plt.title("Decision Tree Structure")
plt.show()

# Порівняння фактичних та прогнозованих значень
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', edgecolors='black', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
