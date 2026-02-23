import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Завантаження даних
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
data.columns = ['Passengers']

print("Перші 5 рядків даних:")
print(data.head())
print(f"\nРозмір датасету: {data.shape}")
print(f"Діапазон дат: {data.index.min()} до {data.index.max()}")

# 1. Візуалізація часового ряду
plt.figure(figsize=(15, 10))

# Основний графік
plt.subplot(3, 2, 1)
plt.plot(data.index, data['Passengers'], linewidth=2)
plt.title('Часовий ряд: Кількість авіапасажирів (1949-1960)')
plt.xlabel('Рік')
plt.ylabel('Кількість пасажирів')
plt.grid(True)

# Декомпозиція ряду
decomposition = seasonal_decompose(data['Passengers'], model='multiplicative', period=12)

# Тренд
plt.subplot(3, 2, 2)
plt.plot(decomposition.trend)
plt.title('Трендова складова')
plt.grid(True)

# Сезонність
plt.subplot(3, 2, 3)
plt.plot(decomposition.seasonal)
plt.title('Сезонна складова')
plt.grid(True)

# Залишкова складова
plt.subplot(3, 2, 4)
plt.plot(decomposition.resid)
plt.title('Залишкова складова')
plt.grid(True)

plt.tight_layout()
plt.show()

# Аналіз структури ряду
print("=== АНАЛІЗ СТРУКТУРИ ЧАСОВОГО РЯДУ ===")
print("1. Наявність тренду: ЗРОСТАЮЧИЙ (чітко видно зростання кількості пасажирів)")
print("2. Сезонність: ВИСОКА (періодичні коливання з періодом 12 місяців)")
print("3. Тип функції тренду: ЛІНІЙНО-ЗРОСТАЮЧИЙ")
print("4. Монотонність: НЕМОНОТОННИЙ (наявні сезонні коливання)")

# 2. Розділення на навчальну та тестову вибірки
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]

print(f"\nРозмір навчальної вибірки: {len(train)}")
print(f"Розмір тестової вибірки: {len(test)}")

# 3. Прогнозування ARIMA
print("\n=== ПРОГНОЗУВАННЯ ARIMA ===")

# Підбір параметрів ARIMA (p,d,q)
model_arima = ARIMA(train['Passengers'], order=(2,1,2), seasonal_order=(1,1,1,12))
model_arima_fit = model_arima.fit()

# Прогноз на тестовій вибірці
forecast_arima = model_arima_fit.forecast(steps=len(test))
forecast_arima = pd.Series(forecast_arima, index=test.index)

# Обчислення метрик якості
mae_arima = mean_absolute_error(test['Passengers'], forecast_arima)
mse_arima = mean_squared_error(test['Passengers'], forecast_arima)
rmse_arima = np.sqrt(mse_arima)

print(f"MAE ARIMA: {mae_arima:.2f}")
print(f"RMSE ARIMA: {rmse_arima:.2f}")
print(f"AIC ARIMA: {model_arima_fit.aic:.2f}")

# 4. Прогнозування Holt-Winters
print("\n=== ПРОГНОЗУВАННЯ HOLT-WINTERS ===")

model_hw = ExponentialSmoothing(train['Passengers'],
                               trend='add',
                               seasonal='mul',
                               seasonal_periods=12)
model_hw_fit = model_hw.fit()

# Прогноз на тестовій вибірці
forecast_hw = model_hw_fit.forecast(len(test))

# Обчислення метрик якості
mae_hw = mean_absolute_error(test['Passengers'], forecast_hw)
mse_hw = mean_squared_error(test['Passengers'], forecast_hw)
rmse_hw = np.sqrt(mse_hw)

print(f"MAE Holt-Winters: {mae_hw:.2f}")
print(f"RMSE Holt-Winters: {rmse_hw:.2f}")

# 5. Порівняльна візуалізація результатів
plt.figure(figsize=(15, 8))

# Графік історичних даних та прогнозів
plt.subplot(2, 1, 1)
plt.plot(train.index, train['Passengers'], label='Навчальні дані', color='blue')
plt.plot(test.index, test['Passengers'], label='Фактичні значення', color='green')
plt.plot(forecast_arima.index, forecast_arima, label='Прогноз ARIMA', color='red', linestyle='--')
plt.plot(forecast_hw.index, forecast_hw, label='Прогноз Holt-Winters', color='orange', linestyle='--')
plt.title('Порівняння методів прогнозування')
plt.xlabel('Дата')
plt.ylabel('Кількість пасажирів')
plt.legend()
plt.grid(True)

# Графік помилок прогнозу
plt.subplot(2, 1, 2)
errors = pd.DataFrame({
    'ARIMA': test['Passengers'] - forecast_arima,
    'Holt-Winters': test['Passengers'] - forecast_hw
})
plt.plot(errors.index, errors['ARIMA'], label='Помилка ARIMA', color='red')
plt.plot(errors.index, errors['Holt-Winters'], label='Помилка Holt-Winters', color='orange')
plt.title('Помилки прогнозування')
plt.xlabel('Дата')
plt.ylabel('Похибка')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Порівняння методів
results_comparison = pd.DataFrame({
    'Метод': ['ARIMA', 'Holt-Winters'],
    'MAE': [mae_arima, mae_hw],
    'RMSE': [rmse_arima, rmse_hw]
})

print("\n=== ПОРІВНЯННЯ МЕТОДІВ ПРОГНОЗУВАННЯ ===")
print(results_comparison)

# Прогноз на майбутній період
future_forecast_arima = model_arima_fit.forecast(steps=12)
future_forecast_hw = model_hw_fit.forecast(12)

print(f"\nПрогноз ARIMA на наступні 12 місяців:")
print(future_forecast_arima.tail(12))