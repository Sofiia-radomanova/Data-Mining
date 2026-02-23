import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def run_olap_analysis():
    print(" Створення OLAP-куба на Python (Pandas) "
          "")

    # 1. Емуляція завантаження даних (або pd.read_csv('sales.csv'))
    data = {
        'Year': [2022, 2022, 2022, 2023, 2023, 2023],
        'Category': ['Electronics', 'Furniture', 'Electronics', 'Electronics', 'Furniture', 'Clothing'],
        'Region': ['Kyiv', 'Kyiv', 'Lviv', 'Kyiv', 'Lviv', 'Kyiv'],
        'Sales': [15000, 3000, 12000, 18000, 4000, 2500]
    }
    df = pd.DataFrame(data)

    print("\n1. Вихідні дані:")
    print(df)

    # 2. Створення КУБА (Pivot Table)
    # Ми хочемо бачити суму продажів в розрізі Років та Категорій
    olap_cube = pd.pivot_table(
        df,
        values='Sales',
        index=['Year', 'Category'],
        columns=['Region'],
        aggfunc='sum',
        fill_value=0  # Замінити пусті значення на 0
    )

    print("\n2. Побудований OLAP-куб (Агрегація: Сума продажів):")
    print(olap_cube)

    # 3. Операція SLICING (Зріз) - Дивимось тільки 2023 рік
    print("\n3. Операція Slicing (Тільки 2023 рік):")
    try:
        slice_2023 = olap_cube.loc[2023]
        print(slice_2023)
    except KeyError:
        print("Дані за 2023 рік відсутні")

    # 4. Операція DICING (Вирізка) - Тільки Електроніка у Києві
    print("\n4. Операція Dicing (Тільки Electronics у Region='Kyiv'):")
    # Фільтруємо вихідний датафрейм, бо це гнучкіше
    dicing = df[(df['Category'] == 'Electronics') & (df['Region'] == 'Kyiv')]
    print(dicing.groupby('Year')['Sales'].sum())

    # Отримуємо унікальні списки для осей (відсортовані)
    years = sorted(df['Year'].unique())
    categories = sorted(df['Category'].unique())

    # Створюємо словники для перекладу "Текст -> Координата"
    # Наприклад: {2022: 0, 2023: 1} та {'Clothing': 0, 'Electronics': 1...}
    year_map = {val: i for i, val in enumerate(years)}
    cat_map = {val: i for i, val in enumerate(categories)}

    # 2. Налаштування 3D полотна
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Підготовка координат для кожного стовпчика
    x_pos = [year_map[y] for y in df['Year']]  # Координати X (0, 1...)
    y_pos = [cat_map[c] for c in df['Category']]  # Координати Y (0, 1, 2...)
    z_pos = np.zeros(len(df))  # Стоять на "землі" (Z=0)

    dx = 0.5  # Ширина стовпчика
    dy = 0.5  # Глибина стовпчика
    dz = df['Sales'].values  # Висота стовпчика (наші продажі)

    # Малюємо стовпчики
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color='#00ceaa', shade=True)

    # --- ВИПРАВЛЕННЯ ОСЕЙ ---

    # 1. Кажемо, ДЕ ставити мітки (по центру стовпчиків: координата + половина ширини)
    # np.arange(len(years)) створить масив [0, 1]
    ax.set_xticks(np.arange(len(years)) + dx / 2)
    ax.set_yticks(np.arange(len(categories)) + dy / 2)

    # 2. Кажемо, ЩО писати (наші реальні назви)
    ax.set_xticklabels(years)
    ax.set_yticklabels(categories)

    # Підписи осей
    ax.set_xlabel('Рік')
    ax.set_ylabel('Категорія')
    ax.set_zlabel('Сума продажів')
    ax.set_title('3D OLAP Куб Продажів')

    plt.show()


if __name__ == "__main__":
    run_olap_analysis()



