import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)

# 2. Підготовка транзакцій
# Датасет має колонки item1, item2, …, item20 (або подібні)
print(df.head(10))

# Перетворимо дані в список транзакцій
transactions = []
for i, row in df.iterrows():
    # прибрати NaN і взяти тільки товари
    items = row.dropna().tolist()
    transactions.append(items)

# Трансформуємо транзакції у формат для mlxtend
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# 3. Застосування FP-Growth
# Виставимо мінімальні значення підтримки та достовірності
min_support = 0.02  # наприклад 2%
frequent_itemsets_fp = fpgrowth(df_trans, min_support=min_support, use_colnames=True)



# Побудова правил за допомогою FP-Growth
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.3)

# припускаю, що frequent_itemsets — результат fpgrowth(..., use_colnames=True)
frequent_itemsets_fp['length'] = frequent_itemsets_fp['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets_fp.groupby('length').size())
print(frequent_itemsets_fp.sort_values('support', ascending=False).head(30))

print("FP-Growth frequent itemsets:")
print(frequent_itemsets_fp.sort_values(by="support", ascending=False).head(20))

pd.set_option('display.max_columns', None)

# --- FP-Growth rules (фільтрація колонок) ---
print("FP-Growth rules:")

rules_fp_filtered = rules_fp[['antecedents', 'consequents', 'support', 'confidence']]
print(
    rules_fp_filtered
        .sort_values(by=["confidence", "support"], ascending=False)
        .head(20)
)

# --- Apriori rules ---
frequent_itemsets_ap = apriori(df_trans, min_support=min_support, use_colnames=True)
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.3)

print("Apriori rules:")

rules_ap_filtered = rules_ap[['antecedents', 'consequents', 'support', 'confidence']]
print(
    rules_ap_filtered
        .sort_values(by=["confidence", "support"], ascending=False)
        .head(20)
)

# 5. Аналіз при різних параметрах
# Наприклад, інший support і confidence

cols = ["antecedents", "consequents", "support", "confidence"]

for supp in [0.01, 0.03]:
    for conf in [0.4, 0.5]:
        fi = fpgrowth(df_trans, min_support=supp, use_colnames=True)
        rs = association_rules(fi, metric="confidence", min_threshold=conf)

        print(f"\n=== При support={supp}, confidence={conf} ===")
        print("Кількість частих itemset:", fi.shape[0])
        print("Кількість правил:", rs.shape[0])

        # Безпечний вибір колонок, бо раптом їх немає
        available_cols = [c for c in cols if c in rs.columns]

        print(
            rs.sort_values(by=["lift", "confidence"], ascending=False)[available_cols].head(10)
        )


import networkx as nx
import matplotlib.pyplot as plt

def plot_rules_graph(rules):
    G = nx.DiGraph()  # Направлений граф

    # Додаємо ребра: antecedent → consequent
    for _, row in rules.iterrows():
        for a in row['antecedents']:
            for c in row['consequents']:
                G.add_edge(a, c, weight=row['confidence'])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)

    # Малюємо вузли
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue")

    # Малюємо стрілки
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='->',
        arrowsize=20,
        width=[G[u][v]['weight'] * 4 for u, v in G.edges()]
    )

    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Graph of Association Rules (FP-Growth)")
    plt.axis('off')
    plt.show()

# Виклик:
plot_rules_graph(rules_fp_filtered)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def support_heatmap(frequent_itemsets):
    pairs = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)]

    matrix = {}
    for _, row in pairs.iterrows():
        a, b = list(row['itemsets'])
        matrix.setdefault(a, {})[b] = row['support']

    df_matrix = pd.DataFrame(matrix).fillna(0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_matrix, annot=True, cmap="crest")
    plt.title("Support Matrix for Frequent Itemsets (Pairs)")
    plt.show()


# Виклик:
support_heatmap(frequent_itemsets_fp)
