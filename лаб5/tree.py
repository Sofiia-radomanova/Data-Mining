import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv('Spotify-2000.csv')
pd.set_option('display.max_columns', None)

#print(data.head())
#print(data.info())
#print(data.isnull().sum())

encoder = LabelEncoder()
data['Top Genre'] = encoder.fit_transform(data['Top Genre'])
data['Length (Duration)'] = data['Length (Duration)'].str.replace(',', '').astype(int)
X = data.drop(columns=['Title', 'Artist','Top Genre'], axis=1)
Y = data['Top Genre']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

full_tree = DecisionTreeClassifier(random_state=2)
full_tree.fit(X_train, y_train)

Y_pred_full=full_tree.predict(X_test)
print("Результати дерева без обмежень:")
print(accuracy_score(y_test, Y_pred_full))

pre_pruned_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10,min_samples_leaf=5,  random_state=2)
pre_pruned_tree.fit(X_train, y_train)
Y_pred_pruned=pre_pruned_tree.predict(X_test)
print("Результати дерева з обрізанням:")
print(accuracy_score(y_test, Y_pred_pruned))

param_grid = {
'max_depth': [3, 5, 10, None],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 5, 10]
}


grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
#cv- крос-валідація, дані діляться на 5 частин, 4 - для навчання, 1 - тест, повторюється 5 разів ( кожна частина один раз стає тестовою)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_ #словник із найкращими знайденими значеннями
best_tree = grid_search.best_estimator_ #уже навчена модель з цими параметрами
y_pred_gridsearch = best_tree.predict(X_test)

gridsearch_acc = accuracy_score(y_test, y_pred_gridsearch)
print(f"Результати дерева з автоматичним підбором параметрів: {gridsearch_acc:.4f} з параметрами: {best_params}")


plt.figure(figsize=(18, 8))  # розмір фігури
plot_tree(full_tree, filled=True)
plt.title("Unpruned Tree")
plt.show()

plt.figure(figsize=(18, 8))  # розмір фігури
plot_tree(pre_pruned_tree, filled=True)
plt.title("Pre-Pruned Tree")
plt.show()

plt.figure(figsize=(18, 8))  # розмір фігури
plot_tree(best_tree, filled=True)
plt.title("Grid search Tree")

plt.show()