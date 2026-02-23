
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


column_names = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]


df = pd.read_csv('mushroom/agaricus-lepiota.data',names=column_names)

pd.set_option('display.max_columns', None)
print(df.head())

print(df.info())
print(df.isnull().sum())

sns.countplot(x='class', data=df)
plt.title('Розподіл класів ')
plt.show()

X = df.drop("class", axis=1)   # усі ознаки
y = df["class"]                # цільова змінна (їстівний / отруйний)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# e → 0
# p → 1
print(label_encoder.classes_)  # ['e' 'p']

X_encoded = pd.get_dummies(X, drop_first=True)

#масштабування
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)