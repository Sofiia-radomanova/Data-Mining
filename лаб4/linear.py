
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd


df = pd.read_csv('bank+marketing/bank/bank-full.csv', sep=';')
pd.set_option('display.max_columns', None)
print(df.head())

print(df.info())
print(df.isnull().sum())

#sns.countplot(x='y', data=df)
#plt.title('Розподіл класів ')
#plt.show()

X = df.drop('y',  axis=1)
Y = df['y']

# Знаходимо категоріальні колонки
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
# One-hot кодування категоріальних ознак
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


