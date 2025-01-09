import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df_0 = pd.read_csv('Classified Data (2)', index_col=0)

scaler = StandardScaler()
scaler.fit(df_0.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df_0.drop('TARGET CLASS', axis=1))
df = pd.DataFrame(df_0, columns=df_0.columns[:-1])

df_features = pd.DataFrame(scaled_features, columns = df.columns)
print(df.columns)
print(df_features.head())

X = df_features
y = df_0['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(X_train, y_train)
pred = knc.predict(X_test)
#print(pred)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# ---------------------------------------------------------------

error_rate = []
for i in range(1, 40):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(X_train, y_train)
    pred_i = knc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error_rate, color = 'green', linestyle = 'dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

knc = KNeighborsClassifier(n_neighbors=12)
knc.fit(X_train, y_train)
pred_1 = knc.predict(X_test)
print(confusion_matrix(y_test, pred_1))
print(classification_report(y_test, pred_1))
