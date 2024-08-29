import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# test with wine dataset
wine = pd.read_csv(r'C:\data\redwinequality.csv')

columns = ['alcohol']
X = wine[columns]
y = wine['quality'].astype(float)

# test with randomly created dataset
#n_samples = 500
#X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
#                                      n_informative=1, noise=10,
#                                      coef=True, random_state=0)
plt.figure(figsize=(5,5))
plt.scatter(X, y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)                


y_pred_train = lr.predict(X_train)

plt.figure(figsize=(5,5))
plt.scatter(X_train, y_train, color = 'gold')
plt.plot(X_train, y_pred_train, color = 'blue')


y_pred_test = lr.predict(X_test)                

plt.figure(figsize=(5,5))
plt.scatter(X_test, y_test, color = 'gold')
plt.plot(X_test, y_pred_test, color = 'blue')


fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(y_train, y_pred_train - y_train, s=80, 
          facecolors='none', edgecolors='b')
ax[1].scatter(y_test,  y_pred_test  - y_test,  s=80, 
          facecolors='none', edgecolors='r')
fig.show()








