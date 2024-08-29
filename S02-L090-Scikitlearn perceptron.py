import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
%matplotlib inline

# from sklearn import datasets
iris = datasets.load_iris()

X = iris.data[:]
y = iris.target[:]

# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# from sklearn.linear_model import Perceptron
perceptron = Perceptron(max_iter=20, eta0=0.1)
perceptron.fit(X_train_std, y_train)

y_pred = perceptron.predict(X_test_std)

y_test
y_pred

[(a,b) for (a, b) in zip(y_pred[y_pred != y_test], y_test[y_pred != y_test])]


bad_results = [(a,b) for (a, b) in zip(y_pred[y_pred != y_test], y_test[y_pred != y_test])]
good_results = [(a,b) for (a, b) in zip(y_pred[y_pred == y_test], y_test[y_pred == y_test])]
bad_results
good_results
print(len(good_results) / len(y_test))

print(perceptron.score(X_test_std, y_test))

print(perceptron.coef_)
print(perceptron.intercept_)
print(perceptron.n_iter_)
print(perceptron.t_)






X, y = datasets.load_digits(return_X_y=True)
X[1]
X[1].reshape(8,8)
len(X)
np.min(X)
np.max(X)

nrows = 10
ncols = 13

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows, ncols))
fig.subplots_adjust(hspace = 0.8)

for row in range(nrows):
    for col in range(ncols):
        picture_number = row*ncols + col
        ax[row, col].imshow(X[picture_number].reshape((8,8)), 
          cmap=plt.cm.gray_r, interpolation='nearest')        
        ax[row, col].set_title(y[picture_number])
        ax[row, col].xaxis.set_visible(False)
        ax[row, col].yaxis.set_visible(False)
fig.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
sc.fit(X)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

perceptron = Perceptron(max_iter=100, random_state=0, shuffle=True, 
                        fit_intercept=True)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
perceptron.score(X_test, y_test)

bad_results = [(a,b,c) for (a,b,c) in zip(X_test[y_test != y_pred], 
                                          y_test[y_test != y_pred],
                                          y_pred[y_test != y_pred] )]
len(bad_results)

nrows = len(bad_results)

fig, ax = plt.subplots(nrows=nrows, figsize=(1,nrows))
fig.subplots_adjust(hspace = 0.8)

for row in range(nrows):
    picture_number = row
    ax[row].imshow(bad_results[picture_number][0].reshape((8,8)), 
          cmap=plt.cm.gray_r, interpolation='nearest')        
    ax[row].set_title("t: {} p:{}".format(bad_results[picture_number][1],
                                          bad_results[picture_number][2]))
    ax[row].xaxis.set_visible(False)
    ax[row].yaxis.set_visible(False)
fig.show()

print(perceptron.coef_)
print(len(perceptron.coef_))
print(perceptron.intercept_)
print(perceptron.n_iter_)
print(perceptron.t_)






