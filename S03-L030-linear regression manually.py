import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
%matplotlib inline

class LinearRegression:
    
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    
    def predict(self, x):
        
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        return self.get_activation(x_1) 
        
    
    def get_activation(self, x):
        
        activation = np.dot(x, self.w)
        return activation
     
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)

        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):

            error = 0
            
            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y - activation), X_1)
            self.w += delta_w
                
            error = np.square(y - activation).sum()/2.0
                
            self.list_of_errors.append(error)
            
            if(self.is_verbose):
                print("Epoch: {}, weights: {}, error {}".format(
                        e, self.w, error))
                
        

wine = pd.read_csv(r'C:\data\redwinequality.csv')

X = wine['alcohol'].values.reshape(-1,1)
y = wine['quality'].values

scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)

lr1 = LinearRegression(eta = 0.0001, epochs=100) 
lr1.fit(X_train, y_train)                
plt.scatter(range(lr1.epochs), lr1.list_of_errors)

y_pred = lr1.predict(X_test)

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, s=80, facecolors='none', edgecolors='r')

plt.figure(figsize=(7, 7))
plt.scatter(X_test, y_test)
plt.plot(X_test,y_pred, color='red')
              
np.count_nonzero(np.rint(y_pred) == y_test) / len(y_test)


round(np.mean(y_test))
np.count_nonzero(round(np.mean(y_test)) == y_test) / len(y_test)












