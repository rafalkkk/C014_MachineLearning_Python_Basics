import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
%matplotlib inline

class Perceptron:
    
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    
    def predict(self, x):
        
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        #activation = self.get_activation(x_1)
        #y_pred = np.where(activation >0, 1, -1)
        #return y_pred
        return np.where(self.get_activation(x_1) > 0, 1, -1)
        
    
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
                
                
                
X = np.array([
    [2., 4.,  20.],  # 2*2 - 4*4 + 20 =   8 > 0
    [4., 3., -10.],  # 2*4 - 4*3 - 10 = -14 < 0
    [5., 6.,  13.],  # 2*5 - 4*6 + 13 =  -1 < 0
    [5., 4.,   8.],  # 2*5 - 4*4 + 8 =    2 > 0
    [3., 4.,   5.],  # 2*3 - 4*4 + 5 =   -5 < 0 
])

y = np.array([1, -1, -1, 1, -1])

(X[:,0] - X[:,0].mean()) / X[:,0].std()

def StandarizeData(X):
    for col in range(X.shape[1]):
        X[:,col] = (X[:,col] - X[:,col].mean()) / X[:,col].std()

X
StandarizeData(X)
X


perceptron = Perceptron(eta=1, epochs=100, is_verbose=True)            
perceptron.fit(X, y)
plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)


df = pd.read_csv(r"C:\data\iris.data", header = None)
df = df.iloc[:100, :].copy()
df[4] = df[4].apply(lambda x: 1 if x == 'Iris-setosa' else -1)
df

X = df.iloc[0:100, :-1].values
y = df[4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train
StandarizeData(X_train)
X_train

p = Perceptron(eta = 0.001, epochs=100) 
p.fit(X_train, y_train)                
plt.scatter(range(p.epochs), p.list_of_errors)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaler.mean_
scaler.scale_
X_train = scaler.transform(X_train)


#### mini demo standaryzacja

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = np.array([2., 7., 8., 10.])
zeros = np.zeros(data.shape[0])

plt.figure(figsize=(10,2))
plt.scatter(x = data, y=zeros, marker='o', s=50, c='b')
plt.scatter(x = [data.mean()], y=0, marker='d', s=100, c='r')
plt.scatter(x = [data.mean() - data.std(), data.mean() + data.std()], 
            y = [0,0], marker='s', s=70, c='y', alpha=0.5)

mean = data.mean()
data_stand = data - mean
data_stand

plt.figure(figsize=(10,2))
plt.scatter(x = data_stand, y=zeros, marker='o', s=50, c='b')
plt.scatter(x = [data_stand.mean()], y=0, marker='d', s=100, c='r')
plt.scatter(x = [data_stand.mean() - data_stand.std(), 
                 data_stand.mean() + data_stand.std()], 
            y = [0,0], marker='s', s=70, c='y', alpha=0.5)

std = data.std()
data_stand = data_stand / std
data_stand

plt.figure(figsize=(10,2))
plt.scatter(x = data_stand, y=zeros, marker='o', s=50, c='b')
plt.scatter(x = [data_stand.mean()], y=0, marker='d', s=100, c='r')
plt.scatter(x = [data_stand.mean() - data_stand.std(), 
                 data_stand.mean() + data_stand.std()], 
            y=[0,0], marker='s', s=70, c='y', alpha=0.5)






              