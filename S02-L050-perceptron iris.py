import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

class Perceptron:
    
    def __init__(self, eta=0.10, epochs=50, is_verbose = False):
        
        self.eta = eta
        self.epochs = epochs
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    
    def predict(self, x):
        
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        return self.__predict(x_1)
        
        
    def __predict(self, x):
        
        total_stimulation = np.dot(x, self.w)       
        y_pred = np.where(total_stimulation>0, 1, -1)
        return y_pred
        
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0], 1))
        X_1 = np.append(X.copy(), ones, axis=1)

        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epochs):

            number_of_errors = 0
            
            y_pred = self.__predict(X_1)
            delta_w = self.eta * np.dot((y - y_pred), X_1)
            self.w += delta_w
                
            number_of_errors = np.count_nonzero(y - y_pred)
                
            self.list_of_errors.append(number_of_errors)
            
            if(self.is_verbose):
                print("Epoch: {}, weights: {}, number of errors {}".format(
                        e, self.w, number_of_errors))
                
                
                
X = np.array([
    [2, 4,  20],  # 2*2 - 4*4 + 20 =   8 > 0
    [4, 3, -10],  # 2*4 - 4*3 - 10 = -14 < 0
    [5, 6,  13],  # 2*5 - 4*6 + 13 =  -1 < 0
    [5, 4,   8],  # 2*5 - 4*4 + 8 =    2 > 0
    [3, 4,   5],  # 2*3 - 4*4 + 5 =   -5 < 0 
])

y = np.array([1, -1, -1, 1, -1])

perceptron = Perceptron(eta=0.1, epochs=100, is_verbose=True)            
perceptron.fit(X, y)

print(perceptron.w)

%matplotlib inline
plt.scatter(range(perceptron.epochs), perceptron.list_of_errors)

print(perceptron.predict(np.array([[1, 2, 3]])))  # 2*1 - 4*2 + 1 = -3 < 0
print(perceptron.predict(np.array([[2, 2, 8]])))  # 2*2 - 4*2 + 8 =  4 > 0
print(perceptron.predict(np.array([[3, 3, 3]])))  # 2*3 - 4*3 + 3 = -3 < 0


df = pd.read_csv(r"C:\data\iris.data", header = None)
df = df.iloc[:100, :].copy()
df[4] = df[4].apply(lambda x: 1 if x == 'Iris-setosa' else -1)
df

X = df.iloc[0:100, :-1].values
y = df[4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

p = Perceptron(eta = 0.5, epochs=10) 
p.fit(X_train, y_train)                

y_pred = p.predict(X_test)

print(list(zip(y_pred, y_test)))
print(np.count_nonzero(y_pred - y_test))

plt.scatter(range(p.epochs), p.list_of_errors)










              