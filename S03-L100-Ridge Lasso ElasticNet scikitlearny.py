import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


wine = pd.read_csv(r'C:\data\redwinequality.csv')
columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

X = wine[columns].values
y = wine['quality'].values.astype(float)



scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



linear = LinearRegression()
linear.fit(X_train, y_train)                

lasso = Lasso(alpha = 0.1)
lasso.fit(X_train, y_train)

ridge = Ridge(alpha = 0.1)
ridge.fit(X_train, y_train)

elastic = ElasticNet(alpha = 0.1, l1_ratio = 0.5)
elastic.fit(X_train, y_train)


coef = pd.DataFrame(data = [linear.coef_, lasso.coef_, ridge.coef_, 
                            elastic.coef_]).transpose()
coef.index = columns
coef.columns = ['Linear', 'Lasso', 'Ridge', 'ElasticNet']
coef

fig, axs = plt.subplots(4, figsize=(10,10))
fig.suptitle('Coefficients in different regressors')
linear_coef = pd.Series(linear.coef_, columns).sort_values()
axs[0].bar(x = linear_coef.index,  height=linear_coef)
lasso_coef = pd.Series(lasso.coef_, columns).sort_values()
axs[1].bar(x = lasso_coef.index, height=lasso_coef)
ridge_coef = pd.Series(ridge.coef_, columns).sort_values()
axs[2].bar(x = ridge_coef.index, height=ridge_coef)
elastic_coef = pd.Series(elastic.coef_, columns).sort_values()
axs[3].bar(x = elastic_coef.index, height=elastic_coef)
fig.show()


#linear_coef = pd.Series(linear.coef_, columns).sort_values()
#linear_coef.plot(kind='bar', title='Linear')
#lasso_coef = pd.Series(lasso.coef_, columns).sort_values()
#lasso_coef.plot(kind='bar', title='Lasso')
#ridge_coef = pd.Series(ridge.coef_, columns).sort_values()
#ridge_coef.plot(kind='bar', title='Ridge')
#elastic_coef = pd.Series(elastic.coef_, columns).sort_values()
#elastic_coef.plot(kind='bar', title='Elastic Net')


r2 = pd.DataFrame(columns = ['train', 'test'], 
                  index = ['linear', 'lasso', 'ridge', 'elastic net'])
r2.loc['linear'] = [r2_score(y_train, linear.predict(X_train)), 
                    r2_score(y_test,  linear.predict(X_test))]
r2.loc['lasso'] = [r2_score(y_train, lasso.predict(X_train)), 
                   r2_score(y_test,  lasso.predict(X_test))]
r2.loc['ridge'] = [r2_score(y_train, ridge.predict(X_train)), 
                   r2_score(y_test,  ridge.predict(X_test))]
r2.loc['elastic net'] = [r2_score(y_train, elastic.predict(X_train)), 
                         r2_score(y_test,  elastic.predict(X_test))]

r2


alpha_list = np.arange(0.01, 1, 0.01)
alpha_list




lasso_coef = pd.DataFrame(columns = columns)
lasso_r2 = []

for a in alpha_list:
    lasso = Lasso(alpha = a)
    lasso.fit(X_train, y_train)
    lasso_coef.loc[a] = lasso.coef_
    
    lasso_r2.append(r2_score(y_train, lasso.predict(X_train)))
    
f = plt.figure(figsize=(10,5))
plt.title('Lasso coefficient changes')
lasso_coef.plot(kind='line', ax=f.gca())
plt.legend(loc='upper right')
plt.show()

plt.plot(alpha_list, lasso_r2)
plt.title('Lasso R2 score')
plt.show()





ridge_coef = pd.DataFrame(columns = columns)
ridge_r2 = []

for a in alpha_list:
    ridge = Ridge(alpha = a)
    ridge.fit(X_train, y_train)
    ridge_coef.loc[a] = ridge.coef_
    
    ridge_r2.append(r2_score(y_train, ridge.predict(X_train)))
 
f = plt.figure(figsize=(10,5))
plt.title('Ridge coefficient changes')
ridge_coef.plot(kind='line', ax=f.gca())
plt.legend(loc='upper right')
plt.show()

plt.plot(alpha_list, ridge_r2)
plt.title('Ridge R2 score')
plt.show()




elastic_coef = pd.DataFrame(columns = columns)
elastic_r2 = []

for a in alpha_list:
    elastic = ElasticNet(alpha = a)
    elastic.fit(X_train, y_train)
    elastic_coef.loc[a] = elastic.coef_
    
    elastic_r2.append(r2_score(y_train, elastic.predict(X_train)))
    
f = plt.figure(figsize=(10,5))
plt.title('ElasticNet coefficient changes')
elastic_coef.plot(kind='line', ax=f.gca())
plt.legend(loc='upper right')
plt.show()

plt.plot(alpha_list, elastic_r2)
plt.title('ElasticNet R2 score')
plt.show()



