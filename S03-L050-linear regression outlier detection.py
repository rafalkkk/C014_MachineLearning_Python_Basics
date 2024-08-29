import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
%matplotlib inline
from sklearn.linear_model import LinearRegression


wine = pd.read_csv(r'C:\data\redwinequality.csv')

plt.figure()
sns.boxplot(wine['alcohol'])
plt.plot()


from scipy import stats

z = np.abs(stats.zscore(wine))
print(z)

threshold = 4
print(np.where(z > threshold))


# removing outliers with Z-score
wine_o_z = wine[(z<threshold).all(axis=1)]
wine_o_z

# detecting outliers with IQR method
Q1 = wine.quantile(0.25)
Q3 = wine.quantile(0.75)
IQR = Q3 - Q1

((wine < (Q1 - 1.5 * IQR)) | (wine > (Q3 +1.5 * IQR)))
((wine < (Q1 - 1.5 * IQR)) | (wine > (Q3 +1.5 * IQR))).iloc[14:18,4:7]

# removing outliers
outlier_condition = ((wine < (Q1 - 1.5 * IQR)) | (wine > (Q3 +1.5 * IQR)))
wine_o_iqr = wine[~outlier_condition.any(axis=1)]
wine_o_iqr


wine["quality"].unique()
wine_o_z["quality"].unique()
wine_o_iqr["quality"].unique()

for i in wine.columns[:-1]:
    fig, axs = plt.subplots(3,1, figsize=(10,7))
    #fig.suptitle(i)
    sns.boxplot(x=wine['quality'], y=wine[i], ax=axs[0])
    sns.boxplot(x=wine_o_z['quality'], y=wine_o_z[i], ax=axs[1])
    sns.boxplot(x=wine_o_iqr['quality'], y=wine_o_iqr[i], ax=axs[2])
    plt.plot()




columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

X = wine[columns]
y = wine['quality'].astype(float)

X = wine_o_z[columns]
y = wine_o_z['quality'].astype(float)

X = wine_o_iqr[columns]
y = wine_o_iqr['quality'].astype(float)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)                

y_pred = lr.predict(X_test)             
   
good_counter = np.count_nonzero(y_test == np.rint(y_pred))
total_counter = len(y_test)
print(good_counter / total_counter)

#plt.figure(figsize=(7, 7))
#plt.scatter(y_test, y_pred, s=80, facecolors='none', edgecolors='r')

#lr.score(X_test, y_test)







