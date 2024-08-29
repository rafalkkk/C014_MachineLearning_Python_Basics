import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

wine = pd.read_csv(r'C:\data\redwinequality.csv')

wine.head()
wine.info()

wine.describe()
wine.columns

for i in wine.columns:
    print(wine[i].describe())



plt.figure()
sns.boxplot(x=wine['quality'], y=wine['alcohol'])
plt.plot()

for i in wine.columns[:-1]:
    plt.figure()
    sns.boxplot(x=wine['quality'], y=wine[i])
    plt.plot()


plt.figure()
sns.barplot(x=wine['quality'], y=wine['alcohol'])
plt.plot()    
 
for i in wine.columns[:-1]:
    plt.figure()
    sns.barplot(x=wine['quality'], y=wine[i])
    plt.plot()    

corr_matrix = np.corrcoef(wine.values.T)

fig, ax = plt.subplots(figsize=(11,11))
sns.set(font_scale=1.1)
sns.heatmap(data = corr_matrix,
            square = True,
            cbar = True,
            annot = True,
            fmt = '.2f',
            annot_kws = {'size' : 10},
            xticklabels = wine.columns,
            yticklabels = wine.columns
)


sns.pairplot(wine, height=1.5)

columns = ['alcohol','volatile acidity', 'sulphates', 'citric acid',
           'total sulfur dioxide','density',  
           'quality']

sns.pairplot(wine[columns], size=1.5)

















