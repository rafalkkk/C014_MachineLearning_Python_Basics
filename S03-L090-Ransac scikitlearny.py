import numpy as np
import pandas as pd
from statsmodels import robust
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
%matplotlib inline



n_samples = 500
n_outliers = 50
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)            
# Add outlier data
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

plt.figure(figsize=(5,5))
plt.scatter(X,y,color='gold')
plt.show()




scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)                

y_pred = lr.predict(X_test)                

line_X = np.arange(X.min(), X.max())[:, np.newaxis]

plt.figure(figsize=(7,5))
plt.scatter(X[:], y[:], color='green', 
            marker='.', label='Inliers')
plt.plot(X_test, y_pred, color='red', label='Linear regression')

plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()





ransac = linear_model.RANSACRegressor(
            base_estimator = LinearRegression(),
            min_samples = 0.7,
            residual_threshold = robust.mad(y),
            max_trials = 100)

ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print("Estimated coefficients (true, linear regression, RANSAC):")
print(lr.coef_, ransac.estimator_.coef_)


# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

plt.scatter(X_train[inlier_mask], y_train[inlier_mask], color='green', marker='.',
            label='Inliers')
plt.scatter(X_train[outlier_mask], y_train[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y, color='red', linewidth=2, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='blue', linewidth=2,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()






wine = pd.read_csv(r'C:\data\redwinequality.csv')
columns = ['alcohol']
X = wine[columns].values
y = wine['quality'].values.astype(float)




scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(X_train, y_train)                

y_pred = lr.predict(X_test)                

line_X = np.arange(X.min(), X.max())[:, np.newaxis]

plt.figure(figsize=(7,5))
plt.scatter(X[:], y[:], color='green', 
            marker='.', label='Inliers')
plt.plot(X_test, y_pred, color='red', label='Linear regression')

plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()





ransac = linear_model.RANSACRegressor(
            base_estimator = LinearRegression(),
            min_samples = 0.7,
            residual_threshold = 1,
            max_trials = 100)

ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print("Estimated coefficients (true, linear regression, RANSAC):")
print(lr.coef_, ransac.estimator_.coef_)


# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)



plt.scatter(X_train[inlier_mask], y_train[inlier_mask], color='green', marker='.',
            label='Inliers')
plt.scatter(X_train[outlier_mask], y_train[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y, color='red', linewidth=2, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='blue', linewidth=2,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()

