import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

animals = pd.read_csv(r"C:\data\animals.csv")

animals = animals[ animals['name'].isin(['Cow','Goat','Donkey','Horse',
                       'Giraffe','Kangaroo','Rabbit','Sheep','Mole','Pig']) ]

animals

plt.figure(figsize=(7, 5))
plt.scatter(animals["body"], animals["brain"], color="blue")

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X = animals["body"].values.reshape(-1,1), y = animals["brain"].values)

brain_pred = lr.predict(X = animals["body"].values.reshape(-1,1))

plt.figure(figsize=(7, 5))
plt.scatter(animals["body"], animals["brain"], color="blue")
plt.plot(animals["body"], brain_pred, color="red", linewidth=2)

new_animals_body = np.array([100, 200, 300, 400])
new_animals_brain = lr.predict(new_animals_body.reshape(-1,1))
new_animals_brain

plt.figure(figsize=(7, 5))
plt.scatter(animals["body"], animals["brain"], color="blue")
plt.plot(animals["body"], brain_pred, color="red", linewidth=2)
plt.scatter(new_animals_body, new_animals_brain , color = 'black', s=100)

print(lr.coef_)
print(lr.intercept_)
