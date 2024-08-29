import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\data\Airbnb listings in Ottawa (May 2016).csv")
df.shape
df.head()

coordinates = df.loc[:,['longitude','latitude']]
coordinates.shape

plt.scatter(df.loc[:,'longitude'], df.loc[:,'latitude'])



WCSS = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(coordinates)
    WCSS.append(kmeans.inertia_)
    
plt.plot(range(1,15),WCSS)
plt.xlabel("Number of K Value(Cluster)")
plt.ylabel("WCSS")
plt.grid()
plt.show()



kmeans =KMeans(n_clusters = 4 ,max_iter=300, random_state= 1)
clusters = kmeans.fit_predict(coordinates) 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


h = 0.001
x_min, x_max = coordinates['longitude'].min(), coordinates['longitude'].max()
y_min, y_max = coordinates['latitude'].min() , coordinates['latitude'].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]) 

plt.figure(1 , figsize = (10 , 4) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel1, origin='lower')

plt.scatter(x= coordinates['longitude'], y = coordinates['latitude'] , 
            c= labels, s=100 )

plt.scatter(x = centroids[:,0], y =  centroids[:,1], 
            s=300 , c='red')

plt.ylabel('Long(y)') , plt.xlabel('Lat(x)')
plt.grid()
plt.title("Clustering")
plt.show()


