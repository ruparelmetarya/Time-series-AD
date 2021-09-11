import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

f = open("data (2).csv")
data = np.genfromtxt(f, delimiter=",")
# print(data)
X_data = np.array(data[:,3:])
print(X_data)
X_len = X_data.size
print X_len
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10000,random_state=0,algorithm="auto",max_iter=100000).fit(X_data)
assignment_values = kmeans.labels_
centroid_values = kmeans.cluster_centers_
print(assignment_values)
# print(centroid_values)
# print(X_data[:, 0])
plt.scatter(X_data[:, 0],X_data[:, 1], c=assignment_values, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()
