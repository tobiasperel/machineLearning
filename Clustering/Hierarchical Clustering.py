# https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/coursera/ML0101EN/ML0101EN-Clus-Hierarchical-Cars-py-v1.ipynb?lti=true
# wget -O cars_clus.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/cars_clus.csv

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

# Generating Random Data

# aca hago una data random le pongo que sean 150 puntos , los centros y el radio de los puntos
X1, y1 = make_blobs(n_samples=150, centers=[[4, 4], [-2, -1], [1, 1], [10, 4]], cluster_std=0.9) # x1 son las cordenadas meintras que y1 es el resulatado del grupo al que pertenecen
# aca le digo que dentro de los parametros que le pase, que me lo genere random
plt.scatter(X1[:, 0], X1[:, 1], marker='o')
plt.show()
print(X1)
print(y1)
# Agglomerative Clustering

agglom = AgglomerativeClustering(n_clusters=4, linkage='average') # aca decis que haya 4 grupos y que el criterio que vas a usar es promedio
# si repasamos los apuntas habia 4 criterios: single linkage --> tomas los dos puntos mas cerca de grupos distintos, Complete --> tomas los mas lejanos
# average --> el promedio y Centroid --> el centro de los grupos. El que mas se recomienda usar es uno de los ultimos 2 (mas average que otra cosa).
agglom.fit(X1, y1)

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6, 4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0) # creas un minimo un maximo valor de los puntos

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min) # tomas el promedio de x1

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
            color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.), # no se de donde saca los numeros como 10 y 9 
            fontdict={'weight': 'bold', 'size': 9})
'''
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
# plt.axis('off')
'''

# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()

#Dendrogram Associated for the Agglomerative Hierarchical Clustering

