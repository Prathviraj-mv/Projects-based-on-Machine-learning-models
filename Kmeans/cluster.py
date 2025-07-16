import seaborn as sns
import pandas as pd
import numpy as np
from fontTools.subset import subset
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.cluster import k_means, KMeans
from  sklearn.datasets import make_blobs

data =make_blobs(n_samples=200 ,n_features=2,centers=4,cluster_std=1.8,random_state=54)
kmeans =KMeans(n_clusters=4)
kmeans.fit(data[0])
print(kmeans.labels_)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.subplot(1,2,2)
plt.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

plt.show()
