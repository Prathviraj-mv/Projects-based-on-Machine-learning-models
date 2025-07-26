import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.preprocessing import StandardScaler


cancer =load_breast_cancer()
print(cancer.keys())
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df)

scalr =StandardScaler()
scalr.fit(df)

sc =scalr.transform(df)

from sklearn.decomposition import PCA
pca =PCA(n_components=2)
pca.fit(sc)
pcc = pca.transform(sc)
plt.figure(figsize=(6,6))
plt.scatter(pcc[:,0],pcc[:,1],c=cancer['target'])
plt.show()

