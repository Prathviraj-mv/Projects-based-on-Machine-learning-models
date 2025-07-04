import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from  sklearn.datasets import load_iris

flower = load_iris()
print(flower.keys())
flow =pd.DataFrame(flower['data'],columns=flower['feature_names'])
print(flow)

sv =SVC()
X = flow
y =flower['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
sv.fit(X_train,y_train)

pred =sv.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
