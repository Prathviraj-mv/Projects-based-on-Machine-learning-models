import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer


cancer  = load_breast_cancer()
print(cancer.keys())
df_F =pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df_F.info)

from sklearn.model_selection import train_test_split,GridSearchCV

X=df_F
y=cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.svm import SVC

model =SVC()
model.fit(X_train,y_train)
pred =model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

parm_grid ={
    'C':[0,1,1,10,100,1000],
    'gamma':[1,0.1,0.01,0.001]
}
grid =GridSearchCV(SVC(),parm_grid,verbose=56)
grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_estimator_)

pred =grid.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))










