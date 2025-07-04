import seaborn as sns
import pandas as pd
import numpy as np
from fontTools.subset import subset

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


data =pd.read_csv("titanic_train.csv")
data = data.dropna(subset='Age')

sex =pd.get_dummies(data['Sex'],drop_first=True)
emb =pd.get_dummies(data['Embarked'],drop_first=True)


data =pd.concat([data,sex,emb],axis=1)
data =data.drop(columns=['Sex'])
data =data.drop(columns=['Embarked'])
data =data.drop(columns=['Cabin'])
data =data.drop(columns=['Name'])
data =data.drop(columns=['Ticket'])
data =data.drop(columns=['PassengerId'])
print(data.head())
print(data.columns)


X=data.drop("Survived",axis=1)
y= data["Survived"]

sv =SVC()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
sv.fit(X_train,y_train)

pred =sv.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))



param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

grid =GridSearchCV(SVC(),param_grid,verbose=56)
grid.fit(X_train,y_train)
pred =grid.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
