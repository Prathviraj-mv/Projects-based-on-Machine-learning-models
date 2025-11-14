import pandas as pd

file ="earthquake_data_tsunami.csv"
data = pd.read_csv(file)
print(data.head(5))
print(data.columns)

import matplotlib.pyplot as plt
import seaborn as sns

data_corr =data.corr()["tsunami"].sort_values(ascending=True)
print(data_corr)
sns.heatmap(data.corr(),annot=True)
# sns.pairplot(data=data,hue="tsunami")
# sns.countplot(data =data,x="year",hue="tsunami")
from sklearn.metrics import classification_report, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

X = data.drop(["tsunami"],axis=1)
y = data["tsunami"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier
Rf= RandomForestClassifier(n_estimators=200,n_jobs=3,random_state=42,max_depth=10,min_samples_leaf=5,min_samples_split=4)
Rf.fit(X_train,y_train)

prediction =Rf.predict(X_test)
print(classification_report(y_true=y_test,y_pred=prediction))
conf = confusion_matrix(y_test,prediction)
plt.figure( )
sns.heatmap(conf,annot=True,fmt ='d')
plt.show()




#  precision    recall  f1-score   support

#            0       0.98      0.89      0.93       151
#            1       0.87      0.97      0.92       108

#     accuracy                           0.93       259
#    macro avg       0.92      0.93      0.93       259
# weighted avg       0.93      0.93      0.93       259
