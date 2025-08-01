import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from  sklearn.metrics import  confusion_matrix
from sklearn.tree import DecisionTreeClassifier

data =pd.read_csv("kyphosis.csv")
print(data.head())
# sns.pairplot(data=data,hue='Kyphosis')
print(data.columns)

X = data.drop(columns='Kyphosis')
y = data['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# single tree

DT =DecisionTreeClassifier()
DT.fit(X_train,y_train)
pred = DT.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

from sklearn.ensemble import RandomForestClassifier
rfc =RandomForestClassifier(n_estimators=400)
rfc.fit(X_train,y_train)
rfcp =rfc.predict(X_test)
print(confusion_matrix(y_test,rfcp))
print(classification_report(y_test,rfcp))


plt.show()


"Kyphosis","Age","Number","Start"
"absent",71,3,5
"absent",158,3,14
"present",128,4,5
"absent",2,5,1
"absent",1,4,15
"absent",1,2,16
"absent",61,2,17
"absent",37,3,16
"absent",113,2,16
"present",59,6,12
"present",82,5,14
"absent",148,3,16
"absent",18,5,2
"absent",1,4,12
"absent",168,3,18
"absent",1,3,16
"absent",78,6,15
"absent",175,5,13
"absent",80,5,16
"absent",27,4,9
"absent",22,2,16
"present",105,6,5
"present",96,3,12
"absent",131,2,3
"present",15,7,2
"absent",9,5,13
"absent",8,3,6
"absent",100,3,14
"absent",4,3,16
"absent",151,2,16
"absent",31,3,16
"absent",125,2,11
"absent",130,5,13
"absent",112,3,16
"absent",140,5,11
"absent",93,3,16
"absent",1,3,9
"present",52,5,6
"absent",20,6,9
"present",91,5,12
"present",73,5,1
"absent",35,3,13
"absent",143,9,3
"absent",61,4,1
"absent",97,3,16
"present",139,3,10
"absent",136,4,15
"absent",131,5,13
"present",121,3,3
"absent",177,2,14
"absent",68,5,10
"absent",9,2,17
"present",139,10,6
"absent",2,2,17
"absent",140,4,15
"absent",72,5,15
"absent",2,3,13
"present",120,5,8
"absent",51,7,9
"absent",102,3,13
"present",130,4,1
"present",114,7,8
"absent",81,4,1
"absent",118,3,16
"absent",118,4,16
"absent",17,4,10
"absent",195,2,17
"absent",159,4,13
"absent",18,4,11
"absent",15,5,16
"absent",158,5,14
"absent",127,4,12
"absent",87,4,16
"absent",206,4,10
"absent",11,3,15
"absent",178,4,15
"present",157,3,13
"absent",26,7,13
"absent",120,2,13
"present",42,7,6
"absent",36,4,13
