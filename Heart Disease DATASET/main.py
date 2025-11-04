import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

FILE ="heart.csv"
data = pd.read_csv(FILE)

cP =LabelEncoder()
data["ChestPainType"] =cP.fit_transform(data["ChestPainType"])
rp = LabelEncoder()
data["RestingECG"] = rp.fit_transform(data["RestingECG"])
st = LabelEncoder()
data["ST_Slope"] = st.fit_transform(data["ST_Slope"])
ex = LabelEncoder()
data["ExerciseAngina"] = ex.fit_transform(data["ExerciseAngina"])
sex = LabelEncoder()
data["Sex"] = sex.fit_transform(data["Sex"])
corr_data =data.corr()["HeartDisease"].sort_values(ascending=True)

sns.countplot(data=data,x ="Age",hue ="HeartDisease")
X = data.drop("HeartDisease",axis=1)
y = data["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
LF =LogisticRegression(random_state=42)
LF.fit(X_train,y_train)
prediction = LF.predict(X_test)
print(classification_report(y_true=y_test,y_pred=prediction))
print(round(accuracy_score(y_true=y_test,y_pred=prediction),2))
print(round(precision_score(y_true=y_test,y_pred=prediction),2))
# precision    recall  f1-score   support
#
#            0       0.77      0.89      0.83       123
#            1       0.92      0.82      0.87       180
#
#     accuracy                           0.85       303
#    macro avg       0.85      0.86      0.85       303
# weighted avg       0.86      0.85      0.85       303
# 0.8514851485148515
# 0.9192546583850931
from xgboost import XGBClassifier
xg =XGBClassifier(n_estimators=200,random_state=42)
xg.fit(X_train,y_train)
prediction = xg.predict(X_test)
print(classification_report(y_true=y_test,y_pred=prediction))
print(round(accuracy_score(y_true=y_test,y_pred=prediction),2))
print(round(precision_score(y_true=y_test,y_pred=prediction),2))
# precision    recall  f1-score   support
#
#            0       0.78      0.87      0.82       123
#            1       0.90      0.83      0.87       180
#
#     accuracy                           0.85       303
#    macro avg       0.84      0.85      0.85       303
# weighted avg       0.85      0.85      0.85       303
#
# 0.8481848184818482
# 0.9036144578313253


from sklearn.ensemble import RandomForestClassifier
RF =RandomForestClassifier(n_estimators=500,random_state=42,max_depth=10,min_samples_split=4,min_samples_leaf=2)
RF.fit(X_train,y_train)
prediction = RF.predict(X_test)
print(classification_report(y_true=y_test,y_pred=prediction))
print(round(accuracy_score(y_true=y_test,y_pred=prediction),2))
print(round(precision_score(y_true=y_test,y_pred=prediction),2))
# precision    recall  f1-score   support
#
#            0       0.84      0.88      0.86       123
#            1       0.91      0.88      0.90       180
#
#     accuracy                           0.88       303
#    macro avg       0.88      0.88      0.88       303
# weighted avg       0.88      0.88      0.88       303
#
# 0.8811881188118812
# 0.9137931034482759

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

gS =GridSearchCV(n_jobs=-1,estimator=RF,cv =5,verbose=2,param_grid=param_grid,scoring='accuracy')
gS.fit(X_train,y_train)
BGS =gS.best_estimator_

prediction = BGS.predict(X_test)
print(classification_report(y_true=y_test,y_pred=prediction))
print(round(accuracy_score(y_true=y_test,y_pred=prediction),2))
print(round(precision_score(y_true=y_test,y_pred=prediction),2))
 # precision    recall  f1-score   support

#            0       0.83      0.88      0.85       123
#            1       0.91      0.88      0.90       180
#
#     accuracy                           0.88       303
#    macro avg       0.87      0.88      0.87       303
# weighted avg       0.88      0.88      0.88       303
#
# 0.8778877887788779
# 0.9132947976878613
sns.heatmap(confusion_matrix(y_true=y_test,y_pred=prediction),annot=True, fmt='d')
plt.show()
