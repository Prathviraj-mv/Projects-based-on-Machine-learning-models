# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data =pd.read_csv("breast-cancer.csv")
print(data.head(5))
print(data.columns)

# %%
# total count
sns.countplot(x="diagnosis", data=data, palette=["green", "red"])  

# %%
corr = data.drop(columns=["id", "diagnosis"]).corr()
sns.heatmap(corr, cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# %%
data["diagnosis"] = data["diagnosis"].map({"M":1, "B":0})
corr_with_diag = data.corr()["diagnosis"].sort_values(ascending=False)
print(corr_with_diag)

# %%
print(data.info())
data.isnull().sum()
sns.pairplot(
    data[["diagnosis", "radius_mean", "texture_mean", "area_mean", "perimeter_mean"]],
    hue="diagnosis",
    palette={1:"red", 0:"green"}
)
plt.show()

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,classification_report

# Select your chosen features into X
X = data[[
    "concave points_worst",
    "perimeter_worst",
    "concave points_mean",
    "radius_worst",
    "perimeter_mean",
    "area_worst",
    "radius_mean",
    "area_mean",
    "concavity_mean",
    "concavity_worst",
    "compactness_mean",
    "compactness_worst",
    "radius_se",
    "perimeter_se",
    "area_se",
    "texture_worst",
    "smoothness_worst",
    "symmetry_worst",
    "texture_mean",
    "concave points_se",
    "smoothness_mean",
    "symmetry_mean",
    "fractal_dimension_worst",
    "compactness_se"

]]

y = data["diagnosis"] 
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)


# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
rf =RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_split=5, min_samples_leaf=2, bootstrap=True, n_jobs=-1, random_state=42)
rf.fit(X_train,y_train)
pred =rf.predict(X_test)
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))




