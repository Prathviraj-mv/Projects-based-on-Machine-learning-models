import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


data = pd.read_csv("USA_Housing.csv")
print(data.describe())
print(data.head())

print(data.columns)
X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
          'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

LR = LinearRegression()
LR.fit(X_train, y_train)

print(LR.intercept_)

cpd = pd.DataFrame(LR.coef_, X.columns, columns=['Coeff'])
print(cpd)
prediction =LR.predict(X_test)
print(prediction)
plt.scatter(y_test, prediction)
sns.displot(y_test - prediction)

print(metrics.mean_absolute_error(y_test,prediction))

print(metrics.mean_squared_error(y_test,prediction))

print(np.sqrt(metrics.mean_squared_error(y_test,prediction)))

plt.show()
