# %%
import pandas as pd
path ="ecommerce_customer_churn_dataset.csv"
data =pd.read_csv(path)
pd.set_option("display.max_rows",None)


# %%
from sklearn.preprocessing import LabelEncoder

for col in data.select_dtypes(include="object"):
    le =LabelEncoder()
    data[col] = le.fit_transform(data[col])

# print(data.head(5))    

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,35))
sns.jointplot(data=data,y="Session_Duration_Avg",x ="Age",hue ="Churned")
plt.show()

# %%
plt.figure(figsize=(10,5))
sns.jointplot(data=data,y="Wishlist_Items",x ="Total_Purchases",hue ="Churned")
plt.show()

# %%
plt.figure(figsize=(10,5))
sns.jointplot(data=data,y="Login_Frequency",x ="Membership_Years",hue ="Churned")
plt.show()

# %%
print(data.isnull().sum())
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Session_Duration_Avg"] = data["Session_Duration_Avg"].fillna(data["Session_Duration_Avg"].median())
data["Pages_Per_Session"] = data["Pages_Per_Session"].fillna(data["Pages_Per_Session"].median())
data["Wishlist_Items"] = data["Wishlist_Items"].fillna(0)
data["Days_Since_Last_Purchase"] = data["Days_Since_Last_Purchase"].fillna(data["Days_Since_Last_Purchase"].max())
data["Returns_Rate"] = data["Returns_Rate"].fillna(data["Returns_Rate"].median())
data["Email_Open_Rate"] = data["Email_Open_Rate"].fillna(0)
data["Discount_Usage_Rate"] = data["Discount_Usage_Rate"].fillna(0)
data["Customer_Service_Calls"] = data["Customer_Service_Calls"].fillna(0)
data["Product_Reviews_Written"] = data["Product_Reviews_Written"].fillna(data["Product_Reviews_Written"].median())
data["Social_Media_Engagement_Score"] = data["Social_Media_Engagement_Score"].fillna(data["Social_Media_Engagement_Score"].median())
data["Mobile_App_Usage"] = data["Mobile_App_Usage"].fillna(data["Mobile_App_Usage"].median())
data["Payment_Method_Diversity"] = data["Payment_Method_Diversity"].fillna(0)
data["Credit_Balance"] = data["Credit_Balance"].fillna(data["Credit_Balance"].median())




# %%
corr = data.corr()["Churned"].sort_values(ascending=True)
print(corr)

# %%
plt.figure(figsize=(10,5))
sns.heatmap(data=data.corr())
plt.show()

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score


# %%
data["Activity_Intensity"] = data["Pages_Per_Session"] * data["Login_Frequency"]
data["Value_Per_Session"] = data["Lifetime_Value"] / (data["Login_Frequency"] + 1)
data["Purchase_Frequency"] = data["Total_Purchases"] / (data["Membership_Years"] + 0.1)

drop = [
    'Gender',
    'Membership_Years',
    'City',
    'Country',
    'Churned'
]

    # 'Age',
    # 'Wishlist_Items',
    # 'Session_Duration_Avg',
    # 'Lifetime_Value',
    # 'Login_Frequency',
    # 'Email_Open_Rate',
    # 'Credit_Balance',
    # 'Membership_Years',
    # 'City',
    # 'Returns_Rate',
    # 'Payment_Method_Diversity',
    # 'Product_Reviews_Written',
    # 'Country',
    # 'Days_Since_Last_Purchase',
    # 'Cart_Abandonment_Rate',
    # 'Discount_Usage_Rate',
    # 'Total_Purchases',
    # 'Signup_Quarter'
    
X =data.drop(columns=drop,errors='ignore')


from sklearn.preprocessing import StandardScaler

y= data["Churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF =RandomForestClassifier(n_estimators=300)
RF.fit(X_train,y_train)

prediction = RF.predict(X_test)

print(classification_report(y_pred=prediction,y_true=y_test))



# %%
from sklearn.model_selection import cross_val_score

cv = cross_val_score(RF,X,y,cv =5)
print(cv)

cmf = confusion_matrix(y_test, prediction)
print(cmf)
plt.figure(figsize=(10,5))
sns.heatmap(data=cmf,annot=True)

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RF,
    param_grid=param_grid,
    cv=5,                    
    scoring='f1_macro',      
    n_jobs=-1,                
    verbose=2
)

grid_search.fit(X_train,y_train)
print("Best Params:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print(classification_report(y_test, y_pred))



