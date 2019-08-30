import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from yellowbrick.features.importances import FeatureImportances
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("./datasets/BNP_cardiff_claim_management_train.csv", nrows=20000)

data = data.dropna(axis=0)
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(data.select_dtypes(include=num_types).columns)
non_numerical_columns = [column for column in data.columns if column not in numerical_columns]
data = pd.get_dummies(data, columns=non_numerical_columns)
X_train, x_test, Y_train, y_test = train_test_split(data.drop(['target'], axis=1), data["target"], test_size=0.2,random_state=41)

model = LogisticRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))
rfe = RFE(model,200,step =50)
rfe.fit(X_train,Y_train)
y_pred = rfe.predict(x_test)
print(accuracy_score(y_test,y_pred))
fig = plt.figure()
ax = fig.add_subplot()
viz = FeatureImportances(model, ax = ax, labels = X_train.columns,relative = False)
viz.fit(X_train,Y_train)
viz.poof()