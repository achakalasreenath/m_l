import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute._iterative import IterativeImputer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from yellowbrick.features.importances import FeatureImportances

data = pd.read_csv("./datasets/BNP_cardiff_claim_management_train.csv", nrows=20000)
target_data = data["target"]
feature_data = data.drop(["target"],axis = 1)
columns = feature_data.columns

#feature_data = feature_data.dropna()
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(feature_data.select_dtypes(include=num_types).columns)
non_numerical_columns = [column for column in feature_data.columns if column not in numerical_columns]
feature_data = pd.get_dummies(feature_data, columns=non_numerical_columns)
simple_imputer = IterativeImputer()
imputed_data = simple_imputer.fit_transform(feature_data)
feature_data = pd.DataFrame(imputed_data,columns = columns)
X_train, x_test, Y_train, y_test = train_test_split(feature_data, target_data, test_size=0.4,random_state=41)

model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model.fit(X_train,Y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))