import pandas as pd
import numpy as np
from pandas import Index
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier




def filtering_constant_quasi_constantfeatures_and_duplicate():
    data = pd.read_csv("./datasets/BNP_cardiff_claim_management_train.csv", nrows=20000)
    #data = pd.read_csv("./datasets/Customer Satisfaction_train.csv", nrows=40000)

    #preprocessing
    data = data.dropna(axis=0)
    num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = list(data.select_dtypes(include=num_types).columns)
    non_numerical_columns = [column for column in data.columns if column not in numerical_columns]
    # One Hot Encoding
    #data = pd.get_dummies(data, columns=non_numerical_columns)
    data = data[numerical_columns]



    # Before feature selection
    X_train, x_test, Y_train, y_test = train_test_split(data.drop(['target'], axis=1), data["target"], test_size=0.2,random_state=41)
    model1 = DecisionTreeClassifier(criterion="entropy")

    model1.fit(X_train, Y_train)
    y_pred = model1.predict(x_test)
    print("accuracy score before feature selection", accuracy_score(y_test, y_pred))


    # filters the features that are constant irrespective of the output i.e.. with a variance of 0
    # features with variance > 0 are kept
    # if you use threshold 0.01, quasi-constant features will be removed. features that are 99% similiar throughout the dataset
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    # Removing the features with variance <= 0
    # X_train = constant_filter.transform(X_train)
    # x_test = constant_filter.transform(x_test)
    constant_columns = [column for column in X_train.columns if
                        constant_filter.get_support()[X_train.columns.get_loc(column)] == False]


    X_train = X_train.drop(constant_columns, axis=1)
    print("Before Constant Feature Removal", len(data.columns))
    print("After Constant Feature Removal", len(X_train.columns))
    #model5 = DecisionTreeClassifier(criterion="entropy")
    model1.fit(X_train, Y_train)
    x_test = x_test.drop(constant_columns, axis=1)
    y_pred = model1.predict(x_test)
    print("Accuracy score after constant  feature removal", accuracy_score(y_test, y_pred))


    # quasi-contant filter
    qconstant_filter = VarianceThreshold(threshold=0.001)
    qconstant_filter.fit(X_train)
    qconstant_columns = [column for column in X_train.columns if
                         qconstant_filter.get_support()[X_train.columns.get_loc(column)] == False]


    X_train = X_train.drop(qconstant_columns, axis=1)
    print("After Quasi-Constant Feature Removal", len(X_train.columns))
    #model6 = DecisionTreeClassifier(criterion="entropy")
    model1.fit(X_train, Y_train)
    x_test = x_test.drop(qconstant_columns, axis=1)
    y_pred = model1.predict(x_test)
    print("accuracy score after quasi constant feature removal", accuracy_score(y_test, y_pred))


    # Removing duplicate features
    X_train_T = X_train.T
    # returns a boolean series with False corresponding to unique rows and True to Duplicate rows
    # Drop all the duplicates and keep the first copy
    duplicated = X_train_T.duplicated(keep="first")
    duplicate_columns = X_train_T[duplicated].T.columns
    # returns only unique rows
    X_train_T = X_train_T[~duplicated]
    X_train = X_train_T.T
    print("After duplicate Feature Removal", len(X_train.columns))


    #model2 = DecisionTreeClassifier(criterion="entropy")
    model1.fit(X_train, Y_train)
    # removing columns from the test set
    x_test = x_test.drop(list(duplicate_columns), axis=1)
    y_pred= model1.predict(x_test)
    print("accuracy score after duplicates removal", accuracy_score(y_test, y_pred))


# removing correlated featured
    #model3 = DecisionTreeClassifier(criterion="entropy")
    X_train, x_test, Y_train, y_test = train_test_split(data.drop(['target'], axis=1), data["target"], test_size=0.1,
                                                        random_state=41)
    model1.fit(X_train, Y_train)
    y_pred = model1.predict(x_test)
    print("accuracy score before removing correlated features", accuracy_score(y_test, y_pred))
    correlation_matrix = X_train.corr(method="pearson")
    correlated_features = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            if (i != j) and (correlation_matrix.iloc[i, j] > 0.8):
                correlated_features.append(correlation_matrix.columns[i])

    X_train = X_train.drop(correlated_features, axis=1)
    x_test = x_test.drop(correlated_features, axis=1)
    #model4 = DecisionTreeClassifier(criterion="entropy")
    model1.fit(X_train, Y_train)
    y_pred = model1.predict(x_test)
    print("accuracy score before removing correlated features", accuracy_score(y_test, y_pred))
