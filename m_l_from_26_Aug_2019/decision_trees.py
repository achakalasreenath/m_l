import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
#print(data.target_names)
X = data.data
Y = data.target
X_train,x_test,Y_train,y_test = train_test_split(X,Y,random_state = 47, test_size = 0.25)

# measure to calculate the information gain
clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X_train,Y_train)
train_accuracy_score = accuracy_score(y_true = Y_train,y_pred = clf.predict(X_train))
test_accuracy_score = accuracy_score(y_true = y_test, y_pred = clf.predict(x_test))
print("training",train_accuracy_score,"test",test_accuracy_score)
# 'min_sample_split' --> min number of samples required to perform a split at a node, default value =2,
# tuning this values prevents overfitting
clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=50)
clf.fit(X_train,Y_train)
train_accuracy_score = accuracy_score(y_true = Y_train,y_pred = clf.predict(X_train))
test_accuracy_score = accuracy_score(y_true = y_test, y_pred = clf.predict(x_test))
print("Training",train_accuracy_score,"test",test_accuracy_score)
