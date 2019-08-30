import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
#print(data.target_names)
X = data.data
Y = data.target
X_train,x_test,Y_train,y_test = train_test_split(X,Y,random_state = 47, test_size = 0.25)


model1 = DecisionTreeClassifier(criterion = 'entropy')
model2 = LogisticRegression(random_state=1)
model3 = KNeighborsClassifier()

# MaxVoting Technique
model = VotingClassifier(estimators = [("lr", model2),("dt",model1)])
model.fit(X_train,Y_train)
train_accuracy_score = accuracy_score(y_true = Y_train,y_pred = model.predict(X_train))
test_accuracy_score = accuracy_score(y_true = y_test, y_pred = model.predict(x_test))
print("training",train_accuracy_score,"test",test_accuracy_score)

#Averaging
model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)

test_accuracy_score1 = accuracy_score(y_true = y_test, y_pred = model1.predict(x_test))
test_accuracy_score2 = accuracy_score(y_true = y_test, y_pred = model2.predict(x_test))
test_accuracy_score3 = accuracy_score(y_true = y_test, y_pred = model3.predict(x_test))
average_accuracy = (test_accuracy_score1 + test_accuracy_score2 + test_accuracy_score3 )/3
print("test",average_accuracy)