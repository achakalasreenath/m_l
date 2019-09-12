import random

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, SGDRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pprint as pp
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = []
non_numerical_columns = []
estimator = LinearRegression()


def days(start_and_end_dates):
    return (start_and_end_dates[1] - start_and_end_dates[0]).days


def pre_processing(data):
    data = data.drop(
        columns=["id1", "type", "total_floor_count", "floor_no", "furnished", "price_currency", "tom"], axis=1
    )
    print(data.info())
    for column in data.columns:
        print("Missing Value percentage in {} is {}".format(column,
                                                            data[column].isnull().sum() * 100 / len(data[column])))

    #data["start_date"] = data["start_date"].fillna("8/31/2018")
    #data["end_date"] = data["end_date"].fillna("9/3/2019")
    data = data.dropna(axis = 0, how = "any", subset=["start_date","end_date"])

    start_dates = [datetime.strptime(date, "%m/%d/%Y") for date in data["start_date"]]
    end_dates = [datetime.strptime(date, "%m/%d/%Y") for date in data["end_date"]]
    zip_dates = list(zip(start_dates, end_dates))

    data["No Of Days"] = list(map(days, zip_dates))
    data = data.drop(["end_date"], axis=1)
    #data["building_age"] = data["building_age"].fillna(data["building_age"].mode())

    data["building_age"] = [item.split(" ")[0] for item in np.array(data["building_age"],dtype = str)]
    data["building_age"] = [item.split("-")[0] for item in data["building_age"].values]

    print(data["building_age"])
    data["building_age"] = pd.to_numeric(data["building_age"], errors="coerce")
    print(data["building_age"])

    global numerical_columns
    global non_numerical_columns
    numerical_columns = list(data.select_dtypes(include=num_types).columns)
    numerical_columns.remove("listing_type")
    non_numerical_columns = [column for column in data.columns if column not in numerical_columns]
    non_numerical_columns.append("listing_type")

    # data = data.fillna({numerical_columns[0]:random.uniform(data[numerical_columns[0]].values.min(),data[numerical_columns[0]].values.max())
    #                     ,numerical_columns[1]:random.uniform(data[numerical_columns[1]].values.min(),data[numerical_columns[1]].values.max())
    #                     ,numerical_columns[2]:random.uniform(data[numerical_columns[2]].values.min(),data[numerical_columns[2]].values.max())
    #                     ,numerical_columns[3]:random.uniform(data[numerical_columns[3]].values.min(),data[numerical_columns[3]].values.max())
    #                     })

    data = data.fillna({numerical_columns[0]:data[numerical_columns[0]].mean()
                        ,numerical_columns[1]:data[numerical_columns[1]].mean()
                        ,numerical_columns[2]:data[numerical_columns[2]].mean()
                        ,numerical_columns[3]:data[numerical_columns[3]].mean()
                        })
    data = data.fillna({non_numerical_columns[0]:data[non_numerical_columns[0]].mode(),
                        non_numerical_columns[1]:data[non_numerical_columns[1]].mode(),
                        non_numerical_columns[2]:data[non_numerical_columns[2]].mode(),
                        non_numerical_columns[3]:data[non_numerical_columns[3]].mode(),
                        non_numerical_columns[4]:data[non_numerical_columns[4]].mode(),
                        non_numerical_columns[5]:data[non_numerical_columns[5]].mode(),
                        non_numerical_columns[6]:data[non_numerical_columns[6]].mode()})



    # for column in numerical_columns:
    #     data[column].fillna(axis=0,value=random.uniform(data[column].values.min(),data[column].values.max()),inplace=True)
    #     print(data[column]._is_view)
    # for column in non_numerical_columns:
    #     data[column].fillna(axis = 0,value=data[column].mode(),inplace=True)
    #     print(data[column]._is_view)


    # for column in numerical_columns:
    #     data[column] = preprocessing.scale(data[column])
    data = pd.get_dummies(data, columns=non_numerical_columns)
    data["price"] = [item/data["price"].mean() for item in data["price"].values]
    Y = data["No Of Days"]
    X = data.drop(["No Of Days"], axis=1)

    return X,Y


def plot_learning_curves(X,Y):
    train_sizes, train_scores, test_scores = learning_curve(estimator,X,Y,scoring="neg_mean_squared_error")

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, -train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, -test_mean, color="#111111", label="Cross-validation score")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("neg_mean_squared_error"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def create_polynomial_features(X):
    polynomial_features = PolynomialFeatures(10)
    numerical_columns.remove("No Of Days")
    X_num = X[numerical_columns]
    X_array = polynomial_features.fit_transform(X_num)
    X_df = pd.DataFrame(X_array)
    non_num = list(set(X.columns) - set(numerical_columns))
    X_non_num = X[non_num]
    X = X_non_num.join(X_df, X_non_num.index == X_df.index)
    return X


if __name__ == "__main__":

    data_train = pd.read_csv(r"D:\MOVE\Training\datasets\real_estate_data.csv", nrows=10000)
    data_test = pd.read_csv(r"D:\MOVE\Training\datasets\real_estate_data_test.csv")
    X,Y = pre_processing(data_train)
    #X = create_polynomial_features(X)
    #X_test,Y_test = pre_processing(data_test)
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=41)
    model = estimator
    model.fit(X_train, Y_train)
    MSE = []
    for example in x_test.itertuples():
        index = example[0]
        values = example[1:]
        y_pred = model.predict(np.array(values).reshape(1, -1))
        error = ((y_test[index] - y_pred) * (y_test[index] - y_pred) / len(x_test))
        MSE.append([values[:4],error])


    ypred = model.predict(x_test)
    print("mean squared error",metrics.mean_squared_error(y_test,ypred))
    #plot_learning_curves(X,Y)
    print("r2 score", model.score(x_test,y_test))







