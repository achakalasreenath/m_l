import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from pyspark.sql import SparkSession, SQLContext, Column
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from yellowbrick.model_selection import FeatureImportances


def no_of_days(start_date_list_with_id, end_date_list_with_id):
    start_date_num_list = [(item[0], datetime.strptime(item[1], '%m/%d/%Y')) for item in start_date_list_with_id]
    end_date_num_list = [(item[0], datetime.strptime(item[1], '%m/%d/%Y')) for item in end_date_list_with_id]
    number_of_days = [(item[0][0], (item[1][1] - item[0][1]).days) for item in
                      list(zip(start_date_num_list, end_date_num_list))]
    return number_of_days, start_date_num_list, end_date_num_list


sparkSession = SparkSession.builder.appName("Analysing real_estate_data").getOrCreate()
data_spark = sparkSession.read.format("csv").option("header", "true").load(
    r"D:\MOVE\Training\datasets\real_estate_data.csv")
sqlContext = SQLContext(sparkSession)
data_spark = sqlContext.createDataFrame(data_spark.head(10000), schema=data_spark.schema)
data_spark = data_spark.drop(
    "type", "total_floor_count", "floor_no", "furnished", "price_currency",
)

simple_imputer = SimpleImputer(strategy="constant", missing_values=None, fill_value="8/28/2019")
simple_imputer_2 = SimpleImputer(strategy="constant", missing_values=None, fill_value="8/31/2018")

columns = data_spark.columns

data_spark_start_dates = simple_imputer_2.fit_transform(
    np.array(data_spark.select("id1", "start_date").collect(), dtype='object'))
data_spark_end_dates = simple_imputer.fit_transform(
    np.array(data_spark.select("id1", "end_date").collect(), dtype='object'))

data_spark_start_dates_df = sqlContext.createDataFrame(data_spark_start_dates.tolist(),
                                                       schema=StructType([StructField("id1", StringType(), False),
                                                                          StructField("start_date", StringType(),
                                                                                      False)]))
data_spark_end_dates_df = sqlContext.createDataFrame(data_spark_end_dates.tolist(),
                                                     schema=StructType([StructField("id1", StringType(), False),
                                                                        StructField("end_date", StringType(), False)]))
data_spark = data_spark.drop("start_date", "end_date")
data_spark = data_spark.join(data_spark_start_dates_df, ["id1"])
data_spark = data_spark.join(data_spark_end_dates_df, ["id1"])

start_date_list_with_id = [(item["id1"], item["start_date"]) for item in data_spark.toLocalIterator()]
end_date_list_with_id = [(item["id1"], item["end_date"]) for item in data_spark.toLocalIterator()]
num_of_days, start_date_num_list, end_date_num_list = no_of_days(start_date_list_with_id, end_date_list_with_id)

num_of_days_df = sqlContext.createDataFrame(num_of_days,
                                            schema=StructType([StructField("id1", StringType(), False),
                                                               StructField("Number Of Days", IntegerType(), False)]))
# #start_date_num_list_df = sqlContext.createDataFrame(start_date_num_list,
#                                                     schema=StructType([StructField("id1", StringType(), False),
#                                                                        StructField("Start Date TimeStamp",
#                                                                                    TimestampType(), False)]))
# #end_date_num_list_df = sqlContext.createDataFrame(end_date_num_list,
#                                                   schema=StructType([StructField("id1", StringType(), False),
#                                                                      StructField("End Date TimeStamp", TimestampType(),
#                                                                                  False)]))

data_spark = data_spark.join(num_of_days_df, ["id1"])
# data_spark = data_spark.join(start_date_num_list_df, ["id1"])
# data_spark = data_spark.join(end_date_num_list_df, ["id1"])

data_spark = data_spark.toPandas()
data_spark["price"] = pd.to_numeric(data_spark["price"], errors="coerce")
data_spark["size"] = pd.to_numeric(data_spark["size"], errors="coerce")
data_spark["building_age"] = pd.to_numeric(data_spark["building_age"], errors="coerce")
data_spark["tom"] = pd.to_numeric(data_spark["tom"], errors="coerce")
# data_spark["listing_type"] = pd.to_numeric(data_spark["listing_type"],errors ="coerce")


#data_spark = data_spark.drop(["id1"], axis=1)
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(data_spark.select_dtypes(include=num_types).columns)
# numerical_columns.extend(["Start Date TimeStamp","End Date TimeStamp"])
non_numerical_columns = [column for column in data_spark.columns if column not in numerical_columns]

# data_spark = data_spark.fillna({'sub_type':data_spark['sub_type'].mode(),
#                                 'listing_type':data_spark['listing_type'].mode(),
#                                 'tom':data_spark['tom'].mean(axis = 0,skipna=True),
#                                 'building_age':data_spark['building_age'].mean(axis = 0,skipna=True),
#                                 'room_count':data_spark['room_count'].mode(),
#                                 'size':data_spark['size'].mean(axis = 0,skipna=True),
#                                 'price':data_spark['price'].mean(axis = 0,skipna=True),
#                                 'address':data_spark['address'].mode(),
#                                 'heating_type':data_spark['heating_type'].mode()})

for column in numerical_columns:
    data_spark = data_spark.fillna(data_spark[column].values.mean())
for column in non_numerical_columns:
    data_spark = data_spark.fillna(data_spark[column].mode())


Y = preprocessing.scale(data_spark["Number Of Days"])
data_spark_poly = data_spark[numerical_columns]
polynomial_features = PolynomialFeatures(2)
data_spark_poly_array = polynomial_features.fit_transform(np.array(data_spark_poly))
data_spark_poly_df = pd.DataFrame(data_spark_poly_array)
data_spark = data_spark[non_numerical_columns].join(data_spark_poly_df,data_spark[non_numerical_columns].index == data_spark_poly_df.index)


data_spark = pd.get_dummies(data_spark, columns=non_numerical_columns)

# data_spark["price"] = preprocessing.scale(data_spark["price"])


for column in data_spark.columns:
    if column != "id1":
        data_spark[column] = preprocessing.scale(data_spark[column])
X = data_spark

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

# model = SGDRegressor(loss="squared_loss", penalty="elasticnet", alpha=0.00000001, l1_ratio=0.5, fit_intercept=True)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(x_test)
print(model.score(x_test, y_test))
train_sizes, train_scores, test_scores = learning_curve(LinearRegression(),
                                                        X,
                                                        Y)


# Create means and standard deviations of training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Create means and standard deviations of test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Draw lines
plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

# Draw bands
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

# Create plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
# plt.plot(y_pred[100:200], label="predicted")
# plt.plot(y_test[100:200], label="actual")
# plt.title("AdaBoostRegressor  with scaling, score = {} with features imputed".format(model.score(x_test, y_test)))
# plt.legend()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# viz = FeatureImportances(model,ax = ax )
# viz.fit(X_train[:][:4], Y_train)
# viz.poof()
