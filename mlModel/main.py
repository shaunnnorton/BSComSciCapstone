from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error


from pandas.plotting import scatter_matrix
from matplotlib import pyplot


import numpy as np
import pandas as pd

df = pd.read_csv("mlModel/modelData/transcoding_mesurment.tsv",sep="\t")
df.drop("id",inplace= True, axis=1)
df.drop("umem",inplace= True, axis=1)





samples, features = 14000,18

X = df.drop("utime",axis=1)
Y = df["utime"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=69)

transformer = make_column_transformer((OneHotEncoder(), ["codec", "o_codec"]),remainder="passthrough")

training_transformed= transformer.fit_transform(X_train)
training_transformed_df = pd.DataFrame(training_transformed,
                                       columns=transformer.get_feature_names_out())



enc = OneHotEncoder()


model = make_pipeline(StandardScaler(with_mean=False),SVR(kernel="linear",verbose=True),verbose=True)
model.fit(training_transformed_df,y_train)

