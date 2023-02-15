from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, PredictionErrorDisplay, RocCurveDisplay
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge

from pandas.plotting import scatter_matrix 
from matplotlib import pyplot


import numpy as np
import pandas as pd

df = pd.read_csv("mlModel/modelData/transcoding_mesurment.tsv",sep="\t")
df.drop("id",inplace= True, axis=1)
df.drop("umem",inplace= True, axis=1)

#df.hist()
#scatter_matrix(df)
#pyplot.show()




# samples, features = 14000,18

X = df.drop("utime",axis=1)
Y = df["utime"]

transformer = make_column_transformer((OneHotEncoder(), ["codec", "o_codec"]),remainder="passthrough")
transformed= transformer.fit_transform(X)
transformed_df = pd.DataFrame(transformed,
                            columns=transformer.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(
    transformed_df, Y, test_size=0.3, random_state=69)



# scatter_matrix(training_transformed_df[])
# pyplot.show()

# model = KernelRidge(alpha=1.0)
# model.fit(training_transformed_df,y_train)

# testing_transformed= transformer.fit_transform(X_test)
# testing_transformed_df = pd.DataFrame(testing_transformed,
#                                        columns=transformer.get_feature_names_out())



# enc = OneHotEncoder()


model = make_pipeline(StandardScaler(with_mean=False),BayesianRidge(verbose=True),verbose=True)
model.fit(X_train,y_train)


from joblib import dump

dump({"model":model,
      "X_train":X_train,
      "X_test":X_test, 
      "y_train":y_train, 
      "y_test":y_test}, "./Models/0002.joblib")


print(model.score(X_test,y_test))
print(model.get_params())
