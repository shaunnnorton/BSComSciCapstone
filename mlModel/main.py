from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor

import pandas as pd

df = pd.read_csv("mlModel/modelData/transcoding_mesurment.tsv",sep="\t")
df.drop("id",inplace= True, axis=1)
df.drop("umem",inplace= True, axis=1)


X = df.drop(["utime","i","p","b","i_size","p_size","b_size"],axis=1)
Y = df["utime"]

transformer = make_column_transformer((OneHotEncoder(), ["codec", "o_codec"]),remainder="passthrough")
transformed= transformer.fit_transform(X)
transformed_df = pd.DataFrame(transformed,
                            columns=transformer.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(
    transformed_df, Y, test_size=0.3, random_state=69)



model = make_pipeline(StandardScaler(with_mean=False),DecisionTreeRegressor(max_leaf_nodes=100),verbose=True)
model.fit(X_train,y_train)

from joblib import dump

dump({"model":model,
      "X_train":X_train,
      "X_test":X_test, 
      "y_train":y_train, 
      "y_test":y_test}, "./Models/0005.joblib")


print(model.score(X_test,y_test))
print(model.get_params())
