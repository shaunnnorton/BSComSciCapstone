from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeRegressor

import pandas as pd


# Extracting the data from the dataset. 
# Dataset from the University of California Irving Machine Learning Repository
# Dataset found at https://archive.ics.uci.edu/ml/datasets/Online+Video+Characteristics+and+Transcoding+Time+Dataset
df = pd.read_csv("mlModel/modelData/transcoding_mesurment.tsv",sep="\t")
df.drop("id",inplace= True, axis=1) # Drop the Id column as it is not necessary. 
df.drop("umem",inplace= True, axis=1)  # Drop the umen coluumn as it is not necessary


X = df.drop(["utime","i","p","b","i_size","p_size","b_size"],axis=1)  # Create the input dataset
Y = df["utime"] # Output dataset

# Transform the codec using onehotencoding to make it usable as it is categorical
transformer = make_column_transformer((OneHotEncoder(), ["codec", "o_codec"]),remainder="passthrough")
transformed= transformer.fit_transform(X)
transformed_df = pd.DataFrame(transformed,
                            columns=transformer.get_feature_names_out())

# Split the dataset into training and test sets using a 70-30 split. 
X_train, X_test, y_train, y_test = train_test_split(
    transformed_df, Y, test_size=0.3, random_state=69)


#Create and fit the model using a Pipeline
model = make_pipeline(StandardScaler(with_mean=False),DecisionTreeRegressor(max_leaf_nodes=100),verbose=True)
model.fit(X_train,y_train)


# Store the model using joblib to save future computation
from joblib import dump

# Save the model and dataset 
dump({"model":model,
      "X_train":X_train,
      "X_test":X_test, 
      "y_train":y_train, 
      "y_test":y_test}, "./Models/0005.joblib")


# Print model information after creation including score. 
print(model.score(X_test,y_test))
print(model.get_params())
