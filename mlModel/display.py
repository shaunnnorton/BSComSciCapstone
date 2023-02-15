from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, PredictionErrorDisplay, RocCurveDisplay
from sklearn.kernel_ridge import KernelRidge
from sklearn.inspection import PartialDependenceDisplay



from pandas.plotting import scatter_matrix 
from matplotlib import pyplot


import numpy as np
import pandas as pd

from joblib import load

data = load("./Models/0002.joblib")
model = data["model"]
X_test = data["X_test"]
y_test = data["y_test"]

print(X_test.columns.values)
print(model.score(X_test,y_test))
svr_display = PredictionErrorDisplay.from_estimator(model,X_test,y_test)
#pdd = PartialDependenceDisplay.from_estimator(model,data["X_train"],["remainder__height"])

ax = pyplot.gca()
svr_display.plot(ax=ax)
pyplot.show()