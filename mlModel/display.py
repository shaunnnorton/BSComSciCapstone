from sklearn.metrics import PredictionErrorDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import plot_tree
from matplotlib import pyplot

from joblib import load

data = load("./Models/0004.joblib")
model = data["model"]
X_test = data["X_test"]
y_test = data["y_test"]

print(X_test.columns.values)
print(model.score(X_test,y_test))

print(model.named_steps)
fig = pyplot.figure(figsize=(4,4),dpi=1000)
plot_tree(model['decisiontreeregressor'],filled=True,max_depth=3,rounded=True,feature_names=X_test.columns)

pyplot.savefig("./Figures/test0088.png")





# svr_display = PredictionErrorDisplay.from_estimator(model,X_test,y_test)
# pdd = PartialDependenceDisplay.from_estimator(model,data["X_train"],X_test.columns.values)

# ax = pyplot.gca()
# svr_display.plot(ax=ax)
# ax2 = pyplot.gca()
# pdd.plot(ax=ax2)

# pyplot.show()



