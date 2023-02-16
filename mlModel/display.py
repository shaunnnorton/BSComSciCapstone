from sklearn.metrics import PredictionErrorDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import plot_tree
from matplotlib import pyplot

from joblib import load

# Load the current model from memory
data = load("./Models/0004.joblib")
model = data["model"]
X_test = data["X_test"]
y_test = data["y_test"]

# Print the dataset columns and the models score to the console. 
print(X_test.columns.values)
print(model.score(X_test,y_test))

# Create a model of the fisrt three levels of the decision tree. 
fig = pyplot.figure(figsize=(4,4),dpi=1000)
plot_tree(model['decisiontreeregressor'],filled=True,max_depth=3,rounded=True,feature_names=X_test.columns)
# Save the decisont tree modeld to a png file.
pyplot.savefig("./Figures/test0088.png")




# Create plots fof predicion error display and Pardital dependency display
svr_display = PredictionErrorDisplay.from_estimator(model,X_test,y_test)
pdd = PartialDependenceDisplay.from_estimator(model,data["X_train"],X_test.columns.values)

# Display plots 
ax = pyplot.gca()
svr_display.plot(ax=ax)
ax2 = pyplot.gca()
pdd.plot(ax=ax2)

# Show all plots
pyplot.show()



