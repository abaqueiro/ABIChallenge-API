# data analysis and wrangling
import pandas as pd
import pickle

# machine learning
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

## Acquire data
X_train = pd.read_pickle("../data/X_train.pkl")
Y_train = pd.read_pickle("../data/Y_train.pkl")
X_test  = pd.read_pickle("../data/X_test.pkl")

# load the model from disk
filename = 'random_forest.sav'
loaded_model = pickle.load(open(filename, 'rb'))
Y_pred = loaded_model.predict(X_test)
acc_random_forest = round(loaded_model.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)