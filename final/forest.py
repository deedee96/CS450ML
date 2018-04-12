from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
import numpy as np
import pandas as pd 


df = pd.read_csv("./john_is_an_idiot.csv")

df = df.drop(["created", "vantage_text", "dispo_text", "interviewee_id", "interview_id", "new_id", "X1", "completed", "hour"], axis=1)
df.dispo_id = df.dispo_id.fillna(0)

y = df["shift"]
x = df.drop(["shift"], axis=1)
# print(x.head())
# print(y.head())

# # na's? 
# print(df[df.isnull().any(axis=1)])

# x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3)

# clf = RandomForestClassifier(n_estimators=200)
# kfold = model_selection.KFold(n_splits=5, random_state=7)
# results = model_selection.cross_val_score(clf, x, y, cv=kfold)

#clf = KNeighborsClassifier(n_neighbors=5)
#kfold = model_selection.KFold(n_splits=5, random_state=7)
#results = model_selection.cross_val_score(clf, x, y, cv=kfold)


clf = SVC()
kfold = model_selection.KFold(n_splits=5, random_state=7)
results = model_selection.cross_val_score(clf, x, y, cv=kfold)
#clf.fit(X, y) 
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
 #   max_iter=-1, probability=False, random_state=None, shrinking=True,
  #  tol=0.001, verbose=False)

print(np.mean(results))

