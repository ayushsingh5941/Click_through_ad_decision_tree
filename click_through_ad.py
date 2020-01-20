# categorical and numerical data :
# categorical : represents characteristics is not numeric, it can take numerical value
# eg. 12 can represent months
# Numerical : represents mathematical meaning
# naive bayes can handle both data and SVM require only numerical data
# let's start reading datasets4
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
# from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

file = pd.read_csv('train.csv', nrows=100000)
y_train = file['click']
discarded_columns = ['id', 'click', 'hour', 'device_id', 'device_ip']
columns = list(file.columns[i] for i in range(len(file.columns)) if file.columns[i] not in discarded_columns)
x_train_dict = file[columns].to_dict(orient='records')
# Now we need to transform this dictionary object in ONE HOT ENCODING vectors using Dictvectorizer
# hot encoding convert categorical features with k possible values into binary features
dict_vectorizer = DictVectorizer(sparse=False)
x_train = dict_vectorizer.fit_transform(x_train_dict)
print(len(x_train[0]))

# Simiilarly grab test.csv

file = pd.read_csv('train.csv', nrows=100000)
y_test = file['click']
x_test_dict = file[columns].to_dict(orient='records')
x_test = dict_vectorizer.transform(x_test_dict)
print(len(x_test[0]))

# Now we need to train decision tree using grid search cv, classification metrics should be Auc Roc
# As imbalance in binary case clicked
# Decision trees are prone over fitting, it uses bagging technique
# tree bagging, reduces high variance
parameters = {'max_depth': [3, 10, None]}
# decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
grid_cv = GridSearchCV(random_forest, parameters, cv=3, scoring='roc_auc')
grid_cv.fit(x_train, y_train)
print(grid_cv.best_params_)
# using best estimator
best_estimator = grid_cv.best_estimator_
predict_prob = best_estimator.predict_prob(x_test)[:, 1]
print('Roc and Auc' + roc_auc_score(y_test, predict_prob))
