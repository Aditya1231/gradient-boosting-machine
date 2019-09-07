# --------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


data = pd.read_csv(path)

# Explore the data 
data.sex.value_counts()
data[data['sex']=='Female'].age.mean()
total_len = len(data)
german = (data['native-country']=='Germany').sum()
print(german/total_len*100)

# mean and standard deviation of their age

ages1 = data[data['salary']=='<=50K'].age
ages2 = data[data['salary']=='>50K'].age

print("more: mean, sd", ages1.mean(), ages1.std())
print("less: mean, sd", ages2.mean(), ages2.std())

# Display the statistics of age for each gender of all the races (race feature).
data.groupby(['race','sex']).age.describe()

# encoding the categorical features.
data.salary = data.salary.apply(lambda x:1 if x=='<=50K' else 0)

# Split the data and apply decision tree classifier
X = pd.get_dummies(data.drop('salary',axis=1))
y = data.salary

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=12)
Xtrain1, Xval, ytrain1, yval = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=12)

clf_dt = DecisionTreeClassifier()
clf_dt.fit(Xtrain1, ytrain1)
y_val_score = clf_dt.score(Xval, yval)
y_test_score = clf_dt.score(Xtest, ytest)
print(y_val_score, y_test_score)

# Perform the boosting task

model1 = DecisionTreeClassifier()
model2 = LogisticRegression()
#clf_vt = VotingClassifier(estimators=[('dt', model1), ('lr', model2)], voting='soft')


clf_vt = VotingClassifier(estimators = [('dt',model1),('lr',model2)],voting='soft')
clf_vt.fit(Xtrain1, ytrain1)
clf_vt_y_val_score = clf_vt.score(Xval, yval)
clf_vt_y_test_score = clf_vt.score(Xtest, ytest)
print(clf_vt_y_val_score, clf_vt_y_test_score)


clf_vt2 = VotingClassifier(estimators=[('dt', model1), ('lr', model2)], voting='soft')
clf_vt2.fit(Xtrain, ytrain)
#clf_vt_y_val_score = clf_vt.score(Xval, yval)
clf_vt2_y_test_score = clf_vt2.score(Xtest, ytest)
print(#clf_vt_y_val_score,
clf_vt2_y_test_score)

#  plot a bar plot of the model's top 10 features with it's feature importance score
from sklearn.ensemble import GradientBoostingClassifier
model_10  = GradientBoostingClassifier(n_estimators=10, max_depth=6, subsample=0.8, random_state=12)
model_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6, subsample=0.8, random_state=12)
model_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6, subsample=0.8, random_state=12)


model_10.fit(Xtrain1, ytrain1)
model_50.fit(Xtrain1, ytrain1)
model_100.fit(Xtrain1, ytrain1)

model_10_val_score = model_10.score(Xval, yval)
model_10_test_score = model_10.score(Xtest, ytest)

print(model_10_val_score, model_10_test_score )

model_50_val_score = model_50.score(Xval, yval)
model_50_test_score = model_50.score(Xtest, ytest)

print(model_50_val_score, model_50_test_score )

model_100_val_score = model_100.score(Xval, yval)
model_100_test_score = model_100.score(Xtest, ytest)

print(model_100_val_score, model_100_test_score)

data.shape

feat_imp = pd.DataFrame({"importance":model_50.feature_importances_}, index=Xtrain.columns)
feat_imp.sort_values(by="importance", ascending=False,inplace=True)
feat_imp[:10].sort_values(by="importance", ascending=True).plot.barh()

#  Plot the training and testing error vs. number of trees
train_err_10 = 1 - model_10.score(Xtrain1, ytrain1)
train_err_50 = 1 - model_50.score(Xtrain1, ytrain1)
train_err_100 = 1 - model_100.score(Xtrain1, ytrain1)
training_errors= [train_err_10, train_err_50, train_err_100]
validation_err_10 = 1 - model_10_val_score
validation_err_50 = 1 - model_50_val_score
validation_err_100 = 1 - model_100_val_score
validation_errors = [validation_err_10, validation_err_50,validation_err_100]

testing_err_10 = 1 - model_10_test_score
testing_err_50 = 1 - model_50_test_score
testing_err_100 = 1 - model_100_test_score
testing_errors = [testing_err_10, testing_err_50,testing_err_100]
n_trees = [10,50,100]
plt.plot(n_trees, training_errors, label="Training error")
plt.plot(n_trees, testing_errors, label="Testing error" )
plt.legend(["Training error", "Testing error"])
plt.title("Training vs Testing error by no. of trees")

n_trees = [10,50,100]
plt.plot(n_trees, training_errors, label="Training error")
plt.plot(n_trees, validation_errors, label="Validation error" )
plt.legend(["Training error", "Validation error"])
plt.title("Training vs Validation error by no. of trees")

n_trees = [10,50,100]
plt.plot(n_trees, training_errors, label="Training error")
plt.plot(n_trees, testing_errors, label="Testing error")
plt.plot(n_trees, validation_errors, label="Validation error" )
plt.legend(["Training error", "Testing error", "Validation error"])
plt.title("Training vs Testing vs Validation error by no. of trees")


