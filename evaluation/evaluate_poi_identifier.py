#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score, precision_score, recall_score
print "Accuracy: " , accuracy_score(labels_test, pred)
tp = 0
fp = 0
fn = 0
for i in range(0,len(labels_test)):
    if (labels_test[i] == 1.0) and (pred[i] == 1.0):
        tp += 1
    if (labels_test[i] == 1.0) and (pred[i] == 0.0):
        fn += 1
    if (labels_test[i] == 0.0) and (pred[i] == 1.0):
        fp += 1
print tp, fp, fn

print precision_score(labels_test, pred)
print recall_score(labels_test, pred)