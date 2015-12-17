#!/usr/bin/python
from __future__ import division
import sys
import pickle
import matplotlib.pyplot
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary']

features_list += ['bonus', 'exercised_stock_options', 'total_stock_value',                             'deferral_payments', 'total_payments','loan_advances', 'restricted_stock_deferred',
                  'deferred_income', 'expenses', 'long_term_incentive',
                  'restricted_stock', 'director_fees', 'other']
                  
#Features not used since we will create a combined feature based on these                  
#features_list += ['to_messages', 'from_poi_to_this_person', 'from_messages', 
#               'from_this_person_to_poi']

# You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
# Visualization to remove outlier from lesson 7
#features = ["salary", "bonus"]
#data = featureFormat(data_dict, features)
#for point in data:
#	print point
#	salary = point[0]
#	bonus  = point[1]
#	matplotlib.pyplot.scatter( salary, bonus )
#matplotlib.pyplot.xlabel("salary")
#matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()

### Task 3: Create new feature(s) (Lesson 11)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages == "NaN" or all_messages == "NaN":
         fraction = 0
    else:
        #print ( poi_messages, all_messages )
        fraction =  poi_messages/all_messages
    return fraction

for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi    

features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")

#Check one datapoint
#print data_dict["LAY KENNETH L"]

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Classifiers commented out were tried but did not give needed precision and recall
#from sklearn.naive_bayes import GaussianNB
#clfGNB = GaussianNB()

#from sklearn import svm
#clfSVC = svm.SVC() #kernel='rbf', C=10)
#Getting Error with SVM: Precision or recall may be undefined due to a lack of true positive predicitons.

from sklearn import tree
clfDT = tree.DecisionTreeClassifier()

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
clfADABoost = AdaBoostClassifier(base_estimator = clfDT)
#clfRandForest = RandomForestClassifier()

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
pca = PCA()
kselection = SelectKBest()

# Combine Features from PCA and kselect:
combined_features = FeatureUnion([("pca", pca), ("kselect", kselection)])
# FeatureScale, UseFeatures and Ensemble ADABOOST. DT by itself does not give good results.
estimators = [("minmax", preprocessing.MinMaxScaler()), ("cfeatures", combined_features), ('adaboost', clfADABoost)]
clfpipe = Pipeline(estimators)

#('dt', clfDT),

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# After trying several combinations the following met the criteria of Precision and Recall being atleast 0.3
from sklearn.grid_search import GridSearchCV
params = dict(cfeatures__pca__n_components=[1, 2, 5], \
cfeatures__kselect__k=[1, 2, 5], adaboost__base_estimator__min_samples_split=[10, 40], \
adaboost__n_estimators=[1, 2])
clf = GridSearchCV(clfpipe, param_grid=params)

#Good Info on Precision Recall
#http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

