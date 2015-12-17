#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

i = 0
j = 0
for k in enron_data.keys():
	#print enron_data[k]
	if enron_data[k]["poi"] == True:
		if enron_data[k]["total_payments"] == 'NaN':
			i = i + 1
		j = j + 1
		
print i, j
print (i*100)/j