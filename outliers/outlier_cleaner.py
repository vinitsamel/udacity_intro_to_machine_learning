#!/usr/bin/python

import math
from operator import itemgetter

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    for i in range (0, len(predictions)):
		tup = (ages[i][0], net_worths[i][0], math.pow((net_worths[i] -  predictions[i]), 2))
		cleaned_data.append(tup)
    

    ### your code goes here

    print cleaned_data
    cleaned_data = sorted(cleaned_data, key=itemgetter(2), reverse=False)
    print cleaned_data
    return cleaned_data[0:81]

