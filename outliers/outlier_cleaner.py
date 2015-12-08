#!/usr/bin/python

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

    residuals = (net_worths - predictions)**2
    cleaned_data = sorted(zip(ages, net_worths, residuals), key=itemgetter(2))

    return cleaned_data[:int(round(len(cleaned_data)*.9))]

