#!/usr/bin/python

import os
import sys
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

sys.path.append(os.getcwd() + "/../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'deferred_income', 'salary', 'bonus', 'total_stock_value', 'exercised_stock_options']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

my_dataset = data_dict
clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, features_list)
