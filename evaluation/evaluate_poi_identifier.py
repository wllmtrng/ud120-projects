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

# From Python 3.3 forward, a change to the order in which dictionary keys are
# processed was made such that the orders are randomized each time the code is
# run. This will cause some compatibility problems with the graders and project
#  code, which were run under Python 2.7. To correct for this, add the following
#  argument to the featureFormat call on line 25 of evaluate_poi_identifier.py:
#
# sort_keys = '../tools/python2_lesson14_keys.pkl'
#
# This will open up a file in the tools folder with the Python 2 key order.
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


