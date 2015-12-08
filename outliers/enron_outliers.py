#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
from pandas import DataFrame
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]
data = DataFrame(featureFormat(data_dict, features))

matplotlib.pyplot.scatter(data[0], data[1])
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


# Find salary outliers
max = 0
for k,v in data_dict.iteritems():
    if not isinstance(v["salary"],str) and not isinstance(v["bonus"],str) and v["salary"] > max:
        max = v["salary"]
        print k




