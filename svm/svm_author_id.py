#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC


### features_train and features_test are the features for the training
### and Testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

t0 = time()
clf = SVC(kernel='linear').fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
print "Accuracy: ", clf.score(features_test, labels_test)
print "Testing time:", round(time()-t0, 3), "s"


print "Fitting Classifier to 10% of training set"
t0 = time()
clf = SVC(kernel='linear').fit(features_train[:len(features_train)/100],
                               labels_train[:len(labels_train)/100])
print "training time:", round(time()-t0, 3), "s"

t0 = time()
print "Accuracy: ", clf.score(features_test, labels_test)
print "Testing time:", round(time()-t0, 3), "s"


print "Changing kernel to rbf"

# Manipulating C parameter for better accuracy. The SVM C parameter controls
# the tradeoff between smooth decision boundary and classifying training points
# correctly.
for c in [1., 10., 100., 1000., 10000.]:
    print "C = ", c
    t0 = time()
    clf = SVC(C=c, kernel='rbf').fit(features_train[:len(features_train)/100],
                                   labels_train[:len(labels_train)/100])
    print "Training time:", round(time()-t0, 3), "s"

    t0 = time()
    print "Accuracy: ", clf.score(features_test, labels_test)
    print "Testing time:", round(time()-t0, 3), "s\n"

print "Choose C = 10000 and fit on full training set"
t0 = time()
clf = SVC(C=10000., kernel='rbf').fit(features_train, labels_train)
print "Training time:", round(time()-t0, 3), "s"

t0 = time()
print "Accuracy: ", clf.score(features_test, labels_test)
print "Testing time:", round(time()-t0, 3), "s\n"

# What class does your SVM (0 or 1, corresponding to Sara and Chris
# respectively) predict for element 10 of the test set? The 26th? The 50th?
clf = SVC(C=c, kernel='rbf').fit(features_train[:len(features_train)/100],
                                   labels_train[:len(labels_train)/100])

clf.predict([features_test[10]])
clf.predict([features_test[26]])
clf.predict([features_test[50]])

# There are over 1700 test events--how many are predicted to be in the "Chris"
# (1) class? (Use the RBF kernel, C=10000., and the full training set.)
clf = SVC(C=10000., kernel='rbf').fit(features_train, labels_train)
pred = clf.predict(features_test)
print(sum(pred))


