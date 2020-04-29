import os
import numpy as np
import pandas as pd
import csv
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import matplotlib.pyplot as plt

"""
This function extracts the features and labels from a csv file.

Parameters:
filename : string - file to read from
with_label : boolean - whether the file has labels or not

Returns:
- numpy arrays containing the features and labels if with_label is True
- numpy arrays containing the features if with_label is False
"""
def extract_data(filename,with_label=True):
    data = []    
    f = open(filename,"r")
    d_reader = csv.reader(f,delimiter=",",quotechar="\"")
    first = True
    for line in d_reader:
        if first:
            first = False
            continue
        data.append(line)
#     print(np.shape(data))
    col_count = 324
    x_train = np.empty((0,col_count))
    y_train = np.array([])

    for line in data:
        if with_label:
            x_train = np.append(x_train,np.array(list(map(float,line[1:-2]))).reshape((1,col_count)),axis=0)
            y_train = np.append(y_train,int(float(line[-1])))
        else:
            try:
                x_train = np.append(x_train,np.array(list(map(float,line[1:]))).reshape((1,col_count)),axis=0)
            except ValueError:
                print(line[1:])
#     print(x_train.shape)
#     print(y_train.shape)
    if with_label:
        return x_train, y_train
    else:
        return x_train
    #print(y_train)
    
"""
Trains a model using the given data and a hyperparameter search object

Parameters:
x_train - input data
y_train - target labels for data
hp_search - model_selection object
name - name of the experiment

Returns: best estimator for the given data given the model selector
"""
def train_model(x_train,y_train,hp_search,name,verbose=True):
    hp_search.fit(x_train,y_train)
    print("Best Score: {:.4f}".format(hp_search.best_score_))
    if verbose:
        for k,v in hp_search.best_params_.items():
            print("{} => {}".format(k,v))
        print("Splits: {}".format(hp_search.n_splits_))
    y_out = hp_search.predict(x_train)
    print("{} Train Accuracy: {:.4f}%".format(name,np.mean(y_out == y_train) * 100.0))
    return hp_search.best_estimator_

"""
Trains and prints the result of the training and model selection.

Parameters:
name - name of the test run
x_train - input data
y_train - target labels for data
model_selector - model_selection object
"""
def print_res(name,x_train,y_train,model_selector,verbose=True):
    train_model(x_train,y_train,model_selector,name,verbose)

    # display confusion matrix
    if verbose:
        disp = plot_confusion_matrix(model_selector, x_train, y_train,
                                 display_labels=["Calm","Cheerful","Bravery","Fearful","Sadness","Love"],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    # print(y_out)\
    
"""
Tests and prints the result of the training and model selection.

Parameters:
name - name of the test run
x_test - input data
y_test - target labels for data
model_selector - model_selection object
"""
def test_res(name,x_test,y_test,model_selector,verbose=True):
    y_pred = model_selector.predict(x_test)
    # display confusion matrix
    print("{} Validation Accuracy: {:.2f}%".format(name,np.mean(y_pred == y_test) * 100.0))
    print("{} F1-score: {:.2f}".format(name,f1_score(y_test, y_pred, average='weighted')))
    if verbose:
        disp = plot_confusion_matrix(model_selector, x_test, y_test,
                                 display_labels=["Calm","Cheerful","Bravery","Fearful","Sadness","Love"],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    return np.mean(y_pred == y_test) * 100.0, f1_score(y_test, y_pred, average='weighted')
    
    
"""
This function displays a decision tree.

Parameters:
dt : DecisionTreeClassifier - decision tree object
filename : string - file to read from
"""
def disp_tree(dt,filename):
    classes = ['Brave', 'Cheerful', 'Fearful', 'Love', 'Sadness', 'Calm']
    file = pd.read_csv(filename)
    features = list(file)
    fig, ax = plt.subplots(figsize=(40, 40))
    treefig = tree.plot_tree(dt, class_names=classes, feature_names=features[1:-3], fontsize=12, ax=ax)
    plt.show()