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
from sklearn.tree import export_text

"""
Converts the labels to binary for one vs rest models

Parameters:
y_train - input data
label - the theme of choice

Returns: binary list of labels for one vs rest models
"""
def ovr_labels(y_train, label):
    ovr_list = np.array(list(map(int,y_train == label)))
    return ovr_list

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
Trains a model using the given data and a hyperparameter search object

@ -84,13 +95,24 @@ name - name of the test run
x_train - input data
y_train - target labels for data
model_selector - model_selection object
theme - themes selected for the one vs rest model
"""

def print_res(name,x_train,y_train,model_selector,theme,verbose=True):
    train_model(x_train,y_train,model_selector,name,verbose)

    # display confusion matrix
    if verbose:
        disp = plot_confusion_matrix(model_selector, x_train, y_train,
                                 display_labels=[theme,"Not "+theme],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    
"""
Tests and prints the result of the training and model selection.

Parameters:
name - name of the test run
x_test - input data
y_test - target labels for data
model_selector - model_selection object
theme - theme selected for the one vs rest model
"""
def test_res(name,x_test,y_test,model_selector,theme,verbose=True):
    y_pred = model_selector.predict(x_test)
    print(y_pred)
    print(y_test)
    # display confusion matrix
    print("{} Validation Accuracy: {:.2f}%".format(name,np.mean(y_pred == y_test) * 100.0))
    print("{} F1-score: {:.2f}".format(name,f1_score(y_test, y_pred, average='weighted')))
    if verbose:
        disp = plot_confusion_matrix(model_selector, x_test, y_test,
                                 display_labels=["Not "+theme,theme],
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
"""
This function displays the decision tree in text form.

Parameters:
classifier : DecisionTreeClassifier - decision tree object
features : list - the list of features used by the classifer
"""
def disp_tree_text(classifier, features):
    tree_rules = export_text(classifier, feature_names=features)
    print(tree_rules)
"""
This function returns the decision tree in text form.

Parameters:
classifier : DecisionTreeClassifier - decision tree object
features : list - the list of features used by the classifer

Returns:
The decision tree in text form.
"""

def tree_to_text(classifier, features):
    tree_rules = export_text(classifier, feature_names=features)
    return  tree_rules

"""
This function extracts the individual rules from the decision tree.

Parameter:
rules: the decision tree in text form

Returns:
The rules used by the decision tree.

"""
def extract_rules(rules):
    line = []
    lines = []
    for i in rules:
        if i != '\n' and not i == '|' and not i == '-':
            line.append(i)
        if i == '\n':
            str_line = ''.join(str(e) for e in line)
            str_line = str_line.lstrip()
            lines.append(str_line)
            line = []

    counter = 0
    rule_sets = {}
    rule_set = []
    for i in lines:
        rule_set.append(i)
        if 'class' in i:
            rule_sets[str(counter)] = rule_set
            counter+=1
            rule_set = []
    return rule_sets

"""
This function modifies the rules so that it can be used in the pandas.query() function.

Parameter:
rule_sets: the rules of the decision tree

Returns:
The modified rules.

"""
def transform_rules(rule_sets):
    for i in rule_sets:
        for j in range(len(rule_sets[i])):
            x = rule_sets[i][j]
            if "<=" in x:
                splits = x.split("<=")
                splits[0] = splits[0].rstrip()
                splits[1] = splits[1].lstrip()
                splits[0] = splits[0].replace(' ','_')
                rule_sets[i][j] = splits[0]+" <= "+splits[1]
            elif ">" in x:
                splits = x.split(">")
                splits[0] = splits[0].rstrip()
                splits[1] = splits[1].lstrip()
                splits[0] = splits[0].replace(' ','_')
                rule_sets[i][j] = splits[0]+" > "+splits[1]
            elif "==" in x:
                splits = x.split("==")
                splits[0] = splits[0].rstrip()
                splits[1] = splits[1].lstrip()
                splits[0] = splits[0].replace(' ','_')
                rule_sets[i][j] = splits[0]+" == "+splits[1]
    return(rule_sets)

def build_query(rule):
    c = 0
    for i in rule:
        if c == 0:
            query = i
        else:
            if "class" in i:
                if '0' in i:
                    query+=' and not Themes == 1'
                if '1' in i:
                    query+=' and Themes == 1'
            else:
                query+=' and '+i
        c+=1
    return query