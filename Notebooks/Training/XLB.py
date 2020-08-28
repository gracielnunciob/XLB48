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
    tree_rules = export_text(classifier, feature_names=features, max_depth=99)
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

"""
This function builds the query to be used in the pandas.query() function.

Parameter:
rule: the rule to be examined
theme: the specific theme the one vs rest model is built for

Returns:
The query form of the rule.

"""
def build_query(rule, theme):
    c = 0
    query = ""
    for i in rule:
        if c == 0:
            if "class" in i:
                if '0' in i:
                    query = 'not Themes == '+str(theme)
                if '1' in i:
                    query = 'Themes == '+str(theme)
                    
            else:
                query = i
        else:
            if "class" in i:
                if '0' in i:
                    query+=' and not Themes == '+str(theme)
                if '1' in i:
                    query+=' and Themes == '+str(theme)
            else:
                query+=' and '+i
        c+=1
    return query

"""
This function searches the dataset and returns the count of samples that a satifies the conditions of a rule.

Parameter:
query: the rule in query form to be used in pandas.query()
dataset: the dataframe object

Returns:
The count of samples that a satifies the conditions of a rule.

"""
def db_lookup(query, dataset):
    x = dataset.query(query)
    return x.shape[0]

"""
This function computes the number of samples that satisfy the conditions of each antecedent and precedent.

Parameter:
rules: the set of rules extracted from the decision tree model 
theme: the specific theme the one vs rest model is built for 

Returns:
The count of samples that a satifies the conditions of a rule.
"""
def get_count_compound(rules, theme):
    count = []
    dataset = pd.read_csv("FinalTrainingSet.csv")
    cols = dataset.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, "unicode")) else x)
    col = cols.tolist()
    col[-1] = 'Themes'
    dataset.columns = col
    for i in rules:
        rule = rules[i]
        query = build_query(rule, theme)
        x = db_lookup(query, dataset)
        count.append((query, x))
    return count

"""
This function computes the number of samples that satisfy the conditions of each antecedent.

Parameter:
rules: the set of rules extracted from the decision tree model 
theme: the specific theme the one vs rest model is built for 

Returns:
The count of samples that a satifies the conditions of a rule.
"""
def get_count_A(rules, theme):
    count = []
    dataset = pd.read_csv("FinalTrainingSet.csv")
    cols = dataset.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, "unicode")) else x)
    col = cols.tolist()
    col[-1] = 'Themes'
    dataset.columns = col
    for i in rules:
        rule = rules[i]
        query = build_query(rule[:-1], theme)
        x = db_lookup(query, dataset)
        count.append((query, x))
    return count

"""
This function computes the number of samples that satisfy the conditions of each precedent.

Parameter:
rules: the set of rules extracted from the decision tree model 
theme: the specific theme the one vs rest model is built for 

Returns:
The count of samples that a satifies the conditions of a rule.
"""

def get_count_B(rules, theme):
    count = []
    dataset = pd.read_csv("FinalTrainingSet.csv")
    cols = dataset.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, "unicode")) else x)
    col = cols.tolist()
    col[-1] = 'Themes'
    dataset.columns = col
    for i in rules:
        rule = rules[i]
        if "class" in rule[-1]:
            if '0' in i:
                query = 'not Themes == '+str(theme)
            if '1' in i:
                query = 'Themes == '+str(theme)
                    
        else:
            query = i
        x = db_lookup(query, dataset)
        count.append((query, x))
    return count


"""
This function computes the confidence and lift for each rule.

Parameter:
compound: a list of tuples containing the rule and it's true positives 
A: a list of tuples containing the precedents of the rule and it's true positives
B: a list of tuples containing the antecedents of the rule and it's true positives
total: the total number of samples in the dataset

Returns:
The lists of tuples cotaining the rules and it's confidence and lift.
"""
def compute_confidence_and_lift(compound, A, B, total):
    confidence = []
    lifts = []
    for i in range(len(compound)):
        if A[i][1]>0:
            conf = compound[i][1]/A[i][1]
            confidence.append((compound[i][0], conf))
        else:
            confidence.append((compound[i][0], 0))
            
    for i in range(len(compound)):
        if (B[i][1]/total)>0:
            lift = confidence[i][1]/(B[i][1]/total)
            lifts.append((compound[i][0], lift))
        else:
            lifts.append((compound[i][0], 0))
    return confidence, lifts
"""
This function computes the average confidence and lift.

Parameter:
compound: a list of tuples containing the rule and it's true positives 
A: a list of tuples containing the precedents of the rule and it's true positives
B: a list of tuples containing the antecedents of the rule and it's true positives
total: the total number of samples in the dataset

Returns:
The average confidence and lift.
"""
def avg_lift_and_confidence(compound, A, B, total):
    confidence, lift = compute_confidence_and_lift(compound, A, B, total)
    avg_conf = 0
    avg_lift = 0
    for i in confidence:
        avg_conf+=i[1]
    avg_conf = avg_conf/len(confidence)
    for i in lift:
        avg_lift+=i[1]
    avg_lift = avg_lift/len(lift)
    return avg_conf, avg_lift
"""
This function displays the confidence and lift for each rule.

Parameter:
tree: the decision tree object
features: the list of features after feature selection
theme: the specific theme the one vs rest model is built for

"""
def disp_conf_and_lift(tree, features, theme):
    tree_text = tree_to_text(tree, features)
    rules = extract_rules(tree_text)
    rules = transform_rules(rules)
    comp = get_count_compound(rules, theme)
    A = get_count_A(rules, theme)
    B = get_count_B(rules, theme)
    conf, lift = compute_confidence_and_lift(comp, A, B, 401)
    avg_conf, avg_lift = avg_lift_and_confidence(comp, A, B, 401)
    for i in conf:
        print(i[0]+" Confidence: "+str(i[1]))
    print("Average Confidence: "+str(avg_conf))
    for i in lift:
        print(i[0]+" Lift: "+str(i[1]))
    print("Average Lift: "+str(avg_lift))