import numpy as np

"""
Takes the input of:
    -numfolds = the number of folds given
    -model = the machine learning model
    -parameters = 
    -X = 
    -y
"""
def cross_validation(numfolds,model,parameters,X,y):
    model.params = parameters
    
    # initialization
    folds = [[]for i in range(numfolds)]
    labels = [[]for i in range(numfolds)]
    
    # distribution of data into folds 
    for i, datapoint in enumerate(X):
        folds[i % numfolds].append(datapoint) 
        labels[i % numfolds].append(y[i]) 
        
    # K-fold cross validation
    average_result = 0
    for i in range(numfolds):
        test = folds[i]
        trainlist = []
        labellist = []
        for j, fold in enumerate(folds):
            if (j != i):
                trainlist += fold
                labellist += labels[j]
            
        model.train(np.array(trainlist),np.array(labellist))
        result = model.evaluate(test)
        average_result += result/numfolds
        
    return average_result, model    
 
"""
This function searches for the optimal hyperparameters using Ranodmized Search.

Parameters:
numfold : int - number of folds k for k-fold cross validation
model : AbstractCustomModel - model object for training
parameters : dict - parameters with ranged for hyperparameter search
X : array-like of size (n_samples,n_features) - dataset to train on
y : array-like of size (n_samples,) - ground truth labels
num_iter : int - number of iterations of hyperparameter search to perform
random_state : int - seed for random number generator
"""
def hyperparameter_search(num_folds,model,parameters,X,y,num_iter=200,random_state=879057,interval=10,verbose=False):
    best_result = -1e9
    best_model = None
    best_params = None 
    np.random.seed(random_state)
    for i in range(num_iter):
        # threshold function
        if verbose and (i + 1) % interval == 0:
            print("""Iteration {} / {}
Best Result: {:.2f}""".format(i + 1,num_iter,best_result))
        params = {
            "thresholds" : np.random.normal(0.5, 0.15, parameters["num_features"]), 
            "min_support" : np.random.uniform(
                parameters["min_support_lo"],
                parameters["min_support_hi"],1
            )[0],
            "min_confidence" : np.random.uniform(
                parameters["min_confidence_lo"],
                parameters["min_confidence_hi"],1
            )[0],
            "col_names" : parameters["col_names"],
            "label_names" : parameters["label_names"],
            "label_support" : parameters["label_support"]
        }
        
        # highest threshold
        result, cur_model = cross_validation(num_folds,model,params,X,y)
        if result > best_result:
            best_params = params
            best_result = result
            best_model = cur_model
    best_model.params = best_params
    best_model.train(data)
            
    return best_params,best_result,best_model    
    
