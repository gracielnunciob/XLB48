import numpy as np

"""
Takes the input of:
    -numfolds = the number of folds given
    -model = the machine learning model
    -parameters = 
    -data = 
"""
def cross_validation(numfolds,model,parameters,data):
    model.params = parameters
    
    # initialization
    folds = [[]for i in range(numfolds)]
    
    # distribution of data into folds 
    for i, datapoint in enumerate(data):
        folds[i % numfolds].append(datapoint) 
        
    # K-fold cross validation
    average_result = 0
    for i in range(numfolds):
        test = folds[i]
        trainlist = []
        for j, fold in enumerate(folds):
            if (j != i):
                trainlist += fold
            
        model.train(np.array(trainlist))
        result = model.evaluate(test)
        average_result += result/numfolds
        
    return average_result, model    
 
"""
This function searches for the optimal hyperparameters using Ranodmized Search.

Parameters:
numfold : int - number of folds k for k-fold cross validation
model : AbstractCustomModel - model object for training
parameters : dict - parameters with ranged for hyperparameter search
data : np.ndarray - dataset to train on
num_iter : int - number of iterations of hyperparameter search to perform
random_state : int - seed for random number generator
"""
def hyperparameter_search(num_folds,model,parameters,data,num_iter=200,random_state=879057):
    best_result = -1e9
    best_model = None
    best_params = None 
    np.random.seed(random_state)
    for i in range(num_iter):
        # threshold function
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
        result, cur_model = cross_validation(num_folds,model,params,data)
        if result > best_result:
            best_params = params
            best_result = result
            best_model = cur_model
    best_model.params = best_params
    best_model.train(data)
            
    return best_params,best_result,best_model    
    
