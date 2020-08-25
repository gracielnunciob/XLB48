from apyori import apriori
import numpy as np

"""
This class represents a rule extracted from the apriori algorithm.

Attributes:

left : int[] - indices of antecedent of rule
left_text : string[] - names of elements in antecedent of rule
right : int - index of consequent rule among the target labels
right_text : string - name of target label
label_support : float[] - support for each label in respective order (0-based)
"""
class Rule:

    """
    Default constructor for a rule

    Parameters:
    left : int[]            - indices of the features used
    right : int             - label of the right hand side
    data : np.ndarray       - dataset to use in computing metrics
    col_names : string[]    - list containing feature names of the dataset
    label_names : string[]  - list containing corresponding labels
    label_support : float[] - list containing support values for each target 
                                label in the dataset
    """
    def __init__(self,left,right,data,col_names, label_names, label_support):
        self.left = left
        self.left_text = [col_names[i] for i in left]
        self.right = right
        self.right_text = label_names[self.right - 1]
        self.label_support = label_support
        self.compute_confidence(data)
        self.compute_lift(data)
        self.compute_interestingness(data)

    """
    This method computes for the confidence of a rule as observed in a given 
    dataset

    Parameters:
    data : np.array - dataset to use as basis for computation of confidence 
                      value

    Returns a float indicating the confidence value of this rule.
    """
     def compute_confidence(self,data):
        #gets the column of the emotion
        row_count, emotioncol = data.shape
        emotioncol -= 1
        
        #"right" is just emotion val
        for k in right:
            #This loops through the whole 401
            confcount = 0
            for i in range(row_count):
                #This checks if all is true pa 
                s = True
                for j in left:
                    if(data[i][j] == 1 and data[i][emotioncol] == k and s == True):
                        s = True
                    else:
                        s = False  

                #if all are true pa for that row meaning all the featurelist are 1 for that row, increment 
                if(s == True):
                    confcount += 1
                    
            confidence = (confcount/row_count) / ruleset.support
            self.confidence = confidence

    """
    This method computes for the lift of a rule as observed in a given dataset

    Parameters:
    data : np.array - dataset to use as basis for computation of lift value

    Returns a float indicating the lift value of this rule.
    """
    def compute_lift(self,data):
        conf = compute_confidence(data)
        lift = conf / self.label_support[self.right-1]
        self.lift = lift

    """
    This method computes for the interestingness of a rule as observed in a 
    given dataset

    Parameters:
    data : np.array - dataset to use as basis for computation of 
    interestingness value

    Returns a float indicating the interestingness value of this rule.
    """
    def compute_interestingness(self,data):
        conf = compute_confidence(data)
        interestingness = math.sqrt(conf^2 / (ruleset.support * self.label_support[self.right-1]))
        self.interestingness = interestingness
        

    def __str__(self):
        return """{} -> {}
    confidence: {:.2f}
    lift: {:.2f}
    interstingness: {:.2f}""".format(
            ", ".join(self.left_text),self.right_text,self.confidence,
            self.lift,self.interestingness
    )
        

"""
This abstract class represents a custom machine learning model.
"""
class AbstractCustomModel:
    
    """
    This is the default constructor for this class.
 
    Parameters:
    params : dict - hyperparameters for the model
    """
    def __init__(self,params={}):
        self.params = params
 
    """
    This method trains the model on the given dataset
 
    Parameters:
    x_train : numpy.ndarray - training set data
    """
    def train(self,x_train,y_train=None):
        # TODO
        pass
 
    """
    This method evaluates the performance of the model on the given test set.
 
    Parameters:
    x_test : numpy.ndarray - test set data
 
    Returns:
    A floating point number which measures the performance of the model on the 
    test set.
    """
    def evaluate(self,x_test,y_test=None):
        # TODO
        pass
    
    """
    Probability estimates.

    The returned estimates for all classes are ordered by the label of classes.

    For a multi_class problem, if multi_class is set to be “multinomial” the 
    softmax function is used to find the predicted probability of each class. 
    Else use a one-vs-rest approach, i.e calculate the probability of each class 
    assuming it to be positive using the logistic function. and normalize these 
    values across all the classes.
    
    Parameters:
    X : array-like of shape (n_samples, n_features) 
        - Vector to be scored, where n_samples is the number of samples and 
          n_features is the number of features.
    
    Returns:
    T : array-like of shape (n_samples, n_classes)
        - Returns the probability of the sample for each class in the model, 
          where classes are ordered as they are in self.classes_.
    """
    def predict_proba(self, X):
        # TODO
        pass
    
    """
    Predict logarithm of probability estimates.

    The returned estimates for all classes are ordered by the label of classes.
    
    Parameters:
    X : array-like of shape (n_samples, n_features) 
        - Vector to be scored, where n_samples is the number of samples and 
          n_features is the number of features.
    
    Returns:
    T : array-like of shape (n_samples, n_classes)
        - Returns the log-probability of the sample for each class in the 
          model, where classes are ordered as they are in self.classes_.
    """
    def predict_log_proba(self,X):
        return np.log(self.predict_proba(X))
        
    """
    Predict class labels for samples in X.
    
    Parameters:
    X : array_like or sparse matrix, shape (n_samples, n_features)
        - Samples.

    Returns:
    C : array, shape [n_samples] - Predicted class label per sample.
    """
    def predict(self,X):
        probas = self.predict_proba(X)
        return np.argmax(probas,axis=1)

    """
    Return the mean accuracy on the given test data and labels.

    In multi-label classification, this is the subset accuracy which is a harsh
    metric since you require for each sample that each label set be correctly predicted.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        - Test samples.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        - True labels for X.
    
    Returns:
    score : float - Mean accuracy of self.predict(X) wrt. y.
    """
    def score(self,X,y):
        return np.mean(self.predict(X) == y)
 

"""
This class is an adapter for the Apyori library.
"""
class APyoriAdapter(AbstractCustomModel):
 
    """
    This is the default constructor for this class.
 
    Parameters:
    params : dict - hyperparameters for the model
    """
    def __init__(self,params={}):
        self.params = params
        super().__init__(params)
 

    def convert_to_transaction(self,row):
        return [(i,j)for i,j in enumerate(row)]

    """
    This method trains the model on the given dataset
 
    Parameters:
    x_train : numpy.ndarray - training set data
    """
    def train(self,x_train,y_train=None):
        x_train = self.discretize_dataset(x_train,self.params["thresholds"])
        data_2 = [[i for i,j in enumerate(row) if j==1 ] for row in x_train]
        #if running takes long we raise min support 
        itemsets = list(apriori(data_2,min_support=self.params["min_support"]))
        ctr = 0
        self.ruleset = []
        for ruleset in itemsets:
            for i in range(1,7):
                cur_rule = Rule(
                    list(ruleset.items),i,x_train,self.params["col_names"],\
                    self.params["label_names"],self.params["label_support"]
                )
                if cur_rule.confidence > self.params["min_confidence"] - 1e-9:
                    self.ruleset.append(cur_rule)
        return self.ruleset
            
    """
    This method evaluates the performance of the model on the given test set.
 
    Parameters:
    x_test : numpy.ndarray - test set data
 
    Returns:
    A floating point number which measures the performance of the model on the 
    test set.
    """
    def evaluate(self,x_test=None,y_test=None):
        ave_interestingness = 0.0
        ctr = 0
        
        for rule in self.ruleset:
            ave_interestingness += rule.interestingness
            ctr += 1
            
        ave_interestingness /= ctr
    
        return ave_interestingness

    """
    Probability estimates.

    The returned estimates for all classes are ordered by the label of classes.

    For a multi_class problem, if multi_class is set to be “multinomial” the 
    softmax function is used to find the predicted probability of each class. 
    Else use a one-vs-rest approach, i.e calculate the probability of each class 
    assuming it to be positive using the logistic function. and normalize these 
    values across all the classes.
    
    Parameters:
    X : array-like of shape (n_samples, n_features) 
        - Vector to be scored, where n_samples is the number of samples and 
          n_features is the number of features.
    
    Returns:
    T : array-like of shape (n_samples, n_classes)
        - Returns the probability of the sample for each class in the model, 
          where classes are ordered as they are in self.classes_.
    """
    def predict_proba(self, X):
        X = self.discretize_dataset(X,self.params["thresholds"])
        data_2 = [[i for i,j in enumerate(row) if j==1 ] for row in X]
        res = []
        # for each row in test set
        for row in data_2:
            label_ctr = [0] * 6
            # check against each rule
            for rule in self.ruleset:
                all_ok = True
                
                # check for absence of any element of antecedent
                for l_ind in rule.left:
                    if l_ind not in row:
                        all_ok = False
                        break
                
                # if entire antecedent is present
                if all_ok:
                    # add log of confidence
                    label_ctr[rule.right - 1] += np.log(rule.confidence)
            temp_row = np.exp(np.array(label_ctr))
            temp_row = temp_row / np.sum(temp_row)
            res.append(temp_row)
        return np.array(res)
    
    """
    discretize_dataset function transforms the dataset into binary o

    Takes the input of:
    -data = 
    -thresholds = 
    """
    def discretize_dataset(self,data,thresholds):
        temp = [row[::]for row in data]
        data = temp
        for i, datapoint in enumerate(data):
            for j, threshold in enumerate(thresholds):
                if datapoint[j] >= threshold:
                    datapoint[j] = 1
                else: 
                    datapoint[j] = 0
        return data            
