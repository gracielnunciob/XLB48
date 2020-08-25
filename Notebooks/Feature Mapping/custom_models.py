from apyori import apriori

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
    left : int[] - indices of the features used
    right : int - label of the right hand side
    data : np.ndarray - dataset to use in computing metrics
    col_names : string[] - list containing feature names of the dataset
    label_names : string[] - list containing corresponding labels
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
    This method computes for the confidence of a rule as observed in a given dataset

    Parameters:
    data : np.array - dataset to use as basis for computation of confidence value

    Returns a float indicating the confidence value of this rule.
    """
    def compute_confidence(self,data):
        # TOOD
        # replace this with actual computation
        self.confidence = 0.0 

    """
    This method computes for the lift of a rule as observed in a given dataset

    Parameters:
    data : np.array - dataset to use as basis for computation of lift value

    Returns a float indicating the lift value of this rule.
    """
    def compute_lift(self,data):
        # TODO
        self.lift = 0.0 

    """
    This method computes for the interestingness of a rule as observed in a given dataset

    Parameters:
    data : np.array - dataset to use as basis for computation of interestingness value

    Returns a float indicating the interestingness value of this rule.
    """
    def compute_interestingness(self,data):
        # TODO
        self.interestingness = 0.0 

    def __str__(self):
        return """{} -> {}
    confidence: {:.2f}
    lift: {:.2f}
    interstingness: {:.2f}""".format(", ".join(self.left_text),self.right_text,self.confidence,self.lift,self.interestingness)
        

"""
This abstract class represents a custom machine learning model.
"""
class AbstractCustomModel:
    
    """
    This is the default constructor for this class.
 
    Parameters:
    params : dict - hyperparameters for the model
    """
    def __init__(self,params):
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
        self.results = []
        for ruleset in itemsets:
#             print(ruleset)
            for i in range(1,7):
#                 print("Adding i = {}".format(i))
                self.results.append(Rule(
                    list(ruleset.items),i,x_train,self.params["col_names"],\
                    self.params["label_names"],self.params["label_support"]
                ))
#                 print(self.results[-1])
        return self.results
            
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
        
        for rule in self.results:
            ave_interestingness += rule.interestingness
            ctr += 1
            
        ave_interestingness /= ctr
    
        return ave_interestingness
    
    
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
