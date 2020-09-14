import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def get_top_k(x_train, model_selector, k):    
    print(sorted(model_selector.predict_proba(x_train)[:,1], reverse=True)[:k])
    probas = model_selector.predict_proba(x_train)[:,1]
    preds = x_train
    pp = np.append(preds, probas.reshape(probas.shape[0], 1), axis=1)

    #sort by highest predict_proba scores
    sortedpp = pp[pp[:, pp.shape[1]-1].argsort()[::-1]]
    top_k = sortedpp[:k]
    
    return top_k

def get_top_ave(top_k):
    topave = np.empty((1, 69))
    col = 0

    #get average of each column except last (probas)
    for col in range(top_k.shape[1]-1):
        topave[0, col] = top_k[:, col].mean()
    return topave