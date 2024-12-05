from LIMESegment.Utils.explanations import LIMESegment, LEFTIST
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class LSClassifier(ClassifierMixin, BaseEstimator):
    
    def __init__(self, tsmodel):
        self.tsmodel = tsmodel      

    def fit(self,X,y):
        X_reshaped = X.reshape(X.shape[0],X.shape[2], X.shape[1])
        self.tsmodel.fit(X_reshaped,y)  
        return self
    # def predict_proba(self,X):
    #     X_reshaped = X.reshape(X.shape[0],X.shape[2], X.shape[1])
    #     return self.tsmodel.predict_proba(X_reshaped)
    def predict(self,X):
        X_reshaped = X.reshape(X.shape[0],X.shape[2], X.shape[1])
        return self.tsmodel.predict(X_reshaped)
    

def convert_ls_explanation(lsexp, tslen):
    fullsm = np.zeros(tslen)
    for i in range(len(lsexp[0])):
        if i < len(lsexp[0])-1:
            fullsm[lsexp[1][i]:lsexp[1][i+1]] = lsexp[0][i]
        else:
            fullsm[lsexp[1][i]:] = lsexp[0][i]
    return fullsm

def explain_with_LIMESegment(example, model, model_type='class', distance='dtw', n=100, window_size=None, cp=None, f=None):
    wrapped_model = LSClassifier(model)
    ts = example.reshape(example.shape[-1],1)
    explanations = LIMESegment(ts, wrapped_model, model_type=model_type, distance=distance, n=n)
    return convert_ls_explanation(explanations, example.shape[-1])

def explain_with_LEFTIST(example, model, X_background, model_type="class", n=100):
    wrapped_model = LSClassifier(model)
    ts = example.reshape(example.shape[-1],1)
    X_bg_reshaped = X_background.reshape(X_background.shape[0],X_background.shape[2], X_background.shape[1])
    explanations = LEFTIST(ts, wrapped_model, X_bg_reshaped, model_type=model_type, n=n)
    return convert_ls_explanation(explanations, example.shape[-1])