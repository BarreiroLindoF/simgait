"""
Created on Thu Jun 27 16:38:19 2019

@author: flavio.barreiro
"""

from abc import ABC, abstractmethod
from model import SupervisedModel
import numpy as np
import torch.nn as nn


# Abstract class allowing for polymorphism operations
class Tester(ABC):
    
    def __init__(self, model):
        if not isinstance(model, SupervisedModel):
            raise TypeError('A SupervisedModel or a Pytorch model has to be used!')
        self.model = model
    
    @abstractmethod 
    def test(self, X, y):
        pass
    
    def test_accuracy(self, x_train, x_test, y_train, y_test):
        self.model.fit(x_train, y_train)
        
        predictions = self.model.predict(x_test)
        
        accuracy = np.mean(y_test == predictions)
        
        return accuracy
    

from sklearn.model_selection import train_test_split

# Split the X and y into two datasets : train and test
# the class will do the training and accuracy testing
# automaticaly 
class SplitValidationTester(Tester):
    
    def __init__(self, model, test_percentage):
        super().__init__(model)
        self.percentage = test_percentage
        
    def test(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.percentage)
        
        return self.test_accuracy(x_train, x_test, y_train, y_test)
    
    
from sklearn.model_selection import KFold

# Split the X and y into two datasets : train and test
# If kfold = 10 then the training will equal to 9/10 of the dataset
# and 1/10 will be used for testing. 
# 10 different tests will be done.
# the class will do the training and accuracy testing
# automaticaly and each test accuracy is returned
class KFoldTester(Tester):
    
    def __init__(self, model, kfolds):
        super().__init__(model)
        self.nb_kfolds = kfolds
        self.kfolds = KFold(kfolds)

    def test(self, X, y):
        accuracy_hist = []
        for train_index, test_index in self.kfolds.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
                        
            accuracy = self.test_accuracy(x_train, x_test, y_train, y_test)
            
            accuracy_hist.append(accuracy)
            
        return accuracy_hist