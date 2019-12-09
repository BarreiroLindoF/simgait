from model import *
import json
from tester import SplitValidationTester
import numpy as np
import sys
    
def main():
    
    
    X = np.load("C:\\Users\\lucas\\Desktop\\gaitmasteris\\data\\extracted\\random_forest_formated\\x.npy")
    y = np.load("C:\\Users\\lucas\\Desktop\\gaitmasteris\\data\\extracted\\random_forest_formated\\y.npy")
    print(X)
    print(y)
    
    
    #RandomForest
    model = RandomForest()
    tester = SplitValidationTester(model, 0.2)
    accuracy = tester.test(X, y)
    print("Random Forest accuracy : " + str(accuracy))

    '''
    preprocess_data_cnn()
    train_cnn()
    
    preprocess_data_rnn()
    train_rnn()
    '''

    '''
    #Try models with known algorithms like Knn, Decision Tree...
    ###############################
    #Get the x and y data (x : windows of frame where the events could be, y: the events to predict with their frames)
    _, _, x, y = pre_processing.prepare_data(data)
    #KNN
    model = models.kNN(1)
    tester = SplitValidationTester(model, 0.2)
    accuracy = tester.test(x, y)
    print("kNN accuracy : " + str(accuracy))
    
    #DecisionTree
    model = models.DecisionTree()
    tester = SplitValidationTester(model, 0.2)
    accuracy = tester.test(x, y)
    print("Decision Tree accuracy : " + str(accuracy))
    
    ###############################
    
    #Try models with some NN
    ###############################
    #MLP
    _, _, x, y = pre_processing.prepare_data_nn(data, is_mlp = True)
    model = models.NeuralNetwork()
    model.fit(x,y)
    model.predict(x)
    #CNN
    _, _, x, y = pre_processing.prepare_data_nn(is_mlp = False)
    model = models.CNN()
    model.fit(x,y)
    model.predict(x)
    ###############################
    '''
    
main()