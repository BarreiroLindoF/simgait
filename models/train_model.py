from model import *
import json
from tester import SplitValidationTester
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import sys
    
def main():
    
    
    X_train = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\Width200MarkersAngles\\CV2\\1d_X_train.npy")
    y_train = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\Width200MarkersAngles\\CV2\\1d_y_train.npy")
    X_test = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\Width200MarkersAngles\\CV2\\1d_X_test.npy")
    y_test = np.load("D:\\Users\\Flavio\\Documents\\Research Project\\gait\\data\\normalized\\Width200MarkersAngles\\CV2\\1d_y_test.npy")

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #RandomForest
    model = RandomForestClassifier(1)
    # 100 DT = 44%
    # 1000 DT = 46%
    # 10 DT = 39%
    # 1 DT no random forest = 36%
    # 1 DT with RF = 32%


    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(y_test == predictions)
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
