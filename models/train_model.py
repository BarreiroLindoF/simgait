from model import *
import json
from tester import SplitValidationTester
from sklearn.ensemble import RandomForestClassifier
import os

import numpy as np
import sys

# Project directory to make this code runnable on any windows system (to be changed on mac)
project_dir = os.path.expanduser(os.path.dirname(os.getcwd()))

def get_cv_dir_names(directory):
    # Get all directory names for cross validation
    dir_names = os.listdir(os.path.join(project_dir, "data", "models_prepared", "cnn_formated",  directory))
    dir_names = filter(lambda k: 'CV' in k, dir_names)

    return dir_names


def main():

    dir_names = get_cv_dir_names("Width100MarkersOnly")

    lst_accuracy = []

    for folder in dir_names:
        X_train = np.load(os.path.join(project_dir, "data", "models_prepared", "cnn_formated", "Width100MarkersOnly", folder, "1d_X_train.npy"))
        y_train = np.load(os.path.join(project_dir, "data", "models_prepared", "cnn_formated", "Width100MarkersOnly", folder, "1d_y_train.npy"))
        X_test = np.load(os.path.join(project_dir, "data", "models_prepared", "cnn_formated", "Width100MarkersOnly", folder, "1d_X_test.npy"))
        y_test = np.load( os.path.join(project_dir, "data", "models_prepared", "cnn_formated", "Width100MarkersOnly", folder, "1d_y_test.npy"))

        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

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
        lst_accuracy.append(accuracy)
        print("Random Forest accuracy for ", folder, " : " + str(accuracy))

    print("Random Forest accuracy mean : ", np.mean(np.asarray(lst_accuracy)))

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
